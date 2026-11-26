"""Evaluation entry point. The scorer instantiates DoomAgent from this file.

Contract: docs/CONTRACT.md §6.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch

SUBMISSION_DIR = Path(__file__).resolve().parent
WEIGHTS_DIR = SUBMISSION_DIR / "weights"
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from model import NUM_ACTIONS, SpatialDoomNet, preprocess_obs


class DoomAgent:
    def __init__(self, observation_space, action_space, device: str):
        """Called once. Load weights from WEIGHTS_DIR onto `device` ("cuda" or "cpu")."""
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = torch.device(device)

        self.net = SpatialDoomNet(hidden_dim=256).to(self.device)
        ckpt_path = WEIGHTS_DIR / "policy.pt"
        if ckpt_path.exists():
            state = torch.load(ckpt_path, map_location=self.device, weights_only=True)
            model_state = state["model"] if isinstance(state, dict) and "model" in state else state
            self.net.load_state_dict(model_state)
        self.net.eval()

        self.hx: torch.Tensor | None = None
        self.grid: torch.Tensor | None = None
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        self.temp = float(os.environ.get("DOOM_EVAL_TEMP", "0.75"))
        self.reset()

    def reset(self) -> None:
        """Called before every map. Clear recurrent state, memory buffers and trackers here."""
        self.hx = torch.zeros(1, 1, self.net.hidden_dim, dtype=torch.float32, device=self.device)
        self.grid = self.net.grid_mem.init_grid(1, self.device)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)

    @torch.inference_mode()
    def act(self, observation: dict) -> np.ndarray:
        """Return a MultiBinary action: int8 array of shape (9,) with values in {0, 1}."""
        obs_t = preprocess_obs(observation, self.prev_action, self.device)
        logits, _, self.hx, self.grid, _, _ = self.net.forward_step(obs_t, self.hx, self.grid)
        probs = torch.sigmoid(logits / self.temp).squeeze(0)
        action = torch.bernoulli(probs).to(torch.int8).cpu().numpy()
        self.prev_action = action
        return action
