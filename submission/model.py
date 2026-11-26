"""Custom spatial-memory neural architecture for zero-shot DOOM navigation & combat.

Combines:
  1. Circular 1D-CNN over 64 ego-centric rays (with hit-type embeddings & trig angles).
  2. Masked Multi-Head Self-Attention over variable-length visible entities (up to 32).
  3. Dead-reckoned 15x15 egocentric semantic memory grid (visited, wall, door,
     locked_door, enemy, item) updated from raycasts + velocity/turn dead reckoning.
  4. GRU recurrent core + 9-way Bernoulli actor, value head, and training-only
     auxiliary heads (DFP-style future vitals & privileged loop-closure).
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_RAYS = 64
RAY_RANGE = 1024.0
NUM_HIT_TYPES = 6
MAX_ENTITIES = 32
NUM_ENTITY_TYPES = 12
VITALS_DIM = 17
NUM_ACTIONS = 9

# Dead-reckoned egocentric semantic grid constants
GRID_SIZE = 15
GRID_CELL_UNITS = 128.0  # covers ~1920 x 1920 map units around the player
GRID_CHANNELS = 6  # [visited, wall, door, locked_door, enemy, item]

# Approximate Doom turn angle per 4-tic step when TURN_LEFT / TURN_RIGHT is held
# (Doom's normal turn speed is ~10-11.25 deg/step at 4 tics; calibrated for grid rotation)
TURN_RAD_PER_STEP = math.radians(11.25)
TICS_PER_STEP = 4.0


def preprocess_obs(obs: dict, prev_action: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    """Convert a single `doom_env` observation dict + prev_action into batched (B=1) tensors."""
    ray_dist = torch.as_tensor(obs["rays"]["distance"], dtype=torch.float32, device=device).unsqueeze(0)
    ray_type = torch.as_tensor(obs["rays"]["hit_type"], dtype=torch.long, device=device).unsqueeze(0)
    ent_feat = torch.as_tensor(obs["visible_entities"]["features"], dtype=torch.float32, device=device).unsqueeze(0)
    ent_type = torch.as_tensor(obs["visible_entities"]["entity_type"], dtype=torch.long, device=device).unsqueeze(0)
    ent_mask = torch.as_tensor(obs["visible_entities"]["mask"], dtype=torch.float32, device=device).unsqueeze(0)
    vitals = torch.as_tensor(obs["vitals"], dtype=torch.float32, device=device).unsqueeze(0)
    prev_act = torch.as_tensor(prev_action, dtype=torch.float32, device=device).unsqueeze(0)
    return {
        "ray_dist": ray_dist,
        "ray_type": ray_type,
        "ent_feat": ent_feat,
        "ent_type": ent_type,
        "ent_mask": ent_mask,
        "vitals": vitals,
        "prev_act": prev_act,
    }


def normalize_vitals(vitals: torch.Tensor) -> torch.Tensor:
    """Scale raw `vitals` (B, 17) into well-behaved neural network ranges."""
    out = vitals.clone()
    out[..., 0] = out[..., 0] / 100.0  # health
    out[..., 1] = out[..., 1] / 100.0  # armor
    out[..., 2] = out[..., 2] / 7.0    # selected_weapon
    out[..., 3:8] = torch.clamp(out[..., 3:8] / 50.0, 0.0, 4.0)  # ammo
    # 8:14 are 0/1 key flags
    out[..., 14:17] = torch.clamp(out[..., 14:17] / 20.0, -2.0, 2.0)  # velocities
    return out


class EgocentricGridMemory(nn.Module):
    """Differentiable/vectorized dead-reckoned 15x15 egocentric semantic map."""

    def __init__(self, grid_size: int = GRID_SIZE, cell_units: float = GRID_CELL_UNITS) -> None:
        super().__init__()
        self.grid_size = grid_size
        self.cell_units = cell_units
        self.half = grid_size // 2

        angles = torch.arange(NUM_RAYS, dtype=torch.float32) * (2.0 * math.pi / NUM_RAYS)
        # In player frame: +x = forward, +y = left (matches ray index 0=ahead, 16=left)
        self.register_buffer("ray_cos", torch.cos(angles), persistent=False)
        self.register_buffer("ray_sin", torch.sin(angles), persistent=False)

        # Base normalized sampling coordinates for grid_sample: (1, H, W, 2)
        lin = torch.linspace(-1.0, 1.0, grid_size)
        gy, gx = torch.meshgrid(lin, lin, indexing="ij")
        self.register_buffer("base_gx", gx.unsqueeze(0), persistent=False)
        self.register_buffer("base_gy", gy.unsqueeze(0), persistent=False)

    def init_grid(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros(batch_size, GRID_CHANNELS, self.grid_size, self.grid_size, device=device)

    @torch.no_grad()
    def update(
        self,
        grid: torch.Tensor,
        ray_dist: torch.Tensor,
        ray_type: torch.Tensor,
        vitals: torch.Tensor,
        prev_act: torch.Tensor,
    ) -> torch.Tensor:
        """Shift/rotate `grid` by dead-reckoned motion and write current rays into it."""
        B = grid.shape[0]
        device = grid.device
        # 1. Dead-reckoning warp (in-place-safe out-of-place grid_sample)
        # Forward/right velocity in map units per step
        vel_fwd = vitals[:, 14] * TICS_PER_STEP
        vel_right = vitals[:, 15] * TICS_PER_STEP
        dtheta = (prev_act[:, 4] - prev_act[:, 5]) * TURN_RAD_PER_STEP  # +left, -right

        # Convert world shift to normalized [-1, 1] grid coordinates
        half_span = (self.grid_size * 0.5) * self.cell_units
        dx_norm = (vel_fwd / half_span).view(B, 1, 1)
        dy_norm = (-vel_right / half_span).view(B, 1, 1)  # +y in image = left/right

        cos_t = torch.cos(dtheta).view(B, 1, 1)
        sin_t = torch.sin(dtheta).view(B, 1, 1)

        gx = self.base_gx.expand(B, -1, -1)
        gy = self.base_gy.expand(B, -1, -1)
        # Inverse warp: sample old grid at rotated+translated coords
        src_x = cos_t * gx - sin_t * gy + dx_norm
        src_y = sin_t * gx + cos_t * gy + dy_norm
        sample_grid = torch.stack([src_x, src_y], dim=-1)

        warped = F.grid_sample(
            grid, sample_grid, mode="bilinear", padding_mode="zeros", align_corners=True
        )
        # Preserve static map structure (visited, walls, doors, locked doors); decay dynamic entities
        decay = torch.tensor([0.9995, 0.999, 0.9995, 1.0, 0.70, 0.95], device=device).view(1, GRID_CHANNELS, 1, 1)
        warped = warped * decay

        # 2. Mark player's current center cell as visited (channel 0)
        warped[:, 0, self.half, self.half] = 1.0

        # 3. Project 64 rays onto grid cells (vectorized scatter)
        dist_units = ray_dist * RAY_RANGE  # (B, 64)
        px = dist_units * self.ray_cos.unsqueeze(0)  # +forward -> +x
        py = -dist_units * self.ray_sin.unsqueeze(0)  # +left -> -y in grid

        ix = torch.clamp(torch.round(px / self.cell_units).long() + self.half, 0, self.grid_size - 1)
        iy = torch.clamp(torch.round(py / self.cell_units).long() + self.half, 0, self.grid_size - 1)

        # Map hit_type (1=wall, 2=door, 3=locked_door, 4=enemy, 5=item) -> channel (1..5)
        valid = (ray_type > 0) & (ray_dist < 0.999)
        b_idx = torch.arange(B, device=device).unsqueeze(1).expand_as(ix)
        ch_idx = torch.clamp(ray_type, 1, 5)

        b_flat = b_idx[valid]
        c_flat = ch_idx[valid]
        y_flat = iy[valid]
        x_flat = ix[valid]
        warped[b_flat, c_flat, y_flat, x_flat] = 1.0

        return warped


class SpatialDoomNet(nn.Module):
    """Compact, custom spatial-memory actor-critic network."""

    def __init__(self, hidden_dim: int = 256) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim

        # 1. Circular 1D-CNN over 64 rays
        angles = torch.arange(NUM_RAYS, dtype=torch.float32) * (2.0 * math.pi / NUM_RAYS)
        self.register_buffer("ray_trig", torch.stack([torch.cos(angles), torch.sin(angles)], dim=0).unsqueeze(0), persistent=False)
        self.ray_type_emb = nn.Embedding(NUM_HIT_TYPES, 8)
        # Input channels: 1 (dist) + 8 (hit_type) + 2 (cos, sin) = 11
        self.ray_conv = nn.Sequential(
            nn.Conv1d(11, 32, kernel_size=5, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(32, 48, kernel_size=5, stride=2, padding=2, padding_mode="circular"),
            nn.ReLU(),
            nn.Conv1d(48, 64, kernel_size=3, stride=2, padding=1, padding_mode="circular"),
            nn.ReLU(),
        )
        # 64 -> 32 -> 16; adaptive pool to 4 quadrants (front/left/back/right) -> 64 * 4 = 256 -> 96
        self.ray_fc = nn.Sequential(nn.Linear(64 * 4, 96), nn.ReLU())

        # 2. Relational Entity Self-Attention (handles 0..32 entities)
        self.ent_type_emb = nn.Embedding(NUM_ENTITY_TYPES, 12)
        self.ent_proj = nn.Sequential(nn.Linear(5 + 12, 64), nn.ReLU())
        self.ent_attn = nn.MultiheadAttention(embed_dim=64, num_heads=4, batch_first=True)
        self.ent_norm = nn.LayerNorm(64)
        self.ent_fc = nn.Sequential(nn.Linear(128, 64), nn.ReLU())

        # 3. Egocentric Semantic Memory Grid + CNN reader
        self.grid_mem = EgocentricGridMemory()
        self.grid_cnn = nn.Sequential(
            nn.Conv2d(GRID_CHANNELS, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 15 -> 7
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((3, 3)),  # 32 * 9 = 288
            nn.Flatten(),
            nn.Linear(32 * 9, 64),
            nn.ReLU(),
        )

        # 4. Vitals + previous action encoder
        self.vitals_fc = nn.Sequential(nn.Linear(VITALS_DIM + NUM_ACTIONS, 48), nn.ReLU())

        # 5. Recurrent GRU core
        fusion_dim = 96 + 64 + 64 + 48  # 272
        self.fusion_fc = nn.Sequential(nn.Linear(fusion_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)

        # 6. Policy (9 Bernoullis) & Value heads
        self.actor = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.critic = nn.Linear(hidden_dim, 1)

        # 7. Training-only auxiliary heads
        # Predicts delta vitals (health, armor, ammo, 6 keys = 9 dims)
        self.aux_future_vitals = nn.Linear(hidden_dim, 9)
        # Predicts loop-closure ("visited near here before?")
        self.aux_loop_closure = nn.Linear(hidden_dim, 1)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv1d, nn.Conv2d)):
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def encode_step(
        self,
        ray_dist: torch.Tensor,
        ray_type: torch.Tensor,
        ent_feat: torch.Tensor,
        ent_type: torch.Tensor,
        ent_mask: torch.Tensor,
        vitals: torch.Tensor,
        prev_act: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        B = ray_dist.shape[0]

        # Rays: (B, 11, 64)
        rt_emb = self.ray_type_emb(ray_type).permute(0, 2, 1)  # (B, 8, 64)
        r_in = torch.cat([ray_dist.unsqueeze(1), rt_emb, self.ray_trig.expand(B, -1, -1)], dim=1)
        r_conv = self.ray_conv(r_in)  # (B, 64, 16)
        r_quad = F.adaptive_avg_pool1d(r_conv, 4).reshape(B, -1)
        ray_z = self.ray_fc(r_quad)

        # Entities: (B, 32, 64)
        et_emb = self.ent_type_emb(ent_type)
        e_in = self.ent_proj(torch.cat([ent_feat, et_emb], dim=-1))
        # key_padding_mask: True where padded (mask == 0)
        pad_mask = ent_mask < 0.5
        all_padded = pad_mask.all(dim=1)
        # Avoid NaN in attention when all entities are masked out
        safe_pad = pad_mask.clone()
        safe_pad[all_padded, 0] = False
        attn_out, _ = self.ent_attn(e_in, e_in, e_in, key_padding_mask=safe_pad, need_weights=False)
        e_h = self.ent_norm(e_in + attn_out) * ent_mask.unsqueeze(-1)
        count = ent_mask.sum(dim=1, keepdim=True).clamp(min=1.0)
        e_mean = e_h.sum(dim=1) / count
        e_max = e_h.masked_fill(pad_mask.unsqueeze(-1), -1e4).max(dim=1).values
        e_max = torch.where(all_padded.unsqueeze(-1), torch.zeros_like(e_max), e_max)
        ent_z = self.ent_fc(torch.cat([e_mean, e_max], dim=-1))

        # Grid update + CNN read
        new_grid = self.grid_mem.update(grid, ray_dist, ray_type, vitals, prev_act)
        grid_z = self.grid_cnn(new_grid)

        # Vitals + prev_act
        v_norm = normalize_vitals(vitals)
        vit_z = self.vitals_fc(torch.cat([v_norm, prev_act], dim=-1))

        fused = self.fusion_fc(torch.cat([ray_z, ent_z, grid_z, vit_z], dim=-1))
        return fused, new_grid

    def forward_step(
        self,
        obs_t: dict[str, torch.Tensor],
        hx: torch.Tensor,
        grid: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward for inference & rollout collection."""
        fused, new_grid = self.encode_step(
            obs_t["ray_dist"],
            obs_t["ray_type"],
            obs_t["ent_feat"],
            obs_t["ent_type"],
            obs_t["ent_mask"],
            obs_t["vitals"],
            obs_t["prev_act"],
            grid,
        )
        gru_out, new_hx = self.gru(fused.unsqueeze(1), hx)
        h = gru_out.squeeze(1)
        logits = self.actor(h)
        value = self.critic(h).squeeze(-1)
        aux_vitals = self.aux_future_vitals(h)
        aux_loop = self.aux_loop_closure(h).squeeze(-1)
        return logits, value, new_hx, new_grid, aux_vitals, aux_loop

    def forward_sequence(
        self,
        fused_seq: torch.Tensor,
        hx_init: torch.Tensor,
        dones_seq: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sequence forward (B, T, D) respecting episode `dones` resets for BPTT."""
        B, T, _ = fused_seq.shape
        hx = hx_init
        outs = []
        for t in range(T):
            # Reset hidden state where previous step terminated
            if t > 0:
                alive = (1.0 - dones_seq[:, t - 1]).view(1, B, 1)
                hx = hx * alive
            out_t, hx = self.gru(fused_seq[:, t : t + 1, :], hx)
            outs.append(out_t)
        h_seq = torch.cat(outs, dim=1)  # (B, T, H)
        logits = self.actor(h_seq)
        values = self.critic(h_seq).squeeze(-1)
        aux_vitals = self.aux_future_vitals(h_seq)
        aux_loop = self.aux_loop_closure(h_seq).squeeze(-1)
        return logits, values, aux_vitals, aux_loop
