"""Render side-by-side gameplay + live 15x15 EgocentricGridMemory videos (.mp4 / .gif)."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.agent import DoomAgent
from training.wad_gen import build_wad

OUT_DIR = ROOT / "training" / "demos"

# BGR colors for memory grid channels: [visited, wall, door, locked_door, enemy, item]
GRID_COLORS = np.array(
    [
        [80, 50, 30],     # 0: visited (slate blue)
        [200, 200, 200],  # 1: wall (light gray)
        [255, 220, 40],   # 2: door (cyan)
        [60, 40, 255],    # 3: locked_door (red)
        [40, 140, 255],   # 4: enemy (orange)
        [80, 230, 80],    # 5: item (green)
    ],
    dtype=np.float32,
)


def render_grid_panel(grid_tensor, size: int = 360) -> np.ndarray:
    """Convert (1, 6, 15, 15) grid tensor into a crisp RGB panel."""
    g = grid_tensor.squeeze(0).detach().cpu().numpy()  # (6, 15, 15)
    canvas = np.full((15, 15, 3), 18.0, dtype=np.float32)
    for ch in range(6):
        w = np.clip(g[ch], 0.0, 1.0)[..., None]
        canvas = np.maximum(canvas, w * GRID_COLORS[ch])
    canvas = np.clip(canvas, 0, 255).astype(np.uint8)
    # Mark player at center (7, 7)
    canvas[7, 7] = [255, 255, 255]
    img = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_NEAREST)
    # Draw subtle cell grid lines
    step = size // 15
    for i in range(1, 15):
        cv2.line(img, (i * step, 0), (i * step, size), (35, 35, 35), 1)
        cv2.line(img, (0, i * step), (size, i * step), (35, 35, 35), 1)
    cv2.putText(img, "EgoGrid 15x15", (10, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (230, 230, 230), 1, cv2.LINE_AA)
    return img


def record_episode(wad_path: str | Path, map_name: str, label: str, out_stem: Path, ticks: int = 2100) -> None:
    env = make_env(wad_path, map_name, episode_ticks=ticks, render_buffers=True)
    agent = DoomAgent(env.observation_space, env.action_space, "cpu")
    obs, _ = env.reset(seed=0)
    agent.reset()

    frames_bgr: list[np.ndarray] = []
    outcome = "tick_cap"

    for step_i in range(ticks // 4):
        act = agent.act(obs)
        rgb = obs["rgb"]  # (120, 160, 3) RGB
        bgr_view = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        bgr_view = cv2.resize(bgr_view, (480, 360), interpolation=cv2.INTER_NEAREST)
        grid_bgr = render_grid_panel(agent.grid, size=360)

        top = np.hstack([bgr_view, grid_bgr])  # (360, 840, 3)
        hud = np.full((42, top.shape[1], 3), 24, dtype=np.uint8)
        priv = env.privileged_state()
        hp = int(priv.get("health", 0))
        kills = int(priv.get("kills", 0))
        keys = int(obs["vitals"][8:14].sum())
        hud_txt = f"{label} | Step {step_i:03d} | HP: {hp} | Kills: {kills} | Keys: {keys}"
        cv2.putText(hud, hud_txt, (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (240, 240, 240), 1, cv2.LINE_AA)
        frame = np.vstack([top, hud])
        frames_bgr.append(frame)

        obs, _, term, trunc, info = env.step(act)
        if term or trunc:
            outcome = "EXIT" if info["exit_reached"] else ("DEAD" if term else "TIMEOUT")
            # Render final frame + outcome banner so the ending is clearly visible
            fin_bgr = cv2.resize(cv2.cvtColor(obs["rgb"], cv2.COLOR_RGB2BGR), (480, 360), interpolation=cv2.INTER_NEAREST)
            fin_grid = render_grid_panel(agent.grid, size=360)
            fin_top = np.hstack([fin_bgr, fin_grid])
            banner_col = (40, 200, 60) if outcome == "EXIT" else ((40, 40, 220) if outcome == "DEAD" else (40, 180, 220))
            cv2.putText(fin_top, f"OUTCOME: {outcome}", (110, 185), cv2.FONT_HERSHEY_SIMPLEX, 1.1, banner_col, 3, cv2.LINE_AA)
            fin_hud = np.full((42, fin_top.shape[1], 3), 24, dtype=np.uint8)
            priv = env.privileged_state()
            cv2.putText(
                fin_hud,
                f"{label} | Step {step_i+1:03d} | HP: {int(priv.get('health', 0))} | Kills: {int(priv.get('kills', 0))} | {outcome}",
                (14, 27), cv2.FONT_HERSHEY_SIMPLEX, 0.6, banner_col, 2, cv2.LINE_AA,
            )
            fin_frame = np.vstack([fin_top, fin_hud])
            for _ in range(12):
                frames_bgr.append(fin_frame)
            break

    env.close()

    # Save MP4
    out_mp4 = out_stem.with_suffix(".mp4")
    h, w, _ = frames_bgr[0].shape
    writer = cv2.VideoWriter(str(out_mp4), cv2.VideoWriter_fourcc(*"mp4v"), 12.0, (w, h))
    for f in frames_bgr:
        writer.write(f)
    writer.release()

    # Save lightweight GIF (every 2nd frame)
    out_gif = out_stem.with_suffix(".gif")
    pil_frames = [Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_bgr[::2]]
    pil_frames[0].save(
        out_gif,
        save_all=True,
        append_images=pil_frames[1:],
        duration=120,
        loop=0,
    )
    print(f"Saved {out_mp4.name} & {out_gif.name}: {len(frames_bgr)} steps -> {outcome}")


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    record_episode(ROOT / "maps" / "two_rooms.wad", "MAP01", "Starter: two_rooms", OUT_DIR / "demo_two_rooms", ticks=1200)

    with tempfile.TemporaryDirectory() as td:
        wad = Path(td) / "demo_maps.wad"
        mnames = build_wad(wad, [(2000, 1), (2001, 2), (2002, 3), (2003, 4)])
        record_episode(wad, mnames[0], "Unseen Tier 1 (seed=2000)", OUT_DIR / "demo_tier1", ticks=2100)
        record_episode(wad, mnames[1], "Unseen Tier 2 (seed=2001)", OUT_DIR / "demo_tier2", ticks=2100)
        record_episode(wad, mnames[2], "Unseen Tier 3 (seed=2002)", OUT_DIR / "demo_tier3", ticks=3500)
        record_episode(wad, mnames[3], "Unseen Tier 4 (seed=2003)", OUT_DIR / "demo_tier4", ticks=4200)


if __name__ == "__main__":
    main()
