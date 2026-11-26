"""Training-only DAgger (Dataset Aggregation) trainer for Tiers 1-4.

Uses a privileged training-only room-graph expert (`generate_map_with_meta` +
`env.privileged_state()`) to teach `SpatialDoomNet` exact long-horizon Tier 3/4
progression (keycard -> backtrack to locked door -> unlock -> exit) plus combat
and door discipline. At evaluation, `submission/agent.py` uses ONLY pure neural weights.
"""

from __future__ import annotations

import argparse
import math
import random
import sys
import tempfile
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = ROOT / "submission"
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "policy.pt"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from doom_env import make_env
from model import NUM_ACTIONS, SpatialDoomNet, preprocess_obs
from training.wad_gen import build_wad, generate_map_with_meta


class TrainingOracleExpert:
    """Privileged BFS + combat expert used ONLY to label DAgger training trajectories."""

    def __init__(self, meta: dict) -> None:
        self.adj = meta["adj"]
        self.edge_kind = meta["edge_kind"]
        self.key_placements = meta["key_placements"]
        self.exit_room = meta["exit_room"]
        self.room_centers = meta["room_centers"]
        self.stride = meta["stride"]
        self.use_cd = 0
        self.strafe_dir = 1

    def _nearest_room(self, x: float, y: float) -> tuple[int, int]:
        return min(
            self.room_centers,
            key=lambda r: (self.room_centers[r][0] - x) ** 2 + (self.room_centers[r][1] - y) ** 2,
        )

    def _bfs_next_room(self, src: tuple[int, int], dst: tuple[int, int], has_key: bool) -> tuple[int, int] | None:
        if src == dst:
            return dst
        q = deque([src])
        parent: dict[tuple[int, int], tuple[int, int] | None] = {src: None}
        while q:
            u = q.popleft()
            if u == dst:
                break
            for v in self.adj.get(u, []):
                if v in parent:
                    continue
                ek = self.edge_kind[(min(u, v), max(u, v))][0]
                if ek == "lock" and not has_key:
                    continue
                parent[v] = u
                q.append(v)
        if dst not in parent:
            return None
        cur = dst
        while parent[cur] is not None and parent[cur] != src:
            cur = parent[cur]
        return cur

    def act(self, obs: dict, priv: dict, step_i: int) -> np.ndarray:
        action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        px, py = float(priv["x"]), float(priv["y"])
        ang = math.radians(float(priv["angle"]))
        has_key = int(obs["vitals"][8:14].sum()) > 0

        # 1. Pick high-level goal: keycard first, then exit
        cur_room = self._nearest_room(px, py)
        goal_room = self.exit_room
        if self.key_placements and not has_key:
            goal_room = self.key_placements[0][0]

        nxt_room = self._bfs_next_room(cur_room, goal_room, has_key=has_key)
        if nxt_room is None:
            nxt_room = cur_room

        # Target waypoint: if we're in cur_room and nxt_room != cur_room, aim toward midpoint (door/hallway)
        # until we cross past the midpoint, then aim at nxt_room center
        cx, cy = self.room_centers[cur_room]
        nx, ny = self.room_centers[nxt_room]
        if nxt_room != cur_room:
            mx, my = (cx + nx) * 0.5, (cy + ny) * 0.5
            # Check if we are still on cur_room's side of the doorway
            d_cur = (px - cx) ** 2 + (py - cy) ** 2
            d_mid = (px - mx) ** 2 + (py - my) ** 2
            tx, ty = (mx, my) if d_mid > 48.0**2 and d_cur < (nx - cx) ** 2 + (ny - cy) ** 2 else (nx, ny)
        else:
            tx, ty = nx, ny

        # 2. Check for visible enemy in combat range
        ent_m = obs["visible_entities"]["mask"]
        ent_t = obs["visible_entities"]["entity_type"]
        ent_f = obs["visible_entities"]["features"]
        best_enemy = None
        for i in range(len(ent_m)):
            if ent_m[i] == 1 and ent_t[i] in (1, 2, 3):
                dist_u = float(ent_f[i, 0]) * 1024.0
                if dist_u < 480.0:
                    best_enemy = (dist_u, float(ent_f[i, 1]), float(ent_f[i, 2]))
                    break

        if best_enemy is not None and obs["vitals"][3] > 0:
            dist_u, s_ang, c_ang = best_enemy
            # Aim toward enemy & shoot when aligned
            if s_ang > 0.12:
                action[4] = 1  # TURN_LEFT
            elif s_ang < -0.12:
                action[5] = 1  # TURN_RIGHT
            if abs(s_ang) < 0.25 and c_ang > 0.75:
                action[6] = 1  # ATTACK
            # Circle-strafe while advancing/fighting
            if step_i % 20 == 0:
                self.strafe_dir *= -1
            action[2 if self.strafe_dir > 0 else 3] = 1
            if dist_u > 220.0:
                action[0] = 1
        else:
            # 3. Navigate toward (tx, ty)
            desired = math.atan2(ty - py, tx - px)
            diff = (desired - ang + math.pi) % (2.0 * math.pi) - math.pi
            if diff > 0.15:
                action[4] = 1  # TURN_LEFT
            elif diff < -0.15:
                action[5] = 1  # TURN_RIGHT
            if abs(diff) < 1.1:
                action[0] = 1  # MOVE_FORWARD

        # 4. Door USE discipline: press USE once when facing a closed door/locked_door
        self.use_cd = max(0, self.use_cd - 1)
        r_types = obs["rays"]["hit_type"]
        r_dists = obs["rays"]["distance"]
        front_rays = [0, 1, 2, 62, 63]
        door_ahead = any(
            (r_types[ri] == 2 or (r_types[ri] == 3 and has_key)) and r_dists[ri] * 1024.0 < 120.0
            for ri in front_rays
        )
        if door_ahead and self.use_cd == 0:
            action[7] = 1  # USE
            self.use_cd = 12  # wait ~3s before pressing USE again

        # 5. Wall unstick strafe if velocity is very low while trying to move forward
        spd = math.hypot(float(obs["vitals"][14]), float(obs["vitals"][15]))
        if action[0] == 1 and spd < 1.5 and not door_ahead:
            action[2 if (step_i // 8) % 2 == 0 else 3] = 1

        return action


def train_dagger(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    net = SpatialDoomNet(hidden_dim=256).to(device)
    if args.resume and WEIGHTS_PATH.exists():
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt, strict=False)
        print(f"[DAgger] Resumed weights from {WEIGHTS_PATH}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    # Replay buffer of whole episodes (truncated to chunks of length T for BPTT)
    dataset: list[dict[str, torch.Tensor]] = []
    tiers = args.tiers or [1, 2, 3, 4]
    t0 = time.perf_counter()
    total_steps = 0

    with tempfile.TemporaryDirectory() as td:
        for it in range(1, args.iters + 1):
            beta = max(0.20, 1.0 - 0.80 * ((it - 1) / max(1, args.iters - 1)))
            exits, keys_got, kills_tot, ep_cnt = 0, 0, 0, 0

            # Collect `episodes_per_iter` episodes across tiers
            for ep in range(args.episodes_per_iter):
                tier = tiers[(it * args.episodes_per_iter + ep) % len(tiers)]
                seed = args.seed + it * 100 + ep
                _, meta = generate_map_with_meta(seed, tier)
                wad = Path(td) / f"dag_{it}_{ep}.wad"
                build_wad(wad, [(seed, tier)])
                ticks = {1: 2100, 2: 2100, 3: 3500, 4: 4200}.get(tier, 2800)

                env = make_env(wad, "MAP01", episode_ticks=ticks, render_buffers=False)
                obs, _ = env.reset(seed=seed)
                expert = TrainingOracleExpert(meta)

                hx = torch.zeros(1, 1, net.hidden_dim, device=device)
                grid = net.grid_mem.init_grid(1, device)
                prev_act = np.zeros(NUM_ACTIONS, dtype=np.int8)

                ep_obs, ep_exp = [], []
                net.eval()
                for step_i in range(ticks // 4):
                    priv = env.privileged_state()
                    exp_act = expert.act(obs, priv, step_i)

                    obs_t = preprocess_obs(obs, prev_act, device)
                    with torch.no_grad():
                        logits, _, hx, grid, _, _ = net.forward_step(obs_t, hx, grid)
                        stud_act = torch.bernoulli(torch.sigmoid(logits / 0.75)).squeeze(0).to(torch.int8).cpu().numpy()

                    act_exec = exp_act if random.random() < beta else stud_act
                    ep_obs.append({k: v.squeeze(0).cpu() for k, v in obs_t.items()})
                    ep_exp.append(torch.from_numpy(exp_act.astype(np.float32)))

                    obs, _, term, trunc, info = env.step(act_exec)
                    prev_act = act_exec
                    total_steps += 1
                    if term or trunc:
                        exits += int(info["exit_reached"])
                        keys_got += int(obs["vitals"][8:14].sum())
                        kills_tot += int(env.privileged_state()["kills"])
                        ep_cnt += 1
                        break
                env.close()

                # Chunk episode into T=64 sequences
                T = 64
                for s in range(0, len(ep_obs), T):
                    chunk_o = ep_obs[s : s + T]
                    chunk_e = ep_exp[s : s + T]
                    if len(chunk_o) < 16:
                        continue
                    dataset.append(
                        {
                            "ray_dist": torch.stack([x["ray_dist"] for x in chunk_o]),
                            "ray_type": torch.stack([x["ray_type"] for x in chunk_o]),
                            "ent_feat": torch.stack([x["ent_feat"] for x in chunk_o]),
                            "ent_type": torch.stack([x["ent_type"] for x in chunk_o]),
                            "ent_mask": torch.stack([x["ent_mask"] for x in chunk_o]),
                            "vitals": torch.stack([x["vitals"] for x in chunk_o]),
                            "prev_act": torch.stack([x["prev_act"] for x in chunk_o]),
                            "expert": torch.stack(chunk_e),
                        }
                    )
                if len(dataset) > 800:
                    dataset = dataset[-800:]

            # Supervised BPTT updates on aggregated dataset
            net.train()
            losses = []
            for _ in range(args.train_steps_per_iter):
                batch = random.sample(dataset, k=min(16, len(dataset)))
                # Pad/trim to min len in batch for fast batched BPTT
                L = min(x["ray_dist"].shape[0] for x in batch)
                B = len(batch)
                hx = torch.zeros(1, B, net.hidden_dim, device=device)
                grid = net.grid_mem.init_grid(B, device)

                loss_sum = 0.0
                for t in range(L):
                    obs_t = {
                        k: torch.stack([b[k][t] for b in batch]).to(device)
                        for k in ("ray_dist", "ray_type", "ent_feat", "ent_type", "ent_mask", "vitals", "prev_act")
                    }
                    tgt = torch.stack([b["expert"][t] for b in batch]).to(device)
                    logits, _, hx, grid, _, _ = net.forward_step(obs_t, hx, grid)
                    loss_sum = loss_sum + F.binary_cross_entropy_with_logits(logits, tgt)
                loss = loss_sum / L
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                optimizer.step()
                losses.append(float(loss.item()))

            dt = time.perf_counter() - t0
            print(
                f"[DAgger {it}/{args.iters} | Steps {total_steps} | beta={beta:.2f} | {dt:.0f}s] "
                f"DemoExit={exits}/{ep_cnt} ({100*exits/max(1,ep_cnt):.0f}%) | Keys={keys_got} | Kills={kills_tot} | BCE={np.mean(losses):.3f}"
            )
            WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": net.state_dict(), "step": total_steps}, WEIGHTS_PATH)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=15)
    ap.add_argument("--episodes-per-iter", type=int, default=8)
    ap.add_argument("--train-steps-per-iter", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--tiers", type=int, nargs="+", default=[2, 3, 4])
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    train_dagger(args)


if __name__ == "__main__":
    main()
