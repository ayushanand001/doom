"""Recurrent PPO + Auxiliary Tasks + PLR training pipeline for SpatialDoomNet.

Trains on `doom_env` (render_buffers=False) across 495+ procedurally generated
Tiers 1-4 WADs with:
  - Parallel multiprocessing workers (CPU-friendly, thousands of steps/sec)
  - Prioritized Level Replay (PLR) over map seeds/tiers
  - Reward shaping (exit, keycards, kills, visited-area exploration, USE anti-spam)
  - CleanRL-style Recurrent PPO (9 independent Bernoullis, GAE, clipped value/policy)
  - Auxiliary losses: DFP-style future-vitals prediction + privileged loop-closure
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_DIR = ROOT / "submission"
WEIGHTS_PATH = SUBMISSION_DIR / "weights" / "policy.pt"
GEN_MAPS_DIR = ROOT / "training" / "generated_maps"

if str(SUBMISSION_DIR) not in sys.path:
    sys.path.insert(0, str(SUBMISSION_DIR))

from model import NUM_ACTIONS, SpatialDoomNet


def _discover_maps(tiers: set[int] | None = None) -> list[tuple[str, str]]:
    """Return list of (wad_path, map_name) across generated packs + two_rooms.wad."""
    maps: list[tuple[str, str]] = []
    if tiers is None or 1 in tiers:
        maps.append((str(ROOT / "maps" / "two_rooms.wad"), "MAP01"))
    if GEN_MAPS_DIR.exists():
        from doom_env.wad import list_maps, read_wad

        for pack_idx, wad in enumerate(sorted(GEN_MAPS_DIR.glob("*.wad"))):
            for m_idx, mname in enumerate(list_maps(read_wad(wad))):
                seed = pack_idx * 99 + m_idx
                tier = (seed % 4) + 1
                if tiers is None or tier in tiers:
                    maps.append((str(wad), mname))
    return maps


def _worker_loop(
    worker_id: int,
    conn,
    map_pool: list[tuple[str, str]],
    episode_ticks: int,
    seed: int,
) -> None:
    """Subprocess running one `doom_env` instance with reward shaping & map switching."""
    from doom_env import make_env

    rng = random.Random(seed + worker_id * 1009)
    env = None
    cur_map = None

    # Per-episode tracking for reward shaping & aux targets
    visited_cells: dict[tuple[int, int], int] = {}
    opened_doors: set[int] = set()
    prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
    use_cooldown = 0
    prev_kills = 0
    prev_dmg = 0.0
    prev_keys = 0
    prev_items = 0
    prev_hp = 100.0
    step_idx = 0

    def load_map(wad_path: str, map_name: str):
        nonlocal env, cur_map
        if env is not None:
            env.close()
        env = make_env(wad_path, map_name, episode_ticks=episode_ticks, render_buffers=False)
        cur_map = (wad_path, map_name)

    def reset_episode(map_idx: int):
        nonlocal visited_cells, opened_doors, prev_action, use_cooldown, prev_kills, prev_dmg, prev_keys, prev_items, prev_hp, step_idx
        wad_path, map_name = map_pool[map_idx]
        if cur_map != (wad_path, map_name):
            load_map(wad_path, map_name)
        obs, _ = env.reset(seed=rng.randint(0, 2**30))
        priv = env.privileged_state()
        visited_cells = {(int(priv["x"] // 96), int(priv["y"] // 96)): 0}
        opened_doors = set()
        prev_action = np.zeros(NUM_ACTIONS, dtype=np.int8)
        use_cooldown = 0
        prev_kills = int(priv.get("kills", 0))
        prev_dmg = float(priv.get("damage_taken", 0.0))
        prev_keys = int(sum(int(k) for k in priv.get("keys", []))) if isinstance(priv.get("keys"), (list, tuple, np.ndarray)) else int(obs["vitals"][8:14].sum())
        prev_items = int(priv.get("items", 0))
        prev_hp = float(priv.get("health", 100.0))
        step_idx = 0
        return obs, priv

    try:
        while True:
            cmd, payload = conn.recv()
            if cmd == "close":
                break
            elif cmd == "reset":
                obs, _ = reset_episode(int(payload))
                conn.send((obs, prev_action.copy()))
            elif cmd == "step":
                action, next_map_idx = payload
                action = np.asarray(action, dtype=np.int8)
                obs, _, term, trunc, info = env.step(action)
                priv = env.privileged_state()
                step_idx += 1

                # --- Reward shaping ---
                reward = -0.002  # tiny living cost to encourage brisk progression
                if info.get("exit_reached", False):
                    frac_left = max(0.0, 1.0 - priv["tic"] / max(1, episode_ticks))
                    hp_bonus = 0.02 * max(0.0, float(priv.get("health", 0.0)))
                    reward += 10.0 + 2.0 * frac_left + hp_bonus
                if priv.get("dead", False):
                    reward -= 5.0

                kills = int(priv.get("kills", 0))
                if kills > prev_kills:
                    reward += 2.5 * (kills - prev_kills)
                prev_kills = kills

                dmg = float(priv.get("damage_taken", 0.0))
                if dmg > prev_dmg:
                    reward -= 0.04 * (dmg - prev_dmg)
                prev_dmg = dmg

                items = int(priv.get("items", 0))
                if items > prev_items:
                    reward += 0.3 * (items - prev_items)
                prev_items = items

                hp = float(priv.get("health", 100.0))
                if hp > prev_hp and prev_hp < 75.0:
                    reward += 1.2
                prev_hp = hp

                cur_keys = int(obs["vitals"][8:14].sum())
                if cur_keys > prev_keys:
                    reward += 4.0 * (cur_keys - prev_keys)
                prev_keys = cur_keys

                # Progression bonus: door / locked door opened
                door_lock = priv.get("door_lock")
                sec_ceil = priv.get("sector_ceiling")
                if door_lock is not None and sec_ceil is not None:
                    for s_idx, lock_val in enumerate(door_lock):
                        if lock_val >= 0 and s_idx not in opened_doors and sec_ceil[s_idx] > 8.0:
                            opened_doors.add(s_idx)
                            reward += 5.0 if lock_val > 0 else 0.8

                # Exploration bonus: newly visited 96x96 area
                cell = (int(priv["x"] // 96), int(priv["y"] // 96))
                first_visit_step = visited_cells.get(cell)
                loop_closure = 0.0
                if first_visit_step is None:
                    visited_cells[cell] = step_idx
                    reward += 0.15
                elif step_idx - first_visit_step > 8:
                    loop_closure = 1.0

                # Door USE anti-spam penalty (prevents bouncing doors)
                if action[7] == 1 and use_cooldown > 0:
                    reward -= 0.05
                use_cooldown = 4 if action[7] == 1 else max(0, use_cooldown - 1)

                done = bool(term or trunc)
                ep_info = None
                if done:
                    ep_info = {
                        "exit": float(info.get("exit_reached", False)),
                        "kills": kills,
                        "keys": cur_keys,
                        "areas": len(visited_cells),
                        "steps": step_idx,
                    }
                    obs, _ = reset_episode(int(next_map_idx))
                else:
                    prev_action = action

                conn.send((obs, prev_action.copy(), float(reward), done, loop_closure, ep_info))
    finally:
        if env is not None:
            env.close()
        conn.close()


class PLRSampler:
    """Prioritized Level Replay (Jiang et al., 2021) over the training map pool."""

    def __init__(self, num_maps: int, temp: float = 0.5, staleness_coef: float = 0.2) -> None:
        self.num_maps = num_maps
        self.temp = temp
        self.staleness_coef = staleness_coef
        self.scores = np.ones(num_maps, dtype=np.float64)
        self.last_seen = np.zeros(num_maps, dtype=np.int64)
        self.step = 0

    def sample(self) -> int:
        self.step += 1
        # Rank-based score priority + staleness mix
        ranks = np.argsort(np.argsort(-self.scores)) + 1
        p_score = (1.0 / ranks) ** (1.0 / self.temp)
        p_score /= p_score.sum()

        stale = np.maximum(0, self.step - self.last_seen).astype(np.float64)
        p_stale = stale / stale.sum() if stale.sum() > 0 else np.ones(self.num_maps) / self.num_maps

        probs = (1.0 - self.staleness_coef) * p_score + self.staleness_coef * p_stale
        idx = int(np.random.choice(self.num_maps, p=probs))
        self.last_seen[idx] = self.step
        return idx

    def update(self, map_idx: int, score: float) -> None:
        self.scores[map_idx] = 0.7 * self.scores[map_idx] + 0.3 * max(1e-4, score)


def obs_batch_to_tensors(obs_list: list[dict], prev_acts: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    return {
        "ray_dist": torch.as_tensor(np.stack([o["rays"]["distance"] for o in obs_list]), dtype=torch.float32, device=device),
        "ray_type": torch.as_tensor(np.stack([o["rays"]["hit_type"] for o in obs_list]), dtype=torch.long, device=device),
        "ent_feat": torch.as_tensor(np.stack([o["visible_entities"]["features"] for o in obs_list]), dtype=torch.float32, device=device),
        "ent_type": torch.as_tensor(np.stack([o["visible_entities"]["entity_type"] for o in obs_list]), dtype=torch.long, device=device),
        "ent_mask": torch.as_tensor(np.stack([o["visible_entities"]["mask"] for o in obs_list]), dtype=torch.float32, device=device),
        "vitals": torch.as_tensor(np.stack([o["vitals"] for o in obs_list]), dtype=torch.float32, device=device),
        "prev_act": torch.as_tensor(prev_acts, dtype=torch.float32, device=device),
    }


def extract_vitals_target(vitals: torch.Tensor) -> torch.Tensor:
    """9-dim normalized target for DFP future-vitals head: [health, armor, ammo, 6 keys]."""
    return torch.cat(
        [
            vitals[..., 0:2] / 100.0,
            vitals[..., 3:4] / 50.0,
            vitals[..., 8:14],
        ],
        dim=-1,
    )


def train(args: argparse.Namespace) -> None:
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    tiers = set(args.tiers) if args.tiers else None
    map_pool = _discover_maps(tiers)
    print(f"[Train] Discovered {len(map_pool)} maps (tiers={sorted(tiers) if tiers else 'all'}). Device={device}, Workers={args.num_envs}")
    plr = PLRSampler(len(map_pool))

    ctx = mp.get_context("spawn")
    workers = []
    conns = []
    cur_map_indices = [plr.sample() for _ in range(args.num_envs)]
    for i in range(args.num_envs):
        parent_conn, child_conn = ctx.Pipe()
        p = ctx.Process(
            target=_worker_loop,
            args=(i, child_conn, map_pool, args.episode_ticks, args.seed),
            daemon=True,
        )
        p.start()
        child_conn.close()
        workers.append(p)
        conns.append(parent_conn)

    # Initial reset
    obs_list = []
    prev_acts = []
    for i, c in enumerate(conns):
        c.send(("reset", cur_map_indices[i]))
    for c in conns:
        o, pa = c.recv()
        obs_list.append(o)
        prev_acts.append(pa)
    prev_acts_arr = np.stack(prev_acts)

    net = SpatialDoomNet(hidden_dim=256).to(device)
    if args.resume and WEIGHTS_PATH.exists():
        ckpt = torch.load(WEIGHTS_PATH, map_location=device, weights_only=True)
        net.load_state_dict(ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt)
        print(f"[Train] Resumed weights from {WEIGHTS_PATH}")

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)

    N = args.num_envs
    T = args.num_steps
    total_updates = max(1, args.total_steps // (N * T))

    hx = torch.zeros(1, N, net.hidden_dim, dtype=torch.float32, device=device)
    grid = net.grid_mem.init_grid(N, device)

    recent_returns: list[float] = []
    recent_exits: list[float] = []
    recent_areas: list[int] = []
    ep_returns = np.zeros(N, dtype=np.float64)
    best_score = -1e9
    global_step = 0
    t_start = time.perf_counter()

    try:
        for update in range(1, total_updates + 1):
            # LR annealing
            frac = 1.0 - (update - 1.0) / total_updates
            optimizer.param_groups[0]["lr"] = args.lr * max(0.1, frac)

            # Rollout storage (T, N, ...)
            buf_ray_dist = torch.zeros(T, N, 64, dtype=torch.float32, device=device)
            buf_ray_type = torch.zeros(T, N, 64, dtype=torch.long, device=device)
            buf_ent_feat = torch.zeros(T, N, 32, 5, dtype=torch.float32, device=device)
            buf_ent_type = torch.zeros(T, N, 32, dtype=torch.long, device=device)
            buf_ent_mask = torch.zeros(T, N, 32, dtype=torch.float32, device=device)
            buf_vitals = torch.zeros(T, N, 17, dtype=torch.float32, device=device)
            buf_prev_act = torch.zeros(T, N, NUM_ACTIONS, dtype=torch.float32, device=device)
            buf_grids = torch.zeros(T, N, *grid.shape[1:], dtype=torch.float32, device=device)

            buf_actions = torch.zeros(T, N, NUM_ACTIONS, dtype=torch.float32, device=device)
            buf_logprobs = torch.zeros(T, N, dtype=torch.float32, device=device)
            buf_rewards = torch.zeros(T, N, dtype=torch.float32, device=device)
            buf_dones = torch.zeros(T, N, dtype=torch.float32, device=device)
            buf_values = torch.zeros(T, N, dtype=torch.float32, device=device)
            buf_loop_targets = torch.zeros(T, N, dtype=torch.float32, device=device)

            hx_init = hx.clone()

            net.eval()
            for t in range(T):
                obs_t = obs_batch_to_tensors(obs_list, prev_acts_arr, device)
                buf_ray_dist[t] = obs_t["ray_dist"]
                buf_ray_type[t] = obs_t["ray_type"]
                buf_ent_feat[t] = obs_t["ent_feat"]
                buf_ent_type[t] = obs_t["ent_type"]
                buf_ent_mask[t] = obs_t["ent_mask"]
                buf_vitals[t] = obs_t["vitals"]
                buf_prev_act[t] = obs_t["prev_act"]
                buf_grids[t] = grid

                with torch.no_grad():
                    logits, value, new_hx, new_grid, _, _ = net.forward_step(obs_t, hx, grid)
                    dist = torch.distributions.Bernoulli(logits=logits)
                    action = dist.sample()
                    logprob = dist.log_prob(action).sum(dim=-1)

                buf_actions[t] = action
                buf_logprobs[t] = logprob
                buf_values[t] = value

                act_np = action.cpu().numpy().astype(np.int8)
                next_map_picks = [plr.sample() for _ in range(N)]
                for i, c in enumerate(conns):
                    c.send(("step", (act_np[i], next_map_picks[i])))

                next_obs = []
                next_pa = []
                for i, c in enumerate(conns):
                    o, pa, rew, done, loop_c, ep_info = c.recv()
                    next_obs.append(o)
                    next_pa.append(pa)
                    buf_rewards[t, i] = rew
                    buf_dones[t, i] = float(done)
                    buf_loop_targets[t, i] = loop_c
                    ep_returns[i] += rew
                    if done:
                        if ep_info is not None:
                            recent_returns.append(float(ep_returns[i]))
                            recent_exits.append(ep_info["exit"])
                            recent_areas.append(ep_info["areas"])
                        ep_returns[i] = 0.0
                        cur_map_indices[i] = next_map_picks[i]
                        new_hx[:, i, :] = 0.0
                        new_grid[i] = 0.0

                obs_list = next_obs
                prev_acts_arr = np.stack(next_pa)
                hx = new_hx
                grid = new_grid
                global_step += N

            # Bootstrap next value
            with torch.no_grad():
                obs_next = obs_batch_to_tensors(obs_list, prev_acts_arr, device)
                _, next_value, _, _, _, _ = net.forward_step(obs_next, hx, grid)

            # GAE
            advantages = torch.zeros_like(buf_rewards)
            lastgaelam = 0.0
            for t in reversed(range(T)):
                if t == T - 1:
                    nextnonterminal = 1.0 - buf_dones[t]
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - buf_dones[t]
                    nextvalues = buf_values[t + 1]
                delta = buf_rewards[t] + args.gamma * nextvalues * nextnonterminal - buf_values[t]
                lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
                advantages[t] = lastgaelam
            returns = advantages + buf_values

            # Update PLR priorities with mean |advantage| per env
            mean_adv = advantages.abs().mean(dim=0).cpu().numpy()
            for i in range(N):
                plr.update(cur_map_indices[i], float(mean_adv[i]))

            # DFP future vitals target: vitals_{t+4} - vitals_t (masked by no-done)
            vit_tgt_all = extract_vitals_target(buf_vitals)  # (T, N, 9)
            future_delta_vitals = torch.zeros_like(vit_tgt_all)
            if T > 4:
                future_delta_vitals[:-4] = vit_tgt_all[4:] - vit_tgt_all[:-4]

            # PPO optimization epochs (full sequence minibatches over N envs)
            net.train()
            # Transpose to (N, T, ...) for sequence BPTT
            b_ray_dist = buf_ray_dist.transpose(0, 1).reshape(N * T, 64)
            b_ray_type = buf_ray_type.transpose(0, 1).reshape(N * T, 64)
            b_ent_feat = buf_ent_feat.transpose(0, 1).reshape(N * T, 32, 5)
            b_ent_type = buf_ent_type.transpose(0, 1).reshape(N * T, 32)
            b_ent_mask = buf_ent_mask.transpose(0, 1).reshape(N * T, 32)
            b_vitals = buf_vitals.transpose(0, 1).reshape(N * T, 17)
            b_prev_act = buf_prev_act.transpose(0, 1).reshape(N * T, NUM_ACTIONS)
            b_grids = buf_grids.transpose(0, 1).reshape(N * T, *grid.shape[1:])

            b_actions = buf_actions.transpose(0, 1)
            b_logprobs = buf_logprobs.transpose(0, 1)
            b_adv = advantages.transpose(0, 1)
            b_adv_norm = (b_adv - b_adv.mean()) / (b_adv.std() + 1e-8)
            b_returns = returns.transpose(0, 1)
            b_values = buf_values.transpose(0, 1)
            b_dones = buf_dones.transpose(0, 1)
            b_loop = buf_loop_targets.transpose(0, 1)
            b_fut_vit = future_delta_vitals.transpose(0, 1)

            for _ in range(args.update_epochs):
                fused_flat, _ = net.encode_step(
                    b_ray_dist, b_ray_type, b_ent_feat, b_ent_type, b_ent_mask, b_vitals, b_prev_act, b_grids
                )
                fused_seq = fused_flat.view(N, T, -1)
                logits_seq, values_seq, aux_vit_seq, aux_loop_seq = net.forward_sequence(fused_seq, hx_init, b_dones)

                dist = torch.distributions.Bernoulli(logits=logits_seq)
                new_logprobs = dist.log_prob(b_actions).sum(dim=-1)
                entropy = dist.entropy().sum(dim=-1).mean()

                ratio = torch.exp(new_logprobs - b_logprobs)
                pg_loss1 = -b_adv_norm * ratio
                pg_loss2 = -b_adv_norm * torch.clamp(ratio, 1.0 - args.clip_coef, 1.0 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                v_loss_unclipped = (values_seq - b_returns) ** 2
                v_clipped = b_values + torch.clamp(values_seq - b_values, -args.clip_coef, args.clip_coef)
                v_loss = 0.5 * torch.max(v_loss_unclipped, (v_clipped - b_returns) ** 2).mean()

                aux_vit_loss = F.mse_loss(aux_vit_seq[:, :-4], b_fut_vit[:, :-4]) if T > 4 else 0.0
                aux_loop_loss = F.binary_cross_entropy_with_logits(aux_loop_seq, b_loop)

                loss = pg_loss - args.ent_coef * entropy + args.vf_coef * v_loss + 0.1 * aux_vit_loss + 0.1 * aux_loop_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                optimizer.step()

            if update % args.log_interval == 0 or update == total_updates:
                fps = global_step / max(1e-6, time.perf_counter() - t_start)
                mean_ret = float(np.mean(recent_returns[-50:])) if recent_returns else 0.0
                exit_rate = float(np.mean(recent_exits[-50:])) if recent_exits else 0.0
                mean_area = float(np.mean(recent_areas[-50:])) if recent_areas else 0.0
                print(
                    f"[Upd {update}/{total_updates} | Steps {global_step}] "
                    f"FPS={fps:.0f} | Ret={mean_ret:.2f} | Exit={exit_rate*100:.1f}% | Areas={mean_area:.1f} | Ent={entropy.item():.2f} | Loss={loss.item():.3f}"
                )
                WEIGHTS_PATH.parent.mkdir(parents=True, exist_ok=True)
                torch.save({"model": net.state_dict(), "step": global_step}, WEIGHTS_PATH)
                if recent_returns and mean_ret >= best_score:
                    best_score = mean_ret
                    torch.save({"model": net.state_dict(), "step": global_step}, WEIGHTS_PATH.parent / "best_policy.pt")

    finally:
        for c in conns:
            try:
                c.send(("close", None))
            except Exception:
                pass
        for p in workers:
            p.join(timeout=2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-steps", type=int, default=500_000)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--episode-ticks", type=int, default=2100)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-coef", type=float, default=0.2)
    parser.add_argument("--ent-coef", type=float, default=0.015)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiers", type=int, nargs="+", default=None)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
