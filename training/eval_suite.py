"""Full Tier 1-4 unseen-map & sandboxed evaluation suite matching PS1 scoring."""

from __future__ import annotations

import argparse
import resource
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from doom_env import make_env
from submission.agent import DoomAgent
from training.wad_gen import build_wad


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--num-maps", type=int, default=50)
    ap.add_argument("--seed-offset", type=int, default=2000)
    args = ap.parse_args()

    ckpt_path = ROOT / "submission" / "weights" / "policy.pt"
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    print(f"Checkpoint step: {ckpt.get('step', '?')} | Unseen maps: {args.num_maps}")

    with tempfile.TemporaryDirectory() as td:
        wad = Path(td) / "eval_suite.wad"
        specs = [(args.seed_offset + i, (i % 4) + 1) for i in range(args.num_maps)]
        mnames = build_wad(wad, specs)

        env0 = make_env(wad, "MAP01", render_buffers=False)
        t_init0 = time.perf_counter()
        agent = DoomAgent(env0.observation_space, env0.action_space, "cpu")
        init_s = time.perf_counter() - t_init0
        env0.close()

        reset_times, first_acts, all_acts = [], [], []
        invalid_cnt = 0
        stats: dict[int, list[dict]] = {1: [], 2: [], 3: [], 4: []}

        for idx, (mname, (_, tier)) in enumerate(zip(mnames, specs)):
            ticks = {1: 2100, 2: 2100, 3: 3500, 4: 4200}[tier]
            env = make_env(wad, mname, episode_ticks=ticks, render_buffers=False)
            obs, _ = env.reset(seed=idx)
            priv0 = env.privileged_state()
            visited = {(int(priv0["x"] // 96), int(priv0["y"] // 96))}
            opened_doors, unlocked_doors = set(), set()

            t_r0 = time.perf_counter()
            agent.reset()
            reset_times.append(time.perf_counter() - t_r0)

            for step_i in range(ticks // 4):
                t_a0 = time.perf_counter()
                act = agent.act(obs)
                dt_ms = (time.perf_counter() - t_a0) * 1000.0
                if step_i == 0:
                    first_acts.append(dt_ms)
                all_acts.append(dt_ms)

                act_arr = np.asarray(act)
                if act_arr.shape != (9,) or not np.all((act_arr == 0) | (act_arr == 1)):
                    invalid_cnt += 1

                obs, _, term, trunc, info = env.step(act_arr.astype(np.int8))
                priv = env.privileged_state()
                visited.add((int(priv["x"] // 96), int(priv["y"] // 96)))
                dl, sc = priv.get("door_lock"), priv.get("sector_ceiling")
                if dl is not None and sc is not None:
                    for si, lv in enumerate(dl):
                        if lv >= 0 and sc[si] > 8.0:
                            opened_doors.add(si)
                            if lv > 0:
                                unlocked_doors.add(si)
                if term or trunc:
                    break

            priv = env.privileged_state()
            exited = int(info["exit_reached"])
            stats[tier].append(
                {
                    "exit": exited,
                    "rem_hp": float(priv["health"]) if exited else 0.0,
                    "rem_time": max(0.0, 1.0 - priv["tic"] / ticks) if exited else 0.0,
                    "keys": int(obs["vitals"][8:14].sum()),
                    "doors": len(opened_doors),
                    "unlocked": len(unlocked_doors),
                    "areas": len(visited),
                    "kills": int(priv["kills"]),
                    "dmg": float(priv["damage_taken"]),
                    "dead": int(priv["dead"]),
                }
            )
            env.close()

        rss_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
        print(
            f"SANDBOX: init={init_s:.2f}s/60s | reset_max={max(reset_times)*1000:.1f}ms/10000ms | "
            f"first_act_max={max(first_acts):.1f}ms/10000ms | act_avg={np.mean(all_acts):.2f}ms/40ms | "
            f"act_p99={np.percentile(all_acts, 99):.2f}ms | act_max={max(all_acts):.1f}ms/5000ms | "
            f"invalid={invalid_cnt} | peak_ram={rss_mb:.0f}MB/8192MB"
        )
        for t in (1, 2, 3, 4):
            rows = stats[t]
            n = len(rows)
            exits = sum(r["exit"] for r in rows)
            print(
                f"Tier {t} ({n} maps): "
                f"Exit={exits}/{n} ({100*exits/n:.0f}%) | "
                f"Keys={sum(r['keys'] for r in rows)} | "
                f"Doors={sum(r['doors'] for r in rows)} | "
                f"Unlocked={sum(r['unlocked'] for r in rows)} | "
                f"AvgAreas={np.mean([r['areas'] for r in rows]):.1f} | "
                f"Kills={sum(r['kills'] for r in rows)} | "
                f"AvgDmg={np.mean([r['dmg'] for r in rows]):.1f} | "
                f"Deaths={sum(r['dead'] for r in rows)}/{n}"
            )


if __name__ == "__main__":
    main()
