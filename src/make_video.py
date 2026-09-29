from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

from layouts import get_layout
from policies import make_policy
from train import EVAL_SEED_BASE
from warehouse_env import WarehouseEnv, draw_frame

TITLES = {"orca": "ORCA/RVO", "heuristic": "Heuristic", "mapf": "MAPF", "ppo": "CTDE PPO"}


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--layout", default="small1")
    p.add_argument("--algos", default="orca,ppo")
    p.add_argument("--demand", default="standard")
    p.add_argument("--density", default="high")
    p.add_argument("--seconds", type=float, default=40.0)
    p.add_argument("--episode", type=int, default=0, help="evaluation episode index")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--tag", default="main")
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--out", default="videos")
    p.add_argument("--name", default=None, help="output file name (default: from layout and algos)")
    a = p.parse_args(argv)

    lay = get_layout(a.layout).name
    algos = a.algos.split(",")
    envs, pols = [], []
    for algo in algos:
        ckpt = None
        if algo == "ppo":
            found = sorted(glob.glob(os.path.join(a.runs_dir, a.tag, lay, "seed*", "final.pt")))
            if not found:
                raise SystemExit(f"No trained policy for {lay}; run train.py first.")
            ckpt = found[0]
        env = WarehouseEnv(lay, a.demand, a.density, max_steps=10**9)
        pol = make_policy(algo, ckpt)
        obs, _ = env.reset_arrays(seed=EVAL_SEED_BASE + a.episode)
        pol.reset(env)
        envs.append([env, obs])
        pols.append(pol)

    n = len(algos)
    cols = 2 if n > 1 else 1
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(7.5 * cols, 5.2 * rows), dpi=100,
                             squeeze=False)
    for ax in axes.flat[n:]:
        ax.axis("off")
    fig.patch.set_facecolor("white")

    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, a.name or f"{lay}_{'_vs_'.join(algos)}.mp4")
    steps_per_frame = max(1, int(round(WarehouseEnv.SIM_HZ / a.fps)))
    n_frames = int(a.seconds * a.fps)
    writer = FFMpegWriter(fps=a.fps, bitrate=3000)
    partial = path[:-4] + ".partial.mp4"
    with writer.saving(fig, partial, dpi=100):
        for f in range(n_frames):
            for k, ((env, obs), pol) in enumerate(zip(envs, pols)):
                for _ in range(steps_per_frame):
                    obs, _, _, _ = env.step_arrays(pol.act(env, obs))
                envs[k][1] = obs
                ax = axes.flat[k]
                draw_frame(env, ax)
                m = env.episode_metrics()
                ax.set_title(f"{TITLES.get(algos[k], algos[k])}\n"
                             f"deliveries {m['deliveries']}   "
                             f"collisions {m['contact_events']}   "
                             f"ISO-zone entries {m['violations']}",
                             fontsize=10, fontfamily="monospace")
            fig.suptitle(f"{get_layout(lay).label}  |  demand {a.demand}, "
                         f"density {a.density}  |  t = {f / a.fps:4.1f} s", fontsize=11)
            writer.grab_frame()
            if f % a.fps == 0:
                print(f"  {f / a.fps:4.0f}s / {a.seconds:.0f}s", flush=True)
    os.replace(partial, path)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
