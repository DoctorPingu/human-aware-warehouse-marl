from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import os
import signal
import time
from multiprocessing import Pool
from typing import Dict, List, Optional

from layouts import DEMAND_LEVELS, DENSITY_LEVELS, LAYOUTS, STRESS_LEVEL
from train import EVAL_SEED_BASE

BASELINE_ALGOS = ["orca", "heuristic", "mapf"]
PAUSED_EXIT = 3
METRIC_FIELDS = ["orders_per_hour", "congestion_prop", "collision_rate_100",
                 "violations_per_interaction", "deliveries", "collisions_rh",
                 "collisions_rr", "safety_steps", "interactions", "violations",
                 "min_separation", "wall_hits", "steps",
                 "distance_per_delivery", "distance_m"]
ROW_FIELDS = ["suite", "algo", "tag", "train_layout", "layout", "demand",
              "density", "perturb", "episode", "unit", "env_seed", "train_seed", "seconds"] + METRIC_FIELDS


def find_checkpoints(runs_dir: str, tag: str, layout: str, name: str = "final.pt") -> List[str]:
    return sorted(glob.glob(os.path.join(runs_dir, tag, layout, "seed*", name)))


PERTURBATIONS = {
    "nominal": {},
    "speed_x0.8": {"human_speed_scale": 0.8},
    "speed_x1.2": {"human_speed_scale": 1.2},
    "avoid_0.30m": {"human_avoid_r": 0.30},
    "avoid_0.40m": {"human_avoid_r": 0.40},
    "robots_x1.5": {"robot_scale": 1.5},
}
SENSITIVITY_LEVELS = ["nominal", "speed_x0.8", "speed_x1.2", "avoid_0.30m", "avoid_0.40m"]
DEMAND_ARM_LAYOUTS = ["small_1_parallel_aisle", "large_1_fulfilment"]
CURRICULUM_TARGET = "large_1_fulfilment"
UNITS = 10
SENSITIVITY_LAYOUTS = ["small_1_parallel_aisle", "large_1_fulfilment"]


def _job(label: str, algo: str, tag: str, train_layout: str, layout: str,
         demand: str, density: str, episodes: range, ckpts: List[str],
         perturb: str = "nominal") -> Dict:
    return dict(label=label, algo=algo, tag=tag, train_layout=train_layout,
                layout=layout, demand=demand, density=density, perturb=perturb,
                episodes=list(episodes), ckpts=ckpts)


def build_jobs(suite: str, runs_dir: str, n_eps: int, layouts: List[str]) -> List[Dict]:
    jobs: List[Dict] = []
    eps = range(n_eps)
    factorial = [(d, s) for d in DEMAND_LEVELS for s in DENSITY_LEVELS]

    def ppo(tag, train_layout, layout, dem, den, label="ppo", perturb="nominal",
            name="final.pt"):
        ck = find_checkpoints(runs_dir, tag, train_layout, name)
        if not ck:
            print(f"  [skip] no checkpoints for tag={tag} layout={train_layout}")
            return
        jobs.append(_job(label, "ppo", tag, train_layout, layout, dem, den, eps, ck, perturb))

    if suite == "smoke":
        for algo in BASELINE_ALGOS:
            jobs.append(_job(algo, algo, "", "", layouts[0], "standard", "standard",
                             range(2), []))
        ppo("main", layouts[0], layouts[0], "standard", "standard")
        if jobs and jobs[-1]["algo"] == "ppo":
            jobs[-1]["episodes"] = [0, 1]

    elif suite == "main":
        for lay in layouts:
            for dem, den in factorial:
                for algo in BASELINE_ALGOS:
                    jobs.append(_job(algo, algo, "", "", lay, dem, den, eps, []))
                ppo("main", lay, lay, dem, den)

    elif suite == "calibration":
        for lay in layouts:
            for dem, den in factorial:
                jobs.append(_job("orca", "orca", "", "", lay, dem, den, eps, []))
                ppo("v1", lay, lay, dem, den, label="ppo_v1")
                ppo("main", lay, lay, dem, den, label="ppo_v2")

    elif suite == "ablation":
        for lay in layouts:
            for dem, den in factorial:
                jobs.append(_job("orca", "orca", "", "", lay, dem, den, eps, []))
                ppo("main", lay, lay, dem, den, label="ppo_full")
                ppo("throughput_only", lay, lay, dem, den, label="ppo_throughput_only")
                ppo("safety_only", lay, lay, dem, den, label="ppo_safety_only")

    elif suite == "generalisation":
        for train_lay in LAYOUTS:
            for lay in layouts:
                ppo("main", train_lay, lay, "standard", "standard",
                    label="ppo_transfer" if train_lay != lay else "ppo")
        for lay in layouts:
            for dem in DEMAND_LEVELS:
                for algo in BASELINE_ALGOS:
                    jobs.append(_job(algo, algo, "", "", lay, dem, STRESS_LEVEL, eps, []))
                ppo("main", lay, lay, dem, STRESS_LEVEL)
    elif suite == "sensitivity":
        for lay in [l for l in SENSITIVITY_LAYOUTS if l in layouts]:
            for pert in SENSITIVITY_LEVELS:
                for algo in BASELINE_ALGOS:
                    jobs.append(_job(algo, algo, "", "", lay, "standard", "high", eps, [], pert))
                ppo("main", lay, lay, "standard", "high", perturb=pert)
    elif suite == "robot_stress":
        for lay in layouts:
            for dem in DEMAND_LEVELS:
                for algo in BASELINE_ALGOS:
                    jobs.append(_job(algo, algo, "", "", lay, dem, "standard", eps, [],
                                     "robots_x1.5"))
                ppo("main", lay, lay, dem, "standard", perturb="robots_x1.5")
    elif suite == "demand":
        for lay in [l for l in DEMAND_ARM_LAYOUTS if l in layouts]:
            for dem, den in factorial:
                jobs.append(_job("orca", "orca", "", "", lay, dem, den, eps, []))
                ppo("main", lay, lay, dem, den, label="ppo_mixed")
                ppo("std_only", lay, lay, dem, den, label="ppo_std_only")
    elif suite == "curriculum":
        lay = CURRICULUM_TARGET
        if lay in layouts:
            for dem, den in factorial:
                jobs.append(_job("orca", "orca", "", "", lay, dem, den, eps, []))
                ppo("main", lay, lay, dem, den, label="ppo_direct")
                ppo("curriculum", lay, lay, dem, den, label="ppo_curriculum")
    elif suite == "progress":
        snaps = sorted({int(os.path.basename(p)[5:-3]) for p in glob.glob(
            os.path.join(runs_dir, "main", "*", "seed*", "snap_*.pt"))})
        for lay in layouts:
            for dem, den in factorial:
                for snap in snaps:
                    ppo("main", lay, lay, dem, den, label=f"ppo_{snap / 1e6:g}m",
                        name=f"snap_{snap}.pt")
    else:
        raise SystemExit(f"unknown suite {suite!r}")
    return jobs


def _part_path(out_dir: str, suite: str, job: Dict) -> str:
    from warehouse_env import EPISODE_STEPS
    key = json.dumps({k: job[k] for k in sorted(job) if k != "ckpts"}, sort_keys=True)
    key += f"|steps={EPISODE_STEPS}|eval_seed_base={EVAL_SEED_BASE}"
    key += "|" + "|".join(f"{os.path.basename(os.path.dirname(c))}/{os.path.basename(c)}:"
                          f"{int(os.path.getmtime(c))}" for c in job["ckpts"])
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(out_dir, "parts", suite, f"{job['label']}_{job['layout']}_{h}.csv")


def run_job(args) -> str:
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    suite, out_dir, job = args
    path = _part_path(out_dir, suite, job)
    if os.path.exists(path):
        return path
    import torch
    torch.set_num_threads(1)
    from policies import make_policy, run_episode
    from warehouse_env import WarehouseEnv

    env = WarehouseEnv(job["layout"], job["demand"], job["density"],
                       **PERTURBATIONS[job.get("perturb", "nominal")])
    policies = ([make_policy("ppo", c) for c in job["ckpts"]] if job["algo"] == "ppo"
                else [make_policy(job["algo"])])
    rows = []
    for k in job["episodes"]:
        pi = k % len(policies)
        seed = EVAL_SEED_BASE + k
        t0 = time.perf_counter()
        m = run_episode(env, policies[pi], seed)
        train_seed = (os.path.basename(os.path.dirname(job["ckpts"][pi]))
                      if job["ckpts"] else "")
        rows.append({"suite": suite, "algo": job["label"], "tag": job["tag"],
                     "train_layout": job["train_layout"], "layout": env.layout.name,
                     "demand": job["demand"], "density": job["density"],
                     "perturb": job.get("perturb", "nominal"),
                     "episode": k, "env_seed": seed, "train_seed": train_seed,
                     "unit": pi if job["ckpts"] else k % job.get("units", UNITS),
                     "seconds": round(time.perf_counter() - t0, 3),
                     **{f: m[f] for f in METRIC_FIELDS}})
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)
    return path


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", required=True,
                   choices=["smoke", "main", "calibration", "ablation", "generalisation",
                            "sensitivity", "robot_stress", "demand", "curriculum", "progress"])
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--jobs", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--units", type=int, default=UNITS,
                   help="seed units for baselines; match the number of training seeds")
    p.add_argument("--out", default="results")
    p.add_argument("--layouts", default="all",
                   help="comma list of names/aliases, or 'all'. "
                        "Calibration and ablation default to small1.")
    a = p.parse_args(argv)

    from layouts import get_layout
    if a.layouts == "all":
        lays = (["small_1_parallel_aisle"] if a.suite in ("calibration", "ablation")
                else list(LAYOUTS))
    else:
        lays = [get_layout(x).name for x in a.layouts.split(",")]

    jobs = build_jobs(a.suite, a.runs_dir, a.episodes, lays)
    for job in jobs:
        if not job["ckpts"]:
            job["units"] = a.units
    n_eps = sum(len(j["episodes"]) for j in jobs)
    print(f"Suite '{a.suite}': {len(jobs)} jobs, {n_eps} episodes, {a.jobs} workers")
    t0 = time.time()
    work = [(a.suite, a.out, j) for j in jobs]
    parts = []
    os.makedirs(a.out, exist_ok=True)
    prog_file = os.path.join(a.out, "eval_progress.json")
    cached = sum(os.path.exists(_part_path(a.out, a.suite, j)) for j in jobs)

    def report(done: int) -> None:
        tmp = prog_file + ".tmp"
        with open(tmp, "w") as f:
            json.dump({"suite": a.suite, "done": done, "total": len(work),
                       "cached": cached, "started": t0, "updated": time.time()}, f)
        os.replace(tmp, prog_file)

    def _stop(signum, frame):
        raise SystemExit(PAUSED_EXIT)
    signal.signal(signal.SIGTERM, _stop)

    report(0)
    if a.jobs > 1:
        with Pool(a.jobs) as pool:
            for i, path in enumerate(pool.imap_unordered(run_job, work), 1):
                parts.append(path)
                print(f"  [{i:>3}/{len(work)}] {os.path.basename(path)}", flush=True)
                report(i)
    else:
        for i, w in enumerate(work, 1):
            parts.append(run_job(w))
            print(f"  [{i:>3}/{len(work)}] {os.path.basename(parts[-1])}", flush=True)
            report(i)

    out_csv = os.path.join(a.out, f"{a.suite}.csv")
    os.makedirs(a.out, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=ROW_FIELDS)
        w.writeheader()
        for part in sorted(parts):
            with open(part) as pf:
                w.writerows(csv.DictReader(pf))
    print(f"Wrote {out_csv} ({n_eps} episodes) in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
