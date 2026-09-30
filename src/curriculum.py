import argparse
import json
import os
import signal
import subprocess
import sys
from typing import List, Tuple

PY = sys.executable
SRC = os.path.dirname(os.path.abspath(__file__))
HERE = os.path.dirname(SRC)
PAUSED_EXIT = 3
STAGES = [
    ("curriculum_stage1", "small_1_parallel_aisle", 0.3, "standard"),
    ("curriculum_stage2", "medium_1_twin_block", 0.3, "mix"),
    ("curriculum", "large_1_fulfilment", 0.4, "mix"),
]


def stage_plan(total: int) -> List[Tuple[str, str, int, str]]:
    return [(tag, lay, int(round(total * frac)), dem) for tag, lay, frac, dem in STAGES]


def trained_steps(run_dir: str) -> int:
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            return int(json.load(f).get("timesteps", 0))
    except (OSError, ValueError):
        return 0


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--timesteps", type=int, required=True)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--threads", type=int, default=1)
    a = p.parse_args()

    state = {"child": None, "stop": False}

    def stop(signum, frame):
        state["stop"] = True
        if state["child"] is not None and state["child"].poll() is None:
            state["child"].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    prev = None
    for i, (tag, lay, steps, dem) in enumerate(stage_plan(a.timesteps), 1):
        run_dir = os.path.join(HERE, a.runs_dir, tag, lay, f"seed{a.seed}")
        final = os.path.join(run_dir, "final.pt")
        if os.path.exists(final) and trained_steps(run_dir) >= steps:
            print(f"Stage {i} ({lay}) already done", flush=True)
            prev = final
            continue
        if state["stop"]:
            sys.exit(PAUSED_EXIT)
        cmd = [PY, "-u", os.path.join(SRC, "train.py"), "--layout", lay, "--seed", str(a.seed), "--tag", tag,
               "--reward", "v2", "--timesteps", str(steps), "--demand", dem,
               "--runs-dir", a.runs_dir, "--threads", str(a.threads)]
        if os.path.exists(os.path.join(run_dir, "latest.pt")):
            cmd.append("--resume")
        elif prev is not None:
            cmd += ["--init-from", prev]
        print(f"Stage {i}: {lay}, {steps:,} steps, demand {dem}", flush=True)
        state["child"] = subprocess.Popen(cmd, cwd=HERE)
        rc = state["child"].wait()
        state["child"] = None
        if rc == PAUSED_EXIT or state["stop"]:
            sys.exit(PAUSED_EXIT)
        if rc != 0:
            sys.exit(rc)
        prev = final
    print("Curriculum complete", flush=True)


if __name__ == "__main__":
    main()
