from __future__ import annotations

import argparse
import csv
import glob
import heapq
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from layouts import LAYOUTS

PY = sys.executable
HERE = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(HERE, "run_state.json")
PAUSE = os.path.join(HERE, "PAUSE")
STATUS_TXT = os.path.join(HERE, "status.txt")
PAUSED_EXIT = 3

MAIN_ORDER = ["large_2_robotic_grid", "large_1_fulfilment", "medium_1_twin_block",
              "medium_2_fishbone", "small_2_l_shaped", "small_1_parallel_aisle"]
SMALL = ["small_1_parallel_aisle"]
TRAIN_SETS = {
    "train": [("main", "v2", MAIN_ORDER, [])],
    "demand_arm": [("std_only", "v2", ["large_1_fulfilment", "small_1_parallel_aisle"],
                    ["--demand", "standard"])],
    "calibrate": [("v1", "v1", SMALL, [])],
    "ablation": [("throughput_only", "throughput_only", SMALL, []),
                 ("safety_only", "safety_only", SMALL, [])],
}
ALL_TRAINING = ["curriculum", "train", "demand_arm", "calibrate", "ablation"]
SUITES = ["main", "calibration", "ablation", "generalisation", "sensitivity",
          "robot_stress", "demand", "curriculum", "progress"]
VIDEO_ALGOS = "orca,heuristic,mapf,ppo"

FALLBACK_SPS = 1150
STARTUP_S = 60
EVAL_S_PER_EPISODE = 7.0
ANALYZE_S = 5 * 60
VIDEO_S = 5 * 60


def run(cmd: List[str], log: str = None) -> int:
    print("  $", " ".join(cmd), flush=True)
    if log is None:
        return subprocess.call(cmd, cwd=HERE)
    with open(log, "w") as f:
        return subprocess.call(cmd, cwd=HERE, stdout=f, stderr=subprocess.STDOUT)


def fmt_dur(sec: float) -> str:
    sec = max(0, int(sec))
    if sec < 60:
        return "<1 min"
    h, m = divmod(round(sec / 60), 60)
    return f"{h}h {m:02d}m" if h else f"{m} min"


def clock(t: float) -> str:
    return time.strftime("%H:%M", time.localtime(t))


def read_state() -> Dict:
    try:
        with open(STATE) as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def write_state(**upd) -> None:
    s = read_state()
    s.update(upd)
    with open(STATE + ".tmp", "w") as f:
        json.dump(s, f, indent=2)
    os.replace(STATE + ".tmp", STATE)


def pid_alive(pid: Optional[int]) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def pause_requested() -> bool:
    return os.path.exists(PAUSE)


def planned_runs(train_stages: List[str], a) -> List[Dict]:
    from curriculum import stage_plan
    seeds = [int(s) for s in a.seeds.split(",")]
    out = []
    for st in train_stages:
        if st == "curriculum":
            for s in seeds:
                for tag, lay, steps, _ in stage_plan(a.timesteps):
                    out.append({"tag": tag, "reward": "v2", "layout": lay, "seed": s,
                                "steps": steps, "extra": [], "chain": f"curriculum{s}",
                                "dir": os.path.join(HERE, a.runs_dir, tag, lay, f"seed{s}"),
                                "log": os.path.join("logs", f"curriculum_seed{s}.log")})
            continue
        for tag, reward, lays, extra in TRAIN_SETS[st]:
            for lay in lays:
                for s in seeds:
                    out.append({"tag": tag, "reward": reward, "layout": lay, "seed": s,
                                "steps": a.timesteps, "extra": extra, "chain": None,
                                "dir": os.path.join(HERE, a.runs_dir, tag, lay, f"seed{s}"),
                                "log": os.path.join("logs", f"{tag}_{lay}_seed{s}.log")})
    return out


def _trained_steps(run_dir: str) -> int:
    try:
        with open(os.path.join(run_dir, "config.json")) as f:
            return int(json.load(f).get("timesteps", 0))
    except (OSError, ValueError):
        return 0


def run_done(r: Dict, a) -> bool:
    return (os.path.exists(os.path.join(r["dir"], "final.pt"))
            and _trained_steps(r["dir"]) >= r["steps"])


def last_progress(r: Dict) -> Tuple[int, float]:
    try:
        with open(os.path.join(r["dir"], "progress.csv"), newline="") as f:
            rows = list(csv.DictReader(f))
        if rows:
            return int(float(rows[-1]["step"])), float(rows[-1]["fps"] or 0)
    except (OSError, ValueError, KeyError):
        pass
    return 0, 0.0


def train_cmd(r: Dict, a) -> List[str]:
    cmd = [PY, "-u", "train.py", "--layout", r["layout"], "--seed", str(r["seed"]),
           "--tag", r["tag"], "--reward", r["reward"], "--timesteps", str(r["steps"]),
           "--runs-dir", a.runs_dir, "--threads", "1"] + r["extra"]
    if r["tag"] == "main" and a.snapshots:
        cmd += ["--snapshots", a.snapshots]
    if os.path.exists(os.path.join(r["dir"], "latest.pt")):
        cmd.append("--resume")
    return cmd


def training_estimate(runs: List[Dict], a, slots: int) -> Dict:
    info = []
    for r in runs:
        done = run_done(r, a)
        step, fps = (r["steps"], 0.0) if done else last_progress(r)
        info.append({**r, "done": done, "step": min(step, r["steps"]), "fps": fps})
    speeds = [i["fps"] for i in info if i["fps"] > 0]
    median = sorted(speeds)[len(speeds) // 2] if speeds else FALLBACK_SPS
    by_lay: Dict[str, List[float]] = {}
    for i in info:
        if i["fps"] > 0:
            by_lay.setdefault(i["layout"], []).append(i["fps"])
    for i in info:
        if i["done"]:
            i["left"] = 0.0
            continue
        rate = i["fps"] or (sorted(by_lay[i["layout"]])[len(by_lay[i["layout"]]) // 2]
                            if i["layout"] in by_lay else median)
        i["left"] = (i["steps"] - i["step"]) / max(rate, 1.0) + (0 if i["step"] else STARTUP_S)
    pending = [i for i in info if not i["done"]]
    started = [i for i in pending if i["step"] > 0]
    waiting = [i for i in pending if i["step"] == 0]
    jobs: Dict[str, List[float]] = {}
    for i in started + waiting:
        key = i["chain"] or f"{i['tag']}/{i['layout']}/{i['seed']}"
        jobs.setdefault(key, [0.0, 0.0])
        jobs[key][0] += i["left"]
        jobs[key][1] = max(jobs[key][1], 1.0 if i["step"] > 0 else 0.0)
    free = [0.0] * max(1, slots)
    for left, _ in sorted(jobs.values(), key=lambda v: -v[1]):
        t = heapq.heappop(free) + left
        heapq.heappush(free, t)
    total = sum(i["steps"] for i in info)
    return {"runs": info, "left": max(free) if pending else 0.0,
            "n_done": len(info) - len(pending), "n_started": len(started),
            "n_waiting": len(waiting),
            "frac": sum(i["step"] for i in info) / total if total else 1.0}


def suite_episodes(suite: str, episodes: int) -> int:
    n_lay = len(LAYOUTS)
    jobs = {"main": n_lay * 4 * 4, "calibration": 4 * 3, "ablation": 4 * 4,
            "generalisation": n_lay * n_lay + n_lay * 2 * 4,
            "sensitivity": 2 * 5 * 4,
            "robot_stress": n_lay * 2 * 4,
            "demand": 2 * 4 * 3,
            "curriculum": 4 * 3,
            "progress": n_lay * 4 * 2}[suite]
    return jobs * episodes


def eval_estimate(a, suites: List[str], current: Optional[str]) -> float:
    ej = a.eval_jobs or max(1, (os.cpu_count() or 4) - 2)
    per_ep = EVAL_S_PER_EPISODE / max(1.0, 0.6 * ej)
    left, measured = 0.0, None
    if current:
        try:
            with open(os.path.join(HERE, "results", "eval_progress.json")) as f:
                p = json.load(f)
            if p.get("suite") == current:
                fresh = p["done"] - p["cached"]
                if fresh >= 3:
                    per_job = (p["updated"] - p["started"]) / fresh
                    measured = per_job / a.episodes
                    left += (p["total"] - p["done"]) * per_job
                else:
                    left += suite_episodes(current, a.episodes) * per_ep
        except (OSError, ValueError, KeyError):
            left += suite_episodes(current, a.episodes) * per_ep
    rest = suites[suites.index(current) + 1:] if current in suites else suites
    for s in rest:
        left += suite_episodes(s, a.episodes) * (measured or per_ep)
    return left


def status_report(a, st: Dict) -> Tuple[str, str]:
    now = time.time()
    plan, phase = st.get("plan", []), st.get("phase", "")
    lines = [f"Warehouse RL progress   (checked {clock(now)}, started {clock(st.get('started', now))}, "
             f"{fmt_dur(now - st.get('started', now))} ago)",
             f"Command: python run_all.py {' '.join(st.get('argv', []))}"]
    alive = pid_alive(st.get("pid"))
    if phase == "finished":
        state = "finished"
    elif alive:
        state = "PAUSING (saving and stopping)" if pause_requested() else f"running ({phase})"
    elif phase == "paused":
        state = f"paused (during {st.get('paused_phase', '?')})"
    elif phase == "stopped":
        state = "stopped: a training run failed (see run.log)"
    else:
        state = "stopped unexpectedly (closed window, crash, shutdown or power cut)"
    lines.append(f"State: {state}")
    if not alive and phase != "finished":
        lines.append("To continue:  nohup python run_all.py resume >> run.log 2>&1 &")
    lines.append("")

    left_total, short = 0.0, ""
    runs = planned_runs(st.get("train_stages", []), a)
    if runs:
        est = training_estimate(runs, a, a.jobs)
        lines.append(f"Training: {est['n_done']} of {len(runs)} runs done, "
                     f"{est['n_started']} in progress, {est['n_waiting']} waiting   "
                     f"({est['frac']:.0%} of all training steps)")
        for i in est["runs"]:
            if i["done"]:
                continue
            speed = f"{i['fps']:,.0f} steps/s" if i["fps"] else "not started"
            lines.append(f"  {i['tag']:<18} {i['layout']:<23} seed {i['seed']:<2}  "
                         f"{i['step']:>10,} / {i['steps']:,}  {i['step'] / i['steps']:4.0%}  "
                         f"{speed:>16}  ~{fmt_dur(i['left'])} left")
        if est["left"]:
            lines.append(f"Training time left: about {fmt_dur(est['left'])}")
        left_total += est["left"]
        short = f"training {est['frac']:.0%} ({est['n_done']}/{len(runs)} done) | "
        lines.append("")

    after = []
    eff = st.get("paused_phase") if phase == "paused" else phase
    upcoming = plan[plan.index(eff):] if eff in plan else ([] if phase == "finished" else plan)
    if "evaluate" in upcoming:
        todo = [s for s in SUITES if s not in st.get("eval_done", [])]
        cur = st.get("eval_suite") if eff == "evaluate" and st.get("eval_suite") in todo else None
        e = eval_estimate(a, todo, cur)
        left_total += e
        after.append(f"evaluation ~{fmt_dur(e)}")
        if eff == "evaluate":
            short = f"evaluating {cur or ''} | "
    if "analyze" in upcoming:
        left_total += ANALYZE_S
        after.append(f"analysis ~{fmt_dur(ANALYZE_S)}")
    if "video" in upcoming:
        n_v = len([l for l in MAIN_ORDER if not os.path.exists(video_path(l))])
        v = (VIDEO_S * a.video_seconds / 40.0 * -(-n_v // max(1, min(a.jobs, 6)))
             if n_v else 0)
        left_total += v
        if v:
            after.append(f"videos ~{fmt_dur(v)}")
        if eff == "video":
            short = f"videos ({len(MAIN_ORDER) - n_v}/{len(MAIN_ORDER)} done) | "
    if after:
        lines.append("Still to come (rough estimates): " + ", ".join(after))
    if phase != "finished":
        lines.append(f"Everything done in about {fmt_dur(left_total)}"
                     f"{', around ' + clock(now + left_total) if alive else ' of running time'}")
    lines.append("\nEstimates settle down after the first 10 to 15 minutes of each stage.")
    short = (f"  [progress {clock(now)}] {short}all done in ~{fmt_dur(left_total)} "
             f"(around {clock(now + left_total)})")
    return "\n".join(lines), short


def run_watched(cmd: List[str]) -> int:
    print("  $", " ".join(cmd), flush=True)
    p = subprocess.Popen(cmd, cwd=HERE)
    while p.poll() is None:
        if pause_requested():
            _stop(p)
            return PAUSED_EXIT
        time.sleep(2)
    return p.returncode


def _stop(p: subprocess.Popen) -> None:
    p.send_signal(signal.SIGTERM)
    try:
        p.wait(timeout=180)
    except subprocess.TimeoutExpired:
        p.kill()
        p.wait()


def run_pool(jobs: List[Tuple[List[str], str]], n: int, a=None,
             label: str = "runs") -> str:
    os.makedirs(os.path.join(HERE, "logs"), exist_ok=True)
    if not jobs:
        print("  nothing to do")
        return "done"
    print(f"  {len(jobs)} {label}, {n} at a time. Live logs: logs/   "
          f"Progress: python run_all.py status", flush=True)
    running, queue, failed = [], list(jobs), []
    t0 = last = time.time()
    while queue or running:
        while queue and len(running) < n and not pause_requested():
            cmd, log = queue.pop(0)
            f = open(os.path.join(HERE, log), "a")
            running.append((subprocess.Popen(cmd, cwd=HERE, stdout=f,
                                             stderr=subprocess.STDOUT), log, f))
            print(f"  [start] {log}", flush=True)
        if pause_requested():
            print(f"\n  Pause requested: saving {len(running)} running job(s)...", flush=True)
            for p, log, f in running:
                p.send_signal(signal.SIGTERM)
            for p, log, f in running:
                try:
                    p.wait(timeout=180)
                except subprocess.TimeoutExpired:
                    p.kill()
                    p.wait()
                f.close()
            return "paused"
        time.sleep(5)
        for p, log, f in running[:]:
            if p.poll() is not None:
                f.close()
                running.remove((p, log, f))
                ok = p.returncode == 0
                if not ok:
                    failed.append(log)
                print(f"  [{'done' if ok else f'FAILED ({p.returncode})'}] {log}  "
                      f"({fmt_dur(time.time() - t0)} elapsed, {len(queue)} queued, "
                      f"{len(running)} running)", flush=True)
        if a is not None and time.time() - last >= 60:
            last = time.time()
            report, short = status_report(a, read_state())
            print(short, flush=True)
            with open(STATUS_TXT, "w") as f:
                f.write(report + "\n")
    if failed:
        print("\n  Some jobs failed. Read the end of these logs, then run "
              "`python run_all.py resume`; finished jobs are skipped:")
        for log in failed:
            print("   ", log)
        return "failed"
    return "done"


def stage_train(stages: List[str], a) -> str:
    jobs, chains = [], set()
    for r in planned_runs(stages, a):
        name = f"{r['tag']}/{r['layout']}/seed{r['seed']}"
        if r["chain"]:
            if r["chain"] in chains:
                continue
            stages_of = [x for x in planned_runs(["curriculum"], a) if x["chain"] == r["chain"]]
            if all(run_done(x, a) for x in stages_of):
                print(f"  [done] curriculum seed{r['seed']}")
            else:
                jobs.append(([PY, "-u", "curriculum.py", "--seed", str(r["seed"]),
                              "--timesteps", str(a.timesteps), "--runs-dir", a.runs_dir],
                             r["log"]))
            chains.add(r["chain"])
            continue
        if os.path.exists(os.path.join(r["dir"], "final.pt")):
            if run_done(r, a):
                print(f"  [done] {name}")
                continue
            print(f"  [extend] {name} to {r['steps']:,} steps")
        jobs.append((train_cmd(r, a), r["log"]))
    return run_pool(jobs, a.jobs, a, "training jobs")


def stage_evaluate(a) -> str:
    ej = a.eval_jobs or max(1, (os.cpu_count() or 4) - 2)
    result = "done"
    for suite in SUITES:
        if suite in read_state().get("eval_done", []):
            print(f"\n== Evaluate: {suite} == already done")
            continue
        print(f"\n== Evaluate: {suite} ==", flush=True)
        write_state(eval_suite=suite)
        rc = run_watched([PY, "evaluate.py", "--suite", suite, "--episodes", str(a.episodes),
                          "--jobs", str(ej), "--runs-dir", a.runs_dir,
                          "--units", str(len(a.seeds.split(",")))])
        if rc == PAUSED_EXIT:
            return "paused"
        if rc != 0:
            print(f"  evaluate.py failed on suite {suite}; continuing with the others")
            result = "failed"
            continue
        write_state(eval_done=read_state().get("eval_done", []) + [suite])
    return result


def video_path(layout: str) -> str:
    return os.path.join(HERE, "videos", f"{layout}_comparison.mp4")


def stage_video(a) -> str:
    if not shutil.which("ffmpeg"):
        print("  ffmpeg not found, skipping videos (sudo apt install ffmpeg, then "
              "python run_all.py video)")
        return "done"
    jobs = []
    for lay in MAIN_ORDER:
        partial = video_path(lay)[:-4] + ".partial.mp4"
        if os.path.exists(partial):
            os.remove(partial)
        if os.path.exists(video_path(lay)):
            print(f"  [done] {os.path.relpath(video_path(lay), HERE)}")
            continue
        if not glob.glob(os.path.join(HERE, a.runs_dir, "main", lay, "seed*", "final.pt")):
            print(f"  [skip] no trained policy for {lay}")
            continue
        jobs.append(([PY, "-u", "make_video.py", "--layout", lay, "--algos", VIDEO_ALGOS,
                      "--runs-dir", a.runs_dir, "--seconds", str(a.video_seconds),
                      "--name", os.path.basename(video_path(lay))],
                     os.path.join("logs", f"video_{lay}.log")))
    res = run_pool(jobs, max(1, min(a.jobs, 6)), a, "videos")
    if res == "failed":
        print("  Some videos failed. The results are unaffected; rerun later with "
              "`python run_all.py video`.")
        return "done"
    return res


def stage_check(a):
    print("\n== Check: layouts, environment, a tiny training run, a tiny evaluation ==")
    if run([PY, "layouts.py"]) != 0:
        raise SystemExit("\nCheck failed at layouts.py. Send the output above.")
    if run([PY, "warehouse_env.py"]) != 0:
        raise SystemExit("\nCheck failed at warehouse_env.py. Send the output above.")

    tmp = tempfile.mkdtemp(prefix="warehouse_check_")
    runs, res = os.path.join(tmp, "runs"), os.path.join(tmp, "results")
    steps = [
        ("train.py", [PY, "train.py", "--layout", "small1", "--timesteps", "40000",
                      "--bc-steps", "8000", "--max-steps", "2000", "--runs-dir", runs]),
        ("evaluate.py", [PY, "evaluate.py", "--suite", "smoke", "--runs-dir", runs,
                         "--out", res, "--jobs", "1"]),
        ("analyze.py", [PY, "analyze.py", "--results", res, "--runs-dir", runs]),
    ]
    t0 = time.time()
    rate = None
    for name, cmd in steps:
        if run(cmd) != 0:
            print(f"\nCheck failed at {name}. Its working files are kept in:\n  {tmp}")
            print("\nAfter fixing, rerun just that step with:")
            print("  " + " ".join(f'"{c}"' if " " in c else c for c in cmd))
            print("\nOr rerun the whole check (starts a fresh folder):")
            print("  python run_all.py check")
            raise SystemExit(1)
        if name == "train.py":
            rate = 40000 / (time.time() - t0)
    shutil.rmtree(tmp, ignore_errors=True)
    per_run = a.timesteps / rate / 60
    print(f"\nAll checks passed. This machine trains at roughly {rate:,.0f} steps/s "
          f"per job (including startup), so one {a.timesteps:,}-step run takes about "
          f"{per_run:.0f} min on a small layout, and up to twice that on a large one.")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["check", "train", "calibrate", "ablation", "demand_arm",
                                     "curriculum", "evaluate", "analyze", "video", "all",
                                     "status", "pause", "resume"])
    p.add_argument("--jobs", type=int, default=8,
                   help="parallel training runs; about half your CPU threads")
    p.add_argument("--eval-jobs", type=int, default=None,
                   help="parallel evaluation workers (default: cores - 2)")
    p.add_argument("--seeds", default="10,11,12,13,14,15,16,17,18,19")
    p.add_argument("--timesteps", type=int, default=10_000_000)
    p.add_argument("--snapshots", default="3000000,6000000")
    p.add_argument("--episodes", type=int, default=30)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--video-seconds", type=float, default=40.0)
    p.add_argument("--no-video", action="store_true", help="skip the video stage of 'all'")
    return p


def cmd_status() -> None:
    st = read_state()
    if not st:
        print("No run recorded yet in this folder.")
        return
    a = build_parser().parse_args(st["argv"])
    print(status_report(a, st)[0])


def cmd_pause() -> None:
    st = read_state()
    if not pid_alive(st.get("pid")) or st.get("phase") == "finished":
        print("Nothing is running, so there is nothing to pause.")
        return
    open(PAUSE, "w").close()
    print("Pause requested. Running jobs will save a checkpoint and stop, usually within\n"
          "a minute. Check with `python run_all.py status`. To continue later:\n"
          "  nohup python run_all.py resume >> run.log 2>&1 &")


def main(argv=None):
    signal.signal(signal.SIGPIPE, signal.SIG_DFL)
    argv = list(sys.argv[1:] if argv is None else argv)
    a = build_parser().parse_args(argv)
    if a.stage == "status":
        return cmd_status()
    if a.stage == "pause":
        return cmd_pause()
    if a.stage == "check":
        return stage_check(a)

    st = read_state()
    if pid_alive(st.get("pid")) and st.get("pid") != os.getpid() \
            and st.get("phase") not in ("finished", "paused"):
        raise SystemExit(f"A run is already going (pid {st['pid']}). Use "
                         f"`python run_all.py status`, or `pause` it first. If you are "
                         f"sure nothing is running, delete run_state.json.")
    if a.stage == "resume":
        if not st or st.get("phase") == "finished":
            raise SystemExit("Nothing to resume: no unfinished run is recorded in this folder.")
        print(f"Resuming: python run_all.py {' '.join(st['argv'])}")
        argv = st["argv"]
        a = build_parser().parse_args(argv)
        keep = {"eval_done": st.get("eval_done", []), "started": st.get("started", time.time())}
    else:
        keep = {"eval_done": [], "started": time.time()}
    if pause_requested():
        os.remove(PAUSE)

    if a.stage == "all":
        train_stages = list(ALL_TRAINING)
        plan = ["train", "evaluate", "analyze"] + ([] if a.no_video else ["video"])
    elif a.stage in TRAIN_SETS or a.stage == "curriculum":
        train_stages, plan = [a.stage], ["train"]
    else:
        train_stages, plan = [], [a.stage]
    write_state(argv=argv, pid=os.getpid(), plan=plan, train_stages=train_stages,
                phase="", **keep)
    signal.signal(signal.SIGTERM, lambda *_: open(PAUSE, "w").close())
    signal.signal(signal.SIGINT, lambda *_: open(PAUSE, "w").close())

    t0 = time.time()
    outcome = "done"
    for phase in plan:
        write_state(phase=phase)
        if phase == "train":
            print(f"\n== Train: {', '.join(train_stages)} "
                  f"({len(planned_runs(train_stages, a))} runs in one queue) ==", flush=True)
            res = stage_train(train_stages, a)
        elif phase == "evaluate":
            res = stage_evaluate(a)
        elif phase == "analyze":
            print("\n== Analyse ==", flush=True)
            rc = run_watched([PY, "analyze.py", "--runs-dir", a.runs_dir,
                              "--final-steps", str(a.timesteps)])
            res = "paused" if rc == PAUSED_EXIT else ("done" if rc == 0 else "failed")
            if res == "done":
                print("\nOpen results/analysis/summary.md first.")
        else:
            print("\n== Videos: one four-way comparison per layout ==", flush=True)
            res = stage_video(a)
        if res == "paused":
            if pause_requested():
                os.remove(PAUSE)
            write_state(phase="paused", paused_phase=phase)
            print("\nPaused. Everything finished so far is saved. To continue:\n"
                  "  nohup python run_all.py resume >> run.log 2>&1 &", flush=True)
            return
        if res == "failed":
            outcome = "finished with failures (see above)"
            if phase == "train":
                write_state(phase="stopped")
                print("\nStopping here: later stages need every training run. Fix the "
                      "failed run, then `python run_all.py resume`.")
                return
    write_state(phase="finished", finished=time.time())
    report, _ = status_report(a, read_state())
    with open(STATUS_TXT, "w") as f:
        f.write(report + "\n")
    total = time.time() - read_state().get("started", t0)
    print(f"\nAll stages {outcome}. This session: {fmt_dur(time.time() - t0)}; "
          f"since the first start (including any pauses): {fmt_dur(total)}.")


if __name__ == "__main__":
    main()
