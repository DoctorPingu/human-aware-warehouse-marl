from __future__ import annotations

import argparse
import csv
import json
import os
import signal
import sys
import time
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from layouts import DEMAND_LEVELS, DENSITY_LEVELS, get_layout
from networks import Actor, Critic, RunningMeanStd
from warehouse_env import EPISODE_STEPS, WarehouseEnv

TRAIN_SEED_BLOCK = 100_000
EVAL_SEED_BASE = 3_500_000_000

LOG_KEYS = ["orders_per_hour", "congestion_prop", "collision_rate_100",
            "violations_per_interaction", "deliveries", "collisions_rh",
            "collisions_rr", "min_separation"]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="CTDE PPO for warehouse coordination")
    p.add_argument("--layout", default="small1")
    p.add_argument("--demand", default="mix", help="standard | peak | mix")
    p.add_argument("--density", default="mix", help="standard | high | mix")
    p.add_argument("--reward", default="v2", help="v1 | v2 | throughput_only | safety_only")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--tag", default="main", help="run group name, e.g. main, v1, ablation")
    p.add_argument("--timesteps", type=int, default=3_000_000, help="environment steps")
    p.add_argument("--n-envs", type=int, default=8)
    p.add_argument("--rollout", type=int, default=256, help="steps per env per update")
    p.add_argument("--epochs", type=int, default=10,
                   help="4 in the proof-of-concept; 10 learned markedly faster in testing")
    p.add_argument("--minibatch", type=int, default=512, help="robot-samples per minibatch")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--lr-min", type=float, default=1e-5)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae-lambda", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.003,
                   help="0.01 in the proof-of-concept; lowered because after behaviour "
                        "cloning it pulled the policy back towards random")
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--max-grad-norm", type=float, default=0.5)
    p.add_argument("--max-steps", type=int, default=EPISODE_STEPS,
                   help="episode length in steps (default: 10 simulated minutes)")
    p.add_argument("--save-every", type=int, default=100_000)
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--device", default="cpu",
                   help="cpu is usually faster for networks this small; see README")
    p.add_argument("--threads", type=int, default=1,
                   help="torch threads; keep 1 when running several jobs in parallel")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--init-from", default=None,
                   help="warm-start weights from another checkpoint (curriculum)")
    p.add_argument("--bc-steps", type=int, default=40_000,
                   help="env steps of ORCA demonstrations for behaviour cloning; 0 = off "
                        "(pure PPO from scratch)")
    p.add_argument("--bc-epochs", type=int, default=15)
    p.add_argument("--bc-noise", type=float, default=0.2,
                   help="chance of executing a random action while recording the expert "
                        "label, so the policy also learns to recover from mistakes")
    p.add_argument("--bc-coef", type=float, default=0.5,
                   help="weight of the imitation anchor at the start of PPO")
    p.add_argument("--anchor-steps", type=int, default=2_000_000,
                   help="environment steps over which the imitation anchor fades to zero")
    p.add_argument("--snapshots", default="",
                   help="comma-separated steps at which to keep a copy of the policy")
    p.add_argument("--critic-warmup", type=int, default=10,
                   help="updates that train only the critic after BC, so an untrained "
                        "critic cannot wreck the cloned policy")
    return p.parse_args(argv)


def conditions(demand: str, density: str) -> List[tuple]:
    dems = DEMAND_LEVELS if demand == "mix" else (demand,)
    dens = DENSITY_LEVELS if density == "mix" else (density,)
    return [(a, b) for a in dems for b in dens]


class VecEnv:
    def __init__(self, args):
        conds = conditions(args.demand, args.density)
        self.envs = [WarehouseEnv(args.layout, *conds[e % len(conds)],
                                  reward_preset=args.reward,
                                  max_steps=args.max_steps)
                     for e in range(args.n_envs)]
        self.n = len(self.envs)
        self.R = self.envs[0].N_ROBOTS
        self._base = [args.seed * TRAIN_SEED_BLOCK + e * 10_000 for e in range(self.n)]
        self._ep = [0] * self.n

    def reset(self):
        obs, gs = [], []
        for e, env in enumerate(self.envs):
            o, info = env.reset_arrays(seed=self._base[e])
            obs.append(o); gs.append(info["global_state"])
        return np.stack(obs), np.stack(gs)

    def step(self, actions: np.ndarray):
        obs, gs, rew, done = [], [], [], np.zeros(self.n, bool)
        final_obs, final_gs, finished = {}, {}, []
        for e, env in enumerate(self.envs):
            o, r, trunc, info = env.step_arrays(actions[e])
            g = info["global_state"]
            if trunc:
                done[e] = True
                final_obs[e], final_gs[e] = o, g
                finished.append(info["episode"])
                self._ep[e] += 1
                o, rinfo = env.reset_arrays(seed=self._base[e] + self._ep[e])
                g = rinfo["global_state"]
            obs.append(o); gs.append(g); rew.append(r)
        return (np.stack(obs), np.stack(gs), np.stack(rew), done,
                final_obs, final_gs, finished)


def critic_input(obs: np.ndarray, gs: np.ndarray) -> np.ndarray:
    E, R, _ = obs.shape
    return np.concatenate([np.broadcast_to(gs[:, None, :], (E, R, gs.shape[-1])), obs], -1)


PAUSED_EXIT = 3
_stop_requested = False


def _request_stop(signum, frame):
    global _stop_requested
    _stop_requested = True


def save_ckpt(path, actor, critic, optim, rms, step, update, args, env):
    torch.save({
        "actor": actor.state_dict(), "critic": critic.state_dict(),
        "optim": optim.state_dict(), "rms": rms.state_dict(),
        "step": step, "update": update,
        "obs_dim": env.OBS_DIM, "act_dim": env.ACT_DIM,
        "critic_dim": env.GLOBAL_DIM + env.OBS_DIM,
        "meta": {"layout": args.layout, "reward": args.reward, "seed": args.seed,
                 "tag": args.tag, "timesteps": step},
        "rng": {"np": np.random.get_state(), "torch": torch.get_rng_state()},
    }, path + ".tmp")
    os.replace(path + ".tmp", path)


def behaviour_clone(args, venv: "VecEnv", actor: Actor, device):
    from policies import OrcaPolicy
    rng = np.random.default_rng(args.seed + 12345)
    experts = [OrcaPolicy() for _ in venv.envs]
    obs, _ = venv.reset()
    for ex, env in zip(experts, venv.envs):
        ex.reset(env)
    X, Y = [], []
    steps = max(1, args.bc_steps // venv.n)
    for _ in range(steps):
        labels = np.stack([ex.act(env, obs[e]) for e, (ex, env)
                           in enumerate(zip(experts, venv.envs))])
        X.append(obs.reshape(-1, obs.shape[-1]).copy())
        Y.append(labels.reshape(-1).copy())
        executed = labels.copy()
        flip = rng.random(labels.shape) < args.bc_noise
        executed[flip] = rng.integers(0, venv.envs[0].ACT_DIM, int(flip.sum()))
        obs, _, _, done, _, _, _ = venv.step(executed)
        for e in np.flatnonzero(done):
            experts[e].reset(venv.envs[e])
    X = torch.as_tensor(np.concatenate(X), device=device)
    Y = torch.as_tensor(np.concatenate(Y), dtype=torch.long, device=device)

    opt = torch.optim.Adam(actor.parameters(), lr=1e-3)
    n = X.shape[0]
    for _ in range(args.bc_epochs):
        perm = torch.randperm(n, device=device)
        for s in range(0, n, 1024):
            idx = perm[s:s + 1024]
            loss = nn.functional.cross_entropy(actor(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    with torch.no_grad():
        acc = (actor(X).argmax(-1) == Y).float().mean().item()
    return ({"bc_samples": int(n), "bc_accuracy": acc, "bc_loss": float(loss.item())},
            (X, Y))


def train(args) -> str:
    signal.signal(signal.SIGTERM, _request_stop)
    signal.signal(signal.SIGINT, _request_stop)
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)

    layout = get_layout(args.layout).name
    run_dir = os.path.join(args.runs_dir, args.tag, layout, f"seed{args.seed}")
    os.makedirs(run_dir, exist_ok=True)

    venv = VecEnv(args)
    env0 = venv.envs[0]
    E, R, T = venv.n, venv.R, args.rollout
    cdim = env0.GLOBAL_DIM + env0.OBS_DIM

    actor = Actor(env0.OBS_DIM, env0.ACT_DIM).to(device)
    critic = Critic(cdim).to(device)
    params = list(actor.parameters()) + list(critic.parameters())
    optim = torch.optim.Adam(params, lr=args.lr, eps=1e-5)
    rms = RunningMeanStd()

    step, update = 0, 0
    latest = os.path.join(run_dir, "latest.pt")
    if args.resume and os.path.exists(latest):
        ck = torch.load(latest, map_location=device, weights_only=False)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        optim.load_state_dict(ck["optim"]); rms.load_state_dict(ck["rms"])
        step, update = ck["step"], ck["update"]
        np.random.set_state(ck["rng"]["np"]); torch.set_rng_state(ck["rng"]["torch"])
        print(f"Resumed from {latest} at step {step:,}")
    elif args.init_from:
        ck = torch.load(args.init_from, map_location=device, weights_only=False)
        actor.load_state_dict(ck["actor"]); critic.load_state_dict(ck["critic"])
        rms.load_state_dict(ck["rms"])
        print(f"Warm-started from {args.init_from}")

    bc_info, bc_data = {}, None
    bc_path = os.path.join(run_dir, "bc_data.pt")
    if step > 0 and os.path.exists(bc_path) and step < args.anchor_steps:
        saved = torch.load(bc_path, map_location=device, weights_only=False)
        bc_data = (saved["obs"], saved["act"])
    if step == 0 and not args.init_from and args.bc_steps > 0:
        t_bc = time.time()
        bc_info, bc_data = behaviour_clone(args, venv, actor, device)
        torch.save({"obs": bc_data[0].cpu(), "act": bc_data[1].cpu()}, bc_path)
        print(f"Behaviour cloning: {bc_info['bc_samples']:,} samples, "
              f"accuracy {bc_info['bc_accuracy']:.1%} in {time.time() - t_bc:.0f}s")

    cfg_path = os.path.join(run_dir, "config.json")
    prev_cfg = {}
    if args.resume and os.path.exists(cfg_path):
        with open(cfg_path) as f:
            prev_cfg = json.load(f)
    with open(cfg_path, "w") as f:
        json.dump({**prev_cfg, **vars(args), **bc_info, "layout_full": layout, "n_robots": R,
                   "conditions": conditions(args.demand, args.density)}, f, indent=2)

    prog_path = os.path.join(run_dir, "progress.csv")
    fields = (["step", "update", "fps", "episodes", "return"] + LOG_KEYS
              + ["pg_loss", "vf_loss", "entropy", "clip_frac", "lr"])
    new_file = not (args.resume and os.path.exists(prog_path))
    if not new_file:
        with open(prog_path, newline="") as f:
            kept = [r for r in csv.DictReader(f) if int(float(r["step"])) <= step]
        with open(prog_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(kept)
    prog = open(prog_path, "w" if new_file else "a", newline="")
    writer = csv.DictWriter(prog, fieldnames=fields)
    if new_file:
        writer.writeheader()

    n_updates = max(1, args.timesteps // (E * T))
    buf_obs = np.zeros((T, E, R, env0.OBS_DIM), np.float32)
    buf_cin = np.zeros((T, E, R, cdim), np.float32)
    buf_act = np.zeros((T, E, R), np.int64)
    buf_logp = np.zeros((T, E, R), np.float32)
    buf_rew = np.zeros((T, E, R), np.float32)
    buf_done = np.zeros((T, E), np.float32)
    buf_val = np.zeros((T, E, R), np.float32)
    ep_return = np.zeros(E)

    obs, gs = venv.reset()
    print(f"Training {layout} | reward={args.reward} seed={args.seed} | "
          f"{E} envs x {R} robots | {args.timesteps:,} steps | {n_updates} updates")
    t_start = time.time()
    snapshots = sorted(int(s) for s in args.snapshots.split(",") if s.strip())
    step_start = step
    if _stop_requested and step == 0:
        prog.close()
        print("Paused before training started; nothing to save yet. It will "
              "start from the beginning when resumed.", flush=True)
        sys.exit(PAUSED_EXIT)
    recent: List[Dict] = []
    recent_ret: List[float] = []

    while update < n_updates:
        lr = max(args.lr_min, args.lr * (1.0 - update / n_updates))
        for g in optim.param_groups:
            g["lr"] = lr

        for t in range(T):
            cin = critic_input(obs, gs)
            with torch.no_grad():
                o_t = torch.as_tensor(obs, device=device).reshape(E * R, -1)
                dist = torch.distributions.Categorical(logits=actor(o_t))
                a = dist.sample()
                lp = dist.log_prob(a)
                v = critic(torch.as_tensor(cin, device=device).reshape(E * R, -1))
            a_np = a.cpu().numpy().reshape(E, R)
            buf_obs[t], buf_cin[t], buf_act[t] = obs, cin, a_np
            buf_logp[t] = lp.cpu().numpy().reshape(E, R)
            buf_val[t] = v.cpu().numpy().reshape(E, R) * rms.std + rms.mean

            obs, gs, rew, done, fobs, fgs, finished = venv.step(a_np)
            ep_return += rew.mean(1)
            for e in fobs:
                fc = critic_input(fobs[e][None], fgs[e][None])
                with torch.no_grad():
                    fv = critic(torch.as_tensor(fc, device=device).reshape(R, -1))
                rew[e] += args.gamma * (fv.cpu().numpy() * rms.std + rms.mean)
                recent_ret.append(ep_return[e]); ep_return[e] = 0.0
            recent.extend(finished)
            buf_rew[t], buf_done[t] = rew, done
            step += E

        with torch.no_grad():
            nv = critic(torch.as_tensor(critic_input(obs, gs), device=device)
                        .reshape(E * R, -1)).cpu().numpy().reshape(E, R)
        nv = nv * rms.std + rms.mean
        adv = np.zeros((T, E, R), np.float32)
        last = np.zeros((E, R), np.float32)
        for t in reversed(range(T)):
            nxt = nv if t == T - 1 else buf_val[t + 1]
            nonterm = (1.0 - buf_done[t])[:, None]
            delta = buf_rew[t] + args.gamma * nxt * nonterm - buf_val[t]
            last = delta + args.gamma * args.gae_lambda * nonterm * last
            adv[t] = last
        ret = adv + buf_val
        rms.update(ret)

        N = T * E * R
        b_obs = torch.as_tensor(buf_obs.reshape(N, -1), device=device)
        b_cin = torch.as_tensor(buf_cin.reshape(N, -1), device=device)
        b_act = torch.as_tensor(buf_act.reshape(N), device=device)
        b_logp = torch.as_tensor(buf_logp.reshape(N), device=device)
        b_adv = torch.as_tensor(adv.reshape(N), device=device)
        b_ret = torch.as_tensor(((ret - rms.mean) / rms.std).reshape(N), device=device)

        bc_w = (args.bc_coef * max(0.0, 1.0 - step / args.anchor_steps)
                if bc_data is not None and args.anchor_steps > 0 else 0.0)
        pg_l, vf_l, ent_l, cf_l = [], [], [], []
        for _ in range(args.epochs):
            perm = torch.randperm(N, device=device)
            for s in range(0, N, args.minibatch):
                idx = perm[s:s + args.minibatch]
                dist = torch.distributions.Categorical(logits=actor(b_obs[idx]))
                ratio = (dist.log_prob(b_act[idx]) - b_logp[idx]).exp()
                a_mb = b_adv[idx]
                a_mb = (a_mb - a_mb.mean()) / (a_mb.std() + 1e-8)
                pg = torch.max(-a_mb * ratio,
                               -a_mb * ratio.clamp(1 - args.clip, 1 + args.clip)).mean()
                vf = 0.5 * ((critic(b_cin[idx]) - b_ret[idx]) ** 2).mean()
                ent = dist.entropy().mean()
                if update < args.critic_warmup and args.bc_steps > 0 and not args.resume:
                    loss = args.vf_coef * vf
                else:
                    loss = pg + args.vf_coef * vf - args.ent_coef * ent
                    if bc_w > 0:
                        j = torch.randint(0, bc_data[0].shape[0], (idx.numel(),),
                                          device=device)
                        loss = loss + bc_w * nn.functional.cross_entropy(
                            actor(bc_data[0][j]), bc_data[1][j])
                optim.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(params, args.max_grad_norm)
                optim.step()
                pg_l.append(pg.item()); vf_l.append(vf.item()); ent_l.append(ent.item())
                cf_l.append(((ratio - 1).abs() > args.clip).float().mean().item())
        update += 1

        fps = (step - step_start) / max(1e-9, time.time() - t_start)
        row = {"step": step, "update": update, "fps": round(fps),
               "episodes": len(recent),
               "return": float(np.mean(recent_ret)) if recent_ret else "",
               "pg_loss": np.mean(pg_l), "vf_loss": np.mean(vf_l),
               "entropy": np.mean(ent_l), "clip_frac": np.mean(cf_l), "lr": lr}
        for k in LOG_KEYS:
            row[k] = float(np.nanmean([m[k] for m in recent])) if recent else ""
        writer.writerow(row)
        prog.flush()
        if recent:
            print(f"  {step:>10,} | {fps:5.0f} sps | orders/h {row['orders_per_hour']:6.0f} | "
                  f"coll/100 {row['collision_rate_100']:5.2f} | "
                  f"viol/int {row['violations_per_interaction']:.2f} | "
                  f"cong {row['congestion_prop']:.2f} | ent {row['entropy']:.2f}")
            recent, recent_ret = [], []

        if step % args.save_every < E * T or update == n_updates or _stop_requested:
            save_ckpt(latest, actor, critic, optim, rms, step, update, args, env0)
        for snap in snapshots:
            snap_path = os.path.join(run_dir, f"snap_{snap}.pt")
            if step >= snap and not os.path.exists(snap_path):
                save_ckpt(snap_path, actor, critic, optim, rms, step, update, args, env0)
        if _stop_requested and update < n_updates:
            prog.close()
            print(f"Paused at step {step:,}; checkpoint saved. "
                  f"Rerun with --resume to continue.", flush=True)
            sys.exit(PAUSED_EXIT)

    final = os.path.join(run_dir, "final.pt")
    save_ckpt(final, actor, critic, optim, rms, step, update, args, env0)
    prog.close()
    mins = (time.time() - t_start) / 60
    print(f"Done in {mins:.1f} min -> {final}")
    return final


if __name__ == "__main__":
    train(parse_args())
