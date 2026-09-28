from __future__ import annotations

import heapq
import math
from functools import lru_cache
from typing import Dict, List, Optional, Tuple

import numpy as np

from warehouse_env import WarehouseEnv
from layouts import point_in_poly, polygon_edges

_DIRS = WarehouseEnv._ACTIONS / np.maximum(
    np.linalg.norm(WarehouseEnv._ACTIONS, axis=1, keepdims=True), 1e-9)


def steer(vec: np.ndarray, stop_within: float = 0.05) -> int:
    n = float(np.hypot(vec[0], vec[1]))
    if n < stop_within:
        return 0
    return int(1 + np.argmax(_DIRS[1:] @ (vec / n)))


class Policy:
    name = "base"

    def reset(self, env: WarehouseEnv) -> None:
        pass

    def act(self, env: WarehouseEnv, obs: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class OrcaPolicy(Policy):
    name = "orca"

    def act(self, env, obs):
        out = np.zeros(env.N_ROBOTS, np.intp)
        for i in range(env.N_ROBOTS):
            out[i] = steer(env._target_of(i) - env._rpos[i])
        return out


class HeuristicPolicy(Policy):
    name = "heuristic"
    HUMAN_STOP_R = 1.5
    ROBOT_YIELD_R = 1.0
    AHEAD_COS = 0.3
    STANDOFF_S = 2.0

    def reset(self, env):
        self._waited = np.zeros(env.N_ROBOTS, np.intp)
        self._timeout = int(self.STANDOFF_S * env.SIM_HZ)

    def act(self, env, obs):
        out = np.zeros(env.N_ROBOTS, np.intp)
        for i in range(env.N_ROBOTS):
            p = env._rpos[i]
            to_goal = env._target_of(i) - p
            dist = float(np.hypot(*to_goal))
            if dist < 0.05:
                continue
            heading = to_goal / dist
            blocked = False

            dh = env._hpos - p
            nh = np.linalg.norm(dh, axis=1)
            close = nh < self.HUMAN_STOP_R
            if close.any():
                cos = (dh[close] @ heading) / np.maximum(nh[close], 1e-9)
                blocked = bool((cos > self.AHEAD_COS).any())

            if not blocked:
                for j in range(i):
                    d = env._rpos[j] - p
                    nd = float(np.hypot(*d))
                    if nd < self.ROBOT_YIELD_R and (d @ heading) / nd > self.AHEAD_COS:
                        blocked = True
                        break

            if blocked and self._waited[i] < self._timeout:
                self._waited[i] += 1
                out[i] = 0
            else:
                self._waited[i] = 0
                out[i] = steer(to_goal)
        return out


_NEIGH = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1),
          (1, 1), (1, -1), (-1, 1), (-1, -1)]


@lru_cache(maxsize=None)
def _grid(layout_name: str, cell: float, clearance: float):
    from layouts import get_layout
    lay = get_layout(layout_name)
    nx, ny = int(lay.width / cell), int(lay.height / cell)
    free = np.zeros((nx, ny), bool)
    c2 = clearance * clearance
    edges = [polygon_edges(p) for p in lay.shelves]
    for gx in range(nx):
        for gy in range(ny):
            x, y = (gx + 0.5) * cell, (gy + 0.5) * cell
            if not point_in_poly(x, y, lay.footprint):
                continue
            ok = True
            for poly, eds in zip(lay.shelves, edges):
                if point_in_poly(x, y, poly):
                    ok = False
                    break
                for (x1, y1, x2, y2) in eds:
                    vx, vy = x2 - x1, y2 - y1
                    l2 = vx * vx + vy * vy
                    t = 0.0 if l2 == 0 else max(0.0, min(1.0, ((x - x1) * vx + (y - y1) * vy) / l2))
                    if (x - x1 - t * vx) ** 2 + (y - y1 - t * vy) ** 2 < c2:
                        ok = False
                        break
                if not ok:
                    break
            free[gx, gy] = ok
    return free


class MapfPolicy(Policy):
    name = "mapf"
    CELL = 0.5
    CLEARANCE = 0.40
    TICK = 8
    HORIZON = 120
    MAX_EXPAND = 20000

    def reset(self, env):
        self._free = _grid(env.layout.name, self.CELL, self.CLEARANCE)
        self._nx, self._ny = self._free.shape
        self._paths: List[List[Tuple[int, int]]] = [[] for _ in range(env.N_ROBOTS)]
        self._t0 = np.zeros(env.N_ROBOTS, np.intp)
        self._goal_idx = np.full(env.N_ROBOTS, -1, np.intp)
        self._dist_cache: Dict[Tuple[int, int], np.ndarray] = {}

    def _to_cell(self, p) -> Tuple[int, int]:
        gx = min(self._nx - 1, max(0, int(p[0] / self.CELL)))
        gy = min(self._ny - 1, max(0, int(p[1] / self.CELL)))
        if self._free[gx, gy]:
            return gx, gy
        best, bd = (gx, gy), 1e9
        for r in range(1, 6):
            for dx in range(-r, r + 1):
                for dy in range(-r, r + 1):
                    x, y = gx + dx, gy + dy
                    if 0 <= x < self._nx and 0 <= y < self._ny and self._free[x, y]:
                        d = dx * dx + dy * dy
                        if d < bd:
                            best, bd = (x, y), d
            if bd < 1e9:
                return best
        return gx, gy

    def _center(self, c) -> np.ndarray:
        return np.array([(c[0] + 0.5) * self.CELL, (c[1] + 0.5) * self.CELL])

    def _dist_to(self, goal) -> np.ndarray:
        d = self._dist_cache.get(goal)
        if d is not None:
            return d
        d = np.full((self._nx, self._ny), np.inf)
        d[goal] = 0.0
        q = [goal]
        for (x, y) in q:
            for dx, dy in _NEIGH[1:]:
                nx_, ny_ = x + dx, y + dy
                if (0 <= nx_ < self._nx and 0 <= ny_ < self._ny
                        and self._free[nx_, ny_] and d[nx_, ny_] == np.inf):
                    d[nx_, ny_] = d[x, y] + 1
                    q.append((nx_, ny_))
        self._dist_cache[goal] = d
        return d

    def _reserved(self, me: int, now: int):
        cells, edges = set(), set()
        for j, path in enumerate(self._paths):
            if j == me or not path:
                continue
            k0 = now - self._t0[j]
            for k in range(max(0, k0), len(path)):
                t = self._t0[j] + k
                cells.add((path[k], t))
                if k + 1 < len(path):
                    edges.add((path[k + 1], path[k], t))
            last = path[-1]
            for t in range(self._t0[j] + len(path), now + self.HORIZON + 1):
                cells.add((last, t))
        return cells, edges

    def _plan(self, i: int, start, goal, now: int):
        dist = self._dist_to(goal)
        if not np.isfinite(dist[start]):
            return [start]
        cells, edges = self._reserved(i, now)
        openq = [(dist[start], 0, start, now)]
        parent = {(start, now): None}
        best = (dist[start], (start, now))
        n = 0
        while openq and n < self.MAX_EXPAND:
            f, g, c, t = heapq.heappop(openq)
            n += 1
            if c == goal:
                best = (0, (c, t))
                break
            if dist[c] < best[0]:
                best = (dist[c], (c, t))
            if t - now >= self.HORIZON:
                continue
            for dx, dy in _NEIGH:
                nc = (c[0] + dx, c[1] + dy)
                if not (0 <= nc[0] < self._nx and 0 <= nc[1] < self._ny
                        and self._free[nc]):
                    continue
                nt = t + 1
                if (nc, nt) in cells or (c, nc, t) in edges:
                    continue
                key = (nc, nt)
                if key in parent:
                    continue
                parent[key] = (c, t)
                step = 0.0 if (dx, dy) == (0, 0) else (1.414 if dx and dy else 1.0)
                heapq.heappush(openq, (g + step + dist[nc], g + step, nc, nt))
        path, node = [], best[1]
        while node is not None:
            path.append(node[0])
            node = parent.get(node)
        return path[::-1] or [start]

    def act(self, env, obs):
        now = env._step_count // self.TICK
        out = np.zeros(env.N_ROBOTS, np.intp)
        for i in range(env.N_ROBOTS):
            if env._task_idx[i] != self._goal_idx[i] or not self._paths[i]:
                self._goal_idx[i] = env._task_idx[i]
                start = self._to_cell(env._rpos[i])
                goal = self._to_cell(env._target_of(i))
                self._paths[i] = self._plan(i, start, goal, now)
                self._t0[i] = now
            path = self._paths[i]
            k = min(len(path) - 1, max(0, now - self._t0[i] + 1))
            target = self._center(path[k])
            if k == len(path) - 1:
                target = env._target_of(i)
            out[i] = steer(target - env._rpos[i])
        return out


class PPOPolicy(Policy):
    name = "ppo"

    def __init__(self, checkpoint: str, deterministic: bool = True):
        import torch
        from networks import Actor
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
        self.actor = Actor(ckpt["obs_dim"], ckpt["act_dim"])
        self.actor.load_state_dict(ckpt["actor"])
        self.actor.eval()
        self.deterministic = deterministic
        self.meta = ckpt.get("meta", {})
        self._torch = torch

    def act(self, env, obs):
        with self._torch.no_grad():
            logits = self.actor(self._torch.as_tensor(obs, dtype=self._torch.float32))
            if self.deterministic:
                a = logits.argmax(-1)
            else:
                a = self._torch.distributions.Categorical(logits=logits).sample()
        return a.numpy().astype(np.intp)


BASELINES = {"orca": OrcaPolicy, "heuristic": HeuristicPolicy, "mapf": MapfPolicy}


def make_policy(name: str, checkpoint: Optional[str] = None) -> Policy:
    if name == "ppo":
        if checkpoint is None:
            raise ValueError("ppo needs a checkpoint path")
        return PPOPolicy(checkpoint)
    return BASELINES[name]()


def run_episode(env: WarehouseEnv, policy: Policy, seed: int,
                on_step=None) -> Dict[str, float]:
    obs, _ = env.reset_arrays(seed=seed)
    policy.reset(env)
    while True:
        obs, _, done, info = env.step_arrays(policy.act(env, obs))
        if on_step is not None:
            on_step(env)
        if done:
            return info["episode"]
