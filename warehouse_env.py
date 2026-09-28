from __future__ import annotations

import math
from typing import Dict, Optional, Tuple

import numpy as np
import gymnasium as gym
from gymnasium import spaces

try:
    import rvo2
except ImportError as exc:
    raise ImportError(
        "rvo2 is required. Activate the habitat conda environment, or "
        "install Python-RVO2 from https://github.com/sybrenstuvel/Python-RVO2"
    ) from exc

from layouts import Layout, get_layout


REWARD_PRESETS: Dict[str, Dict[str, float]] = {
    "v1": {
        "delivery": 10.0, "progress": 0.5, "safety": -0.5, "collision": 0.0,
        "contact": -2.0, "time": -0.10, "wall": -1.0, "safety_shaped": 0.0,
    },
    "v2": {
        "delivery": 10.0, "progress": 0.5, "safety": -1.2, "collision": -3.0,
        "contact": -0.25, "time": -0.01, "wall": -1.0, "safety_shaped": 1.0,
    },
    "throughput_only": {
        "delivery": 10.0, "progress": 0.5, "safety": 0.0, "collision": 0.0,
        "contact": 0.0, "time": -0.01, "wall": -1.0, "safety_shaped": 1.0,
    },
    "safety_only": {
        "delivery": 0.0, "progress": 0.0, "safety": -1.2, "collision": -3.0,
        "contact": -0.25, "time": -0.01, "wall": -1.0, "safety_shaped": 1.0,
    },
}


EPISODE_SECONDS = 600
EPISODE_STEPS = int(EPISODE_SECONDS * 24)


class WarehouseEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 24}

    ROBOT_R = 0.25
    HUMAN_R = 0.20
    SAFETY_R = 0.80
    CONTACT_TOL = 0.05
    INTERACT_R = 2.00
    ROBOT_SPD = 1.6
    HUMAN_SPD = 1.2
    SIM_HZ = 24.0
    CONGESTED_SPD = 0.30

    HUMAN_SPEED_JITTER = 0.15
    HUMAN_PAUSE_MAX_S = 2.5
    HUMAN_PAUSE_PROB = 0.5
    ROBOT_START_JITTER = 0.15
    ROBOT_STAGGER_MAX_S = 2.0
    DWELL_JITTER = 0.40

    OBS_DIM = 26
    N_NEAR_ROBOTS = 3
    N_NEAR_HUMANS = 2
    MAX_ROBOTS = 4
    MAX_HUMANS = 7
    GLOBAL_DIM = MAX_ROBOTS * 6 + MAX_HUMANS * 2
    ACT_DIM = 9

    WAYPOINT_TOL = 0.50
    HUMAN_WP_TOL = 0.40

    _ACTIONS = np.array([
        [0.0, 0.0], [0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0],
        [0.7, 0.7], [-0.7, 0.7], [0.7, -0.7], [-0.7, -0.7],
    ], dtype=np.float64)

    def __init__(self,
                 layout: str | Layout = "small1",
                 demand: str = "standard",
                 density: str = "standard",
                 reward_preset: str = "v2",
                 render_mode: Optional[str] = None,
                 max_steps: int = None,
                 human_speed_scale: float = 1.0,
                 human_avoid_r: Optional[float] = None,
                 robot_scale: float = 1.0):
        super().__init__()
        self.robot_scale = float(robot_scale)
        self.human_speed_scale = float(human_speed_scale)
        self.human_avoid_r = float(self.HUMAN_R if human_avoid_r is None else human_avoid_r)

        self.layout: Layout = (layout if isinstance(layout, Layout)
                               else get_layout(layout))
        self.demand = demand
        self.density = density
        self.render_mode = render_mode
        self.max_steps = int(EPISODE_STEPS if max_steps is None else max_steps)

        if reward_preset not in REWARD_PRESETS:
            raise KeyError(f"Unknown reward preset {reward_preset!r}. "
                           f"Valid: {sorted(REWARD_PRESETS)}")
        self.reward_preset = reward_preset
        w = REWARD_PRESETS[reward_preset]
        self.R_DELIVERY = w["delivery"]
        self.R_PROGRESS = w["progress"]
        self.R_SAFETY = w["safety"]
        self.R_COLLISION = w["collision"]
        self.R_CONTACT = w["contact"]
        self.R_TIME = w["time"]
        self.R_WALL = w["wall"]
        self._safety_shaped = bool(w["safety_shaped"])

        lay = self.layout
        self.WW, self.WH = lay.width, lay.height
        base = lay.n_robots
        self.N_ROBOTS = base if self.robot_scale == 1.0 else int(math.ceil(base * self.robot_scale))
        self.N_HUMANS = lay.n_humans(density)
        starts, tasks = list(lay.robot_starts), list(lay.robot_tasks)
        pick, deliv = list(lay.pickup_idx), list(lay.delivery_idx)
        for j in range(self.N_ROBOTS - base):
            a, b = starts[j % base], starts[(j + 1) % base]
            starts.append(((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0))
            tasks.append(tasks[j % base])
            pick.append(pick[j % base])
            deliv.append(deliv[j % base])
        if (self.N_ROBOTS > self.MAX_ROBOTS and self.robot_scale == 1.0) \
                or self.N_HUMANS > self.MAX_HUMANS:
            raise ValueError(
                f"{lay.name}/{density} needs {self.N_ROBOTS} robots and "
                f"{self.N_HUMANS} humans; limits are {self.MAX_ROBOTS}/"
                f"{self.MAX_HUMANS}. Raising them changes GLOBAL_DIM, "
                f"so every checkpoint would need retraining.")

        self._dt = 1.0 / self.SIM_HZ
        self._dwell_base = lay.dwell(demand) * self.SIM_HZ

        box = spaces.Box(-1.0, 1.0, (self.OBS_DIM,), np.float32)
        self._keys = tuple(f"robot_{i}" for i in range(self.N_ROBOTS))
        self.observation_space = spaces.Dict({k: box for k in self._keys})
        self.action_space = spaces.Dict(
            {k: spaces.Discrete(self.ACT_DIM) for k in self._keys})
        self.global_state_space = spaces.Box(
            -2.0, 2.0, (self.GLOBAL_DIM,), np.float32)

        self._tasks = [np.asarray(r, np.float64) for r in tasks]
        self._task_len = np.array([len(r) for r in self._tasks], np.intp)
        self._pickup_idx = np.asarray(pick, np.intp)
        self._delivery_idx = np.asarray(deliv, np.intp)
        self._starts = np.asarray(starts, np.float64)
        self._hroutes = [np.asarray(lay.human_routes[h], np.float64)
                         for h in range(self.N_HUMANS)]
        self._hroute_len = np.array([len(r) for r in self._hroutes], np.intp)
        self._hspeeds_nominal = np.asarray(
            lay.human_speeds[:self.N_HUMANS], np.float64)

        n_all = self.N_HUMANS + self.N_ROBOTS
        iu, ju = np.triu_indices(n_all, k=1)
        rr = (iu >= self.N_HUMANS) & (ju >= self.N_HUMANS)
        rh = (iu >= self.N_HUMANS) ^ (ju >= self.N_HUMANS)
        keep = rr | rh
        self._pi, self._pj = iu[keep], ju[keep]
        self._prr, self._prh = rr[keep], rh[keep]
        self._pri = np.where(self._pi >= self.N_HUMANS,
                             self._pi - self.N_HUMANS, -1)
        self._prj = np.where(self._pj >= self.N_HUMANS,
                             self._pj - self.N_HUMANS, -1)
        self._rh_idx = np.flatnonzero(self._prh)

        radius = np.where(np.arange(n_all) < self.N_HUMANS,
                          self.HUMAN_R, self.ROBOT_R)
        self._coll_r2 = (radius[self._pi] + radius[self._pj]
                         + self.CONTACT_TOL) ** 2
        self._safe_r2 = self.SAFETY_R ** 2
        self._inter_r2 = self.INTERACT_R ** 2
        self._inv_ww, self._inv_wh = 2.0 / self.WW, 2.0 / self.WH
        self._inv_spd = 1.0 / self.ROBOT_SPD
        self._cong_spd2 = self.CONGESTED_SPD ** 2

        R, H = self.N_ROBOTS, self.N_HUMANS
        self._rpos = np.zeros((R, 2))
        self._rvel = np.zeros((R, 2))
        self._hpos = np.zeros((H, 2))
        self._all_pos = np.zeros((n_all, 2))
        self._task_idx = np.zeros(R, np.intp)
        self._carrying = np.zeros(R, bool)
        self._wait = np.zeros(R, np.intp)
        self._hwp_idx = np.ones(H, np.intp)
        self._hwait = np.zeros(H, np.intp)
        self._hspeeds = self._hspeeds_nominal.copy()
        self._hspeeds_norm = np.zeros(H, np.float32)
        self._prev_dist = np.zeros(R)
        self._in_zone = np.zeros(self._rh_idx.size, bool)
        self._in_contact = np.zeros(self._pi.size, bool)
        self._in_inter = np.zeros(self._rh_idx.size, bool)
        self._step_count = 0

        self._obs_buf = np.zeros((R, self.OBS_DIM), np.float32)
        self._rew_buf = np.zeros(R)
        self._global_buf = np.zeros(self.GLOBAL_DIM, np.float32)
        self._reset_counters()

        self._sim = self._build_sim()
        self._has_set_vel = hasattr(self._sim, "setAgentVelocity")

    def _reset_counters(self) -> None:
        self.episode_deliveries = 0
        self.episode_collisions_rr = 0
        self.episode_collisions_rh = 0
        self.episode_contact_events = 0
        self.episode_safety_evts = 0
        self.episode_wall_hits = 0
        self.episode_distance = 0.0
        self.episode_interactions = 0
        self.episode_violations = 0
        self.episode_min_hr = np.inf
        self.episode_active_steps = 0
        self.episode_congested_steps = 0

    def _build_sim(self):
        sim = rvo2.PyRVOSimulator(self._dt, 5.0, 15, 8.0, 2.0,
                                  self.ROBOT_R, self.ROBOT_SPD)
        for i in range(self.N_ROBOTS):
            sim.addAgent(tuple(self._starts[i]), 5.0, 15, 8.0, 2.0,
                         self.ROBOT_R, self.ROBOT_SPD, (0.0, 0.0))
        for h in range(self.N_HUMANS):
            sim.addAgent(tuple(self._hroutes[h][0]), 5.0, 10, 5.0, 1.5,
                         self.human_avoid_r, self.HUMAN_SPD * self.human_speed_scale,
                         (0.0, 0.0))
        for (x1, y1, x2, y2) in self.layout.wall_segs:
            sim.addObstacle([(x1, y1), (x2, y2)])
        for poly in self.layout.shelves:
            sim.addObstacle([tuple(map(float, p)) for p in poly])
        sim.processObstacles()
        return sim

    def _clamp(self, p: np.ndarray, radius: float) -> bool:
        m = radius + 0.10
        hit = False
        if p[0] < m:
            p[0] = m; hit = True
        elif p[0] > self.WW - m:
            p[0] = self.WW - m; hit = True
        if p[1] < m:
            p[1] = m; hit = True
        elif p[1] > self.WH - m:
            p[1] = self.WH - m; hit = True
        return hit

    def _target_of(self, i: int) -> np.ndarray:
        return self._tasks[i][self._task_idx[i]]

    def _dwell_frames(self) -> int:
        j = 1.0 + self.np_random.uniform(-self.DWELL_JITTER, self.DWELL_JITTER)
        return max(1, int(round(self._dwell_base * j)))

    def _write_obs(self, i: int) -> None:
        buf = self._obs_buf[i]
        rp = self._rpos[i]
        tgt = self._target_of(i)
        c = lambda v: -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)

        buf[0] = rp[0] * self._inv_ww - 1.0
        buf[1] = rp[1] * self._inv_wh - 1.0
        buf[2] = c(self._rvel[i, 0] * self._inv_spd)
        buf[3] = c(self._rvel[i, 1] * self._inv_spd)
        buf[4] = c((tgt[0] - rp[0]) * 0.1)
        buf[5] = c((tgt[1] - rp[1]) * 0.1)
        buf[6] = 1.0 if self._carrying[i] else 0.0

        o = 7
        if self.N_ROBOTS > 1:
            d = self._rpos - rp
            d2 = np.einsum("ij,ij->i", d, d)
            d2[i] = np.inf
            order = np.argsort(d2)
        for k in range(self.N_NEAR_ROBOTS):
            if self.N_ROBOTS > 1 and k < self.N_ROBOTS - 1:
                j = order[k]
                buf[o] = c(d[j, 0] * 0.1)
                buf[o + 1] = c(d[j, 1] * 0.1)
                buf[o + 2] = 1.0 if self._carrying[j] else 0.0
            else:
                buf[o] = 1.0; buf[o + 1] = 1.0; buf[o + 2] = 0.0
            o += 3

        dh = self._hpos - rp
        dh2 = np.einsum("ij,ij->i", dh, dh)
        horder = np.argsort(dh2)
        for k in range(self.N_NEAR_HUMANS):
            if k < self.N_HUMANS:
                h = horder[k]
                buf[o] = c(dh[h, 0] * 0.1)
                buf[o + 1] = c(dh[h, 1] * 0.1)
                buf[o + 2] = self._hspeeds_norm[h]
            else:
                buf[o] = 1.0; buf[o + 1] = 1.0; buf[o + 2] = 0.0
            o += 3

        buf[o] = min(1.0, rp[1] * 0.2)
        buf[o + 1] = min(1.0, (self.WH - rp[1]) * 0.2)
        buf[o + 2] = min(1.0, rp[0] * 0.2)
        buf[o + 3] = min(1.0, (self.WW - rp[0]) * 0.2)

    def _fill_obs(self) -> np.ndarray:
        for i in range(self.N_ROBOTS):
            self._write_obs(i)
        return self._obs_buf

    def get_global_state(self) -> np.ndarray:
        g = self._global_buf
        g.fill(0.0)
        c = lambda v: -1.0 if v < -1.0 else (1.0 if v > 1.0 else v)
        o = 0
        for i in range(min(self.N_ROBOTS, self.MAX_ROBOTS)):
            g[o] = self._rpos[i, 0] * self._inv_ww - 1.0
            g[o + 1] = self._rpos[i, 1] * self._inv_wh - 1.0
            g[o + 2] = c(self._rvel[i, 0] * self._inv_spd)
            g[o + 3] = c(self._rvel[i, 1] * self._inv_spd)
            g[o + 4] = 1.0 if self._carrying[i] else 0.0
            g[o + 5] = self._task_idx[i] / self._task_len[i]
            o += 6
        o = self.MAX_ROBOTS * 6
        for h in range(self.N_HUMANS):
            g[o] = self._hpos[h, 0] * self._inv_ww - 1.0
            g[o + 1] = self._hpos[h, 1] * self._inv_wh - 1.0
            o += 2
        return g.copy()

    def episode_metrics(self) -> Dict[str, float]:
        steps = max(1, self._step_count)
        hours = steps / self.SIM_HZ / 3600.0
        contacts = self.episode_collisions_rh + self.episode_collisions_rr
        return {
            "orders_per_hour": self.episode_deliveries / hours,
            "congestion_prop": (self.episode_congested_steps
                                / max(1, self.episode_active_steps)),
            "collision_rate_100": 100.0 * contacts / steps,
            "violations_per_interaction": (self.episode_violations
                                           / max(1, self.episode_interactions)),
            "distance_per_delivery": (self.episode_distance / self.episode_deliveries
                                      if self.episode_deliveries else float("nan")),
            "distance_m": self.episode_distance,
            "deliveries": self.episode_deliveries,
            "collisions_rh": self.episode_collisions_rh,
            "collisions_rr": self.episode_collisions_rr,
            "contact_events": self.episode_contact_events,
            "safety_steps": self.episode_safety_evts,
            "interactions": self.episode_interactions,
            "violations": self.episode_violations,
            "min_separation": (float(np.sqrt(self.episode_min_hr))
                               if np.isfinite(self.episode_min_hr)
                               else float("nan")),
            "wall_hits": self.episode_wall_hits,
            "steps": self._step_count,
        }

    def reset_arrays(self, seed: Optional[int] = None):
        super().reset(seed=seed)
        rng = self.np_random
        R, H = self.N_ROBOTS, self.N_HUMANS

        self._rpos[:] = self._starts + rng.uniform(
            -self.ROBOT_START_JITTER, self.ROBOT_START_JITTER, (R, 2))
        self._rvel.fill(0.0)
        self._task_idx.fill(0)
        self._carrying.fill(False)
        self._wait[:] = rng.integers(
            0, int(self.ROBOT_STAGGER_MAX_S * self.SIM_HZ) + 1, R)

        for h in range(H):
            route = self._hroutes[h]
            for _ in range(20):
                k = int(rng.integers(0, self._hroute_len[h]))
                nxt = (k + 1) % self._hroute_len[h]
                t = rng.uniform(0.0, 1.0)
                p = route[k] + t * (route[nxt] - route[k])
                if np.min(np.sum((self._rpos - p) ** 2, 1)) > 1.2 ** 2:
                    break
            self._hpos[h] = p
            self._hwp_idx[h] = nxt
        self._hwait.fill(0)
        self._hspeeds[:] = self._hspeeds_nominal * (1.0 + rng.uniform(
            -self.HUMAN_SPEED_JITTER, self.HUMAN_SPEED_JITTER, H))
        self._hspeeds[:] = np.minimum(self._hspeeds, self.HUMAN_SPD) * self.human_speed_scale
        self._hspeeds_norm[:] = self._hspeeds / self.HUMAN_SPD

        self._in_zone.fill(False)
        self._in_inter.fill(False)
        self._in_contact.fill(False)
        self._step_count = 0
        self._reset_counters()

        sim, zero = self._sim, (0.0, 0.0)
        for i in range(R):
            sim.setAgentPosition(i, (float(self._rpos[i, 0]),
                                     float(self._rpos[i, 1])))
            sim.setAgentPrefVelocity(i, zero)
            if self._has_set_vel:
                sim.setAgentVelocity(i, zero)
        for h in range(H):
            a = R + h
            sim.setAgentPosition(a, (float(self._hpos[h, 0]),
                                     float(self._hpos[h, 1])))
            sim.setAgentPrefVelocity(a, zero)
            if self._has_set_vel:
                sim.setAgentVelocity(a, zero)

        info = {"global_state": self.get_global_state(),
                "layout": self.layout.name, "demand": self.demand,
                "density": self.density}
        return self._fill_obs().copy(), info

    def reset(self, seed=None, options=None):
        obs, info = self.reset_arrays(seed)
        return {k: obs[i] for i, k in enumerate(self._keys)}, info

    def step_arrays(self, actions: np.ndarray):
        self._step_count += 1
        sim, rng = self._sim, self.np_random
        R, H = self.N_ROBOTS, self.N_HUMANS
        rew = self._rew_buf
        rew.fill(self.R_TIME)

        for i in range(R):
            t = self._tasks[i][self._task_idx[i]]
            dx, dy = t[0] - self._rpos[i, 0], t[1] - self._rpos[i, 1]
            self._prev_dist[i] = (dx * dx + dy * dy) ** 0.5

        for i in range(R):
            if self._wait[i] > 0:
                sim.setAgentPrefVelocity(i, (0.0, 0.0))
            else:
                d = self._ACTIONS[int(actions[i])]
                sim.setAgentPrefVelocity(i, (float(d[0] * self.ROBOT_SPD),
                                             float(d[1] * self.ROBOT_SPD)))

        for h in range(H):
            a = R + h
            if self._hwait[h] > 0:
                self._hwait[h] -= 1
                sim.setAgentPrefVelocity(a, (0.0, 0.0))
                continue
            tgt = self._hroutes[h][self._hwp_idx[h]]
            dx, dy = tgt[0] - self._hpos[h, 0], tgt[1] - self._hpos[h, 1]
            dist = (dx * dx + dy * dy) ** 0.5
            if dist < 1e-6:
                sim.setAgentPrefVelocity(a, (0.0, 0.0))
            else:
                s = min(self._hspeeds[h], dist * 1.5) / dist
                sim.setAgentPrefVelocity(a, (float(dx * s), float(dy * s)))

        sim.doStep()

        for i in range(R):
            px, py = sim.getAgentPosition(i)
            p = self._rpos[i]
            self.episode_distance += ((px - p[0]) ** 2 + (py - p[1]) ** 2) ** 0.5
            p[0], p[1] = px, py
            if self._clamp(p, self.ROBOT_R):
                rew[i] += self.R_WALL
                self.episode_wall_hits += 1
                sim.setAgentPosition(i, (float(p[0]), float(p[1])))
            vx, vy = sim.getAgentVelocity(i)
            self._rvel[i, 0], self._rvel[i, 1] = vx, vy

            if self._wait[i] > 0:
                self._wait[i] -= 1
                continue

            self.episode_active_steps += 1
            if vx * vx + vy * vy < self._cong_spd2:
                self.episode_congested_steps += 1

            cur = self._task_idx[i]
            tgt = self._tasks[i][cur]
            dx, dy = tgt[0] - p[0], tgt[1] - p[1]
            if dx * dx + dy * dy < self.WAYPOINT_TOL ** 2:
                self._task_idx[i] = (cur + 1) % self._task_len[i]
                self._wait[i] = self._dwell_frames()
                if cur == self._delivery_idx[i]:
                    rew[i] += self.R_DELIVERY
                    self.episode_deliveries += 1
            self._carrying[i] = (self._pickup_idx[i] <= self._task_idx[i]
                                 < self._delivery_idx[i])

            nt = self._tasks[i][self._task_idx[i]]
            ndx, ndy = nt[0] - p[0], nt[1] - p[1]
            nd = (ndx * ndx + ndy * ndy) ** 0.5
            if nd < self._prev_dist[i]:
                rew[i] += self.R_PROGRESS * (self._prev_dist[i] - nd)

        for h in range(H):
            a = R + h
            px, py = sim.getAgentPosition(a)
            p = self._hpos[h]
            p[0], p[1] = px, py
            if self._clamp(p, self.HUMAN_R):
                sim.setAgentPosition(a, (float(p[0]), float(p[1])))
            if self._hwait[h] > 0:
                continue
            tgt = self._hroutes[h][self._hwp_idx[h]]
            dx, dy = tgt[0] - p[0], tgt[1] - p[1]
            if dx * dx + dy * dy < self.HUMAN_WP_TOL ** 2:
                self._hwp_idx[h] = (self._hwp_idx[h] + 1) % self._hroute_len[h]
                if rng.random() < self.HUMAN_PAUSE_PROB:
                    self._hwait[h] = int(rng.uniform(0.0, self.HUMAN_PAUSE_MAX_S)
                                         * self.SIM_HZ)

        self._all_pos[:H] = self._hpos
        self._all_pos[H:] = self._rpos
        diff = self._all_pos[self._pi] - self._all_pos[self._pj]
        d2 = np.einsum("ij,ij->i", diff, diff)

        collided = d2 < self._coll_r2
        if collided.any():
            self.episode_collisions_rr += int(np.count_nonzero(collided & self._prr))
            self.episode_collisions_rh += int(np.count_nonzero(collided & self._prh))
            onset = collided & ~self._in_contact
            for idx in np.flatnonzero(collided):
                pen = self.R_CONTACT + (self.R_COLLISION if onset[idx] else 0.0)
                for r in (self._pri[idx], self._prj[idx]):
                    if r >= 0:
                        rew[r] += pen
            self.episode_contact_events += int(np.count_nonzero(onset))
        self._in_contact[:] = collided

        near = (~collided) & (d2 < self._safe_r2) & self._prh
        if near.any():
            idxs = np.flatnonzero(near)
            self.episode_safety_evts += idxs.size
            depth = (1.0 - np.sqrt(d2[idxs]) / self.SAFETY_R
                     if self._safety_shaped else np.ones(idxs.size))
            for n, idx in enumerate(idxs):
                pen = self.R_SAFETY * float(depth[n])
                for r in (self._pri[idx], self._prj[idx]):
                    if r >= 0:
                        rew[r] += pen

        rh_d2 = d2[self._rh_idx]
        if rh_d2.size:
            self.episode_min_hr = min(self.episode_min_hr, float(rh_d2.min()))
            in_inter = rh_d2 < self._inter_r2
            in_zone = rh_d2 < self._safe_r2
            self.episode_interactions += int(np.count_nonzero(in_inter & ~self._in_inter))
            self.episode_violations += int(np.count_nonzero(in_zone & ~self._in_zone))
            self._in_inter[:] = in_inter
            self._in_zone[:] = in_zone

        truncated = self._step_count >= self.max_steps
        info = {"global_state": self.get_global_state(),
                "deliveries": self.episode_deliveries,
                "collisions_rh": self.episode_collisions_rh,
                "collisions_rr": self.episode_collisions_rr,
                "step": self._step_count}
        if truncated:
            info["episode"] = self.episode_metrics()
        return self._fill_obs().copy(), rew.copy(), truncated, info

    def step(self, actions: Dict[str, int]):
        act = np.fromiter((actions[k] for k in self._keys), np.intp,
                          count=self.N_ROBOTS)
        obs, rew, trunc, info = self.step_arrays(act)
        return ({k: obs[i] for i, k in enumerate(self._keys)},
                {k: float(rew[i]) for i, k in enumerate(self._keys)},
                False, trunc, info)

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig = plt.figure(figsize=(12, 8), dpi=80)
        draw_frame(self, fig.add_subplot(111))
        fig.canvas.draw()
        frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
        plt.close(fig)
        return frame

    def close(self):
        self._sim = None


def draw_frame(env: WarehouseEnv, ax) -> None:
    from matplotlib.patches import Circle, Polygon, Rectangle
    lay = env.layout
    ax.clear()
    ax.set_facecolor("#EFF1ED")
    ax.set_xlim(-1, env.WW + 1)
    ax.set_ylim(-1, env.WH + 1)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.add_patch(Polygon(lay.footprint, closed=True, facecolor="#EDEFF2",
                         edgecolor="#9AA0A8", lw=1.4))
    for r, col in ((lay.receiving, "#8FBEE8"), (lay.shipping, "#F2937A"),
                   (lay.staging, "#A5D8C4")):
        ax.add_patch(Rectangle(r[:2], r[2], r[3], facecolor=col, lw=0))
    for r in lay.aisles:
        ax.add_patch(Rectangle(r[:2], r[2], r[3], facecolor="#B9C9DC", lw=0))
    for poly in lay.shelves:
        ax.add_patch(Polygon(poly, closed=True, facecolor="#F5C24F",
                             edgecolor="#BD8C22", lw=0.6))
    for (px, py) in lay.charge_pads:
        ax.add_patch(Rectangle((px - 0.75, py - 0.75), 1.5, 1.5,
                               facecolor="#5FC244", lw=0, alpha=0.8))
    for (px, py) in lay.pick_stations:
        ax.add_patch(Rectangle((px - 0.75, py - 0.55), 1.5, 1.1,
                               facecolor="#8A7DE0", lw=0))
    for (px, py) in lay.pd_points:
        ax.add_patch(Rectangle((px - 0.8, py - 0.8), 1.6, 1.6,
                               facecolor="#2BA8C4", lw=0))
    for h in range(env.N_HUMANS):
        p = env._hpos[h]
        ax.add_patch(Circle(p, env.SAFETY_R, fill=False, edgecolor="#3C5A8C",
                            ls="--", lw=0.6, alpha=0.5))
        ax.add_patch(Circle(p, env.HUMAN_R, color="#3C5A8C", zorder=5))
    for i in range(env.N_ROBOTS):
        p = env._rpos[i]
        ax.add_patch(Circle(p, env.ROBOT_R, color="#F2793A", zorder=6))
        ax.text(p[0], p[1], str(i + 1), ha="center", va="center",
                fontsize=6, color="white", fontweight="bold", zorder=7)


class SingleAgentWarehouseWrapper(gym.Wrapper):
    def __init__(self, env: WarehouseEnv, robot_idx: int = 0):
        super().__init__(env)
        self._key = f"robot_{robot_idx}"
        self.observation_space = env.observation_space[self._key]
        self.action_space = env.action_space[self._key]
        self._others = {k: 0 for k in env._keys if k != self._key}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[self._key], info

    def step(self, action):
        obs, rew, term, trunc, info = self.env.step(
            {**self._others, self._key: action})
        return obs[self._key], rew[self._key], term, trunc, info


if __name__ == "__main__":
    import time
    from layouts import LAYOUTS

    print("Warehouse Environment v5 - Sanity Check")
    print("=" * 76)
    for nm in LAYOUTS:
        for density in ("standard", "high", "stress"):
            env = WarehouseEnv(layout=nm, density=density, max_steps=300)
            obs, info = env.reset_arrays(seed=1)
            assert obs.shape == (env.N_ROBOTS, env.OBS_DIM)
            assert info["global_state"].shape == (env.GLOBAL_DIM,)
            t0 = time.perf_counter()
            for _ in range(300):
                obs, rew, trunc, info = env.step_arrays(
                    env.np_random.integers(0, env.ACT_DIM, env.N_ROBOTS))
            dt = time.perf_counter() - t0
            m = info["episode"]
            print(f"{env.layout.label:<32} {density:<9} "
                  f"{env.N_ROBOTS}R/{env.N_HUMANS}H  {300 / dt:6.0f} steps/s  "
                  f"interactions={m['interactions']:<3} "
                  f"violations={m['violations']}")
            env.close()

    def run(seed):
        e = WarehouseEnv("small1", max_steps=200)
        e.reset_arrays(seed=seed)
        for _ in range(200):
            _, _, _, info = e.step_arrays(np.zeros(e.N_ROBOTS, np.intp))
        return e._hpos.copy()
    a, b, c = run(7), run(7), run(8)
    assert np.allclose(a, b), "same seed must reproduce the same episode"
    assert not np.allclose(a, c), "different seeds must differ"
    print("=" * 76)
    print("Seeding verified: identical under the same seed, varied across seeds.")
