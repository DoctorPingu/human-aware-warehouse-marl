"""
Warehouse Multi-Agent Gymnasium Environment
Honours Project: Deep Reinforcement Learning for Multi-Agent Coordination
in Mixed Human-Robot Warehouse Environments
Mohammed Al-Khudhair

This module defines the WarehouseEnv class, a multi-agent Gymnasium-compatible
environment for training autonomous mobile robots (AMRs) to coordinate in a
shared warehouse space alongside human workers.

Architecture:
    - Centralised Training with Decentralised Execution (CTDE)
    - Each robot has its own local observation space
    - A shared global state is available during training for the centralised critic
    - Reward function combines throughput, safety, and efficiency signals

Observation space per robot (23-dimensional):
    - Own position (x, y): 2
    - Own velocity (vx, vy): 2
    - Own task target (tx, ty): 2
    - Carrying item (binary): 1
    - Nearest 3 robots: relative (dx, dy, carrying) each = 9
    - Nearest 2 humans: relative (dx, dy, speed) each = 6
    - Nearest wall distance (4 directions): 4 (added to reach dim 26)
    Total: 26

Action space per robot (Discrete 9):
    0: stay
    1-4: cardinal movement (N, S, E, W)
    5-8: diagonal movement (NE, NW, SE, SW)

Global state (for centralised critic):
    All robot positions, velocities, carry states, and task indices
    All human positions
    Concatenated into a single flat vector

Reward function:
    +10.0  on successful delivery to pick station
    +0.5   per step making progress toward current target
    -0.5   per step in safety zone of a human (ISO 10218 proximity)
    -2.0   per step in collision distance of any other agent
    -0.1   per step (small time penalty to encourage efficiency)
    -1.0   per step pressing against a wall (hard clamp triggered)
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
import rvo2
from typing import Dict, List, Tuple, Optional, Any

class WarehouseEnv(gym.Env):
    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 24}

    # ── World constants ───────────────────────────────────────────
    WW = 34.0
    WH = 24.0

    ROBOT_R     = 0.25
    HUMAN_R     = 0.20
    SAFETY_R    = 0.80
    COLLISION_R = 0.44
    ROBOT_SPD   = 1.6
    HUMAN_SPD   = 1.2
    STEP_SIZE   = 0.055

    N_ROBOTS = 3
    N_HUMANS = 3

    OBS_DIM    = 26
    GLOBAL_DIM = N_ROBOTS * 6 + N_HUMANS * 2

    # ── Action deltas: stay, N, S, E, W, NE, NW, SE, SW ──────────
    _ACTIONS = np.array([
        [0.0,  0.0 ],
        [0.0,  1.0 ],
        [0.0, -1.0 ],
        [1.0,  0.0 ],
        [-1.0,  0.0],
        [0.7,  0.7 ],
        [-0.7,  0.7],
        [0.7, -0.7 ],
        [-0.7, -0.7],
    ], dtype=np.float32)

    # ── Reward shaping weights ────────────────────────────────────
    R_DELIVERY   =  10.0
    R_PROGRESS   =   0.5
    R_SAFETY_PEN =  -0.5
    R_COLLISION  =  -2.0
    R_TIME       =  -0.1
    R_WALL_HIT   =  -1.0

    # ── Layout ────────────────────────────────────────────────────
    SHELF_COLS  = [5.0, 10.0, 15.0, 20.0, 25.0]
    SHELF_ROWS_L = [1.5,  5.0,  8.5]
    SHELF_ROWS_U = [14.5, 18.0, 21.5]
    SHELF_W = 2.0
    SHELF_D = 0.85

    CX = [2.0, 8.5, 13.5, 18.5, 23.5, 28.5, 32.0]
    CY = [3.7, 7.2, 12.0, 16.7, 20.2]

    STATIONS = [(30.8, 4.0),(30.8, 12.0),(30.8, 20.0)]

    ROBOT_STARTS = [
        (8.5,  3.7),
        (13.5, 3.7),
        (18.5, 3.7),
    ]

    ROBOT_TASKS = [
        [(8.5,3.7),(8.5,7.2),(8.5,12.0),(28.5,12.0),(31.5,12.0),(30.8,4.0)],
        [(13.5,3.7),(13.5,7.2),(13.5,12.0),(28.5,12.0),(31.5,12.0),(30.8,12.0)],
        [(18.5,3.7),(18.5,7.2),(18.5,12.0),(28.5,12.0),(31.5,12.0),(30.8,20.0)],
    ]

    PICKUP_IDX   = [1, 1, 1]
    DELIVERY_IDX = [5, 5, 5]

    HUMAN_WPS = [
        [(2.5,12.0),(13.5,12.0),(13.5,3.7),(8.5,3.7),(8.5,12.0),(28.5,12.0)],
        [(8.5,3.7),(8.5,7.2),(18.5,7.2),(18.5,3.7),(8.5,3.7)],
        [(18.5,16.7),(18.5,12.0),(8.5,12.0),(8.5,16.7),(18.5,16.7)],
    ]
    HUMAN_SPEEDS = [1.0, 0.85, 0.85]

    WALL_SEGS = [
        (0.0, 0.0, WW,  0.0),
        (WW,  0.0, WW,  WH ),
        (WW,  WH,  0.0, WH ),
        (0.0, WH,  0.0, 0.0),
        (4.0, 0.0, 4.0, 9.5),
        (4.0,14.5, 4.0, WH ),
        (30.0,0.0,30.0, 9.5),
        (30.0,14.5,30.0,WH ),
    ]

    def __init__(self, render_mode: Optional[str] = None,
                 max_steps: int = 2000):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self._step_count = 0

        # ── Observation and action spaces ─────────────────────────
        low  = np.full(self.OBS_DIM, -1.0, dtype=np.float32)
        high = np.full(self.OBS_DIM,  1.0, dtype=np.float32)
        single_obs = spaces.Box(low=low, high=high, dtype=np.float32)

        self.observation_space = spaces.Dict({
            f'robot_{i}': single_obs for i in range(self.N_ROBOTS)
        })

        self.action_space = spaces.Dict({
            f'robot_{i}': spaces.Discrete(9) for i in range(self.N_ROBOTS)
        })

        self.global_state_space = spaces.Box(
            low  = np.full(self.GLOBAL_DIM, -2.0, dtype=np.float32),
            high = np.full(self.GLOBAL_DIM,  2.0, dtype=np.float32),
            dtype=np.float32,
        )

        # ── Pre-build shelves list ────────────────────────────────
        self.shelves = []
        for cx in self.SHELF_COLS:
            for ry in self.SHELF_ROWS_L + self.SHELF_ROWS_U:
                self.shelves.append((cx, ry, self.SHELF_W, self.SHELF_D))

        # Internal state (initialised in reset)
        self.robot_pos      : List[np.ndarray] = []
        self.robot_vel      : List[np.ndarray] = []
        self.robot_task_idx : List[int]         = []
        self.robot_carrying : List[bool]        = []
        self.robot_wait     : List[int]         = []
        self.human_pos      : List[np.ndarray] = []
        self.human_wp_idx   : List[int]         = []
        self._sim           : Optional[rvo2.PyRVOSimulator] = None

        # Episode statistics
        self.episode_deliveries  = 0
        self.episode_collisions  = 0
        self.episode_safety_evts = 0
        self.episode_wall_hits   = 0

        # Render buffer
        self._render_frame: Optional[np.ndarray] = None

    def _build_sim(self) -> rvo2.PyRVOSimulator:
        DT = 1.0 / 24.0
        sim = rvo2.PyRVOSimulator(DT, 5.0, 15, 8.0, 2.0,
                                   self.ROBOT_R, self.ROBOT_SPD)
        for i in range(self.N_ROBOTS):
            sim.addAgent(tuple(self.robot_pos[i]),
                         5.0,15,8.0,2.0,
                         self.ROBOT_R, self.ROBOT_SPD,(0.0,0.0))
        for h in range(self.N_HUMANS):
            sim.addAgent(tuple(self.human_pos[h]),
                         5.0,10,5.0,1.5,
                         self.HUMAN_R, self.HUMAN_SPD,(0.0,0.0))
        for (x1,y1,x2,y2) in self.WALL_SEGS:
            sim.addObstacle([(x1,y1),(x2,y2)])
        for (sx,sy,sw,sd) in self.shelves:
            sim.addObstacle([(sx,sy),(sx+sw,sy),(sx+sw,sy+sd),(sx,sy+sd)])
        sim.processObstacles()
        return sim

    def _clamp(self, pos: np.ndarray, radius: float) -> Tuple[np.ndarray, bool]:
        m = radius + 0.10
        p = pos.copy(); hit = False
        if p[0] < m:          p[0] = m;          hit = True
        if p[0] > self.WW-m:  p[0] = self.WW-m;  hit = True
        if p[1] < m:          p[1] = m;          hit = True
        if p[1] > self.WH-m:  p[1] = self.WH-m;  hit = True
        return p, hit

    def _normalise_pos(self, pos: np.ndarray) -> np.ndarray:
        return np.array([pos[0]/self.WW*2-1, pos[1]/self.WH*2-1],
                        dtype=np.float32)

    def _normalise_delta(self, delta: np.ndarray,
                          max_dist: float = 10.0) -> np.ndarray:
        return np.clip(delta / max_dist, -1.0, 1.0).astype(np.float32)

    def _wall_distances(self, pos: np.ndarray) -> np.ndarray:
        return np.array([
            min(1.0, pos[1]           / 5.0),
            min(1.0, (self.WH-pos[1]) / 5.0),
            min(1.0, pos[0]           / 5.0),
            min(1.0, (self.WW-pos[0]) / 5.0),
        ], dtype=np.float32)

    def _get_obs(self, robot_idx: int) -> np.ndarray:
        rp = self.robot_pos[robot_idx]
        rv = self.robot_vel[robot_idx]
        task_idx = self.robot_task_idx[robot_idx]
        tasks = self.ROBOT_TASKS[robot_idx]
        tgt = np.array(tasks[min(task_idx, len(tasks)-1)])
        carrying = float(self.robot_carrying[robot_idx])

        own = np.concatenate([
            self._normalise_pos(rp),
            np.clip(rv / self.ROBOT_SPD, -1, 1).astype(np.float32),
            self._normalise_delta(tgt - rp),
            [carrying],
        ])

        other_robots = []
        dists = [(np.linalg.norm(self.robot_pos[j]-rp), j)
                 for j in range(self.N_ROBOTS) if j != robot_idx]
        dists.sort()
        for k in range(3):
            if k < len(dists):
                _, j = dists[k]
                delta = self._normalise_delta(self.robot_pos[j] - rp)
                other_robots.extend([delta[0], delta[1],
                                      float(self.robot_carrying[j])])
            else:
                other_robots.extend([1.0, 1.0, 0.0])

        humans = []
        hdists = [(np.linalg.norm(self.human_pos[h]-rp), h)
                  for h in range(self.N_HUMANS)]
        hdists.sort()
        for k in range(2):
            if k < len(hdists):
                _, h = hdists[k]
                delta = self._normalise_delta(self.human_pos[h] - rp)
                spd   = float(self.HUMAN_SPEEDS[h] / self.HUMAN_SPD)
                humans.extend([delta[0], delta[1], spd])
            else:
                humans.extend([1.0, 1.0, 0.0])

        walls = self._wall_distances(rp)

        obs = np.concatenate([own, other_robots, humans, walls])
        assert obs.shape == (self.OBS_DIM,), \
            f"Obs shape mismatch: {obs.shape} vs {self.OBS_DIM}"
        return obs.astype(np.float32)

    def get_global_state(self) -> np.ndarray:
        parts = []
        for i in range(self.N_ROBOTS):
            parts.append(self._normalise_pos(self.robot_pos[i]))
            parts.append(np.clip(self.robot_vel[i]/self.ROBOT_SPD,-1,1
                                 ).astype(np.float32))
            parts.append(np.array([float(self.robot_carrying[i]),
                                    self.robot_task_idx[i] /
                                    len(self.ROBOT_TASKS[i])],
                                   dtype=np.float32))
        for h in range(self.N_HUMANS):
            parts.append(self._normalise_pos(self.human_pos[h]))
        return np.concatenate(parts).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.robot_pos      = [np.array(p,dtype=float)
                               for p in self.ROBOT_STARTS]
        self.robot_vel      = [np.zeros(2) for _ in range(self.N_ROBOTS)]
        self.robot_task_idx = [0]*self.N_ROBOTS
        self.robot_carrying = [False]*self.N_ROBOTS
        self.robot_wait     = [0]*self.N_ROBOTS
        self.human_pos      = [np.array(self.HUMAN_WPS[h][0],dtype=float)
                               for h in range(self.N_HUMANS)]
        self.human_wp_idx   = [1]*self.N_HUMANS
        self._step_count    = 0

        self.episode_deliveries  = 0
        self.episode_collisions  = 0
        self.episode_safety_evts = 0
        self.episode_wall_hits   = 0

        self._sim = self._build_sim()

        obs = {f'robot_{i}': self._get_obs(i) for i in range(self.N_ROBOTS)}
        info = {
            'global_state': self.get_global_state(),
            'deliveries': 0,
            'collisions': 0,
        }
        return obs, info

    def step(self, actions: Dict[str, int]):
        assert self._sim is not None, "Call reset() before step()"
        self._step_count += 1
        rewards    = {f'robot_{i}': self.R_TIME for i in range(self.N_ROBOTS)}
        prev_dists = [np.linalg.norm(
                          np.array(self.ROBOT_TASKS[i][
                              min(self.robot_task_idx[i],
                                  len(self.ROBOT_TASKS[i])-1)]) -
                          self.robot_pos[i])
                      for i in range(self.N_ROBOTS)]

        # ── Apply actions as preferred velocities to RVO2 ─────────
        for i in range(self.N_ROBOTS):
            act = int(actions[f'robot_{i}'])
            delta = self._ACTIONS[act]
            if self.robot_wait[i] > 0:
                self._sim.setAgentPrefVelocity(i,(0.0,0.0))
            else:
                pref = delta * self.ROBOT_SPD
                self._sim.setAgentPrefVelocity(i, tuple(pref.astype(float)))

        # ── Human preferred velocities ────────────────────────────
        for h in range(self.N_HUMANS):
            tgt  = np.array(self.HUMAN_WPS[h][self.human_wp_idx[h]])
            diff = tgt - self.human_pos[h]
            dist = np.linalg.norm(diff)
            if dist < 0.35:
                self._sim.setAgentPrefVelocity(self.N_ROBOTS+h,(0.0,0.0))
            else:
                spd = min(self.HUMAN_SPEEDS[h], dist*1.5)
                self._sim.setAgentPrefVelocity(
                    self.N_ROBOTS+h, tuple((diff/dist*spd).astype(float)))

        self._sim.doStep()

        # ── Update robot positions ────────────────────────────────
        for i in range(self.N_ROBOTS):
            pos = np.array(self._sim.getAgentPosition(i))
            pos, hit = self._clamp(pos, self.ROBOT_R)
            if hit:
                rewards[f'robot_{i}']    += self.R_WALL_HIT
                self.episode_wall_hits   += 1
            vel = np.array(self._sim.getAgentVelocity(i))
            self.robot_pos[i] = pos
            self.robot_vel[i] = vel
            self._sim.setAgentPosition(i, tuple(pos))

            if self.robot_wait[i] > 0:
                self.robot_wait[i] -= 1
                continue

            tasks   = self.ROBOT_TASKS[i]
            cur_idx = self.robot_task_idx[i]
            tgt     = np.array(tasks[cur_idx])
            dist    = np.linalg.norm(tgt - pos)

            if dist < 0.50:
                self.robot_task_idx[i] = (cur_idx+1) % len(tasks)
                self.robot_wait[i]     = int(24 * 0.5)
                if cur_idx == self.DELIVERY_IDX[i]:
                    rewards[f'robot_{i}'] += self.R_DELIVERY
                    self.episode_deliveries += 1

            self.robot_carrying[i] = (
                self.PICKUP_IDX[i] <=
                self.robot_task_idx[i] <
                self.DELIVERY_IDX[i]
            )

            new_dist = np.linalg.norm(
                np.array(tasks[self.robot_task_idx[i]]) - pos)
            if new_dist < prev_dists[i]:
                rewards[f'robot_{i}'] += self.R_PROGRESS * (
                    prev_dists[i] - new_dist)

        # ── Update human positions ────────────────────────────────
        for h in range(self.N_HUMANS):
            pos = np.array(self._sim.getAgentPosition(self.N_ROBOTS+h))
            pos, _ = self._clamp(pos, self.HUMAN_R)
            self.human_pos[h] = pos
            self._sim.setAgentPosition(self.N_ROBOTS+h, tuple(pos))
            tgt = np.array(self.HUMAN_WPS[h][self.human_wp_idx[h]])
            if np.linalg.norm(tgt - pos) < 0.40:
                self.human_wp_idx[h] = (
                    (self.human_wp_idx[h]+1) % len(self.HUMAN_WPS[h]))

        # ── Proximity and collision checks ────────────────────────
        all_pos = self.human_pos + self.robot_pos
        n = len(all_pos)
        for i in range(n):
            for j in range(i+1, n):
                d = np.linalg.norm(all_pos[i]-all_pos[j])
                if d < self.COLLISION_R:
                    self.episode_collisions += 1
                    ri = i - self.N_HUMANS
                    rj = j - self.N_HUMANS
                    if ri >= 0:
                        rewards[f'robot_{ri}'] += self.R_COLLISION
                    if rj >= 0:
                        rewards[f'robot_{rj}'] += self.R_COLLISION
                elif d < self.SAFETY_R:
                    self.episode_safety_evts += 1
                    ri = i - self.N_HUMANS
                    rj = j - self.N_HUMANS
                    human_involved = (i < self.N_HUMANS or j < self.N_HUMANS)
                    if human_involved:
                        if ri >= 0:
                            rewards[f'robot_{ri}'] += self.R_SAFETY_PEN
                        if rj >= 0:
                            rewards[f'robot_{rj}'] += self.R_SAFETY_PEN

        # ── Observations ──────────────────────────────────────────
        obs = {f'robot_{i}': self._get_obs(i) for i in range(self.N_ROBOTS)}

        terminated = False
        truncated  = self._step_count >= self.max_steps

        info = {
            'global_state'   : self.get_global_state(),
            'deliveries'     : self.episode_deliveries,
            'collisions'     : self.episode_collisions,
            'safety_events'  : self.episode_safety_evts,
            'wall_hits'      : self.episode_wall_hits,
            'step'           : self._step_count,
        }

        return obs, rewards, terminated, truncated, info

    def render(self):
        if self.render_mode is None:
            return None
        try:
            import matplotlib
            matplotlib.use('Agg')
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            return None

        fig, ax = plt.subplots(figsize=(12,8), dpi=80)
        ax.set_facecolor('#0A1828')
        fig.patch.set_facecolor('#060E1A')
        ax.set_xlim(0, self.WW)
        ax.set_ylim(0, self.WH)
        ax.set_aspect('equal')
        ax.axis('off')

        hw = mpatches.Rectangle((0,10.5),self.WW,3.0,
             facecolor='#0D2540',edgecolor='none')
        ax.add_patch(hw)

        for (sx,sy,sw,sd) in self.shelves:
            shelf = mpatches.Rectangle((sx,sy),sw,sd,
                    facecolor='#122438',edgecolor='#1C3C6A',lw=0.5)
            ax.add_patch(shelf)

        for (dx,dy) in [(0.8,4.0),(0.8,12.0),(0.8,20.0)]:
            ax.add_patch(mpatches.Rectangle((dx,dy-0.8),1.1,1.6,
                facecolor='#3A2810',edgecolor='#E8A018',lw=0.8))

        for (px,py) in self.STATIONS:
            ax.add_patch(mpatches.Rectangle((px,py-0.8),1.2,1.6,
                facecolor='#182E22',edgecolor='#48B870',lw=1.0))

        for i, pos in enumerate(self.robot_pos):
            c = '#3498DB'
            circle = plt.Circle(pos, self.ROBOT_R, color=c, zorder=5)
            ax.add_patch(circle)
            safe   = plt.Circle(pos, self.SAFETY_R,
                                fill=False, edgecolor=c,
                                linestyle='--', lw=0.5, alpha=0.3)
            ax.add_patch(safe)
            if self.robot_carrying[i]:
                box = mpatches.Rectangle((pos[0]-0.15,pos[1]-0.15),
                      0.30,0.30,facecolor='#D4622A',edgecolor='none',zorder=6)
                ax.add_patch(box)
            ax.text(pos[0],pos[1],f'R{i+1}',ha='center',va='center',
                    fontsize=5,color='white',fontweight='bold',zorder=7)

        for h, pos in enumerate(self.human_pos):
            circle = plt.Circle(pos, self.HUMAN_R,
                                color='#FF6B35', zorder=5)
            ax.add_patch(circle)
            ax.text(pos[0],pos[1],f'W{h+1}',ha='center',va='center',
                    fontsize=5,color='white',fontweight='bold',zorder=7)

        t    = self._step_count / 24.0
        info = (f"Step: {self._step_count}  "
                f"t={t:.1f}s  "
                f"Deliveries: {self.episode_deliveries}  "
                f"Collisions: {self.episode_collisions}")
        ax.set_title(info, color='#AAC4E0', fontsize=9,
                     fontfamily='monospace')

        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        frame = np.frombuffer(fig.canvas.tostring_rgb(),
                              dtype=np.uint8).reshape(h_px, w_px, 3)
        plt.close(fig)
        return frame

    def close(self):
        self._sim = None


# ── Convenience: vectorised single-agent wrapper ──────────────────
class SingleAgentWarehouseWrapper(gym.Wrapper):
    """
    Wraps WarehouseEnv to expose a single robot as a standard Gymnasium
    env for algorithms that do not support multi-agent setups natively.
    Other robots follow a fixed ORCA policy during training of robot_idx.
    """
    def __init__(self, env: WarehouseEnv, robot_idx: int = 0):
        super().__init__(env)
        self.robot_idx        = robot_idx
        self.observation_space= env.observation_space[f'robot_{robot_idx}']
        self.action_space     = env.action_space[f'robot_{robot_idx}']
        self._other_actions   = {
            f'robot_{i}': 0
            for i in range(env.N_ROBOTS) if i != robot_idx
        }

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return obs[f'robot_{self.robot_idx}'], info

    def step(self, action):
        actions = {**self._other_actions,
                   f'robot_{self.robot_idx}': action}
        obs, rewards, terminated, truncated, info = self.env.step(actions)
        return (obs[f'robot_{self.robot_idx}'],
                rewards[f'robot_{self.robot_idx}'],
                terminated, truncated, info)


# ── Quick sanity check ────────────────────────────────────────────
if __name__ == '__main__':
    print("Warehouse Environment Sanity Check")
    print("=" * 40)

    env = WarehouseEnv(max_steps=200)
    obs, info = env.reset(seed=42)

    print(f"Observation keys    : {list(obs.keys())}")
    print(f"Obs dim per robot   : {obs['robot_0'].shape}")
    print(f"Global state dim    : {info['global_state'].shape}")
    print(f"Action space        : {env.action_space['robot_0']}")
    print(f"Obs space bounds    : [{env.observation_space['robot_0'].low.min():.1f}, "
          f"{env.observation_space['robot_0'].high.max():.1f}]")

    total_reward = {f'robot_{i}': 0.0 for i in range(env.N_ROBOTS)}
    for step in range(200):
        actions = {f'robot_{i}': env.action_space[f'robot_{i}'].sample()
                   for i in range(env.N_ROBOTS)}
        obs, rewards, terminated, truncated, info = env.step(actions)
        for k in total_reward:
            total_reward[k] += rewards[k]
        if truncated:
            break

    print(f"\nRandom policy 200 steps:")
    print(f"  Total rewards      : { {k: round(v,2) for k,v in total_reward.items()} }")
    print(f"  Deliveries         : {info['deliveries']}")
    print(f"  Collisions         : {info['collisions']}")
    print(f"  Safety events      : {info['safety_events']}")
    print(f"  Wall hits          : {info['wall_hits']}")
    print(f"\nEnvironment check passed. Ready for training.")
    env.close()
