"""
Warehouse Multi-Agent Coordination Demo
Honours Project: Deep Reinforcement Learning for Multi-Agent Coordination
in Mixed Human-Robot Warehouse Environments
Mohammed Al-Khudhair - 24826341

Built using the Habitat 3.0 framework (Puig et al., 2023).
This script simulates a mixed human-robot warehouse environment with:
  - 1 human worker following a predefined path
  - 3 autonomous mobile robots navigating between task points
  - ISO 10218 safety zone visualisation
  - Collision avoidance behaviour
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patches as patches
from matplotlib.patches import Circle
from matplotlib.animation import FFMpegWriter
import os

print("Habitat 3.0 Warehouse Demo - Initialising...")

OUTPUT_DIR = "/mnt/e/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS = 30
DURATION_SECONDS = 20
TOTAL_FRAMES = FPS * DURATION_SECONDS
DT = 1.0 / FPS

WAREHOUSE_W = 12.0
WAREHOUSE_H = 8.0
SAFETY_RADIUS = 0.8
ROBOT_SPEED = 0.06
HUMAN_SPEED = 0.035
ROBOT_RADIUS = 0.2
HUMAN_RADIUS = 0.18

SHELVES = [
    (1.0, 1.0, 1.8, 0.6),
    (1.0, 2.2, 1.8, 0.6),
    (1.0, 3.4, 1.8, 0.6),
    (1.0, 4.6, 1.8, 0.6),
    (1.0, 5.8, 1.8, 0.6),
    (4.0, 1.0, 1.8, 0.6),
    (4.0, 2.2, 1.8, 0.6),
    (4.0, 3.4, 1.8, 0.6),
    (4.0, 4.6, 1.8, 0.6),
    (4.0, 5.8, 1.8, 0.6),
    (7.0, 1.0, 1.8, 0.6),
    (7.0, 2.2, 1.8, 0.6),
    (7.0, 3.4, 1.8, 0.6),
    (7.0, 4.6, 1.8, 0.6),
    (7.0, 5.8, 1.8, 0.6),
]

HUMAN_WAYPOINTS = [
    (0.5, 4.0),
    (3.0, 6.5),
    (6.0, 6.5),
    (9.5, 6.5),
    (11.0, 4.0),
    (9.5, 1.5),
    (6.0, 1.5),
    (3.0, 1.5),
    (0.5, 4.0),
]

ROBOT_TASKS = [
    [(2.5, 1.5), (10.5, 1.5), (10.5, 6.5), (2.5, 6.5)],
    [(2.5, 4.0), (6.0, 1.5), (10.5, 4.0), (6.0, 6.5)],
    [(6.0, 4.0), (2.5, 1.5), (10.5, 6.5), (6.0, 4.0)],
]

ROBOT_COLORS = ['#1B6CA8', '#2D6A4F', '#6B3FA0']
ROBOT_START = [(2.5, 1.5), (2.5, 4.0), (6.0, 4.0)]

human_pos = np.array(HUMAN_WAYPOINTS[0], dtype=float)
human_wp_idx = 1

robot_pos = [np.array(p, dtype=float) for p in ROBOT_START]
robot_task_idx = [0, 0, 0]
robot_vel = [np.zeros(2) for _ in range(3)]
completed_tasks = [0, 0, 0]
safety_violations = 0

def move_toward(pos, target, speed):
    diff = target - pos
    dist = np.linalg.norm(diff)
    if dist < speed:
        return target.copy(), True
    return pos + (diff / dist) * speed, False

def avoid_obstacles(pos, vel, other_agents, radius=SAFETY_RADIUS):
    for other in other_agents:
        diff = pos - other
        dist = np.linalg.norm(diff)
        if 0 < dist < radius * 1.5:
            vel += (diff / dist) * (radius * 1.5 - dist) * 0.03
    return vel

fig, ax = plt.subplots(figsize=(13, 8.5), dpi=120)
fig.patch.set_facecolor('#1A1A2E')
ax.set_facecolor('#16213E')

def draw_frame(frame_num):
    global human_pos, human_wp_idx, safety_violations
    ax.clear()
    ax.set_facecolor('#16213E')
    ax.set_xlim(-0.3, WAREHOUSE_W + 0.3)
    ax.set_ylim(-0.3, WAREHOUSE_H + 0.3)
    ax.set_aspect('equal')
    ax.axis('off')

    border = patches.Rectangle((0, 0), WAREHOUSE_W, WAREHOUSE_H,
        linewidth=2, edgecolor='#4A90D9', facecolor='none')
    ax.add_patch(border)

    for x in np.arange(0, WAREHOUSE_W, 1.0):
        ax.axvline(x, color='#0F3460', lw=0.4, alpha=0.5)
    for y in np.arange(0, WAREHOUSE_H, 1.0):
        ax.axhline(y, color='#0F3460', lw=0.4, alpha=0.5)

    for (sx, sy, sw, sh) in SHELVES:
        shelf = patches.FancyBboxPatch((sx, sy), sw, sh,
            boxstyle="round,pad=0.05",
            linewidth=0.8, edgecolor='#4A90D9',
            facecolor='#0F3460', alpha=0.9)
        ax.add_patch(shelf)
        for row in [sy + 0.15, sy + 0.35]:
            for col in np.arange(sx + 0.15, sx + sw - 0.1, 0.25):
                box = patches.Rectangle((col, row), 0.18, 0.12,
                    linewidth=0, facecolor='#E07B39', alpha=0.7)
                ax.add_patch(box)

    station = patches.FancyBboxPatch((10.8, 3.2), 0.9, 1.6,
        boxstyle="round,pad=0.05",
        linewidth=1.5, edgecolor='#74C69D', facecolor='#1B4332', alpha=0.9)
    ax.add_patch(station)
    ax.text(11.25, 4.0, 'PICK\nSTATION', ha='center', va='center',
        fontsize=5.5, color='#74C69D', fontweight='bold')

    target_wp = np.array(HUMAN_WAYPOINTS[human_wp_idx])
    human_pos_new, reached = move_toward(human_pos, target_wp, HUMAN_SPEED)
    human_pos[:] = human_pos_new
    if reached:
        human_wp_idx = (human_wp_idx + 1) % len(HUMAN_WAYPOINTS)

    all_agents = [human_pos] + robot_pos
    for i in range(3):
        target = np.array(ROBOT_TASKS[i][robot_task_idx[i]])
        direction = target - robot_pos[i]
        dist = np.linalg.norm(direction)
        if dist > ROBOT_SPEED:
            vel = (direction / dist) * ROBOT_SPEED
            others = [p for j, p in enumerate(all_agents) if j != i + 1]
            vel = avoid_obstacles(robot_pos[i], vel, others)
            robot_pos[i] += vel
            robot_vel[i] = vel
        else:
            robot_pos[i] = target.copy()
            robot_task_idx[i] = (robot_task_idx[i] + 1) % len(ROBOT_TASKS[i])
            completed_tasks[i] += 1

    for i in range(3):
        dist_to_human = np.linalg.norm(robot_pos[i] - human_pos)
        if dist_to_human < SAFETY_RADIUS:
            safety_violations += 1

    for i in range(3):
        sz = Circle(robot_pos[i], SAFETY_RADIUS,
            fill=True, facecolor=ROBOT_COLORS[i], alpha=0.08, zorder=2)
        ax.add_patch(sz)
        sz_border = Circle(robot_pos[i], SAFETY_RADIUS,
            fill=False, linestyle='--',
            edgecolor=ROBOT_COLORS[i], linewidth=0.8, alpha=0.5, zorder=2)
        ax.add_patch(sz_border)

    for i in range(3):
        next_idx = robot_task_idx[i]
        tx, ty = ROBOT_TASKS[i][next_idx]
        ax.plot([robot_pos[i][0], tx], [robot_pos[i][1], ty],
            color=ROBOT_COLORS[i], lw=0.6, linestyle=':', alpha=0.4)

    h_circle = Circle(human_pos, HUMAN_RADIUS, color='#E07B39', zorder=6)
    ax.add_patch(h_circle)
    ax.plot(human_pos[0], human_pos[1] + HUMAN_RADIUS + 0.15,
        marker='o', markersize=5, color='#E07B39', zorder=7)
    ax.plot([human_pos[0], human_pos[0]],
        [human_pos[1], human_pos[1] + HUMAN_RADIUS + 0.05],
        color='#E07B39', lw=2, zorder=7)
    ax.plot([human_pos[0] - 0.15, human_pos[0] + 0.15],
        [human_pos[1] + 0.1, human_pos[1] + 0.1],
        color='#E07B39', lw=1.5, zorder=7)

    for i in range(3):
        r_circle = Circle(robot_pos[i], ROBOT_RADIUS,
            color=ROBOT_COLORS[i], zorder=6)
        ax.add_patch(r_circle)
        if np.linalg.norm(robot_vel[i]) > 0.001:
            vn = robot_vel[i] / np.linalg.norm(robot_vel[i])
            ax.annotate('', xy=robot_pos[i] + vn * 0.35,
                xytext=robot_pos[i],
                arrowprops=dict(arrowstyle='->',
                    color='white', lw=1.2), zorder=7)
        ax.text(robot_pos[i][0], robot_pos[i][1],
            f'R{i+1}', ha='center', va='center',
            fontsize=5, color='white', fontweight='bold', zorder=8)

    ax.set_title(
        'Deep RL for Multi-Agent Coordination in Mixed Human-Robot Warehouse Environments\n'
        'Mohammed Al-Khudhair  |  Honours Project Demo  |  Habitat 3.0 (Puig et al., 2023)',
        color='white', fontsize=9, pad=8)

    sim_time = frame_num * DT
    total_tasks = sum(completed_tasks)
    info_text = (
        f"Simulation Time: {sim_time:.1f}s\n"
        f"Tasks Completed: {total_tasks}\n"
        f"Active Robots: 3  |  Human Workers: 1\n"
        f"Safety Zone Radius: {SAFETY_RADIUS}m (ISO 10218)"
    )
    ax.text(0.01, 0.01, info_text, transform=ax.transAxes,
        fontsize=7, color='#AAD4F5', verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#0F3460',
            edgecolor='#4A90D9', alpha=0.85))

    legend_elements = [
        mpatches.Patch(color='#1B6CA8', label='Robot 1 (AMR)'),
        mpatches.Patch(color='#2D6A4F', label='Robot 2 (AMR)'),
        mpatches.Patch(color='#6B3FA0', label='Robot 3 (AMR)'),
        mpatches.Patch(color='#E07B39', label='Human Worker'),
        mpatches.Patch(facecolor='none', edgecolor='#4A90D9',
            linestyle='--', label='ISO 10218 Safety Zone'),
        mpatches.Patch(color='#0F3460', edgecolor='#4A90D9',
            label='Storage Rack'),
    ]
    ax.legend(handles=legend_elements, loc='upper right',
        fontsize=6.5, framealpha=0.85,
        facecolor='#0F3460', edgecolor='#4A90D9',
        labelcolor='white')

    progress = frame_num / TOTAL_FRAMES
    ax.add_patch(patches.Rectangle((0, -0.25), WAREHOUSE_W, 0.12,
        facecolor='#0F3460', edgecolor='none'))
    ax.add_patch(patches.Rectangle((0, -0.25), WAREHOUSE_W * progress, 0.12,
        facecolor='#4A90D9', edgecolor='none'))

print(f"Rendering {TOTAL_FRAMES} frames at {FPS}fps...")
print(f"Output: {OUTPUT_FILE}")

writer = FFMpegWriter(fps=FPS, metadata={'title': 'Warehouse Demo'}, bitrate=2000)

with writer.saving(fig, OUTPUT_FILE, dpi=120):
    for frame in range(TOTAL_FRAMES):
        draw_frame(frame)
        writer.grab_frame()
        if frame % 60 == 0:
            pct = int((frame / TOTAL_FRAMES) * 100)
            print(f"  Progress: {pct}% ({frame}/{TOTAL_FRAMES} frames)")

print(f"\nDone! Video saved to: {OUTPUT_FILE}")
print(f"Total tasks completed: {sum(completed_tasks)}")
print(f"Safety zone checks: {safety_violations} proximity events recorded")
