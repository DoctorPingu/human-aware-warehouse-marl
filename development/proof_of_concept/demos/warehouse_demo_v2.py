"""
Warehouse Multi-Agent Coordination Demo v2 - Isometric View
Honours Project: Deep Reinforcement Learning for Multi-Agent Coordination
in Mixed Human-Robot Warehouse Environments
Mohammed Al-Khudhair - 24826341
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
import os

print("Warehouse Demo v2 - Isometric View - Initialising...")

OUTPUT_DIR = "/mnt/e/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v2.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 30
DURATION = 30
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

TW = 0.55
TH = 0.28
TZ = 0.45

def iso(x, y, z=0.0):
    sx = (x - y) * TW
    sy = (x + y) * TH - z * TZ
    return sx, sy

WW = 22
WH = 16

SHELF_H = 1.0
SHELF_D = 0.8
SHELF_W = 2.2

SHELVES = []
for col_x in [2.0, 5.5, 9.0, 12.5, 16.0]:
    for row_y in [1.5, 3.5, 5.5, 7.5, 9.5, 11.5]:
        SHELVES.append((col_x, row_y, SHELF_W, SHELF_D, SHELF_H))

C_FLOOR       = '#0D1B2A'
C_FLOOR_LINE  = '#1A2E42'
C_SHELF_TOP   = '#1E3A5F'
C_SHELF_LEFT  = '#152D4A'
C_SHELF_RIGHT = '#264870'
C_SHELF_EDGE  = '#2A5F8F'
C_BOX_A       = '#E07B39'
C_BOX_B       = '#C4532A'
C_STATION     = '#1B4332'
C_STATION_ED  = '#74C69D'
C_DOCK        = '#2C3E50'
C_DOCK_ED     = '#F39C12'
C_CHARGE      = '#2D1B69'
C_CHARGE_ED   = '#9B59B6'
C_HUMAN       = '#FF6B35'
C_SAFETY      = '#E74C3C'
C_BG          = '#060D16'

ROBOT_COLORS = [
    '#3498DB','#2ECC71','#9B59B6','#E74C3C','#F39C12',
    '#1ABC9C','#E91E63','#FF5722','#00BCD4','#8BC34A'
]

STATIONS    = [(19.5, 4.0), (19.5, 8.0), (19.5, 12.0)]
DOCK        = (0.5, 8.0)
CHARGE_BAYS = [(18.5, 0.8), (19.5, 0.8), (20.5, 0.8)]

HUMAN_WPS = [
    (1.5, 8.0),(4.0,13.5),(8.0,13.5),(13.0,13.5),
    (18.0,12.0),(18.0,8.0),(18.0,4.0),(13.0,2.0),
    (8.0,2.0),(4.0,2.0),(1.5,8.0),
]

def make_circuit(pts):
    return pts + [pts[0]]

ROBOT_TASKS = [
    make_circuit([(3.5,2.0),(3.5,13.5),(19.0,4.0)]),
    make_circuit([(7.0,2.0),(7.0,13.5),(19.0,8.0)]),
    make_circuit([(10.5,2.0),(10.5,13.5),(19.0,12.0)]),
    make_circuit([(14.0,2.0),(14.0,13.5),(19.0,4.0)]),
    make_circuit([(17.5,2.0),(17.5,7.0),(19.0,8.0)]),
    make_circuit([(3.5,8.0),(10.5,8.0),(19.0,12.0)]),
    make_circuit([(7.0,5.0),(14.0,5.0),(19.0,4.0)]),
    make_circuit([(10.5,11.0),(17.5,11.0),(19.0,8.0)]),
    make_circuit([(4.0,11.0),(11.0,4.0),(19.0,12.0)]),
    make_circuit([(1.5,4.0),(18.0,4.0),(18.0,12.0)]),
]

ROBOT_STARTS = [
    (3.5,2.0),(7.0,2.0),(10.5,2.0),(14.0,2.0),(17.5,2.0),
    (3.5,8.0),(7.0,5.0),(10.5,11.0),(4.0,11.0),(1.5,4.0),
]

human_pos    = np.array(HUMAN_WPS[0], dtype=float)
human_wp_idx = 1
human_phase  = 0.0
HUMAN_SPEED  = 0.03

N_ROBOTS      = 10
ROBOT_SPEED   = 0.055
SAFETY_RADIUS = 0.9

robot_pos      = [np.array(ROBOT_STARTS[i], dtype=float) for i in range(N_ROBOTS)]
robot_task_idx = [0] * N_ROBOTS
robot_vel      = [np.zeros(2) for _ in range(N_ROBOTS)]
robot_state    = ['move'] * N_ROBOTS
robot_wait     = [0] * N_ROBOTS
completed      = [0] * N_ROBOTS
safety_events  = 0

def repulse(pos, others, radius):
    force = np.zeros(2)
    for o in others:
        d = pos - o
        dist = np.linalg.norm(d)
        if 0 < dist < radius * 1.6:
            force += (d / dist) * (radius * 1.6 - dist) * 0.025
    return force

def draw_iso_rect(ax, x, y, w, d, z_base, z_top,
                  c_top, c_left, c_right, ec, alpha=1.0, zorder=2):
    tl = iso(x,     y,     z_top)
    tr = iso(x + w, y,     z_top)
    br = iso(x + w, y + d, z_top)
    bl = iso(x,     y + d, z_top)
    ax.add_patch(Polygon([tl,tr,br,bl], closed=True,
        facecolor=c_top, edgecolor=ec, lw=0.5, alpha=alpha, zorder=zorder))
    tr2 = iso(x+w, y,   z_top);  tr3 = iso(x+w, y,   z_base)
    br2 = iso(x+w, y+d, z_top);  br3 = iso(x+w, y+d, z_base)
    ax.add_patch(Polygon([tr2,br2,br3,tr3], closed=True,
        facecolor=c_right, edgecolor=ec, lw=0.5, alpha=alpha, zorder=zorder))
    bl2 = iso(x,   y+d, z_top);  bl3 = iso(x,   y+d, z_base)
    ax.add_patch(Polygon([bl2,br2,br3,bl3], closed=True,
        facecolor=c_left, edgecolor=ec, lw=0.5, alpha=alpha, zorder=zorder))

def draw_iso_floor_tile(ax, x, y, color, ec, alpha=1.0, zorder=1):
    c1=iso(x,y,0); c2=iso(x+1,y,0); c3=iso(x+1,y+1,0); c4=iso(x,y+1,0)
    ax.add_patch(Polygon([c1,c2,c3,c4], closed=True,
        facecolor=color, edgecolor=ec, lw=0.3, alpha=alpha, zorder=zorder))

def draw_robot(ax, x, y, color, label, vel, zorder=8):
    rh = 0.35
    rr = 0.22
    sx2, sy2 = iso(x, y, 0)
    ax.add_patch(Ellipse((sx2, sy2), rr*2.2*TW*2, rr*TH*2.5,
        facecolor='black', alpha=0.25, zorder=zorder-1))
    top_sx, top_sy = iso(x, y, rh)
    cx2 = (top_sx+sx2)/2
    cy2 = (top_sy+sy2)/2
    ax.add_patch(Ellipse((cx2, cy2+(top_sy-sy2)*0.2),
        rr*2*TW*2.2, abs(top_sy-sy2)+rr*TH*1.5,
        facecolor=color, edgecolor='white', lw=0.6, alpha=0.95, zorder=zorder))
    ax.add_patch(Ellipse((top_sx, top_sy), rr*2*TW*2.2, rr*TH*2.2,
        facecolor=color, edgecolor='white', lw=0.5, alpha=1.0, zorder=zorder+1))
    if np.linalg.norm(vel) > 0.001:
        vn = vel / np.linalg.norm(vel)
        ax.annotate('', xy=(top_sx+vn[0]*TW*0.6,
                             top_sy+(vn[0]*TH*0.3-vn[1]*TH*0.3)),
                    xytext=(top_sx, top_sy),
                    arrowprops=dict(arrowstyle='->', color='white', lw=1.0),
                    zorder=zorder+2)
    ax.text(top_sx, top_sy+0.06, label, ha='center', va='center',
            fontsize=4.5, color='white', fontweight='bold', zorder=zorder+3)

def draw_humanoid(ax, x, y, phase, zorder=10):
    h = 0.9
    sx2, sy2 = iso(x, y, h)
    bx2, by2 = iso(x, y, 0)
    leg_swing = np.sin(phase) * 0.25
    arm_swing = -np.sin(phase) * 0.18
    body_bob  = abs(np.sin(phase)) * 0.03
    sy_b = sy2 + body_bob * TZ
    ax.add_patch(Circle((sx2, sy_b+0.10), 0.085,
        facecolor=C_HUMAN, edgecolor='#FF8C5A', lw=0.8, zorder=zorder+1))
    torso_bx = bx2 + (sx2-bx2)*0.55
    torso_by = by2 + (sy_b-by2)*0.45
    ax.plot([sx2, torso_bx], [sy_b, torso_by],
            color=C_HUMAN, lw=1.8, solid_capstyle='round', zorder=zorder)
    arm_len = 0.13
    ax.plot([sx2, sx2-arm_len*TW*(1+arm_swing)],
            [sy_b-0.02, sy_b-arm_len*TH+arm_swing*0.04],
            color=C_HUMAN, lw=1.4, solid_capstyle='round', zorder=zorder)
    ax.plot([sx2, sx2+arm_len*TW*(1-arm_swing)],
            [sy_b-0.02, sy_b-arm_len*TH-arm_swing*0.04],
            color=C_HUMAN, lw=1.4, solid_capstyle='round', zorder=zorder)
    leg_len = 0.16
    ax.plot([torso_bx, torso_bx-leg_swing*TW*0.8],
            [torso_by, torso_by-leg_len*TH+leg_swing*0.05],
            color=C_HUMAN, lw=1.6, solid_capstyle='round', zorder=zorder)
    ax.plot([torso_bx, torso_bx+leg_swing*TW*0.8],
            [torso_by, torso_by-leg_len*TH-leg_swing*0.05],
            color=C_HUMAN, lw=1.6, solid_capstyle='round', zorder=zorder)

corners_screen = [iso(x,y,0) for x,y in [(0,0),(WW,0),(WW,WH),(0,WH)]]
X_MIN = min(c[0] for c in corners_screen) - 1.5
X_MAX = max(c[0] for c in corners_screen) + 1.5
Y_MIN = min(c[1] for c in corners_screen) - 1.2
Y_MAX = max(c[1] for c in corners_screen) + 1.2

fig, ax = plt.subplots(figsize=(16, 10), dpi=110)
fig.patch.set_facecolor(C_BG)

def draw_frame(frame):
    global human_pos, human_wp_idx, human_phase, safety_events
    ax.clear()
    ax.set_facecolor(C_BG)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX+0.8)
    ax.set_aspect('equal')
    ax.axis('off')

    for gy in range(WH):
        for gx in range(WW):
            shade = C_FLOOR if (gx+gy)%2==0 else '#0F2030'
            draw_iso_floor_tile(ax, gx, gy, shade, C_FLOOR_LINE, zorder=1)

    for ax_x in [4.5, 8.0, 11.5, 15.0, 18.5]:
        p1=iso(ax_x,0,0); p2=iso(ax_x,WH,0)
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                color='#1A3A55', lw=0.6, linestyle=':', alpha=0.7, zorder=2)

    for (sx2,sy2,sw,sd,sh) in sorted(SHELVES, key=lambda s: -(s[0]+s[1])):
        draw_iso_rect(ax, sx2, sy2, sw, sd, 0, sh,
                      C_SHELF_TOP, C_SHELF_LEFT, C_SHELF_RIGHT, C_SHELF_EDGE,
                      alpha=0.95, zorder=3)
        for bx2 in np.arange(sx2+0.15, sx2+sw-0.15, 0.45):
            bh = 0.22 + np.random.RandomState(int(bx2*10+sy2*7)).uniform(0,0.15)
            draw_iso_rect(ax, bx2, sy2+0.1, 0.28, 0.5, sh, sh+bh,
                          C_BOX_A, C_BOX_B, '#D4622A', '#FF8C5A',
                          alpha=0.9, zorder=4)

    for (px2,py2) in STATIONS:
        draw_iso_rect(ax, px2-0.4, py2-0.6, 0.8, 1.2, 0, 0.6,
                      C_STATION, '#0D2B1C', '#1A3828', C_STATION_ED,
                      alpha=0.95, zorder=3)
        lx,ly = iso(px2, py2, 0.65)
        ax.text(lx, ly, 'PICK', ha='center', va='bottom',
                fontsize=4.5, color=C_STATION_ED, fontweight='bold', zorder=5)

    dx2,dy2 = DOCK
    draw_iso_rect(ax, dx2-0.6, dy2-1.0, 1.2, 2.0, 0, 0.4,
                  C_DOCK, '#1C2B38', '#243444', C_DOCK_ED, alpha=0.95, zorder=3)
    lx,ly = iso(dx2, dy2, 0.45)
    ax.text(lx, ly, 'DOCK', ha='center', va='bottom',
            fontsize=4.5, color=C_DOCK_ED, fontweight='bold', zorder=5)

    for (cx2,cy2) in CHARGE_BAYS:
        draw_iso_rect(ax, cx2-0.25, cy2-0.25, 0.5, 0.5, 0, 0.15,
                      C_CHARGE, '#1E0D3C', '#2D1B55', C_CHARGE_ED,
                      alpha=0.9, zorder=3)
        lx,ly = iso(cx2, cy2, 0.2)
        ax.text(lx, ly, '+', ha='center', va='bottom',
                fontsize=7, color=C_CHARGE_ED, fontweight='bold', zorder=5)

    target_wp = np.array(HUMAN_WPS[human_wp_idx])
    diff = target_wp - human_pos
    dist = np.linalg.norm(diff)
    if dist < HUMAN_SPEED:
        human_pos[:] = target_wp
        human_wp_idx = (human_wp_idx+1) % len(HUMAN_WPS)
    else:
        human_pos += (diff/dist) * HUMAN_SPEED
    human_phase += 0.22

    all_positions = [human_pos.copy()] + [r.copy() for r in robot_pos]
    for i in range(N_ROBOTS):
        if robot_state[i] == 'wait':
            robot_wait[i] -= 1
            if robot_wait[i] <= 0:
                robot_state[i] = 'move'
            robot_vel[i] *= 0.0
            continue
        target = np.array(ROBOT_TASKS[i][robot_task_idx[i]])
        diff2  = target - robot_pos[i]
        dist2  = np.linalg.norm(diff2)
        if dist2 < ROBOT_SPEED:
            robot_pos[i] = target.copy()
            robot_task_idx[i] = (robot_task_idx[i]+1) % len(ROBOT_TASKS[i])
            completed[i] += 1
            robot_state[i] = 'wait'
            robot_wait[i]  = int(FPS*0.4)
        else:
            vel = (diff2/dist2) * ROBOT_SPEED
            others = [all_positions[j] for j in range(len(all_positions)) if j != i+1]
            vel += repulse(robot_pos[i], others, SAFETY_RADIUS)
            robot_pos[i] += vel
            robot_vel[i]  = vel

    for i in range(N_ROBOTS):
        if np.linalg.norm(robot_pos[i]-human_pos) < SAFETY_RADIUS:
            safety_events += 1

    for i in range(N_ROBOTS):
        rx,ry = iso(robot_pos[i][0], robot_pos[i][1], 0)
        ax.add_patch(Ellipse((rx,ry), SAFETY_RADIUS*TW*4, SAFETY_RADIUS*TH*4,
            facecolor=ROBOT_COLORS[i], alpha=0.06, zorder=5))
        ax.add_patch(Ellipse((rx,ry), SAFETY_RADIUS*TW*4, SAFETY_RADIUS*TH*4,
            facecolor='none', edgecolor=ROBOT_COLORS[i],
            linestyle='--', lw=0.7, alpha=0.45, zorder=5))

    for i in range(N_ROBOTS):
        nxt = np.array(ROBOT_TASKS[i][robot_task_idx[i]])
        p1=iso(robot_pos[i][0], robot_pos[i][1], 0.05)
        p2=iso(nxt[0], nxt[1], 0.05)
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                color=ROBOT_COLORS[i], lw=0.5, linestyle=':', alpha=0.35, zorder=6)

    for i in sorted(range(N_ROBOTS), key=lambda i: -(robot_pos[i][0]+robot_pos[i][1])):
        draw_robot(ax, robot_pos[i][0], robot_pos[i][1],
                   ROBOT_COLORS[i], f'R{i+1}', robot_vel[i], zorder=7)

    draw_humanoid(ax, human_pos[0], human_pos[1], human_phase, zorder=10)

    hsx,hsy = iso(human_pos[0], human_pos[1], 0)
    ax.add_patch(Ellipse((hsx,hsy), SAFETY_RADIUS*TW*5, SAFETY_RADIUS*TH*5,
        facecolor=C_SAFETY, alpha=0.07, zorder=6))

    ax.set_title(
        'Deep RL for Multi-Agent Coordination in Mixed Human-Robot Warehouse Environments\n'
        'Mohammed Al-Khudhair  |  Honours Project Demo  |  Habitat 3.0 (Puig et al., 2023)',
        color='#AAD4F5', fontsize=9.5, pad=8, fontfamily='monospace')

    panel = (
        f"  Simulation Time : {frame*DT:5.1f}s\n"
        f"  Robots Active   : {N_ROBOTS}\n"
        f"  Human Workers   : 1\n"
        f"  Tasks Completed : {sum(completed)}\n"
        f"  Safety Radius   : {SAFETY_RADIUS}m (ISO 10218)\n"
        f"  Proximity Events: {safety_events}"
    )
    ax.text(0.01, 0.03, panel, transform=ax.transAxes,
            fontsize=7, color='#AAD4F5', va='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#060D16',
                      edgecolor='#1A3A55', alpha=0.92))

    legend_items = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=ROBOT_COLORS[i], markersize=6, label=f'R{i+1}')
        for i in range(10)
    ] + [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=C_HUMAN, markersize=7, label='Human'),
        Line2D([0],[0], color=C_SAFETY, lw=1.5, linestyle='--', label='Safety Zone'),
    ]
    ax.legend(handles=legend_items, loc='upper right',
              fontsize=5, framealpha=0.9, facecolor='#060D16',
              edgecolor='#1A3A55', labelcolor='#AAD4F5', ncol=2)

    prog = frame / FRAMES
    ax.plot([X_MIN+0.3, X_MAX-0.3], [Y_MIN+0.12, Y_MIN+0.12],
            color='#1A3A55', lw=4, solid_capstyle='round', zorder=15)
    ax.plot([X_MIN+0.3, X_MIN+0.3+(X_MAX-X_MIN-0.6)*prog],
            [Y_MIN+0.12, Y_MIN+0.12],
            color='#4A90D9', lw=4, solid_capstyle='round', zorder=16)

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps)...")
writer = FFMpegWriter(fps=FPS, metadata={'title': 'Warehouse Demo v2'}, bitrate=3000)

with writer.saving(fig, OUTPUT_FILE, dpi=110):
    for frame in range(FRAMES):
        draw_frame(frame)
        writer.grab_frame()
        if frame % FPS == 0:
            pct = int(frame/FRAMES*100)
            print(f"  {pct:3d}%  ({frame}/{FRAMES} frames, "
                  f"tasks: {sum(completed)}, events: {safety_events})")

print(f"\nDone! Saved to: {OUTPUT_FILE}")
