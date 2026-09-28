"""
Warehouse Demo v4 - Side-by-Side Baseline Comparison
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
Left: Simple Repulsion | Right: ORCA/RVO2 Coordination
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
import rvo2, os

print("Warehouse Demo v4 - Initialising...")

OUTPUT_DIR  = "/mnt/e/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v4.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 30
DURATION = 50
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

WW = 22.0
WH = 14.0

N_ROBOTS     = 10
N_HUMANS     = 3
ROBOT_R      = 0.28
HUMAN_R      = 0.22
SAFETY_R     = 0.85
COLLISION_R  = 0.46
ROBOT_SPD    = 1.8
HUMAN_SPD    = 0.75
ROBOT_STEP   = ROBOT_SPD * DT
HUMAN_STEP   = HUMAN_SPD * DT

C_BG        = '#07101C'
C_FLOOR     = '#0A1828'
C_FLOOR_A   = '#0C1E30'
C_HIGHWAY   = '#0C2238'
C_HWLINE    = '#0F2A48'
C_DIVIDER   = '#18344E'
C_ZONE_TXT  = '#1E4A72'
C_SHELF_STD = '#122438'
C_SHELF_STD_E='#1C3C6A'
C_SHELF_CLD = '#0E2236'
C_SHELF_CLD_E='#185886'
C_SHELF_PCK = '#0E1E34'
C_SHELF_PCK_E='#183266'
C_ITEM      = '#D4622A'
C_DOCK_F    = '#3A2810'
C_DOCK_E    = '#E8A018'
C_STAT_F    = '#182E22'
C_STAT_E    = '#48B870'
C_CONV_F    = '#181C1C'
C_CONV_E    = '#3ABAA8'
C_SIMPLE    = '#E74C3C'
C_ORCA      = '#2ECC71'

ROBOT_COLORS = [
    '#3498DB','#2ECC71','#9B59B6','#E74C3C','#F39C12',
    '#1ABC9C','#E91E63','#FF5722','#00BCD4','#8BC34A'
]
HUMAN_COLORS = ['#FF6B35','#FFD93D','#FF4E88']

SHELF_W  = 1.8
SHELF_H  = 0.72
SHELF_WP = 1.55
SHELF_HP = 0.62

SHELVES = []
for cx in [4.0, 6.5]:
    for ry in [1.0, 2.2, 3.4, 4.6]:
        SHELVES.append((cx, ry, SHELF_W, SHELF_H, 'cold'))
    for ry in [8.3, 9.5, 10.7, 11.9]:
        SHELVES.append((cx, ry, SHELF_W, SHELF_H, 'cold'))

for cx in [9.5, 11.5]:
    for ry in [1.0, 2.2, 3.4, 4.6]:
        SHELVES.append((cx, ry, SHELF_W, SHELF_H, 'bulk'))
    for ry in [8.3, 9.5, 10.7, 11.9]:
        SHELVES.append((cx, ry, SHELF_W, SHELF_H, 'bulk'))

for cx in [14.5, 16.5]:
    for ry in [1.0, 2.0, 3.0, 4.0, 5.0]:
        SHELVES.append((cx, ry, SHELF_WP, SHELF_HP, 'pick'))
    for ry in [8.3, 9.3, 10.3, 11.3, 12.3]:
        SHELVES.append((cx, ry, SHELF_WP, SHELF_HP, 'pick'))

DOCKS    = [(0.3, 2.2), (0.3, 7.0), (0.3, 11.8)]
STATIONS = [(20.2, 3.0), (20.2, 7.0), (20.2, 11.0)]
CONVS    = [(21.2, 1.0, 0.7, 5.5), (21.2, 7.5, 0.7, 5.5)]

def mc(p): return p + [p[0]]

ROBOT_STARTS = [
    (5.0,2.0),(7.0,2.0),(10.0,2.0),(12.0,2.0),(15.0,2.0),
    (17.0,2.0),(5.0,9.2),(10.0,9.2),(15.0,9.2),(2.0,7.0),
]

ROBOT_TASKS = [
    mc([(5.0,2.0),(5.0,5.0),(20.2,3.0)]),
    mc([(7.0,2.0),(7.0,5.0),(20.2,3.0)]),
    mc([(10.0,2.0),(10.0,5.0),(20.2,7.0)]),
    mc([(12.0,2.0),(12.0,5.0),(20.2,7.0)]),
    mc([(15.0,2.0),(15.0,5.5),(20.2,11.0)]),
    mc([(17.0,2.0),(17.0,5.5),(20.2,11.0)]),
    mc([(5.0,9.2),(5.0,12.5),(20.2,3.0)]),
    mc([(10.0,9.2),(10.0,12.5),(20.2,7.0)]),
    mc([(15.0,9.2),(15.0,12.5),(20.2,11.0)]),
    mc([(2.0,7.0),(11.0,7.0),(20.2,7.0)]),
]

HUMAN_WPS = [
    mc([(1.0,1.0),(21.0,1.0),(21.0,13.0),(1.0,13.0)]),
    mc([(14.5,2.0),(18.5,2.0),(18.5,5.5),(14.5,5.5),
        (14.5,8.5),(18.5,8.5),(18.5,12.5),(14.5,12.5)]),
    mc([(9.5,1.0),(9.5,5.5),(9.5,8.5),(9.5,13.0),
        (4.5,13.0),(4.5,8.5),(4.5,5.5),(4.5,1.0)]),
]

def make_state():
    return dict(
        robot_pos      = [np.array(p, dtype=float) for p in ROBOT_STARTS],
        robot_vel      = [np.zeros(2) for _ in range(N_ROBOTS)],
        robot_task_idx = [0]*N_ROBOTS,
        robot_state    = ['move']*N_ROBOTS,
        robot_wait     = [0]*N_ROBOTS,
        human_pos      = [np.array(HUMAN_WPS[h][0], dtype=float) for h in range(N_HUMANS)],
        human_wp_idx   = [1]*N_HUMANS,
        human_phase    = [0.0, 1.05, 2.1],
        completed      = [0]*N_ROBOTS,
        collisions     = 0,
        safety_evts    = 0,
    )

S = make_state()
O = make_state()

o_sim = rvo2.PyRVOSimulator(
    DT, 4.5, 15, 8.0, 2.0, ROBOT_R, ROBOT_SPD
)
for i in range(N_ROBOTS):
    o_sim.addAgent(
        tuple(O['robot_pos'][i]),
        4.5, 15, 8.0, 2.0, ROBOT_R, ROBOT_SPD, (0.0, 0.0)
    )
for h in range(N_HUMANS):
    o_sim.addAgent(
        tuple(O['human_pos'][h]),
        4.5, 15, 5.0, 2.0, HUMAN_R, HUMAN_SPD, (0.0, 0.0)
    )

def tally_proximity(state):
    all_p = list(state['human_pos']) + list(state['robot_pos'])
    n = len(all_p)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(all_p[i] - all_p[j])
            if d < COLLISION_R:
                state['collisions'] += 1
            elif d < SAFETY_R:
                state['safety_evts'] += 1

def repulse(pos, others):
    f = np.zeros(2)
    for o in others:
        d = pos - o
        dist = np.linalg.norm(d)
        if 0 < dist < SAFETY_R * 1.7:
            f += (d / dist) * (SAFETY_R * 1.7 - dist) * 1.4
    return f

def step_humans_simple(state):
    for h in range(N_HUMANS):
        tgt  = np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        diff = tgt - state['human_pos'][h]
        dist = np.linalg.norm(diff)
        if dist < HUMAN_STEP * 1.5:
            state['human_pos'][h][:] = tgt
            state['human_wp_idx'][h]  = (state['human_wp_idx'][h]+1) % len(HUMAN_WPS[h])
        else:
            state['human_pos'][h] += (diff/dist) * HUMAN_STEP
        state['human_phase'][h] += 0.20
        state['human_pos'][h] = np.clip(state['human_pos'][h], 0.15, [WW-0.15, WH-0.15])

def step_simple(state):
    step_humans_simple(state)
    all_snap = [p.copy() for p in state['human_pos']] + \
               [p.copy() for p in state['robot_pos']]
    for i in range(N_ROBOTS):
        if state['robot_state'][i] == 'wait':
            state['robot_wait'][i] -= 1
            if state['robot_wait'][i] <= 0:
                state['robot_state'][i] = 'move'
            state['robot_vel'][i] *= 0.0
            continue
        tgt  = np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
        diff = tgt - state['robot_pos'][i]
        dist = np.linalg.norm(diff)
        if dist < ROBOT_STEP * 2:
            state['robot_pos'][i][:]    = tgt
            state['robot_task_idx'][i]  = (state['robot_task_idx'][i]+1) % len(ROBOT_TASKS[i])
            state['completed'][i]      += 1
            state['robot_state'][i]     = 'wait'
            state['robot_wait'][i]      = int(FPS * 0.55)
            state['robot_vel'][i]       = np.zeros(2)
        else:
            others = [all_snap[j] for j in range(len(all_snap)) if j != i+N_HUMANS]
            vel    = (diff/dist) * ROBOT_SPD + repulse(state['robot_pos'][i], others)
            spd    = np.linalg.norm(vel)
            if spd > ROBOT_SPD * 1.6:
                vel = vel / spd * ROBOT_SPD * 1.6
            state['robot_vel'][i]  = vel
            state['robot_pos'][i] += vel * DT
            state['robot_pos'][i]  = np.clip(state['robot_pos'][i], 0.15, [WW-0.15, WH-0.15])
    tally_proximity(state)

def step_orca(state, sim):
    for i in range(N_ROBOTS):
        if state['robot_state'][i] == 'wait':
            sim.setAgentPrefVelocity(i, (0.0, 0.0))
        else:
            tgt  = np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            diff = tgt - state['robot_pos'][i]
            dist = np.linalg.norm(diff)
            if dist < 0.25:
                sim.setAgentPrefVelocity(i, (0.0, 0.0))
            else:
                spd = min(ROBOT_SPD, dist * FPS * 0.55)
                sim.setAgentPrefVelocity(i, tuple((diff/dist) * spd))

    for h in range(N_HUMANS):
        tgt  = np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        diff = tgt - state['human_pos'][h]
        dist = np.linalg.norm(diff)
        if dist < 0.12:
            sim.setAgentPrefVelocity(N_ROBOTS+h, (0.0, 0.0))
        else:
            sim.setAgentPrefVelocity(N_ROBOTS+h, tuple((diff/dist) * HUMAN_SPD))

    sim.doStep()

    for i in range(N_ROBOTS):
        pos = np.clip(np.array(sim.getAgentPosition(i)), 0.15, [WW-0.15, WH-0.15])
        vel = np.array(sim.getAgentVelocity(i))
        state['robot_pos'][i] = pos
        state['robot_vel'][i] = vel
        sim.setAgentPosition(i, tuple(pos))

        if state['robot_state'][i] == 'wait':
            state['robot_wait'][i] -= 1
            if state['robot_wait'][i] <= 0:
                state['robot_state'][i] = 'move'
        else:
            tgt  = np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            if np.linalg.norm(tgt - pos) < 0.38:
                state['robot_task_idx'][i] = (state['robot_task_idx'][i]+1) % len(ROBOT_TASKS[i])
                state['completed'][i]     += 1
                state['robot_state'][i]    = 'wait'
                state['robot_wait'][i]     = int(FPS * 0.55)

    for h in range(N_HUMANS):
        pos = np.clip(np.array(sim.getAgentPosition(N_ROBOTS+h)), 0.15, [WW-0.15, WH-0.15])
        state['human_pos'][h] = pos
        state['human_phase'][h] += 0.20
        sim.setAgentPosition(N_ROBOTS+h, tuple(pos))
        tgt  = np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        if np.linalg.norm(tgt - pos) < HUMAN_STEP * 3.5:
            state['human_wp_idx'][h] = (state['human_wp_idx'][h]+1) % len(HUMAN_WPS[h])

    tally_proximity(state)

def draw_base(ax, title, tc):
    ax.set_facecolor(C_BG)
    ax.set_xlim(-0.5, WW+0.5)
    ax.set_ylim(-0.75, WH+0.5)
    ax.set_aspect('equal')
    ax.axis('off')

    ax.add_patch(Rectangle((0,0), WW, WH, facecolor=C_FLOOR,
        edgecolor=C_DIVIDER, lw=1.5, zorder=1))

    for gx in range(int(WW)):
        for gy in range(int(WH)):
            if (gx+gy)%2==0:
                ax.add_patch(Rectangle((gx,gy),1,1, facecolor=C_FLOOR_A,
                    edgecolor='none', zorder=1))

    zone_defs = [
        (0,    0, 3.5, WH, '#091828', 'RECEIVING'),
        (3.5,  0, 5.5, WH, '#07162A', 'COLD STORAGE'),
        (9.0,  0, 5.0, WH, '#081828', 'BULK STORAGE'),
        (14.0, 0, 5.0, WH, '#071626', 'PICK ZONE'),
        (19.0, 0, 3.0, WH, '#081828', 'SHIPPING'),
    ]
    for (zx,zy,zw,zh,zc,ztxt) in zone_defs:
        ax.add_patch(Rectangle((zx,zy), zw, zh, facecolor=zc,
            edgecolor='none', zorder=1))
        ax.text(zx+zw/2, WH/2, ztxt, ha='center', va='center',
                fontsize=5.5, color=C_ZONE_TXT, alpha=0.55,
                rotation=90, fontfamily='monospace', zorder=2)

    ax.add_patch(Rectangle((0,6.0), WW, 2.0, facecolor=C_HIGHWAY,
        edgecolor='none', zorder=2))
    ax.plot([0,WW],[6.0,6.0], color=C_HWLINE, lw=0.8, ls='--', alpha=0.5, zorder=3)
    ax.plot([0,WW],[8.0,8.0], color=C_HWLINE, lw=0.8, ls='--', alpha=0.5, zorder=3)
    ax.text(WW/2, 7.0, 'MAIN HIGHWAY AISLE', ha='center', va='center',
            fontsize=5.5, color='#1E4870', alpha=0.7, fontfamily='monospace', zorder=3)

    for xd in [3.5,9.0,14.0,19.0]:
        ax.plot([xd,xd],[0,WH], color=C_DIVIDER, lw=0.8, ls=':', alpha=0.5, zorder=3)

    for (sx,sy,sw,sh,zone) in SHELVES:
        if zone=='cold':
            fc,ec = C_SHELF_CLD, C_SHELF_CLD_E
        elif zone=='pick':
            fc,ec = C_SHELF_PCK, C_SHELF_PCK_E
        else:
            fc,ec = C_SHELF_STD, C_SHELF_STD_E
        ax.add_patch(Rectangle((sx,sy), sw, sh, facecolor=fc,
            edgecolor=ec, lw=0.8, zorder=3))
        for ly in np.arange(sy+0.18, sy+sh, 0.18):
            ax.plot([sx+0.06,sx+sw-0.06],[ly,ly], color=ec, lw=0.25, alpha=0.4, zorder=4)
        rng = np.random.RandomState(int(sx*17+sy*11))
        for ix in np.arange(sx+0.10, sx+sw-0.05, 0.26):
            ih = 0.09 + rng.uniform(0, 0.08)
            ax.add_patch(Rectangle((ix-0.07,sy+sh*0.12), 0.14, ih,
                facecolor=C_ITEM, edgecolor='none', alpha=0.65, zorder=4))

    for (dx,dy) in DOCKS:
        ax.add_patch(Rectangle((dx,dy-0.5), 0.9,1.0,
            facecolor=C_DOCK_F, edgecolor=C_DOCK_E, lw=1.0, zorder=3))
        ax.text(dx+0.45, dy, 'D', ha='center', va='center',
                fontsize=6, color=C_DOCK_E, fontweight='bold', zorder=4)

    for (px,py) in STATIONS:
        ax.add_patch(Rectangle((px,py-0.6), 1.0,1.2,
            facecolor=C_STAT_F, edgecolor=C_STAT_E, lw=1.0, zorder=3))
        ax.text(px+0.5, py, 'PS', ha='center', va='center',
                fontsize=6, color=C_STAT_E, fontweight='bold', zorder=4)

    for (cx,cy,cw,ch) in CONVS:
        ax.add_patch(Rectangle((cx,cy), cw,ch,
            facecolor=C_CONV_F, edgecolor=C_CONV_E, lw=0.8, zorder=3))
        ax.text(cx+cw/2, cy+ch/2, 'CV', ha='center', va='center',
                fontsize=5, color=C_CONV_E, fontweight='bold', rotation=90, zorder=4)

    panel_rect = Rectangle((-0.48, -0.68), WW+0.96, 0.62,
        facecolor=C_BG, edgecolor=tc, lw=1.5, alpha=0.9, zorder=14)
    ax.add_patch(panel_rect)

    ax.set_title(title, color=tc, fontsize=13, fontweight='bold',
                 fontfamily='monospace', pad=7)

def draw_agents(ax, state):
    rp = state['robot_pos']
    rv = state['robot_vel']
    hp = state['human_pos']
    hph= state['human_phase']

    for i in range(N_ROBOTS):
        ax.add_patch(Circle(rp[i], SAFETY_R, fill=True,
            facecolor=ROBOT_COLORS[i], alpha=0.055, zorder=5))
        ax.add_patch(Circle(rp[i], SAFETY_R, fill=False,
            edgecolor=ROBOT_COLORS[i], ls='--', lw=0.55, alpha=0.28, zorder=5))

    for h in range(N_HUMANS):
        ax.add_patch(Circle(hp[h], SAFETY_R*0.75, fill=True,
            facecolor=HUMAN_COLORS[h], alpha=0.06, zorder=5))
        ax.add_patch(Circle(hp[h], SAFETY_R*0.75, fill=False,
            edgecolor=HUMAN_COLORS[h], ls=':', lw=0.55, alpha=0.3, zorder=5))

    for i in range(N_ROBOTS):
        nxt = np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
        ax.plot([rp[i][0],nxt[0]],[rp[i][1],nxt[1]],
                color=ROBOT_COLORS[i], lw=0.35, ls=':', alpha=0.22, zorder=5)

    all_p = list(hp) + list(rp)
    n = len(all_p)
    for i in range(n):
        for j in range(i+1,n):
            d = np.linalg.norm(all_p[i]-all_p[j])
            if d < COLLISION_R:
                mid = (all_p[i]+all_p[j])/2
                ax.add_patch(Circle(mid, 0.32, facecolor='red',
                    alpha=0.45, zorder=9))

    for h in range(N_HUMANS):
        col = HUMAN_COLORS[h]
        ax.add_patch(Circle(hp[h], HUMAN_R, facecolor=col,
            edgecolor='white', lw=0.9, zorder=7))
        ax.add_patch(Circle(hp[h], HUMAN_R*0.38, facecolor='white',
            edgecolor='none', alpha=0.75, zorder=8))
        sw = np.sin(hph[h])*0.14
        ax.plot([hp[h][0]-sw, hp[h][0]+sw],
                [hp[h][1]-HUMAN_R*0.7, hp[h][1]+HUMAN_R*0.7],
                color='white', lw=1.1, alpha=0.55, zorder=8)
        ax.text(hp[h][0], hp[h][1]-HUMAN_R-0.18, f'H{h+1}',
                ha='center', va='top', fontsize=5, color=col, zorder=8)

    for i in range(N_ROBOTS):
        ax.add_patch(Circle(rp[i], ROBOT_R, facecolor=ROBOT_COLORS[i],
            edgecolor='white', lw=0.85, zorder=7))
        ax.add_patch(Circle(rp[i], ROBOT_R*0.48, fill=False,
            edgecolor='white', lw=0.45, alpha=0.55, zorder=7))
        spd = np.linalg.norm(rv[i])
        if spd > 0.08:
            vn = rv[i]/spd
            ax.annotate('', xy=rp[i]+vn*(ROBOT_R+0.28),
                xytext=rp[i]+vn*ROBOT_R,
                arrowprops=dict(arrowstyle='->', color='white', lw=0.95), zorder=8)
        ax.text(rp[i][0], rp[i][1], f'R{i+1}',
                ha='center', va='center', fontsize=4.5,
                color='white', fontweight='bold', zorder=9)

def draw_stats_panel(ax, state, frame, algo, col):
    t    = frame*DT
    tot  = sum(state['completed'])
    rate = tot / max(t, 0.5) * 60.0
    txt  = (
        f"  Algorithm       :  {algo}\n"
        f"  Sim Time        :  {t:5.1f}s\n"
        f"  Tasks Completed :  {tot}\n"
        f"  Throughput      :  {rate:.1f} tasks/min\n"
        f"  Collisions      :  {state['collisions']}\n"
        f"  Safety Violations:  {state['safety_evts']}"
    )
    ax.text(0.018, 0.022, txt, transform=ax.transAxes,
            fontsize=8.8, color='#AAD4F5', va='bottom',
            fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.48', facecolor='#050D18',
                      edgecolor=col, alpha=0.94))

def draw_legend_panel(ax):
    items = [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=ROBOT_COLORS[i], markersize=7.5,
               label=f'Robot {i+1}') for i in range(N_ROBOTS)
    ] + [
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor=HUMAN_COLORS[h], markersize=9,
               label=f'Human {h+1}') for h in range(N_HUMANS)
    ] + [
        Line2D([0],[0], color='#E74C3C', lw=1.4, ls='--',
               label='Safety Zone'),
        Line2D([0],[0], marker='o', color='w',
               markerfacecolor='red', markersize=7, alpha=0.6,
               label='Collision'),
    ]
    ax.legend(handles=items, loc='upper right', fontsize=5.8,
              framealpha=0.90, facecolor='#050D18', edgecolor='#1A3A55',
              labelcolor='#AAD4F5', ncol=2, handlelength=1.2)

def draw_progress_bar(ax, frame):
    prog = frame / FRAMES
    bx1, bx2, by = 0.05, WW-0.05, -0.58
    ax.plot([bx1,bx2],[by,by], color='#152A40', lw=5.5,
            solid_capstyle='round', zorder=15)
    ax.plot([bx1, bx1+(bx2-bx1)*prog],[by,by], color='#3A80C8', lw=5.5,
            solid_capstyle='round', zorder=16)

fig, (ax_s, ax_o) = plt.subplots(1, 2, figsize=(24, 11), dpi=90)
fig.patch.set_facecolor(C_BG)
plt.subplots_adjust(left=0.01, right=0.99, top=0.93, bottom=0.02, wspace=0.04)
fig.suptitle(
    'Warehouse Coordination — Baseline Comparison\n'
    'Mohammed Al-Khudhair  |  Honours Project  |  Habitat 3.0 (Puig et al., 2023)',
    color='#8AB4D4', fontsize=14, fontfamily='monospace', y=0.985
)

def render_frame(frame):
    step_simple(S)
    step_orca(O, o_sim)

    ax_s.clear()
    ax_o.clear()

    draw_base(ax_s, 'Baseline: Simple Repulsion', C_SIMPLE)
    draw_agents(ax_s, S)
    draw_stats_panel(ax_s, S, frame, 'Simple Repulsion', C_SIMPLE)
    draw_legend_panel(ax_s)
    draw_progress_bar(ax_s, frame)

    draw_base(ax_o, 'ORCA / RVO2 Coordination', C_ORCA)
    draw_agents(ax_o, O)
    draw_stats_panel(ax_o, O, frame, 'ORCA / RVO2', C_ORCA)
    draw_legend_panel(ax_o)
    draw_progress_bar(ax_o, frame)

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps)...")
print(f"Output -> {OUTPUT_FILE}\n")

writer = FFMpegWriter(fps=FPS, metadata={'title':'Warehouse Demo v4'}, bitrate=4500)

with writer.saving(fig, OUTPUT_FILE, dpi=90):
    for frame in range(FRAMES):
        render_frame(frame)
        writer.grab_frame()
        if frame % (FPS*5) == 0:
            pct = int(frame/FRAMES*100)
            print(
                f"  {pct:3d}%  t={frame*DT:.0f}s  "
                f"simple: {sum(S['completed'])} tasks, {S['collisions']} col  |  "
                f"orca: {sum(O['completed'])} tasks, {O['collisions']} col"
            )

print(f"\nComplete -> {OUTPUT_FILE}")
print(f"Simple repulsion : {sum(S['completed'])} tasks | {S['collisions']} collisions | {S['safety_evts']} safety events")
print(f"ORCA / RVO2      : {sum(O['completed'])} tasks | {O['collisions']} collisions | {O['safety_evts']} safety events")
