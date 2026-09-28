"""
Warehouse Demo v3 - Fixed
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Circle, Ellipse
from matplotlib.animation import FFMpegWriter
from matplotlib.lines import Line2D
import os

OUTPUT_DIR  = "/mnt/e/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v3.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 30
DURATION = 35
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

TW = 0.52
TH = 0.27
TZ = 0.44

def iso(x, y, z=0.0):
    return (y - x) * TW, (x + y) * TH - z * TZ

WW, WH = 26, 18

C_FLOOR      = '#0B1929'
C_FLOOR_ALT  = '#0E2035'
C_FLOOR_LINE = '#172D44'
C_HIGHWAY    = '#0D2240'
C_HIGHWAY_L  = '#1A3860'

C_SHELF_TOP   = '#1B3C6E'
C_SHELF_LIT   = '#264870'
C_SHELF_SHADE = '#122D52'
C_SHELF_EDGE  = '#2A5FAA'

C_COLD_TOP   = '#1A4A5E'
C_COLD_LIT   = '#1E5A70'
C_COLD_SHADE = '#102E40'
C_COLD_EDGE  = '#2080AA'

C_PICK_TOP   = '#163060'
C_PICK_LIT   = '#1C3870'
C_PICK_SHADE = '#0E2040'
C_PICK_EDGE  = '#2050A0'

C_BOX_TOP   = '#E07B39'
C_BOX_LIT   = '#D4622A'
C_BOX_SHADE = '#A3421F'

C_COLD_BOX_TOP   = '#7EC8E3'
C_COLD_BOX_LIT   = '#60AABF'
C_COLD_BOX_SHADE = '#408090'

C_STATION   = '#1B4332'
C_STAT_LIT  = '#1F5040'
C_STAT_EDGE = '#52C77A'

C_DOCK      = '#3D2B10'
C_DOCK_LIT  = '#4A350E'
C_DOCK_EDGE = '#F3A81A'

C_CONV      = '#1C3030'
C_CONV_LIT  = '#203838'
C_CONV_EDGE = '#48C9B0'

C_CHARGE    = '#2D1B69'
C_CHARGE_L  = '#3A2280'
C_CHARGE_E  = '#9B59B6'

C_PALLET    = '#6B4E20'
C_PALLET_L  = '#7A5A25'

C_HUMAN_1 = '#FF6B35'
C_HUMAN_2 = '#FFD93D'
C_HUMAN_3 = '#FF4E88'
C_SAFETY  = '#E74C3C'
C_BG      = '#050D18'
C_ZONE    = '#4A90D9'

ROBOT_COLORS = [
    '#3498DB','#2ECC71','#9B59B6','#E74C3C','#F39C12',
    '#1ABC9C','#E91E63','#FF5722','#00BCD4','#8BC34A'
]
HUMAN_COLORS = [C_HUMAN_1, C_HUMAN_2, C_HUMAN_3]

SHELVES = []
for gy in [1.0, 2.8, 4.6, 10.2, 12.0, 13.8]:
    for gx in [4.5, 6.8]:
        SHELVES.append((gx, gy, 1.8, 0.7, 1.4, 'cold'))
for gy in [1.0, 2.8, 4.6, 10.2, 12.0, 13.8]:
    for gx in [10.5, 12.5]:
        SHELVES.append((gx, gy, 1.8, 0.7, 1.0, 'bulk'))
for gy in [1.0, 2.2, 3.4, 4.6, 10.2, 11.4, 12.6, 13.8]:
    for gx in [15.5, 17.2]:
        SHELVES.append((gx, gy, 1.6, 0.6, 0.7, 'pick'))

DOCKS    = [(0.5,3.0),(0.5,8.5),(0.5,14.0)]
STATIONS = [(20.5,3.0),(20.5,8.5),(20.5,14.0)]
CONVEYORS= [(21.5,1.0,1.2,6.0),(21.5,10.5,1.2,5.0)]
CHARGES  = [(22.5,8.0),(23.5,8.0),(24.5,8.0),(25.5,8.0)]
PALLETS  = [(1.5,1.5),(1.5,3.0),(2.5,1.5),(1.5,13.5),(2.5,15.0)]

HUMAN_WPS = [
    [(1.0,1.0),(1.0,17.0),(13.0,17.0),(25.0,17.0),
     (25.0,9.5),(25.0,1.0),(13.0,1.0),(1.0,1.0)],
    [(15.5,2.0),(18.5,2.0),(19.0,6.5),(19.0,7.5),
     (15.5,7.5),(15.5,10.0),(19.0,10.0),(19.0,16.5),
     (15.5,16.5),(15.5,10.0),(15.5,2.0)],
    [(10.0,1.5),(10.0,7.5),(10.0,9.5),(10.0,16.5),
     (5.0,16.5),(5.0,9.5),(5.0,7.5),(5.0,1.5),(10.0,1.5)],
]
HUMAN_SPEEDS = [0.028, 0.032, 0.030]

def mc(pts): return pts + [pts[0]]

ROBOT_TASKS = [
    mc([(5.5,2.0),(5.5,7.0),(20.5,3.0)]),
    mc([(7.5,2.0),(7.5,7.0),(20.5,8.5)]),
    mc([(11.0,2.0),(11.0,7.0),(20.5,14.0)]),
    mc([(13.0,2.0),(13.0,7.0),(20.5,3.0)]),
    mc([(16.0,2.0),(16.0,7.0),(20.5,8.5)]),
    mc([(18.0,2.0),(18.0,7.0),(20.5,14.0)]),
    mc([(5.5,10.0),(5.5,16.0),(20.5,3.0)]),
    mc([(11.0,10.0),(11.0,16.0),(20.5,8.5)]),
    mc([(16.0,10.0),(16.0,16.0),(20.5,14.0)]),
    mc([(2.0,8.5),(13.0,8.5),(20.5,8.5),(2.0,8.5)]),
]
ROBOT_STARTS = [
    (5.5,2.0),(7.5,2.0),(11.0,2.0),(13.0,2.0),(16.0,2.0),
    (18.0,2.0),(5.5,10.0),(11.0,10.0),(16.0,10.0),(2.0,8.5),
]

N_HUMANS = 3
N_ROBOTS = 10
ROBOT_SPEED   = 0.055
SAFETY_RADIUS = 0.85

human_pos    = [np.array(HUMAN_WPS[h][0], dtype=float) for h in range(N_HUMANS)]
human_wp_idx = [1] * N_HUMANS
human_phase  = [0.0, 1.05, 2.1]

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
        if 0 < dist < radius * 1.7:
            force += (d / dist) * (radius * 1.7 - dist) * 0.022
    return force

def draw_box(ax, x, y, w, d, z0, z1, c_top, c_lit, c_shade, c_edge, alpha=1.0, zo=3):
    ax.add_patch(Polygon(
        [iso(x,y,z1), iso(x+w,y,z1), iso(x+w,y+d,z1), iso(x,y+d,z1)],
        closed=True, facecolor=c_top, edgecolor=c_edge, lw=0.5, alpha=alpha, zorder=zo))
    ax.add_patch(Polygon(
        [iso(x,y,z1), iso(x+w,y,z1), iso(x+w,y,z0), iso(x,y,z0)],
        closed=True, facecolor=c_lit, edgecolor=c_edge, lw=0.5, alpha=alpha, zorder=zo))
    ax.add_patch(Polygon(
        [iso(x,y,z1), iso(x,y+d,z1), iso(x,y+d,z0), iso(x,y,z0)],
        closed=True, facecolor=c_shade, edgecolor=c_edge, lw=0.5, alpha=alpha, zorder=zo))

def draw_tile(ax, x, y, color, ec, zo=1):
    ax.add_patch(Polygon(
        [iso(x,y), iso(x+1,y), iso(x+1,y+1), iso(x,y+1)],
        closed=True, facecolor=color, edgecolor=ec, lw=0.25, alpha=0.9, zorder=zo))

def draw_robot(ax, x, y, color, label, vel, zo=8):
    rh = 0.38
    bx,by = iso(x,y,0)
    tx,ty = iso(x,y,rh)
    ax.add_patch(Ellipse((bx,by), 0.22*TW*4, 0.22*TH*4,
        facecolor='black', alpha=0.3, zorder=zo-1))
    cx2=(bx+tx)/2; cy2=(by+ty)/2
    ax.add_patch(Ellipse((cx2, cy2+(ty-by)*0.15), 0.22*TW*4.5,
        abs(ty-by)+0.22*TH*3,
        facecolor=color, edgecolor='white', lw=0.7, alpha=0.95, zorder=zo))
    ax.add_patch(Ellipse((tx,ty), 0.22*TW*4.5, 0.22*TH*4,
        facecolor=color, edgecolor='white', lw=0.5, alpha=1.0, zorder=zo+1))
    if np.linalg.norm(vel) > 0.001:
        vn = vel / np.linalg.norm(vel)
        ex, ey = iso(x + vn[0]*0.55, y + vn[1]*0.55, rh)
        ax.annotate('', xy=(ex,ey), xytext=(tx,ty),
            arrowprops=dict(arrowstyle='->', color='white', lw=1.1), zorder=zo+2)
    ax.text(tx, ty+0.08, label, ha='center', va='center',
            fontsize=6.5, color='white', fontweight='bold', zorder=zo+3)

def draw_humanoid(ax, x, y, phase, color, zo=10):
    sx_head, sy_head = iso(x, y, 0.95)
    sx_neck, sy_neck = iso(x, y, 0.78)
    sx_hip,  sy_hip  = iso(x, y, 0.42)

    swing = np.sin(phase)
    bob   = abs(swing) * 0.022 * TZ

    ax.add_patch(Circle((sx_head, sy_head + bob), 0.092,
        facecolor=color, edgecolor='white', lw=0.7, zorder=zo+1))

    ax.plot([sx_neck, sx_hip], [sy_neck+bob, sy_hip+bob*0.5],
            color=color, lw=2.2, solid_capstyle='round', zorder=zo)

    arm_off = swing * 0.11
    ax.plot([sx_neck, sx_neck - 0.10 - arm_off],
            [sy_neck+bob, sy_neck + bob - 0.13 - arm_off*0.3],
            color=color, lw=1.7, solid_capstyle='round', zorder=zo)
    ax.plot([sx_neck, sx_neck + 0.10 + arm_off],
            [sy_neck+bob, sy_neck + bob - 0.13 + arm_off*0.3],
            color=color, lw=1.7, solid_capstyle='round', zorder=zo)

    leg_off = swing * 0.10
    ax.plot([sx_hip, sx_hip - 0.07 + leg_off],
            [sy_hip+bob*0.5, sy_hip + bob*0.5 - 0.16 + leg_off*0.4],
            color=color, lw=2.0, solid_capstyle='round', zorder=zo)
    ax.plot([sx_hip, sx_hip + 0.07 - leg_off],
            [sy_hip+bob*0.5, sy_hip + bob*0.5 - 0.16 - leg_off*0.4],
            color=color, lw=2.0, solid_capstyle='round', zorder=zo)

all_corners = [iso(x,y,0) for x,y in [(0,0),(WW,0),(WW,WH),(0,WH)]]
X_MIN = min(c[0] for c in all_corners) - 1.8
X_MAX = max(c[0] for c in all_corners) + 1.8
Y_MIN = min(c[1] for c in all_corners) - 1.5
Y_MAX = max(c[1] for c in all_corners) + 1.5

fig, ax = plt.subplots(figsize=(18, 11), dpi=100)
fig.patch.set_facecolor(C_BG)

def draw_frame(frame):
    global safety_events

    ax.clear()
    ax.set_facecolor(C_BG)
    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(Y_MIN, Y_MAX + 1.0)
    ax.set_aspect('equal')
    ax.axis('off')

    for gy in range(WH):
        for gx in range(WW):
            if 7.5 <= gy + 0.5 <= 9.5:
                col = C_HIGHWAY if (gx+gy)%2==0 else C_HIGHWAY_L
            else:
                col = C_FLOOR if (gx+gy)%2==0 else C_FLOOR_ALT
            draw_tile(ax, gx, gy, col, C_FLOOR_LINE)

    for xd in [4,10,15,20]:
        p1=iso(xd,0,0); p2=iso(xd,WH,0)
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                color='#1A4070', lw=1.0, linestyle='--', alpha=0.6, zorder=2)
    for yd in [7.5,9.5]:
        p1=iso(0,yd,0); p2=iso(WW,yd,0)
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                color='#2060A0', lw=0.8, linestyle=':', alpha=0.5, zorder=2)

    for (zx,zy,ztxt) in [(2.0,8.5,'RECEIVING'),(7.0,8.5,'COLD STORAGE'),
                          (12.5,8.5,'BULK STORAGE'),(17.5,8.5,'PICK ZONE'),
                          (23.0,8.5,'SHIPPING')]:
        lx,ly=iso(zx,zy,0.02)
        ax.text(lx,ly,ztxt,ha='center',va='center',fontsize=6.0,
                color=C_ZONE,alpha=0.55,fontfamily='monospace',zorder=2)

    for (px2,py2) in PALLETS:
        draw_box(ax,px2,py2,0.8,0.6,0,0.18,
                 C_PALLET,C_PALLET_L,'#4A3415','#9B7030',alpha=0.9,zo=3)

    for (dx2,dy2) in DOCKS:
        draw_box(ax,dx2-0.5,dy2-0.8,1.0,1.6,0,0.5,
                 C_DOCK,C_DOCK_LIT,'#251A0A',C_DOCK_EDGE,alpha=0.95,zo=3)
        lx,ly=iso(dx2,dy2,0.55)
        ax.text(lx,ly,'DOCK',ha='center',va='bottom',
                fontsize=7.5,color=C_DOCK_EDGE,fontweight='bold',zorder=5)

    for (cx2,cy2,cw,cd) in CONVEYORS:
        draw_box(ax,cx2,cy2,cw,cd,0,0.25,
                 C_CONV,C_CONV_LIT,'#101F1F',C_CONV_EDGE,alpha=0.92,zo=3)
        lx,ly=iso(cx2+cw/2,cy2+cd/2,0.28)
        ax.text(lx,ly,'CONV',ha='center',va='bottom',
                fontsize=6,color=C_CONV_EDGE,fontweight='bold',zorder=5)

    for (sx2,sy2,sw,sd,sh,zone) in sorted(SHELVES, key=lambda s: -(s[0]+s[1])):
        if zone == 'cold':
            ct,cl,cs,ec = C_COLD_TOP,C_COLD_LIT,C_COLD_SHADE,C_COLD_EDGE
            bt,bl,bs = C_COLD_BOX_TOP,C_COLD_BOX_LIT,C_COLD_BOX_SHADE
        elif zone == 'pick':
            ct,cl,cs,ec = C_PICK_TOP,C_PICK_LIT,C_PICK_SHADE,C_PICK_EDGE
            bt,bl,bs = C_BOX_TOP,C_BOX_LIT,C_BOX_SHADE
        else:
            ct,cl,cs,ec = C_SHELF_TOP,C_SHELF_LIT,C_SHELF_SHADE,C_SHELF_EDGE
            bt,bl,bs = C_BOX_TOP,C_BOX_LIT,C_BOX_SHADE

        draw_box(ax,sx2,sy2,sw,sd,0,sh,ct,cl,cs,ec,alpha=0.95,zo=3)
        rng = np.random.RandomState(int(sx2*13+sy2*7))
        for bx2 in np.arange(sx2+0.12,sx2+sw-0.08,0.40):
            bh2 = sh + 0.18 + rng.uniform(0,0.12)
            draw_box(ax,bx2,sy2+0.08,0.25,sd*0.75,sh,bh2,bt,bl,bs,'#FF9060',
                     alpha=0.88,zo=4)

    for (px2,py2) in STATIONS:
        draw_box(ax,px2-0.4,py2-0.7,0.8,1.4,0,0.65,
                 C_STATION,C_STAT_LIT,'#0A1E15',C_STAT_EDGE,alpha=0.95,zo=3)
        lx,ly=iso(px2,py2,0.70)
        ax.text(lx,ly,'PICK\nSTATION',ha='center',va='bottom',
                fontsize=7.0,color=C_STAT_EDGE,fontweight='bold',zorder=5)

    for (cx2,cy2) in CHARGES:
        draw_box(ax,cx2-0.28,cy2-0.28,0.56,0.56,0,0.18,
                 C_CHARGE,C_CHARGE_L,'#180A3A',C_CHARGE_E,alpha=0.9,zo=3)
        lx,ly=iso(cx2,cy2,0.22)
        ax.text(lx,ly,'+',ha='center',va='bottom',
                fontsize=10,color=C_CHARGE_E,fontweight='bold',zorder=5)

    for h in range(N_HUMANS):
        wps = HUMAN_WPS[h]
        tgt = np.array(wps[human_wp_idx[h]])
        diff= tgt - human_pos[h]
        dist= np.linalg.norm(diff)
        if dist < HUMAN_SPEEDS[h]:
            human_pos[h][:] = tgt
            human_wp_idx[h] = (human_wp_idx[h]+1) % len(wps)
        else:
            human_pos[h] += (diff/dist) * HUMAN_SPEEDS[h]
        human_phase[h] += 0.20

    all_pos = [hp.copy() for hp in human_pos] + [rp.copy() for rp in robot_pos]

    for i in range(N_ROBOTS):
        if robot_state[i] == 'wait':
            robot_wait[i] -= 1
            if robot_wait[i] <= 0:
                robot_state[i] = 'move'
            robot_vel[i] *= 0.0
            continue
        tgt  = np.array(ROBOT_TASKS[i][robot_task_idx[i]])
        diff = tgt - robot_pos[i]
        dist = np.linalg.norm(diff)
        if dist < ROBOT_SPEED:
            robot_pos[i]      = tgt.copy()
            robot_task_idx[i] = (robot_task_idx[i]+1) % len(ROBOT_TASKS[i])
            completed[i]     += 1
            robot_state[i]    = 'wait'
            robot_wait[i]     = int(FPS*0.45)
        else:
            vel = (diff/dist) * ROBOT_SPEED
            others = [all_pos[j] for j in range(len(all_pos)) if j != i+N_HUMANS]
            vel   += repulse(robot_pos[i], others, SAFETY_RADIUS)
            robot_pos[i] += vel
            robot_vel[i]  = vel

    for i in range(N_ROBOTS):
        for h in range(N_HUMANS):
            if np.linalg.norm(robot_pos[i]-human_pos[h]) < SAFETY_RADIUS:
                safety_events += 1

    for i in range(N_ROBOTS):
        rx,ry=iso(robot_pos[i][0],robot_pos[i][1],0)
        ax.add_patch(Ellipse((rx,ry),SAFETY_RADIUS*TW*3.8,SAFETY_RADIUS*TH*3.8,
            facecolor=ROBOT_COLORS[i],alpha=0.07,zorder=5))
        ax.add_patch(Ellipse((rx,ry),SAFETY_RADIUS*TW*3.8,SAFETY_RADIUS*TH*3.8,
            facecolor='none',edgecolor=ROBOT_COLORS[i],
            linestyle='--',lw=0.8,alpha=0.40,zorder=5))

    for i in range(N_ROBOTS):
        nxt=np.array(ROBOT_TASKS[i][robot_task_idx[i]])
        p1=iso(robot_pos[i][0],robot_pos[i][1],0.05)
        p2=iso(nxt[0],nxt[1],0.05)
        ax.plot([p1[0],p2[0]],[p1[1],p2[1]],
                color=ROBOT_COLORS[i],lw=0.5,linestyle=':',alpha=0.3,zorder=6)

    for i in sorted(range(N_ROBOTS),
                    key=lambda i: -(robot_pos[i][0]+robot_pos[i][1])):
        draw_robot(ax,robot_pos[i][0],robot_pos[i][1],
                   ROBOT_COLORS[i],f'R{i+1}',robot_vel[i],zo=7)

    for h in sorted(range(N_HUMANS),
                    key=lambda h: -(human_pos[h][0]+human_pos[h][1])):
        draw_humanoid(ax,human_pos[h][0],human_pos[h][1],
                      human_phase[h],HUMAN_COLORS[h],zo=10)
        hsx,hsy=iso(human_pos[h][0],human_pos[h][1],0)
        ax.add_patch(Ellipse((hsx,hsy),SAFETY_RADIUS*TW*5,SAFETY_RADIUS*TH*5,
            facecolor=HUMAN_COLORS[h],alpha=0.05,zorder=6))

    ax.set_title(
        'Deep RL for Multi-Agent Coordination in Mixed Human-Robot Warehouse Environments\n'
        'Mohammed Al-Khudhair  |  Honours Project Demo  |  Habitat 3.0 (Puig et al., 2023)',
        color='#AAD4F5',fontsize=13,pad=10,fontfamily='monospace')

    panel = (
        f"  Simulation Time :  {frame*DT:5.1f}s\n"
        f"  Robots Active   :  {N_ROBOTS}\n"
        f"  Human Workers   :  {N_HUMANS}\n"
        f"  Tasks Completed :  {sum(completed)}\n"
        f"  Safety Radius   :  {SAFETY_RADIUS}m  (ISO 10218)\n"
        f"  Proximity Events:  {safety_events}"
    )
    ax.text(0.005,0.02,panel,transform=ax.transAxes,fontsize=9.5,
            color='#AAD4F5',va='bottom',fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.55',facecolor='#050D18',
                      edgecolor='#1A3A55',alpha=0.92))

    legend_items = [
        Line2D([0],[0],marker='o',color='w',markerfacecolor=ROBOT_COLORS[i],
               markersize=8,label=f'Robot {i+1}') for i in range(N_ROBOTS)
    ] + [
        Line2D([0],[0],marker='o',color='w',markerfacecolor=C_HUMAN_1,
               markersize=9,label='Human 1'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=C_HUMAN_2,
               markersize=9,label='Human 2'),
        Line2D([0],[0],marker='o',color='w',markerfacecolor=C_HUMAN_3,
               markersize=9,label='Human 3'),
        Line2D([0],[0],color=C_SAFETY,lw=1.8,linestyle='--',label='Safety Zone'),
    ]
    ax.legend(handles=legend_items,loc='upper right',fontsize=7.5,
              framealpha=0.92,facecolor='#050D18',edgecolor='#1A3A55',
              labelcolor='#AAD4F5',ncol=2)

    prog=frame/FRAMES
    bx1=X_MIN+0.4; bx2=X_MAX-0.4; bar_y=Y_MIN+0.15
    ax.plot([bx1,bx2],[bar_y,bar_y],color='#1A3A55',lw=5,
            solid_capstyle='round',zorder=15)
    ax.plot([bx1,bx1+(bx2-bx1)*prog],[bar_y,bar_y],color='#4A90D9',lw=5,
            solid_capstyle='round',zorder=16)

print(f"Rendering {FRAMES} frames...")
writer = FFMpegWriter(fps=FPS,metadata={'title':'Warehouse Demo v3'},bitrate=3500)

with writer.saving(fig,OUTPUT_FILE,dpi=100):
    for frame in range(FRAMES):
        draw_frame(frame)
        writer.grab_frame()
        if frame % FPS == 0:
            print(f"  {int(frame/FRAMES*100):3d}%  tasks={sum(completed)}  events={safety_events}")

print(f"Done -> {OUTPUT_FILE}")