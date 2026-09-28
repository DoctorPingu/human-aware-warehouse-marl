"""
Warehouse Demo v6b - Final Version
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
3 Robots | 3 Humans | ORCA/RVO2 | Junction markings | Uniform inventory
Duration: 50 seconds
"""

import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pyvista as pv
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
import rvo2

print("Warehouse Demo v6b - Initialising...")

OUTPUT_DIR  = "/mnt/e/Honours Project/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v6b.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 24
DURATION = 50
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

WW = 34.0
WH = 24.0
WALL_H = 1.6
WALL_T = 0.25

N_ROBOTS    = 3
N_HUMANS    = 3
ROBOT_R     = 0.25
HUMAN_R     = 0.20
SAFETY_R    = 0.80
COLLISION_R = 0.44
ROBOT_SPD   = 1.6
HUMAN_SPD   = 0.70
ROBOT_STEP  = ROBOT_SPD * DT
HUMAN_STEP  = HUMAN_SPD * DT

ROBOT_COL_HEX = '#3498DB'
HUMAN_COL_HEX = '#FF6B35'

def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2],16)/255.0 for i in (0,2,4))

ROBOT_RGB = hex_rgb(ROBOT_COL_HEX)
HUMAN_RGB = hex_rgb(HUMAN_COL_HEX)
ROBOT_NAMES = ['AMR-1','AMR-2','AMR-3']
HUMAN_NAMES = ['Worker-1','Worker-2','Worker-3']

SHELF_W = 2.0
SHELF_D = 0.85
SHELF_H = 1.0

SHELF_COLS  = [5.0, 10.0, 15.0, 20.0, 25.0]
SHELF_ROWS_L= [1.5,  5.0,  8.5]
SHELF_ROWS_U= [14.5, 18.0, 21.5]

SHELVES = []
for cx in SHELF_COLS:
    for ry in SHELF_ROWS_L:
        SHELVES.append((cx, ry, SHELF_W, SHELF_D, SHELF_H))
    for ry in SHELF_ROWS_U:
        SHELVES.append((cx, ry, SHELF_W, SHELF_D, SHELF_H))

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

DOCKS    = [(0.8, 4.0),(0.8, 12.0),(0.8, 20.0)]
STATIONS = [(30.8, 4.0),(30.8, 12.0),(30.8, 20.0)]
PALLETS  = [(1.6, 6.5),(2.6, 6.5),(1.6, 16.0),(2.6, 16.0)]

CX = [2.0, 7.5, 12.5, 17.5, 22.5, 27.5, 32.0]
CY = [3.25, 6.75, 12.0, 16.25, 19.75]

JUNCTIONS = []
for x in [7.5, 12.5, 17.5, 22.5, 27.5]:
    for y in [3.25, 6.75, 12.0, 16.25, 19.75]:
        JUNCTIONS.append((x, y))

def mc(pts): return pts + [pts[0]]

ROBOT_STARTS = [
    (CX[1], CY[0]),
    (CX[2], CY[0]),
    (CX[3], CY[0]),
]

ROBOT_TASKS = [
    mc([(CX[1],CY[0]),(CX[1],CY[1]),(CX[1],CY[2]),(CX[5],CY[2]),
        (STATIONS[0][0],STATIONS[0][1]),
        (CX[5],CY[2]),(CX[1],CY[2]),(CX[1],CY[3]),(CX[1],CY[4]),
        (CX[1],CY[2]),(CX[1],CY[0])]),
    mc([(CX[2],CY[0]),(CX[2],CY[1]),(CX[2],CY[2]),(CX[4],CY[2]),
        (STATIONS[1][0],STATIONS[1][1]),
        (CX[4],CY[2]),(CX[2],CY[2]),(CX[2],CY[3]),(CX[2],CY[4]),
        (CX[2],CY[2]),(CX[2],CY[0])]),
    mc([(CX[3],CY[0]),(CX[3],CY[1]),(CX[3],CY[2]),(CX[5],CY[2]),
        (STATIONS[2][0],STATIONS[2][1]),
        (CX[5],CY[4]),(CX[3],CY[4]),(CX[3],CY[3]),(CX[3],CY[2]),
        (CX[3],CY[0])]),
]

HUMAN_WPS = [
    mc([(CX[0], CY[2]), (CX[6], CY[2])]),
    mc([(CX[1], 1.0), (CX[1], 22.5)]),
    mc([(CX[5], 1.0), (CX[5], 22.5)]),
]
HUMAN_SPEEDS = [0.030, 0.024, 0.024]

def make_state():
    return dict(
        robot_pos      = [np.array(p,dtype=float) for p in ROBOT_STARTS],
        robot_vel      = [np.zeros(2) for _ in range(N_ROBOTS)],
        robot_task_idx = [0]*N_ROBOTS,
        robot_state    = ['move']*N_ROBOTS,
        robot_wait     = [0]*N_ROBOTS,
        human_pos      = [np.array(HUMAN_WPS[h][0],dtype=float) for h in range(N_HUMANS)],
        human_wp_idx   = [1]*N_HUMANS,
        human_phase    = [0.0,1.05,2.1],
        completed      = [0]*N_ROBOTS,
        collisions     = 0,
        safety_evts    = 0,
        wall_hits      = 0,
    )

O = make_state()

o_sim = rvo2.PyRVOSimulator(DT, 5.0, 15, 8.0, 2.0, ROBOT_R, ROBOT_SPD)
for i in range(N_ROBOTS):
    o_sim.addAgent(tuple(O['robot_pos'][i]),
                   5.0,15,8.0,2.0,ROBOT_R,ROBOT_SPD,(0.0,0.0))
for h in range(N_HUMANS):
    o_sim.addAgent(tuple(O['human_pos'][h]),
                   5.0,15,5.0,2.0,HUMAN_R,HUMAN_SPD,(0.0,0.0))

for (x1,y1,x2,y2) in WALL_SEGS:
    o_sim.addObstacle([(x1,y1),(x2,y2)])

for (sx,sy,sw,sd,sh) in SHELVES:
    o_sim.addObstacle([
        (sx,    sy   ),
        (sx+sw, sy   ),
        (sx+sw, sy+sd),
        (sx,    sy+sd),
    ])

o_sim.processObstacles()

def hard_clamp(pos, radius):
    margin = radius + 0.10
    p = pos.copy()
    clamped = False
    if p[0] < margin:        p[0] = margin;        clamped=True
    if p[0] > WW - margin:   p[0] = WW - margin;   clamped=True
    if p[1] < margin:        p[1] = margin;        clamped=True
    if p[1] > WH - margin:   p[1] = WH - margin;   clamped=True
    return p, clamped

def tally(state):
    ap = list(state['human_pos']) + list(state['robot_pos'])
    n  = len(ap)
    for i in range(n):
        for j in range(i+1, n):
            d = np.linalg.norm(ap[i] - ap[j])
            if d < COLLISION_R:
                state['collisions'] += 1
            elif d < SAFETY_R:
                state['safety_evts'] += 1

def step_orca(state, sim):
    for i in range(N_ROBOTS):
        if state['robot_state'][i] == 'wait':
            sim.setAgentPrefVelocity(i, (0.0, 0.0))
        else:
            tgt  = np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            diff = tgt - state['robot_pos'][i]
            dist = np.linalg.norm(diff)
            if dist < 0.30:
                sim.setAgentPrefVelocity(i, (0.0, 0.0))
            else:
                spd = min(ROBOT_SPD, dist * 2.5)
                sim.setAgentPrefVelocity(i, tuple((diff/dist)*spd))

    for h in range(N_HUMANS):
        tgt  = np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        diff = tgt - state['human_pos'][h]
        dist = np.linalg.norm(diff)
        if dist < 0.20:
            sim.setAgentPrefVelocity(N_ROBOTS+h, (0.0, 0.0))
        else:
            sim.setAgentPrefVelocity(N_ROBOTS+h,
                tuple((diff/dist)*HUMAN_SPEEDS[h]))

    sim.doStep()

    for i in range(N_ROBOTS):
        pos = np.array(sim.getAgentPosition(i))
        pos, hit = hard_clamp(pos, ROBOT_R)
        if hit: state['wall_hits'] += 1
        vel = np.array(sim.getAgentVelocity(i))
        state['robot_pos'][i] = pos
        state['robot_vel'][i] = vel
        sim.setAgentPosition(i, tuple(pos))

        if state['robot_state'][i] == 'wait':
            state['robot_wait'][i] -= 1
            if state['robot_wait'][i] <= 0:
                state['robot_state'][i] = 'move'
        else:
            tgt = np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            if np.linalg.norm(tgt - pos) < 0.50:
                state['robot_task_idx'][i] = \
                    (state['robot_task_idx'][i]+1) % len(ROBOT_TASKS[i])
                state['completed'][i] += 1
                state['robot_state'][i] = 'wait'
                state['robot_wait'][i]  = int(FPS * 0.5)

    for h in range(N_HUMANS):
        pos = np.array(sim.getAgentPosition(N_ROBOTS+h))
        pos, hit = hard_clamp(pos, HUMAN_R)
        if hit: state['wall_hits'] += 1
        state['human_pos'][h]   = pos
        state['human_phase'][h] += 0.20
        sim.setAgentPosition(N_ROBOTS+h, tuple(pos))
        tgt = np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        if np.linalg.norm(tgt - pos) < 0.40:
            state['human_wp_idx'][h] = \
                (state['human_wp_idx'][h]+1) % len(HUMAN_WPS[h])

    tally(state)

ROBOT_BODY = pv.Cylinder(center=(0,0,0.28),direction=(0,0,1),
                          radius=ROBOT_R, height=0.56, resolution=14)
ROBOT_CAP  = pv.Sphere(center=(0,0,0.62), radius=ROBOT_R*0.42)
HUMAN_BODY = pv.Cylinder(center=(0,0,0.38),direction=(0,0,1),
                          radius=HUMAN_R, height=0.76, resolution=12)
HUMAN_HEAD = pv.Sphere(center=(0,0,0.90), radius=HUMAN_R*0.68)
ZONE_RING  = pv.Disc(center=(0,0,0.02), normal=(0,0,1),
                      inner=SAFETY_R-0.07, outer=SAFETY_R+0.07,
                      r_res=1, c_res=30)
COL_SPHERE = pv.Sphere(center=(0,0,0.50), radius=0.34)

def tmesh(mesh, x, y):
    m = mesh.copy()
    m.translate([x,y,0], inplace=True)
    return m

pl = pv.Plotter(off_screen=True, window_size=[2200,1080], border=False)
pl.set_background('#060E1A')

CAM_POS   = (WW/2, WH/2 - 20, 28)
CAM_FOCUS = (WW/2, WH/2 + 2.0, 0)
CAM_UP    = (0, 1, 0)

def build_static():
    floor = pv.Plane(center=(WW/2, WH/2, -0.04), direction=(0,0,1),
                     i_size=WW, j_size=WH)
    pl.add_mesh(floor, color=hex_rgb('#0A1828'),
                show_edges=False, reset_camera=False)

    hw = pv.Box(bounds=(0, WW, 10.5, 13.5, -0.03, 0.02))
    pl.add_mesh(hw, color=hex_rgb('#0D2540'),
                show_edges=False, reset_camera=False)

    for (jx, jy) in JUNCTIONS:
        jm = pv.Disc(center=(jx, jy, 0.005), normal=(0,0,1),
                     inner=0.0, outer=0.45, r_res=1, c_res=20)
        pl.add_mesh(jm, color=hex_rgb('#1F4868'),
                    show_edges=False, reset_camera=False)
        ring = pv.Disc(center=(jx, jy, 0.008), normal=(0,0,1),
                       inner=0.40, outer=0.50, r_res=1, c_res=24)
        pl.add_mesh(ring, color=hex_rgb('#3A6890'),
                    show_edges=False, reset_camera=False)
        x_h = pv.Box(bounds=(jx-0.30, jx+0.30, jy-0.04, jy+0.04, 0.012, 0.020))
        pl.add_mesh(x_h, color=hex_rgb('#5A88B0'),
                    show_edges=False, reset_camera=False)
        x_v = pv.Box(bounds=(jx-0.04, jx+0.04, jy-0.30, jy+0.30, 0.012, 0.020))
        pl.add_mesh(x_v, color=hex_rgb('#5A88B0'),
                    show_edges=False, reset_camera=False)

    for (x1,y1,x2,y2) in WALL_SEGS:
        if x1 == x2:
            wm = pv.Box(bounds=(x1-WALL_T/2, x1+WALL_T/2,
                                min(y1,y2), max(y1,y2),
                                0, WALL_H if x1 in (0.0, WW) else WALL_H*0.7))
        else:
            wm = pv.Box(bounds=(min(x1,x2), max(x1,x2),
                                y1-WALL_T/2, y1+WALL_T/2,
                                0, WALL_H))
        pl.add_mesh(wm, color=hex_rgb('#1C3A5A'), show_edges=True,
                    edge_color=hex_rgb('#2A5A8A'), line_width=0.5,
                    reset_camera=False)

    BOX_W   = 0.30
    BOX_D_F = 0.70
    BOX_HGT = 0.28
    BOX_GAP = 0.08

    for (sx, sy, sw, sd, sh) in SHELVES:
        shelf = pv.Box(bounds=(sx, sx+sw, sy, sy+sd, 0, sh))
        pl.add_mesh(shelf, color=hex_rgb('#122438'), show_edges=True,
                    edge_color=hex_rgb('#1C3C6A'), line_width=0.5,
                    reset_camera=False)

        n_boxes = int((sw - 0.20) // (BOX_W + BOX_GAP))
        total_w = n_boxes * BOX_W + (n_boxes-1) * BOX_GAP
        start_x = sx + (sw - total_w) / 2.0
        box_d   = sd * BOX_D_F
        start_y = sy + (sd - box_d) / 2.0

        for k in range(n_boxes):
            ix = start_x + k * (BOX_W + BOX_GAP)
            b  = pv.Box(bounds=(ix, ix+BOX_W,
                                start_y, start_y+box_d,
                                sh, sh+BOX_HGT))
            pl.add_mesh(b, color=hex_rgb('#D4622A'), show_edges=True,
                        edge_color=hex_rgb('#FF9060'), line_width=0.3,
                        reset_camera=False)

    for (dx, dy) in DOCKS:
        d = pv.Box(bounds=(dx, dx+1.1, dy-0.8, dy+0.8, 0, 0.62))
        pl.add_mesh(d, color=hex_rgb('#3A2810'), show_edges=True,
                    edge_color=hex_rgb('#E8A018'), line_width=1.0,
                    reset_camera=False)
        door = pv.Box(bounds=(dx+0.2, dx+0.9, dy-0.6, dy+0.6, 0, 0.55))
        pl.add_mesh(door, color=hex_rgb('#1A0E04'),
                    show_edges=False, reset_camera=False)

    for (px, py) in STATIONS:
        s = pv.Box(bounds=(px, px+1.2, py-0.8, py+0.8, 0, 0.80))
        pl.add_mesh(s, color=hex_rgb('#182E22'), show_edges=True,
                    edge_color=hex_rgb('#48B870'), line_width=1.2,
                    reset_camera=False)
        belt = pv.Box(bounds=(px+0.1, px+1.1, py-0.7, py+0.7, 0.80, 0.84))
        pl.add_mesh(belt, color=hex_rgb('#1C4A30'),
                    show_edges=False, reset_camera=False)

    for (ppx, ppy) in PALLETS:
        pallet = pv.Box(bounds=(ppx, ppx+0.9, ppy, ppy+0.7, 0, 0.16))
        pl.add_mesh(pallet, color=hex_rgb('#5A4010'), show_edges=True,
                    edge_color=hex_rgb('#8A6020'), line_width=0.5,
                    reset_camera=False)
        for pi in range(2):
            pb = pv.Box(bounds=(ppx+0.08, ppx+0.82,
                                ppy+0.08+pi*0.28, ppy+0.30+pi*0.28,
                                0.16, 0.40))
            pl.add_mesh(pb, color=hex_rgb('#D4622A'),
                        show_edges=False, reset_camera=False)

    pl.camera.position    = CAM_POS
    pl.camera.focal_point = CAM_FOCUS
    pl.camera.up          = CAM_UP
    pl.camera.view_angle  = 44

build_static()
print("Static scene built.")

try:
    FONT_SM = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    FONT_LG = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 22)
    FONT_TL = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 19)
    FONT_LB = ImageFont.truetype(
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 13)
except:
    FONT_SM=FONT_LG=FONT_TL=FONT_LB=ImageFont.load_default()

LEGEND_ITEMS = [
    (ROBOT_COL_HEX, 'AMR Robot (x3)'),
    (HUMAN_COL_HEX, 'Human Worker (x3)'),
    ('#E74C3C',     'Collision Event'),
    ('#3498DB',     'Safety Zone (ISO 10218)'),
    ('#D4622A',     'Inventory Box'),
    ('#48B870',     'Pick Station'),
    ('#E8A018',     'Loading Dock'),
    ('#5A4010',     'Staging Pallet'),
    ('#1C3A5A',     'Wall / Barrier'),
    ('#0D2540',     'Highway Aisle'),
    ('#3A6890',     'Aisle Junction'),
]

def draw_legend(draw, x, y):
    pad=8; rw=14; rh=14; lh=22
    bw=250; bh=len(LEGEND_ITEMS)*lh+pad*2+22
    draw.rectangle([x, y, x+bw, y+bh],
                   fill=(5,13,24,218), outline=(42,90,138), width=1)
    draw.text((x+pad, y+pad), 'LEGEND',
              fill=(138,180,212), font=FONT_LB)
    for k, (col, label) in enumerate(LEGEND_ITEMS):
        ry2 = y + pad + 22 + k*lh
        r,g,b = hex_rgb(col)
        draw.rectangle([x+pad, ry2, x+pad+rw, ry2+rh],
                       fill=(int(r*255), int(g*255), int(b*255), 230))
        draw.text((x+pad+rw+6, ry2), label,
                  fill=(170,212,245), font=FONT_LB)

def overlay(img_arr, frame, state):
    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img, 'RGBA')
    W, H = img.size
    t    = frame * DT

    draw.rectangle([0, 0, W, 54], fill=(6, 14, 26, 248))
    draw.text((W//2, 27),
        "Warehouse Coordination Demo  -  ORCA / RVO2  |  "
        "Mohammed Al-Khudhair  |  Habitat 3.0 (Puig et al., 2023)",
        fill=(138, 180, 212), font=FONT_TL, anchor='mm')

    draw.text((W//2, 76),
        "ORCA / RVO2 Multi-Agent Coordination",
        fill=(46, 204, 113), font=FONT_LG, anchor='mm')

    tot  = sum(state['completed'])
    rate = tot / max(t, 0.5) * 60.0
    lines = [
        f"Algorithm    :  ORCA / RVO2",
        f"Sim Time     :  {t:.1f}s",
        f"Active Robots:  {N_ROBOTS}",
        f"Human Workers:  {N_HUMANS}",
        f"Tasks Done   :  {tot}",
        f"Throughput   :  {rate:.1f} / min",
        f"Collisions   :  {state['collisions']}",
        f"Wall Events  :  {state['wall_hits']}",
        f"Safety Evts  :  {state['safety_evts']}",
    ]
    bw = 340; lh = 22
    bx1 = W - bw - 14; by1 = H - len(lines)*lh - 32
    draw.rectangle([bx1, by1, bx1+bw, by1+len(lines)*lh+22],
                   fill=(5,13,24,220), outline=(46,204,113), width=2)
    for k, line in enumerate(lines):
        col = (46,204,113) if k == 0 else (170,212,245)
        draw.text((bx1+10, by1+11+k*lh), line, fill=col, font=FONT_SM)

    draw_legend(draw, 14, 95)

    prog = frame / FRAMES
    bary = H - 14
    draw.line([14, bary, W-14, bary], fill=(21, 42, 64), width=7)
    draw.line([14, bary, int(14 + (W-28)*prog), bary],
              fill=(58, 128, 200), width=7)

    return np.array(img)

def render_agents(state):
    for i in range(N_ROBOTS):
        rx, ry = state['robot_pos'][i]
        pl.add_mesh(tmesh(ROBOT_BODY, rx, ry), color=ROBOT_RGB,
                    show_edges=False, reset_camera=False, name=f'rb{i}')
        pl.add_mesh(tmesh(ROBOT_CAP, rx, ry), color='white',
                    opacity=0.75, show_edges=False, reset_camera=False,
                    name=f'rc{i}')
        pl.add_mesh(tmesh(ZONE_RING, rx, ry), color=ROBOT_RGB,
                    opacity=0.50, show_edges=False, reset_camera=False,
                    name=f'rz{i}')
        vel = state['robot_vel'][i]
        spd = np.linalg.norm(vel)
        if spd > 0.06:
            vn = vel / spd
            arr = pv.Arrow(start=(rx, ry, 0.72),
                           direction=(vn[0], vn[1], 0), scale=0.60)
            pl.add_mesh(arr, color='white', show_edges=False,
                        reset_camera=False, name=f'ra{i}')
        else:
            pl.add_mesh(pv.Sphere(center=(-99,-99,0), radius=0.01),
                        color='white', opacity=0.0, show_edges=False,
                        reset_camera=False, name=f'ra{i}')
        lm = pv.PolyData(np.array([[rx, ry, 1.18]]))
        lm['labels'] = [ROBOT_NAMES[i]]
        pl.add_point_labels(lm, 'labels', font_size=11,
                            font_family='courier',
                            text_color=ROBOT_COL_HEX,
                            point_color='black', point_size=0,
                            always_visible=True, reset_camera=False,
                            name=f'rl{i}', shape_opacity=0)

    for h in range(N_HUMANS):
        hx, hy = state['human_pos'][h]
        pl.add_mesh(tmesh(HUMAN_BODY, hx, hy), color=HUMAN_RGB,
                    show_edges=False, reset_camera=False, name=f'hb{h}')
        pl.add_mesh(tmesh(HUMAN_HEAD, hx, hy), color=HUMAN_RGB,
                    show_edges=False, reset_camera=False, name=f'hd{h}')
        lm = pv.PolyData(np.array([[hx, hy, 1.32]]))
        lm['labels'] = [HUMAN_NAMES[h]]
        pl.add_point_labels(lm, 'labels', font_size=11,
                            font_family='courier',
                            text_color=HUMAN_COL_HEX,
                            point_color='black', point_size=0,
                            always_visible=True, reset_camera=False,
                            name=f'hl{h}', shape_opacity=0)

    ap = list(state['human_pos']) + list(state['robot_pos'])
    n  = len(ap); k = 0
    for i in range(n):
        for j in range(i+1, n):
            if np.linalg.norm(ap[i]-ap[j]) < COLLISION_R and k < N_ROBOTS:
                mid = (ap[i] + ap[j]) / 2
                pl.add_mesh(tmesh(COL_SPHERE, mid[0], mid[1]),
                            color='red', opacity=0.58,
                            show_edges=False, reset_camera=False,
                            name=f'col{k}')
                k += 1
    for kk in range(k, N_ROBOTS):
        pl.add_mesh(pv.Sphere(center=(-99,-99,0), radius=0.01),
                    color='red', opacity=0.0,
                    show_edges=False, reset_camera=False,
                    name=f'col{kk}')

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps)...")
print(f"Output -> {OUTPUT_FILE}\n")

render_agents(O)
writer = iio.get_writer(OUTPUT_FILE, fps=FPS, macro_block_size=None)

for frame in range(FRAMES):
    step_orca(O, o_sim)
    render_agents(O)

    raw = pl.screenshot(return_img=True)
    if raw.shape[2] == 4:
        raw = raw[:,:,:3]
    final = overlay(raw, frame, O)
    writer.append_data(np.array(final)[:,:,:3])

    if frame % (FPS * 5) == 0:
        pct = int(frame / FRAMES * 100)
        print(f"  {pct:3d}%  t={frame*DT:.0f}s  "
              f"tasks={sum(O['completed'])}  "
              f"col={O['collisions']}  "
              f"wall={O['wall_hits']}")

writer.close()

print(f"\nComplete -> {OUTPUT_FILE}")
print(f"Tasks     : {sum(O['completed'])}")
print(f"Collisions: {O['collisions']}")
print(f"Wall hits : {O['wall_hits']}")
print(f"Safety    : {O['safety_evts']}")
