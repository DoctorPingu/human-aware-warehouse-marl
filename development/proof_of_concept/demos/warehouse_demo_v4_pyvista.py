"""
Warehouse Demo v4 PyVista - 3D Side-by-Side Comparison
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
Left: Simple Repulsion | Right: ORCA/RVO2 | 3 Robots, 1 Human
"""

import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pyvista as pv
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
import rvo2

print("Warehouse Demo v4 PyVista - Initialising...")

OUTPUT_DIR  = "/mnt/e/Honours Project/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v4_pyvista.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 24
DURATION = 35
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

WW = 22.0
WH = 14.0

N_ROBOTS    = 3
N_HUMANS    = 1
ROBOT_R     = 0.28
HUMAN_R     = 0.22
SAFETY_R    = 0.85
COLLISION_R = 0.46
ROBOT_SPD   = 1.8
HUMAN_SPD   = 0.75
ROBOT_STEP  = ROBOT_SPD * DT
HUMAN_STEP  = HUMAN_SPD * DT

def hex_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16)/255.0 for i in (0,2,4))

ROBOT_HEX = ['#3498DB','#E74C3C','#F39C12']
HUMAN_HEX = '#FF6B35'
ROBOT_RGB = [hex_rgb(c) for c in ROBOT_HEX]
HUMAN_RGB = hex_rgb(HUMAN_HEX)

SHELVES = []
for cx in [4.0, 6.5, 9.5, 11.5]:
    for ry in [1.0, 2.2, 3.4, 4.6, 8.3, 9.5, 10.7, 11.9]:
        SHELVES.append((cx, ry, 1.80, 0.72))
for cx in [14.5, 16.5]:
    for ry in [1.0, 2.0, 3.0, 4.0, 5.0, 8.3, 9.3, 10.3, 11.3, 12.3]:
        SHELVES.append((cx, ry, 1.55, 0.62))

DOCKS    = [(0.3,2.2),(0.3,7.0),(0.3,11.8)]
STATIONS = [(20.2,3.0),(20.2,7.0),(20.2,11.0)]

def mc(pts): return pts + [pts[0]]

ROBOT_STARTS = [(5.0,2.0),(10.0,2.0),(15.0,2.0)]

ROBOT_TASKS = [
    mc([(5.0,2.0),(5.0,5.0),(20.2,3.0)]),
    mc([(10.0,2.0),(10.0,5.0),(20.2,7.0)]),
    mc([(15.0,2.0),(15.0,5.5),(20.2,11.0)]),
]

HUMAN_WPS = [
    mc([(9.5,1.0),(9.5,5.5),(9.5,8.5),(9.5,13.0),
        (4.5,13.0),(4.5,8.5),(4.5,5.5),(4.5,1.0)]),
]

def make_state():
    return dict(
        robot_pos      = [np.array(p,dtype=float) for p in ROBOT_STARTS],
        robot_vel      = [np.zeros(2) for _ in range(N_ROBOTS)],
        robot_task_idx = [0]*N_ROBOTS,
        robot_state    = ['move']*N_ROBOTS,
        robot_wait     = [0]*N_ROBOTS,
        human_pos      = [np.array(HUMAN_WPS[h][0],dtype=float) for h in range(N_HUMANS)],
        human_wp_idx   = [1]*N_HUMANS,
        human_phase    = [0.0]*N_HUMANS,
        completed      = [0]*N_ROBOTS,
        collisions     = 0,
        safety_evts    = 0,
    )

S = make_state()
O = make_state()

o_sim = rvo2.PyRVOSimulator(DT,4.5,10,8.0,2.0,ROBOT_R,ROBOT_SPD)
for i in range(N_ROBOTS):
    o_sim.addAgent(tuple(O['robot_pos'][i]),4.5,10,8.0,2.0,ROBOT_R,ROBOT_SPD,(0.0,0.0))
for h in range(N_HUMANS):
    o_sim.addAgent(tuple(O['human_pos'][h]),4.5,10,5.0,2.0,HUMAN_R,HUMAN_SPD,(0.0,0.0))

def tally(state):
    ap = list(state['human_pos'])+list(state['robot_pos'])
    n = len(ap)
    for i in range(n):
        for j in range(i+1,n):
            d = np.linalg.norm(ap[i]-ap[j])
            if d < COLLISION_R:   state['collisions']  += 1
            elif d < SAFETY_R:    state['safety_evts'] += 1

def repulse(pos, others):
    f = np.zeros(2)
    for o in others:
        d=pos-o; dist=np.linalg.norm(d)
        if 0<dist<SAFETY_R*1.7:
            f += (d/dist)*(SAFETY_R*1.7-dist)*1.4
    return f

def step_simple(state):
    for h in range(N_HUMANS):
        tgt=np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        diff=tgt-state['human_pos'][h]; dist=np.linalg.norm(diff)
        if dist<HUMAN_STEP*1.5:
            state['human_pos'][h][:]=tgt
            state['human_wp_idx'][h]=(state['human_wp_idx'][h]+1)%len(HUMAN_WPS[h])
        else:
            state['human_pos'][h]+=(diff/dist)*HUMAN_STEP
        state['human_phase'][h]+=0.20
        state['human_pos'][h]=np.clip(state['human_pos'][h],0.15,[WW-0.15,WH-0.15])
    snap=[p.copy() for p in state['human_pos']]+[p.copy() for p in state['robot_pos']]
    for i in range(N_ROBOTS):
        if state['robot_state'][i]=='wait':
            state['robot_wait'][i]-=1
            if state['robot_wait'][i]<=0: state['robot_state'][i]='move'
            state['robot_vel'][i]=np.zeros(2); continue
        tgt=np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
        diff=tgt-state['robot_pos'][i]; dist=np.linalg.norm(diff)
        if dist<ROBOT_STEP*2:
            state['robot_pos'][i][:]=tgt
            state['robot_task_idx'][i]=(state['robot_task_idx'][i]+1)%len(ROBOT_TASKS[i])
            state['completed'][i]+=1; state['robot_state'][i]='wait'
            state['robot_wait'][i]=int(FPS*0.55); state['robot_vel'][i]=np.zeros(2)
        else:
            others=[snap[j] for j in range(len(snap)) if j!=i+N_HUMANS]
            vel=(diff/dist)*ROBOT_SPD+repulse(state['robot_pos'][i],others)
            spd=np.linalg.norm(vel)
            if spd>ROBOT_SPD*1.6: vel=vel/spd*ROBOT_SPD*1.6
            state['robot_vel'][i]=vel
            state['robot_pos'][i]+=vel*DT
            state['robot_pos'][i]=np.clip(state['robot_pos'][i],0.15,[WW-0.15,WH-0.15])
    tally(state)

def step_orca(state,sim):
    for i in range(N_ROBOTS):
        if state['robot_state'][i]=='wait':
            sim.setAgentPrefVelocity(i,(0.0,0.0))
        else:
            tgt=np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            diff=tgt-state['robot_pos'][i]; dist=np.linalg.norm(diff)
            if dist<0.25: sim.setAgentPrefVelocity(i,(0.0,0.0))
            else:
                spd=min(ROBOT_SPD,dist*FPS*0.55)
                sim.setAgentPrefVelocity(i,tuple((diff/dist)*spd))
    for h in range(N_HUMANS):
        tgt=np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        diff=tgt-state['human_pos'][h]; dist=np.linalg.norm(diff)
        if dist<0.12: sim.setAgentPrefVelocity(N_ROBOTS+h,(0.0,0.0))
        else: sim.setAgentPrefVelocity(N_ROBOTS+h,tuple((diff/dist)*HUMAN_SPD))
    sim.doStep()
    for i in range(N_ROBOTS):
        pos=np.clip(np.array(sim.getAgentPosition(i)),0.15,[WW-0.15,WH-0.15])
        vel=np.array(sim.getAgentVelocity(i))
        state['robot_pos'][i]=pos; state['robot_vel'][i]=vel
        sim.setAgentPosition(i,tuple(pos))
        if state['robot_state'][i]=='wait':
            state['robot_wait'][i]-=1
            if state['robot_wait'][i]<=0: state['robot_state'][i]='move'
        else:
            tgt=np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            if np.linalg.norm(tgt-pos)<0.38:
                state['robot_task_idx'][i]=(state['robot_task_idx'][i]+1)%len(ROBOT_TASKS[i])
                state['completed'][i]+=1; state['robot_state'][i]='wait'
                state['robot_wait'][i]=int(FPS*0.55)
    for h in range(N_HUMANS):
        pos=np.clip(np.array(sim.getAgentPosition(N_ROBOTS+h)),0.15,[WW-0.15,WH-0.15])
        state['human_pos'][h]=pos; state['human_phase'][h]+=0.20
        sim.setAgentPosition(N_ROBOTS+h,tuple(pos))
        tgt=np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        if np.linalg.norm(tgt-pos)<HUMAN_STEP*3.5:
            state['human_wp_idx'][h]=(state['human_wp_idx'][h]+1)%len(HUMAN_WPS[h])
    tally(state)

pl = pv.Plotter(shape=(1,2), off_screen=True,
                window_size=[2400,960], border=False)

CAM_POS   = (WW/2, WH/2 - 13, 21)
CAM_FOCUS = (WW/2, WH/2 + 0.5, 0)
CAM_UP    = (0, 1, 0)

def build_static(sub):
    pl.subplot(0, sub)
    pl.set_background('#07101C')

    pl.add_mesh(pv.Box(bounds=(0,WW,0,WH,-0.15,0)),
                color='#0A1828', show_edges=False, reset_camera=False)

    for gx in range(int(WW)):
        for gy in range(int(WH)):
            if (gx+gy)%2==0:
                pl.add_mesh(pv.Box(bounds=(gx,gx+1,gy,gy+1,-0.12,0)),
                            color='#0C1E30', show_edges=False, reset_camera=False)

    for (zx,zy,zw,zh,zc) in [
        (0,0,3.5,WH,'#091828'),(3.5,0,5.5,WH,'#07162A'),
        (9.0,0,5.0,WH,'#081828'),(14.0,0,5.0,WH,'#071626'),
        (19.0,0,3.0,WH,'#081828')]:
        pl.add_mesh(pv.Box(bounds=(zx,zx+zw,zy,zy+zh,-0.10,0.02)),
                    color=hex_rgb(zc), opacity=0.7, show_edges=False, reset_camera=False)

    pl.add_mesh(pv.Box(bounds=(0,WW,6.0,8.0,-0.05,0.02)),
                color='#0C2238', show_edges=False, reset_camera=False)

    for xd in [3.5,9.0,14.0,19.0]:
        pl.add_mesh(pv.Box(bounds=(xd-0.04,xd+0.04,0,WH,0,1.4)),
                    color='#1A3A55', opacity=0.35, show_edges=False, reset_camera=False)

    for (sx,sy,sw,sh) in SHELVES:
        pl.add_mesh(pv.Box(bounds=(sx,sx+sw,sy,sy+sh,0,0.88)),
                    color='#122438', show_edges=True, edge_color='#1C3C6A',
                    line_width=0.6, reset_camera=False)
        rng = np.random.RandomState(int(sx*17+sy*11))
        for ix in np.arange(sx+0.15, sx+sw-0.1, 0.30):
            ih = 0.12+rng.uniform(0,0.10)
            pl.add_mesh(
                pv.Box(bounds=(ix-0.09,ix+0.09,sy+0.09,sy+sh-0.09,0.88,0.88+ih)),
                color='#D4622A', show_edges=False, reset_camera=False)

    for (dx,dy) in DOCKS:
        pl.add_mesh(pv.Box(bounds=(dx,dx+0.9,dy-0.5,dy+0.5,0,0.55)),
                    color='#3A2810', show_edges=True, edge_color='#E8A018',
                    line_width=1.0, reset_camera=False)

    for (px,py) in STATIONS:
        pl.add_mesh(pv.Box(bounds=(px,px+1.0,py-0.65,py+0.65,0,0.72)),
                    color='#182E22', show_edges=True, edge_color='#48B870',
                    line_width=1.2, reset_camera=False)

    pl.camera.position    = CAM_POS
    pl.camera.focal_point = CAM_FOCUS
    pl.camera.up          = CAM_UP
    pl.camera.view_angle  = 44

build_static(0)
build_static(1)

def update_agents(state, sub, pfx):
    pl.subplot(0, sub)
    rp  = state['robot_pos']
    rv  = state['robot_vel']
    hp  = state['human_pos']
    hph = state['human_phase']

    ZR=0.32; RH=0.64
    ZH=0.44; HH=0.88

    for i in range(N_ROBOTS):
        rx,ry = rp[i]
        pl.add_mesh(
            pv.Cylinder(center=(rx,ry,ZR),direction=(0,0,1),radius=ROBOT_R,height=RH,resolution=18),
            color=ROBOT_RGB[i], show_edges=False, reset_camera=False,
            name=f'{pfx}rb{i}')
        pl.add_mesh(
            pv.Sphere(center=(rx,ry,ZR+RH/2+0.10),radius=ROBOT_R*0.42),
            color='white', opacity=0.7, show_edges=False, reset_camera=False,
            name=f'{pfx}rt{i}')
        pl.add_mesh(
            pv.Cylinder(center=(rx,ry,0.04),direction=(0,0,1),radius=SAFETY_R,height=0.08,resolution=32),
            color=ROBOT_RGB[i], opacity=0.10, show_edges=False, reset_camera=False,
            name=f'{pfx}rz{i}')
        spd=np.linalg.norm(rv[i])
        if spd>0.08:
            vn=rv[i]/spd
            pl.add_mesh(
                pv.Arrow(start=(rx,ry,ZR+RH/2+0.12),direction=(vn[0],vn[1],0),scale=0.72),
                color='white', show_edges=False, reset_camera=False,
                name=f'{pfx}ra{i}')
        else:
            pl.add_mesh(pv.Sphere(center=(0,-99,0),radius=0.01),color='white',
                        opacity=0.0,show_edges=False,reset_camera=False,name=f'{pfx}ra{i}')

    for h in range(N_HUMANS):
        hx,hy=hp[h]
        pl.add_mesh(
            pv.Cylinder(center=(hx,hy,ZH),direction=(0,0,1),radius=HUMAN_R,height=HH,resolution=16),
            color=HUMAN_RGB, show_edges=False, reset_camera=False,
            name=f'{pfx}hb{h}')
        pl.add_mesh(
            pv.Sphere(center=(hx,hy,ZH+HH/2+0.22),radius=HUMAN_R*0.72),
            color=HUMAN_RGB, show_edges=False, reset_camera=False,
            name=f'{pfx}hd{h}')
        pl.add_mesh(
            pv.Cylinder(center=(hx,hy,0.04),direction=(0,0,1),radius=SAFETY_R*0.80,height=0.08,resolution=32),
            color=HUMAN_RGB, opacity=0.10, show_edges=False, reset_camera=False,
            name=f'{pfx}hz{h}')

    ap=list(hp)+list(rp)
    n=len(ap); k=0
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(ap[i]-ap[j])<COLLISION_R and k<N_ROBOTS:
                mid=(ap[i]+ap[j])/2
                pl.add_mesh(pv.Sphere(center=(mid[0],mid[1],0.55),radius=0.36),
                            color='red',opacity=0.55,show_edges=False,reset_camera=False,
                            name=f'{pfx}col{k}')
                k+=1
    for kk in range(k,N_ROBOTS):
        pl.add_mesh(pv.Sphere(center=(0,-99,0),radius=0.01),color='red',
                    opacity=0.0,show_edges=False,reset_camera=False,name=f'{pfx}col{kk}')

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 17)
    FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 21)
    FONT_TL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf", 19)
except:
    FONT_SM = FONT_MD = FONT_LG = FONT_TL = ImageFont.load_default()

def overlay(img_arr, frame):
    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img, 'RGBA')
    W, H = img.size
    t    = frame * DT
    half = W // 2

    draw.rectangle([0,0,W,52], fill=(7,16,28,245))
    draw.text((W//2, 26),
        "Warehouse Coordination — Baseline Comparison  |  "
        "Mohammed Al-Khudhair  |  Habitat 3.0 (Puig et al., 2023)",
        fill=(138,180,212), font=FONT_TL, anchor='mm')

    draw.text((half//2, 72), "Baseline: Simple Repulsion",
              fill=(231,76,60), font=FONT_LG, anchor='mm')
    draw.text((half + half//2, 72), "ORCA / RVO2 Coordination",
              fill=(46,204,113), font=FONT_LG, anchor='mm')

    def stats_box(ox, state, border_col):
        tot  = sum(state['completed'])
        rate = tot / max(t, 0.5) * 60.0
        lines = [
            f"Time        :  {t:.1f}s",
            f"Tasks       :  {tot}",
            f"Throughput  :  {rate:.1f} / min",
            f"Collisions  :  {state['collisions']}",
            f"Safety Evts :  {state['safety_evts']}",
        ]
        bw = 310; lh = 22
        bx1=ox; by1=H-135; bx2=ox+bw; by2=by1+len(lines)*lh+18
        draw.rectangle([bx1,by1,bx2,by2], fill=(5,13,24,215), outline=border_col, width=1)
        for k,line in enumerate(lines):
            draw.text((bx1+10, by1+9+k*lh), line, fill=(170,212,245), font=FONT_SM)

    stats_box(12,  S, (231,76,60))
    stats_box(half+12, O, (46,204,113))

    prog = frame / FRAMES
    bary = H - 14
    draw.line([12, bary, W-12, bary], fill=(21,42,64), width=6)
    draw.line([12, bary, int(12+(W-24)*prog), bary], fill=(58,128,200), width=6)

    return np.array(img)

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps)...")
print(f"Output -> {OUTPUT_FILE}\n")

update_agents(S, 0, 's')
update_agents(O, 1, 'o')

writer = iio.get_writer(OUTPUT_FILE, fps=FPS, macro_block_size=None)

for frame in range(FRAMES):
    step_simple(S)
    step_orca(O, o_sim)

    update_agents(S, 0, 's')
    update_agents(O, 1, 'o')

    raw = pl.screenshot(return_img=True)
    if raw.shape[2] == 4:
        raw = raw[:,:,:3]

    final = overlay(raw, frame)
    writer.append_data(np.array(final)[:,:,:3])

    if frame % (FPS*5) == 0:
        pct = int(frame/FRAMES*100)
        print(f"  {pct:3d}%  t={frame*DT:.0f}s  "
              f"simple={sum(S['completed'])} tasks {S['collisions']} col | "
              f"orca={sum(O['completed'])} tasks {O['collisions']} col")

writer.close()

print(f"\nComplete -> {OUTPUT_FILE}")
print(f"Simple :  {sum(S['completed'])} tasks | {S['collisions']} col | {S['safety_evts']} safety evts")
print(f"ORCA   :  {sum(O['completed'])} tasks | {O['collisions']} col | {O['safety_evts']} safety evts")
