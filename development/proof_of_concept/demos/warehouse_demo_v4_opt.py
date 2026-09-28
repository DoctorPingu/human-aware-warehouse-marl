"""
Warehouse Demo v4 PyVista Optimised
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
Left: Simple Repulsion | Right: ORCA/RVO2 | 3 Robots 1 Human
Optimised: actor transforms instead of mesh rebuilds, single floor plane
"""

import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pyvista as pv
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
import rvo2

print("Warehouse Demo v4 PyVista Optimised - Initialising...")

OUTPUT_DIR  = "/mnt/e/Honours Project/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v4_opt.mp4")
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
    return tuple(int(h[i:i+2],16)/255.0 for i in (0,2,4))

ROBOT_HEX = ['#3498DB','#E74C3C','#F39C12']
HUMAN_HEX = '#FF6B35'
ROBOT_RGB = [hex_rgb(c) for c in ROBOT_HEX]
HUMAN_RGB = hex_rgb(HUMAN_HEX)

SHELVES = []
for cx in [4.0,6.5,9.5,11.5]:
    for ry in [1.0,2.2,3.4,4.6,8.3,9.5,10.7,11.9]:
        SHELVES.append((cx,ry,1.80,0.72))
for cx in [14.5,16.5]:
    for ry in [1.0,2.0,3.0,4.0,5.0,8.3,9.3,10.3,11.3,12.3]:
        SHELVES.append((cx,ry,1.55,0.62))

DOCKS    = [(0.3,2.2),(0.3,7.0),(0.3,11.8)]
STATIONS = [(20.2,3.0),(20.2,7.0),(20.2,11.0)]

def mc(pts): return pts+[pts[0]]

ROBOT_STARTS = [(5.0,2.0),(10.0,2.0),(15.0,2.0)]
ROBOT_TASKS  = [
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
    ap=list(state['human_pos'])+list(state['robot_pos'])
    n=len(ap)
    for i in range(n):
        for j in range(i+1,n):
            d=np.linalg.norm(ap[i]-ap[j])
            if d<COLLISION_R:   state['collisions']  +=1
            elif d<SAFETY_R:    state['safety_evts'] +=1

def repulse(pos,others):
    f=np.zeros(2)
    for o in others:
        d=pos-o; dist=np.linalg.norm(d)
        if 0<dist<SAFETY_R*1.7: f+=(d/dist)*(SAFETY_R*1.7-dist)*1.4
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

# ── Pre-build reusable meshes ─────────────────────────────────────
ROBOT_MESH = pv.Cylinder(center=(0,0,0.32),direction=(0,0,1),
                          radius=ROBOT_R,height=0.64,resolution=14)
ROBOT_TOP  = pv.Sphere(center=(0,0,0.70),radius=ROBOT_R*0.38)
HUMAN_BODY = pv.Cylinder(center=(0,0,0.44),direction=(0,0,1),
                          radius=HUMAN_R,height=0.88,resolution=12)
HUMAN_HEAD = pv.Sphere(center=(0,0,1.02),radius=HUMAN_R*0.65)
ZONE_RING  = pv.Disc(center=(0,0,0.01),normal=(0,0,1),
                      inner=SAFETY_R-0.06,outer=SAFETY_R+0.06,r_res=1,c_res=28)
HUMAN_RING = pv.Disc(center=(0,0,0.01),normal=(0,0,1),
                      inner=SAFETY_R*0.75,outer=SAFETY_R*0.75+0.06,r_res=1,c_res=28)
COL_SPHERE = pv.Sphere(center=(0,0,0.45),radius=0.32)

def translate_mesh(mesh, x, y):
    m = mesh.copy()
    m.translate([x, y, 0], inplace=True)
    return m

# ── Build plotter with static scene ──────────────────────────────
pl = pv.Plotter(shape=(1,2), off_screen=True,
                window_size=[2400,900], border=False)

CAM_POS   = (WW/2, WH/2-13, 20)
CAM_FOCUS = (WW/2, WH/2+0.5, 0)
CAM_UP    = (0,1,0)

def build_static(sub):
    pl.subplot(0,sub)
    pl.set_background('#07101C')

    floor = pv.Plane(center=(WW/2,WH/2,-0.02),direction=(0,0,1),
                     i_size=WW,j_size=WH)
    pl.add_mesh(floor,color='#0A1828',show_edges=False,reset_camera=False)

    hw = pv.Box(bounds=(0,WW,6.0,8.0,-0.04,0.02))
    pl.add_mesh(hw,color='#0C2238',show_edges=False,reset_camera=False)

    for (zx,zw,zc) in [(0,3.5,'#091828'),(3.5,5.5,'#07162A'),
                        (9.0,5.0,'#081828'),(14.0,5.0,'#071626'),
                        (19.0,3.0,'#081828')]:
        zp = pv.Box(bounds=(zx,zx+zw,0,WH,-0.03,0.015))
        pl.add_mesh(zp,color=hex_rgb(zc),opacity=0.75,
                    show_edges=False,reset_camera=False)

    for xd in [3.5,9.0,14.0,19.0]:
        dv = pv.Box(bounds=(xd-0.05,xd+0.05,0,WH,0,1.2))
        pl.add_mesh(dv,color='#1A3A55',opacity=0.30,
                    show_edges=False,reset_camera=False)

    for (sx,sy,sw,sh) in SHELVES:
        shelf = pv.Box(bounds=(sx,sx+sw,sy,sy+sh,0,0.82))
        pl.add_mesh(shelf,color='#122438',show_edges=True,
                    edge_color='#1C3C6A',line_width=0.5,reset_camera=False)
        rng=np.random.RandomState(int(sx*17+sy*11))
        for ix in np.arange(sx+0.18,sx+sw-0.10,0.38):
            ih=0.14+rng.uniform(0,0.10)
            box=pv.Box(bounds=(ix-0.10,ix+0.10,sy+0.10,sy+sh-0.10,0.82,0.82+ih))
            pl.add_mesh(box,color='#D4622A',show_edges=False,reset_camera=False)

    for (dx,dy) in DOCKS:
        d=pv.Box(bounds=(dx,dx+0.9,dy-0.5,dy+0.5,0,0.55))
        pl.add_mesh(d,color='#3A2810',show_edges=True,
                    edge_color='#E8A018',line_width=1.0,reset_camera=False)

    for (px,py) in STATIONS:
        s=pv.Box(bounds=(px,px+1.0,py-0.65,py+0.65,0,0.72))
        pl.add_mesh(s,color='#182E22',show_edges=True,
                    edge_color='#48B870',line_width=1.2,reset_camera=False)

    pl.camera.position    = CAM_POS
    pl.camera.focal_point = CAM_FOCUS
    pl.camera.up          = CAM_UP
    pl.camera.view_angle  = 44

build_static(0)
build_static(1)
print("Static scene built.")

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",15)
    FONT_MD = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",17)
    FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",21)
    FONT_TL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",19)
except:
    FONT_SM=FONT_MD=FONT_LG=FONT_TL=ImageFont.load_default()

def overlay(img_arr, frame):
    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img,'RGBA')
    W,H  = img.size
    t    = frame*DT
    half = W//2

    draw.rectangle([0,0,W,52],fill=(7,16,28,245))
    draw.text((W//2,26),
        "Warehouse Coordination — Baseline Comparison  |  "
        "Mohammed Al-Khudhair  |  Habitat 3.0 (Puig et al., 2023)",
        fill=(138,180,212),font=FONT_TL,anchor='mm')

    draw.text((half//2,72),"Baseline: Simple Repulsion",
              fill=(231,76,60),font=FONT_LG,anchor='mm')
    draw.text((half+half//2,72),"ORCA / RVO2 Coordination",
              fill=(46,204,113),font=FONT_LG,anchor='mm')

    def stats_box(ox,state,bc):
        tot  = sum(state['completed'])
        rate = tot/max(t,0.5)*60.0
        lines=[f"Time        :  {t:.1f}s",
               f"Tasks       :  {tot}",
               f"Throughput  :  {rate:.1f} / min",
               f"Collisions  :  {state['collisions']}",
               f"Safety Evts :  {state['safety_evts']}"]
        bw=310; lh=22
        bx1=ox; by1=H-135; bx2=ox+bw; by2=by1+len(lines)*lh+18
        draw.rectangle([bx1,by1,bx2,by2],fill=(5,13,24,215),outline=bc,width=1)
        for k,line in enumerate(lines):
            draw.text((bx1+10,by1+9+k*lh),line,fill=(170,212,245),font=FONT_SM)

    stats_box(12,S,(231,76,60))
    stats_box(half+12,O,(46,204,113))

    prog=frame/FRAMES
    bary=H-14
    draw.line([12,bary,W-12,bary],fill=(21,42,64),width=6)
    draw.line([12,bary,int(12+(W-24)*prog),bary],fill=(58,128,200),width=6)

    return np.array(img)

def render_agents(state, sub, pfx):
    pl.subplot(0,sub)

    for i in range(N_ROBOTS):
        rx,ry = state['robot_pos'][i]
        pl.add_mesh(translate_mesh(ROBOT_MESH,rx,ry),
                    color=ROBOT_RGB[i],show_edges=False,reset_camera=False,
                    name=f'{pfx}rb{i}')
        pl.add_mesh(translate_mesh(ROBOT_TOP,rx,ry),
                    color='white',opacity=0.7,show_edges=False,reset_camera=False,
                    name=f'{pfx}rt{i}')
        pl.add_mesh(translate_mesh(ZONE_RING,rx,ry),
                    color=ROBOT_RGB[i],opacity=0.55,show_edges=False,reset_camera=False,
                    name=f'{pfx}rz{i}')
        vel=state['robot_vel'][i]
        spd=np.linalg.norm(vel)
        if spd>0.08:
            vn=vel/spd
            arr=pv.Arrow(start=(rx,ry,0.78),direction=(vn[0],vn[1],0),scale=0.65)
            pl.add_mesh(arr,color='white',show_edges=False,reset_camera=False,
                        name=f'{pfx}ra{i}')
        else:
            pl.add_mesh(pv.Sphere(center=(-99,-99,0),radius=0.01),
                        color='white',opacity=0.0,show_edges=False,reset_camera=False,
                        name=f'{pfx}ra{i}')

    for h in range(N_HUMANS):
        hx,hy=state['human_pos'][h]
        pl.add_mesh(translate_mesh(HUMAN_BODY,hx,hy),
                    color=HUMAN_RGB,show_edges=False,reset_camera=False,
                    name=f'{pfx}hb{h}')
        pl.add_mesh(translate_mesh(HUMAN_HEAD,hx,hy),
                    color=HUMAN_RGB,show_edges=False,reset_camera=False,
                    name=f'{pfx}hd{h}')
        pl.add_mesh(translate_mesh(HUMAN_RING,hx,hy),
                    color=HUMAN_RGB,opacity=0.55,show_edges=False,reset_camera=False,
                    name=f'{pfx}hz{h}')

    ap=list(state['human_pos'])+list(state['robot_pos'])
    n=len(ap); k=0
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(ap[i]-ap[j])<COLLISION_R and k<N_ROBOTS:
                mid=(ap[i]+ap[j])/2
                pl.add_mesh(translate_mesh(COL_SPHERE,mid[0],mid[1]),
                            color='red',opacity=0.55,show_edges=False,reset_camera=False,
                            name=f'{pfx}col{k}')
                k+=1
    for kk in range(k,N_ROBOTS):
        pl.add_mesh(pv.Sphere(center=(-99,-99,0),radius=0.01),
                    color='red',opacity=0.0,show_edges=False,reset_camera=False,
                    name=f'{pfx}col{kk}')

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps)...")
print(f"Output -> {OUTPUT_FILE}\n")

writer = iio.get_writer(OUTPUT_FILE,fps=FPS,macro_block_size=None)

render_agents(S,0,'s')
render_agents(O,1,'o')

for frame in range(FRAMES):
    step_simple(S)
    step_orca(O,o_sim)
    render_agents(S,0,'s')
    render_agents(O,1,'o')

    raw=pl.screenshot(return_img=True)
    if raw.shape[2]==4: raw=raw[:,:,:3]
    final=overlay(raw,frame)
    writer.append_data(np.array(final)[:,:,:3])

    if frame%(FPS*5)==0:
        pct=int(frame/FRAMES*100)
        print(f"  {pct:3d}%  t={frame*DT:.0f}s  "
              f"simple={sum(S['completed'])} tasks {S['collisions']} col | "
              f"orca={sum(O['completed'])} tasks {O['collisions']} col")

writer.close()
print(f"\nComplete -> {OUTPUT_FILE}")
print(f"Simple :  {sum(S['completed'])} tasks | {S['collisions']} col | {S['safety_evts']} safety evts")
print(f"ORCA   :  {sum(O['completed'])} tasks | {O['collisions']} col | {O['safety_evts']} safety evts")
