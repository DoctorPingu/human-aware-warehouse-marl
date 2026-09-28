"""
Warehouse Demo v5 - Final Demo
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
Maze layout | 3 robots | 3 humans | Wall boundaries | Labels
TEST VERSION: 5 seconds
"""

import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pyvista as pv
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
import rvo2

print("Warehouse Demo v5 - Initialising...")

OUTPUT_DIR  = "/mnt/e/Honours Project/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v5_test.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 24
DURATION = 5
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

WW = 26.0
WH = 18.0
WALL_H   = 1.8
SHELF_H  = 1.0
AISLE_W  = 2.2

N_ROBOTS    = 3
N_HUMANS    = 3
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

ROBOT_HEX  = ['#3498DB','#E74C3C','#F39C12']
HUMAN_HEX  = ['#FF6B35','#FFD93D','#FF4E88']
ROBOT_RGB  = [hex_rgb(c) for c in ROBOT_HEX]
HUMAN_RGB  = [hex_rgb(c) for c in HUMAN_HEX]
ROBOT_NAME = ['AMR-1','AMR-2','AMR-3']
HUMAN_NAME = ['Worker-1','Worker-2','Worker-3']

C_WALL     = hex_rgb('#1C3A5A')
C_WALL_E   = hex_rgb('#2A5A8A')
C_FLOOR    = hex_rgb('#0A1828')
C_SHELF    = hex_rgb('#122438')
C_SHELF_E  = hex_rgb('#1C3C6A')
C_BOX_A    = hex_rgb('#D4622A')
C_BOX_B    = hex_rgb('#E8852A')
C_BOX_C    = hex_rgb('#B84E20')
C_DOCK_F   = hex_rgb('#3A2810')
C_DOCK_E   = hex_rgb('#E8A018')
C_STAT_F   = hex_rgb('#182E22')
C_STAT_E   = hex_rgb('#48B870')
C_PALLET   = hex_rgb('#5A4010')
C_PILLAR   = hex_rgb('#1A3050')
C_PILLAR_E = hex_rgb('#2A5080')
C_SIGN     = hex_rgb('#0E2840')
C_SIGN_E   = hex_rgb('#4A90D9')

# ── Maze layout: shelf blocks ─────────────────────────────────────
# The warehouse has 5 zones separated by cross-aisles
# Narrow aisles (1.8 wide) force robots to negotiate at junctions
# Layout uses T-junctions and dead-end side aisles

SHELVES = []

# Zone A: Receiving/Cold (x:2-7) - double-deep shelves
for rx in [2.0, 4.5]:
    for ry in [1.0, 2.8, 4.6]:
        SHELVES.append((rx, ry, 2.0, 0.8, 1.2, 'cold'))
    for ry in [10.5, 12.3, 14.1]:
        SHELVES.append((rx, ry, 2.0, 0.8, 1.2, 'cold'))

# Zone B: Bulk (x:8-13) - long aisle shelves
for rx in [8.0, 10.5]:
    for ry in [1.0, 2.5, 4.0, 5.5]:
        SHELVES.append((rx, ry, 2.2, 0.75, 1.0, 'bulk'))
    for ry in [10.5, 12.0, 13.5, 15.0]:
        SHELVES.append((rx, ry, 2.2, 0.75, 1.0, 'bulk'))

# Zone C: Pick (x:14-19) - short dense shelves (maze-like)
for rx in [14.0, 15.8]:
    for ry in [1.0, 2.0, 3.0, 4.0, 5.0]:
        SHELVES.append((rx, ry, 1.6, 0.65, 0.75, 'pick'))
    for ry in [10.8, 11.8, 12.8, 13.8, 14.8]:
        SHELVES.append((rx, ry, 1.6, 0.65, 0.75, 'pick'))

# Dead-end side aisles (creates maze effect)
SHELVES.append((17.8, 1.0, 1.6, 0.65, 0.75, 'pick'))
SHELVES.append((17.8, 2.5, 1.6, 0.65, 0.75, 'pick'))
SHELVES.append((17.8, 13.5, 1.6, 0.65, 0.75, 'pick'))
SHELVES.append((17.8, 15.0, 1.6, 0.65, 0.75, 'pick'))

# ── Internal walls (aisle dividers that create maze corridors) ─────
# Each wall: (x, y, length, axis, height)  axis='x' or 'y'
WALLS = [
    # Perimeter
    (0,   0,   WW,   'x', WALL_H),
    (0,   WH,  WW,   'x', WALL_H),
    (0,   0,   WH,   'y', WALL_H),
    (WW,  0,   WH,   'y', WALL_H),
    # Zone dividers (partial walls with gaps for aisles)
    (7.2, 0,   7.5,  'y', WALL_H * 0.6),
    (7.2, 9.8, 8.2,  'y', WALL_H * 0.6),
    (13.2,0,   7.5,  'y', WALL_H * 0.6),
    (13.2,9.8, 8.2,  'y', WALL_H * 0.6),
    (19.5,0,   7.5,  'y', WALL_H * 0.6),
    (19.5,9.8, 8.2,  'y', WALL_H * 0.6),
    # Cross-aisle walls (horizontal, partial)
    (0,   7.5, 6.5,  'x', WALL_H * 0.4),
    (8.0, 7.5, 4.5,  'x', WALL_H * 0.4),
    (14.0,7.5, 5.0,  'x', WALL_H * 0.4),
]

# ── Pillars at junctions ──────────────────────────────────────────
PILLARS = [
    (7.2, 7.5),(7.2, 0),(7.2, WH),
    (13.2,7.5),(13.2,0),(13.2,WH),
    (19.5,7.5),(19.5,0),(19.5,WH),
]

# ── Dock, stations, pallets ───────────────────────────────────────
DOCKS    = [(0.4, 3.2),(0.4, 9.0),(0.4, 14.8)]
STATIONS = [(23.5,3.5),(23.5,9.0),(23.5,14.5)]
PALLETS  = [(1.2,6.0),(1.2,7.0),(2.0,6.0),(1.2,10.8),(1.2,11.8),(2.0,10.8)]
SIGNS    = [(7.2,7.5,0),(13.2,7.5,0),(19.5,7.5,0)]

# ── Robot/Human setup ─────────────────────────────────────────────
def mc(pts): return pts+[pts[0]]

ROBOT_STARTS = [(3.0,3.0),(9.5,3.0),(15.5,3.0)]
ROBOT_TASKS  = [
    mc([(3.0,3.0),(3.0,6.5),(7.2,9.0),(13.2,9.0),(23.5,3.5)]),
    mc([(9.5,3.0),(9.5,6.5),(13.2,9.0),(19.5,9.0),(23.5,9.0)]),
    mc([(15.5,3.0),(15.5,6.5),(19.5,9.0),(23.5,14.5),(15.5,15.0)]),
]

HUMAN_WPS = [
    mc([(1.5,1.5),(25.0,1.5),(25.0,17.0),(1.5,17.0)]),
    mc([(14.0,3.0),(18.5,3.0),(18.5,6.5),(14.0,6.5),
        (14.0,11.0),(18.5,11.0),(18.5,16.0),(14.0,16.0)]),
    mc([(7.5,2.0),(7.5,7.0),(7.5,10.0),(7.5,17.0),
        (3.5,17.0),(3.5,10.0),(3.5,7.0),(3.5,2.0)]),
]
HUMAN_SPEEDS = [0.028,0.032,0.030]

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
    )

S = make_state()
O = make_state()

o_sim = rvo2.PyRVOSimulator(DT,4.5,12,8.0,2.0,ROBOT_R,ROBOT_SPD)
for i in range(N_ROBOTS):
    o_sim.addAgent(tuple(O['robot_pos'][i]),4.5,12,8.0,2.0,ROBOT_R,ROBOT_SPD,(0.0,0.0))
for h in range(N_HUMANS):
    o_sim.addAgent(tuple(O['human_pos'][h]),4.5,12,5.0,2.0,HUMAN_R,HUMAN_SPD,(0.0,0.0))

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
        if dist<HUMAN_SPEEDS[h]*1.5:
            state['human_pos'][h][:]=tgt
            state['human_wp_idx'][h]=(state['human_wp_idx'][h]+1)%len(HUMAN_WPS[h])
        else:
            state['human_pos'][h]+=(diff/dist)*HUMAN_SPEEDS[h]
        state['human_phase'][h]+=0.20
        state['human_pos'][h]=np.clip(state['human_pos'][h],0.3,[WW-0.3,WH-0.3])
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
            state['robot_pos'][i]=np.clip(state['robot_pos'][i],0.3,[WW-0.3,WH-0.3])
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
        else: sim.setAgentPrefVelocity(N_ROBOTS+h,tuple((diff/dist)*HUMAN_SPEEDS[h]))
    sim.doStep()
    for i in range(N_ROBOTS):
        pos=np.clip(np.array(sim.getAgentPosition(i)),0.3,[WW-0.3,WH-0.3])
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
        pos=np.clip(np.array(sim.getAgentPosition(N_ROBOTS+h)),0.3,[WW-0.3,WH-0.3])
        state['human_pos'][h]=pos; state['human_phase'][h]+=0.20
        sim.setAgentPosition(N_ROBOTS+h,tuple(pos))
        tgt=np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        if np.linalg.norm(tgt-pos)<HUMAN_STEP*3.5:
            state['human_wp_idx'][h]=(state['human_wp_idx'][h]+1)%len(HUMAN_WPS[h])
    tally(state)

# ── Pre-build reusable agent meshes ───────────────────────────────
ROBOT_BODY = pv.Cylinder(center=(0,0,0.32),direction=(0,0,1),
                          radius=ROBOT_R,height=0.64,resolution=14)
ROBOT_TOP  = pv.Sphere(center=(0,0,0.72),radius=ROBOT_R*0.38)
HUMAN_BODY = pv.Cylinder(center=(0,0,0.44),direction=(0,0,1),
                          radius=HUMAN_R,height=0.88,resolution=12)
HUMAN_HEAD = pv.Sphere(center=(0,0,1.02),radius=HUMAN_R*0.65)
ZONE_RING  = pv.Disc(center=(0,0,0.02),normal=(0,0,1),
                      inner=SAFETY_R-0.06,outer=SAFETY_R+0.06,r_res=1,c_res=28)
HUMAN_RING = pv.Disc(center=(0,0,0.02),normal=(0,0,1),
                      inner=SAFETY_R*0.72,outer=SAFETY_R*0.72+0.06,r_res=1,c_res=28)
COL_SPHERE = pv.Sphere(center=(0,0,0.55),radius=0.35)

def tmesh(mesh,x,y):
    m=mesh.copy(); m.translate([x,y,0],inplace=True); return m

# ── Build plotter ─────────────────────────────────────────────────
pl = pv.Plotter(shape=(1,2),off_screen=True,
                window_size=[2560,980],border=False)

CAM_POS   = (WW/2, WH/2-16, 24)
CAM_FOCUS = (WW/2, WH/2+1.0, 0)
CAM_UP    = (0,1,0)

def wall_mesh(x,y,length,axis,height):
    T=0.22
    if axis=='x':
        return pv.Box(bounds=(x,x+length,y-T/2,y+T/2,0,height))
    else:
        return pv.Box(bounds=(x-T/2,x+T/2,y,y+length,0,height))

def build_static(sub):
    pl.subplot(0,sub)
    pl.set_background('#060E1A')

    floor=pv.Plane(center=(WW/2,WH/2,-0.03),direction=(0,0,1),i_size=WW,j_size=WH)
    pl.add_mesh(floor,color=C_FLOOR,show_edges=False,reset_camera=False)

    hw=pv.Box(bounds=(0,WW,7.5,9.5,-0.02,0.02))
    pl.add_mesh(hw,color=hex_rgb('#0C2238'),show_edges=False,reset_camera=False)

    for (wx,wy,wl,wa,wh) in WALLS:
        wm=wall_mesh(wx,wy,wl,wa,wh)
        pl.add_mesh(wm,color=C_WALL,show_edges=True,edge_color=C_WALL_E,
                    line_width=0.4,reset_camera=False)

    for (px,py) in PILLARS:
        pillar=pv.Box(bounds=(px-0.18,px+0.18,py-0.18,py+0.18,0,WALL_H))
        pl.add_mesh(pillar,color=C_PILLAR,show_edges=True,edge_color=C_PILLAR_E,
                    line_width=0.5,reset_camera=False)

    for (sx,sy,sw,sd,sh,zone) in SHELVES:
        if zone=='cold':   fc,ec=hex_rgb('#122438'),hex_rgb('#1C5A8A')
        elif zone=='pick': fc,ec=hex_rgb('#0E1E34'),hex_rgb('#183266')
        else:              fc,ec=C_SHELF,C_SHELF_E
        shelf=pv.Box(bounds=(sx,sx+sw,sy,sy+sd,0,sh))
        pl.add_mesh(shelf,color=fc,show_edges=True,edge_color=ec,
                    line_width=0.5,reset_camera=False)
        rng=np.random.RandomState(int(sx*17+sy*11))
        box_colors=[C_BOX_A,C_BOX_B,C_BOX_C]
        for ix in np.arange(sx+0.18,sx+sw-0.10,0.42):
            ih=0.16+rng.uniform(0,0.14)
            gap=rng.uniform(0,0.06)
            bc=box_colors[rng.randint(0,3)]
            box=pv.Box(bounds=(ix-0.11+gap,ix+0.11-gap,
                               sy+0.10,sy+sd-0.10,sh,sh+ih))
            pl.add_mesh(box,color=bc,show_edges=True,
                        edge_color=hex_rgb('#FF9060'),line_width=0.3,reset_camera=False)

    for (dx,dy) in DOCKS:
        d=pv.Box(bounds=(dx,dx+1.0,dy-0.6,dy+0.6,0,0.6))
        pl.add_mesh(d,color=C_DOCK_F,show_edges=True,edge_color=C_DOCK_E,
                    line_width=1.0,reset_camera=False)
        dk=pv.Box(bounds=(dx+0.1,dx+0.9,dy-0.55,dy+0.55,-0.02,0.06))
        pl.add_mesh(dk,color=hex_rgb('#5A3810'),show_edges=False,reset_camera=False)

    for (px,py) in STATIONS:
        s=pv.Box(bounds=(px,px+1.1,py-0.7,py+0.7,0,0.78))
        pl.add_mesh(s,color=C_STAT_F,show_edges=True,edge_color=C_STAT_E,
                    line_width=1.2,reset_camera=False)
        belt=pv.Box(bounds=(px+0.1,px+1.0,py-0.6,py+0.6,0.78,0.82))
        pl.add_mesh(belt,color=hex_rgb('#1C4A30'),show_edges=False,reset_camera=False)

    for (ppx,ppy) in PALLETS:
        pallet=pv.Box(bounds=(ppx,ppx+0.9,ppy,ppy+0.7,0,0.16))
        pl.add_mesh(pallet,color=C_PALLET,show_edges=True,
                    edge_color=hex_rgb('#8A6020'),line_width=0.5,reset_camera=False)
        for pi in range(2):
            pb=pv.Box(bounds=(ppx+0.08,ppx+0.82,ppy+0.08+pi*0.28,
                              ppy+0.28+pi*0.28,0.16,0.38))
            pl.add_mesh(pb,color=hex_rgb('#D4622A'),show_edges=False,reset_camera=False)

    for (spx,spy,_) in SIGNS:
        sign=pv.Box(bounds=(spx-0.12,spx+0.12,spy-0.05,spy+0.05,WALL_H,WALL_H+0.55))
        pl.add_mesh(sign,color=C_SIGN,show_edges=True,edge_color=C_SIGN_E,
                    line_width=0.8,reset_camera=False)

    pl.camera.position    = CAM_POS
    pl.camera.focal_point = CAM_FOCUS
    pl.camera.up          = CAM_UP
    pl.camera.view_angle  = 46

build_static(0)
build_static(1)
print("Static scene built.")

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",15)
    FONT_LB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",13)
    FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",22)
    FONT_TL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",19)
except:
    FONT_SM=FONT_LB=FONT_LG=FONT_TL=ImageFont.load_default()

def overlay(img_arr,frame,s_state,o_state):
    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img,'RGBA')
    W,H  = img.size
    t    = frame*DT
    half = W//2

    draw.rectangle([0,0,W,54],fill=(6,14,26,248))
    draw.text((W//2,27),
        "Warehouse Coordination — Baseline Comparison  |  "
        "Mohammed Al-Khudhair  |  Habitat 3.0 (Puig et al., 2023)",
        fill=(138,180,212),font=FONT_TL,anchor='mm')

    draw.text((half//2,76),"Baseline: Simple Repulsion",
              fill=(231,76,60),font=FONT_LG,anchor='mm')
    draw.text((half+half//2,76),"ORCA / RVO2 Coordination",
              fill=(46,204,113),font=FONT_LG,anchor='mm')

    def stats_box(ox,state,bc):
        tot  = sum(state['completed'])
        rate = tot/max(t,0.5)*60.0
        lines=[f"Algorithm   :  {'Simple Repulsion' if bc==(231,76,60) else 'ORCA / RVO2'}",
               f"Sim Time    :  {t:.1f}s",
               f"Tasks       :  {tot}",
               f"Throughput  :  {rate:.1f} / min",
               f"Collisions  :  {state['collisions']}",
               f"Safety Evts :  {state['safety_evts']}"]
        bw=340; lh=22
        bx1=ox; by1=H-155; bx2=ox+bw; by2=by1+len(lines)*lh+20
        draw.rectangle([bx1,by1,bx2,by2],fill=(5,13,24,218),outline=bc,width=2)
        for k,line in enumerate(lines):
            col=(231,76,60) if (k==0 and bc==(231,76,60)) else \
                (46,204,113) if (k==0) else (170,212,245)
            draw.text((bx1+10,by1+10+k*lh),line,fill=col,font=FONT_SM)

    stats_box(14,s_state,(231,76,60))
    stats_box(half+14,o_state,(46,204,113))

    prog=frame/FRAMES
    bary=H-14
    draw.line([14,bary,W-14,bary],fill=(21,42,64),width=7)
    draw.line([14,bary,int(14+(W-28)*prog),bary],fill=(58,128,200),width=7)

    return np.array(img)

def render_agents(state,sub,pfx):
    pl.subplot(0,sub)

    for i in range(N_ROBOTS):
        rx,ry=state['robot_pos'][i]
        pl.add_mesh(tmesh(ROBOT_BODY,rx,ry),color=ROBOT_RGB[i],
                    show_edges=False,reset_camera=False,name=f'{pfx}rb{i}')
        pl.add_mesh(tmesh(ROBOT_TOP,rx,ry),color='white',opacity=0.75,
                    show_edges=False,reset_camera=False,name=f'{pfx}rt{i}')
        pl.add_mesh(tmesh(ZONE_RING,rx,ry),color=ROBOT_RGB[i],opacity=0.55,
                    show_edges=False,reset_camera=False,name=f'{pfx}rz{i}')

        vel=state['robot_vel'][i]; spd=np.linalg.norm(vel)
        if spd>0.08:
            vn=vel/spd
            arr=pv.Arrow(start=(rx,ry,0.80),direction=(vn[0],vn[1],0),scale=0.65)
            pl.add_mesh(arr,color='white',show_edges=False,reset_camera=False,
                        name=f'{pfx}ra{i}')
        else:
            pl.add_mesh(pv.Sphere(center=(-99,-99,0),radius=0.01),color='white',
                        opacity=0.0,show_edges=False,reset_camera=False,name=f'{pfx}ra{i}')

        label_mesh = pv.PolyData(np.array([[rx,ry,1.28]]))
        label_mesh['labels'] = [ROBOT_NAME[i]]
        pl.add_point_labels(label_mesh,'labels',font_size=11,font_family='courier',
                            text_color=ROBOT_HEX[i],point_color='black',
                            point_size=0,always_visible=True,reset_camera=False,
                            name=f'{pfx}rl{i}',shape_opacity=0)

    for h in range(N_HUMANS):
        hx,hy=state['human_pos'][h]
        pl.add_mesh(tmesh(HUMAN_BODY,hx,hy),color=HUMAN_RGB[h],
                    show_edges=False,reset_camera=False,name=f'{pfx}hb{h}')
        pl.add_mesh(tmesh(HUMAN_HEAD,hx,hy),color=HUMAN_RGB[h],
                    show_edges=False,reset_camera=False,name=f'{pfx}hd{h}')
        pl.add_mesh(tmesh(HUMAN_RING,hx,hy),color=HUMAN_RGB[h],opacity=0.55,
                    show_edges=False,reset_camera=False,name=f'{pfx}hz{h}')

        label_mesh = pv.PolyData(np.array([[hx,hy,1.55]]))
        label_mesh['labels'] = [HUMAN_NAME[h]]
        pl.add_point_labels(label_mesh,'labels',font_size=11,font_family='courier',
                            text_color=HUMAN_HEX[h],point_color='black',
                            point_size=0,always_visible=True,reset_camera=False,
                            name=f'{pfx}hl{h}',shape_opacity=0)

    ap=list(state['human_pos'])+list(state['robot_pos'])
    n=len(ap); k=0
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(ap[i]-ap[j])<COLLISION_R and k<N_ROBOTS:
                mid=(ap[i]+ap[j])/2
                pl.add_mesh(tmesh(COL_SPHERE,mid[0],mid[1]),color='red',
                            opacity=0.58,show_edges=False,reset_camera=False,
                            name=f'{pfx}col{k}')
                k+=1
    for kk in range(k,N_ROBOTS):
        pl.add_mesh(pv.Sphere(center=(-99,-99,0),radius=0.01),color='red',
                    opacity=0.0,show_edges=False,reset_camera=False,
                    name=f'{pfx}col{kk}')

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps) — TEST RUN...")
print(f"Output -> {OUTPUT_FILE}\n")

render_agents(S,0,'s')
render_agents(O,1,'o')

writer = iio.get_writer(OUTPUT_FILE,fps=FPS,macro_block_size=None)

for frame in range(FRAMES):
    step_simple(S)
    step_orca(O,o_sim)
    render_agents(S,0,'s')
    render_agents(O,1,'o')

    raw=pl.screenshot(return_img=True)
    if raw.shape[2]==4: raw=raw[:,:,:3]
    final=overlay(raw,frame,S,O)
    writer.append_data(np.array(final)[:,:,:3])

    if frame%(FPS)==0:
        pct=int(frame/FRAMES*100)
        print(f"  {pct:3d}%  t={frame*DT:.1f}s  "
              f"simple={sum(S['completed'])} tasks {S['collisions']} col | "
              f"orca={sum(O['completed'])} tasks {O['collisions']} col")

writer.close()
print(f"\nTest complete -> {OUTPUT_FILE}")
print(f"Simple :  {sum(S['completed'])} tasks | {S['collisions']} col | {S['safety_evts']} safety evts")
print(f"ORCA   :  {sum(O['completed'])} tasks | {O['collisions']} col | {O['safety_evts']} safety evts")
