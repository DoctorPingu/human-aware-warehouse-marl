"""
Warehouse Demo v5b - ORCA Only, Single Panel, Wall Collisions
Mohammed Al-Khudhair | Honours Project | Habitat 3.0 (Puig et al., 2023)
3 Robots | 3 Humans | Maze layout | Wall collision | Legend
TEST VERSION: 10 seconds
"""

import os
os.environ['PYVISTA_OFF_SCREEN'] = 'true'

import numpy as np
import pyvista as pv
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont
import rvo2

print("Warehouse Demo v5b - ORCA Only - Initialising...")

OUTPUT_DIR  = "/mnt/e/Honours Project/habitat-lab/demo_output"
OUTPUT_FILE = os.path.join(OUTPUT_DIR, "warehouse_demo_v5b_test.mp4")
os.makedirs(OUTPUT_DIR, exist_ok=True)

FPS      = 24
DURATION = 10
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

WW = 30.0
WH = 20.0
WALL_H  = 1.8
WALL_T  = 0.25

N_ROBOTS    = 3
N_HUMANS    = 3
ROBOT_R     = 0.30
HUMAN_R     = 0.24
SAFETY_R    = 0.90
COLLISION_R = 0.50
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

# ── Wall segments as line obstacles ───────────────────────────────
# Each wall: (x1,y1,x2,y2) — used both visually and as RVO2 obstacles
# Wider aisles: internal walls spaced 3.0 apart instead of 2.2

WALL_SEGS = [
    # Perimeter
    (0,    0,    WW,   0   ),
    (WW,   0,    WW,   WH  ),
    (WW,   WH,   0,    WH  ),
    (0,    WH,   0,    0   ),
    # Zone dividers with gaps at y=8.5 to y=11.5 (3-unit highway gap)
    (8.5,  0,    8.5,  8.0 ),
    (8.5,  12.0, 8.5,  WH  ),
    (15.5, 0,    15.5, 8.0 ),
    (15.5, 12.0, 15.5, WH  ),
    (22.5, 0,    22.5, 8.0 ),
    (22.5, 12.0, 22.5, WH  ),
    # Cross-aisle horizontal walls with gaps at x=8.5, x=15.5, x=22.5
    (0,    10.0, 7.8,  10.0),
    (9.2,  10.0, 14.8, 10.0),
    (16.2, 10.0, 21.8, 10.0),
    (23.2, 10.0, WW,   10.0),
]

# ── Shelf blocks ──────────────────────────────────────────────────
# Wider aisles: shelves placed with 2.8 gap between them
SHELVES = []

# Zone A cold (x:1-8)
for rx in [1.2, 4.0]:
    for ry in [1.0, 3.2, 5.4]:
        SHELVES.append((rx, ry, 2.2, 0.85, 1.2, 'cold'))
    for ry in [11.8, 14.0, 16.2]:
        SHELVES.append((rx, ry, 2.2, 0.85, 1.2, 'cold'))

# Zone B bulk (x:9-15)
for rx in [9.5, 12.2]:
    for ry in [1.0, 3.2, 5.4]:
        SHELVES.append((rx, ry, 2.4, 0.85, 1.0, 'bulk'))
    for ry in [11.8, 14.0, 16.2]:
        SHELVES.append((rx, ry, 2.4, 0.85, 1.0, 'bulk'))

# Zone C pick (x:16-22), shorter denser shelves
for rx in [16.5, 18.8]:
    for ry in [1.0, 2.6, 4.2, 5.8]:
        SHELVES.append((rx, ry, 1.8, 0.75, 0.75, 'pick'))
    for ry in [11.8, 13.4, 15.0, 16.6]:
        SHELVES.append((rx, ry, 1.8, 0.75, 0.75, 'pick'))

# Dead-end maze shelves
SHELVES.append((21.0, 1.0, 1.8, 0.75, 0.75, 'pick'))
SHELVES.append((21.0, 3.5, 1.8, 0.75, 0.75, 'pick'))
SHELVES.append((21.0, 15.5, 1.8, 0.75, 0.75, 'pick'))
SHELVES.append((21.0, 17.5, 1.8, 0.75, 0.75, 'pick'))

DOCKS    = [(0.4, 3.5),(0.4, 10.0),(0.4, 16.5)]
STATIONS = [(26.5, 4.0),(26.5, 10.0),(26.5, 16.0)]
PALLETS  = [(1.0,7.2),(1.8,7.2),(1.0,8.0),(1.0,11.2),(1.8,11.2)]
PILLARS  = [
    (8.5,0),(8.5,10.0),(8.5,WH),
    (15.5,0),(15.5,10.0),(15.5,WH),
    (22.5,0),(22.5,10.0),(22.5,WH),
]

# ── Navigation ────────────────────────────────────────────────────
def mc(pts): return pts+[pts[0]]

ROBOT_STARTS = [(3.5,3.5),(11.0,3.5),(18.5,3.5)]
ROBOT_TASKS  = [
    mc([(3.5,3.5),(3.5,7.5),(8.5,10.0),(15.5,10.0),(26.5,4.0)]),
    mc([(11.0,3.5),(11.0,7.5),(15.5,10.0),(22.5,10.0),(26.5,10.0)]),
    mc([(18.5,3.5),(18.5,7.5),(22.5,10.0),(26.5,16.0),(18.5,17.0)]),
]

HUMAN_WPS = [
    mc([(2.0,2.0),(28.0,2.0),(28.0,18.5),(2.0,18.5)]),
    mc([(16.5,3.5),(20.5,3.5),(20.5,7.5),(16.5,7.5),
        (16.5,12.0),(20.5,12.0),(20.5,18.0),(16.5,18.0)]),
    mc([(9.0,2.5),(9.0,7.5),(9.0,12.0),(9.0,18.0),
        (4.0,18.0),(4.0,12.0),(4.0,7.5),(4.0,2.5)]),
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
        wall_hits      = 0,
    )

O = make_state()

# ── RVO2 simulator with wall obstacles ────────────────────────────
o_sim = rvo2.PyRVOSimulator(DT,5.0,15,8.0,2.0,ROBOT_R,ROBOT_SPD)

for i in range(N_ROBOTS):
    o_sim.addAgent(tuple(O['robot_pos'][i]),5.0,15,8.0,2.0,ROBOT_R,ROBOT_SPD,(0.0,0.0))
for h in range(N_HUMANS):
    o_sim.addAgent(tuple(O['human_pos'][h]),5.0,15,5.0,2.0,HUMAN_R,HUMAN_SPD,(0.0,0.0))

for (x1,y1,x2,y2) in WALL_SEGS:
    o_sim.addObstacle([(x1,y1),(x2,y2)])
o_sim.processObstacles()

# ── Collision detection helpers ───────────────────────────────────
def clamp_to_walls(pos, radius):
    margin = radius + 0.05
    new_pos = pos.copy()
    hit = False
    for (x1,y1,x2,y2) in WALL_SEGS:
        if x1==x2:
            if abs(pos[0]-x1) < margin:
                if pos[0] < x1: new_pos[0] = x1 - margin
                else:            new_pos[0] = x1 + margin
                hit = True
        elif y1==y2:
            if abs(pos[1]-y1) < margin:
                if pos[1] < y1: new_pos[1] = y1 - margin
                else:            new_pos[1] = y1 + margin
                hit = True
    new_pos = np.clip(new_pos, margin, [WW-margin, WH-margin])
    return new_pos, hit

def shelf_collision(pos, radius):
    new_pos = pos.copy()
    hit = False
    for (sx,sy,sw,sd,sh,zone) in SHELVES:
        pad = radius + 0.08
        if (sx-pad < pos[0] < sx+sw+pad) and (sy-pad < pos[1] < sy+sd+pad):
            cx = sx + sw/2; cy = sy + sd/2
            dx = pos[0]-cx; dy = pos[1]-cy
            px = sw/2+pad-abs(dx); py = sd/2+pad-abs(dy)
            if px < py:
                new_pos[0] += np.sign(dx)*px
            else:
                new_pos[1] += np.sign(dy)*py
            hit = True
    return new_pos, hit

def tally(state):
    ap=list(state['human_pos'])+list(state['robot_pos'])
    n=len(ap)
    for i in range(n):
        for j in range(i+1,n):
            d=np.linalg.norm(ap[i]-ap[j])
            if d<COLLISION_R:   state['collisions']  +=1
            elif d<SAFETY_R:    state['safety_evts'] +=1

def step_orca(state,sim):
    for i in range(N_ROBOTS):
        if state['robot_state'][i]=='wait':
            sim.setAgentPrefVelocity(i,(0.0,0.0))
        else:
            tgt=np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            diff=tgt-state['robot_pos'][i]; dist=np.linalg.norm(diff)
            if dist<0.25: sim.setAgentPrefVelocity(i,(0.0,0.0))
            else:
                spd=min(ROBOT_SPD,dist*FPS*0.5)
                sim.setAgentPrefVelocity(i,tuple((diff/dist)*spd))

    for h in range(N_HUMANS):
        tgt=np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        diff=tgt-state['human_pos'][h]; dist=np.linalg.norm(diff)
        if dist<0.12: sim.setAgentPrefVelocity(N_ROBOTS+h,(0.0,0.0))
        else: sim.setAgentPrefVelocity(N_ROBOTS+h,tuple((diff/dist)*HUMAN_SPEEDS[h]))

    sim.doStep()

    for i in range(N_ROBOTS):
        pos=np.array(sim.getAgentPosition(i))
        pos,wall_hit=clamp_to_walls(pos,ROBOT_R)
        pos,shelf_hit=shelf_collision(pos,ROBOT_R)
        if wall_hit or shelf_hit: state['wall_hits']+=1
        pos=np.clip(pos,ROBOT_R+0.05,[WW-ROBOT_R-0.05,WH-ROBOT_R-0.05])
        vel=np.array(sim.getAgentVelocity(i))
        state['robot_pos'][i]=pos; state['robot_vel'][i]=vel
        sim.setAgentPosition(i,tuple(pos))
        if state['robot_state'][i]=='wait':
            state['robot_wait'][i]-=1
            if state['robot_wait'][i]<=0: state['robot_state'][i]='move'
        else:
            tgt=np.array(ROBOT_TASKS[i][state['robot_task_idx'][i]])
            if np.linalg.norm(tgt-pos)<0.42:
                state['robot_task_idx'][i]=(state['robot_task_idx'][i]+1)%len(ROBOT_TASKS[i])
                state['completed'][i]+=1; state['robot_state'][i]='wait'
                state['robot_wait'][i]=int(FPS*0.55)

    for h in range(N_HUMANS):
        pos=np.array(sim.getAgentPosition(N_ROBOTS+h))
        pos,wall_hit=clamp_to_walls(pos,HUMAN_R)
        pos,shelf_hit=shelf_collision(pos,HUMAN_R)
        if wall_hit or shelf_hit: state['wall_hits']+=1
        pos=np.clip(pos,HUMAN_R+0.05,[WW-HUMAN_R-0.05,WH-HUMAN_R-0.05])
        state['human_pos'][h]=pos; state['human_phase'][h]+=0.20
        sim.setAgentPosition(N_ROBOTS+h,tuple(pos))
        tgt=np.array(HUMAN_WPS[h][state['human_wp_idx'][h]])
        if np.linalg.norm(tgt-pos)<HUMAN_STEP*3.5:
            state['human_wp_idx'][h]=(state['human_wp_idx'][h]+1)%len(HUMAN_WPS[h])

    tally(state)

# ── Pre-build reusable meshes ─────────────────────────────────────
ROBOT_BODY = pv.Cylinder(center=(0,0,0.32),direction=(0,0,1),
                          radius=ROBOT_R,height=0.64,resolution=14)
ROBOT_CAP  = pv.Sphere(center=(0,0,0.72),radius=ROBOT_R*0.40)
HUMAN_BODY = pv.Cylinder(center=(0,0,0.44),direction=(0,0,1),
                          radius=HUMAN_R,height=0.88,resolution=12)
HUMAN_HEAD = pv.Sphere(center=(0,0,1.02),radius=HUMAN_R*0.65)
ZONE_RING  = pv.Disc(center=(0,0,0.02),normal=(0,0,1),
                      inner=SAFETY_R-0.07,outer=SAFETY_R+0.07,r_res=1,c_res=30)
HUMAN_RING = pv.Disc(center=(0,0,0.02),normal=(0,0,1),
                      inner=SAFETY_R*0.70,outer=SAFETY_R*0.70+0.07,r_res=1,c_res=30)
COL_SPHERE = pv.Sphere(center=(0,0,0.55),radius=0.36)

def tmesh(mesh,x,y):
    m=mesh.copy(); m.translate([x,y,0],inplace=True); return m

# ── Build plotter ─────────────────────────────────────────────────
pl = pv.Plotter(off_screen=True, window_size=[2200,1080], border=False)

CAM_POS   = (WW/2, WH/2-18, 26)
CAM_FOCUS = (WW/2, WH/2+1.5, 0)
CAM_UP    = (0,1,0)

def build_static():
    pl.set_background('#060E1A')

    floor=pv.Plane(center=(WW/2,WH/2,-0.03),direction=(0,0,1),i_size=WW,j_size=WH)
    pl.add_mesh(floor,color=hex_rgb('#0A1828'),show_edges=False,reset_camera=False)

    hw=pv.Box(bounds=(0,WW,8.0,12.0,-0.02,0.015))
    pl.add_mesh(hw,color=hex_rgb('#0C2238'),show_edges=False,reset_camera=False)

    for (x1,y1,x2,y2) in WALL_SEGS:
        if x1==x2:
            wm=pv.Box(bounds=(x1-WALL_T/2,x1+WALL_T/2,
                              min(y1,y2),max(y1,y2),0,WALL_H))
        else:
            wm=pv.Box(bounds=(min(x1,x2),max(x1,x2),
                              y1-WALL_T/2,y1+WALL_T/2,0,WALL_H))
        pl.add_mesh(wm,color=hex_rgb('#1C3A5A'),show_edges=True,
                    edge_color=hex_rgb('#2A5A8A'),line_width=0.4,reset_camera=False)

    for (px,py) in PILLARS:
        p=pv.Box(bounds=(px-0.22,px+0.22,py-0.22,py+0.22,0,WALL_H))
        pl.add_mesh(p,color=hex_rgb('#1A3050'),show_edges=True,
                    edge_color=hex_rgb('#2A5080'),line_width=0.5,reset_camera=False)

    for (sx,sy,sw,sd,sh,zone) in SHELVES:
        if zone=='cold':   fc,ec=hex_rgb('#122438'),hex_rgb('#1C5A8A')
        elif zone=='pick': fc,ec=hex_rgb('#0E1E34'),hex_rgb('#183266')
        else:              fc,ec=hex_rgb('#122438'),hex_rgb('#1C3C6A')
        shelf=pv.Box(bounds=(sx,sx+sw,sy,sy+sd,0,sh))
        pl.add_mesh(shelf,color=fc,show_edges=True,edge_color=ec,
                    line_width=0.5,reset_camera=False)
        rng=np.random.RandomState(int(sx*17+sy*11))
        box_cols=[hex_rgb('#D4622A'),hex_rgb('#E8852A'),hex_rgb('#B84E20')]
        for ix in np.arange(sx+0.20,sx+sw-0.12,0.48):
            ih=0.18+rng.uniform(0,0.14)
            gap=rng.uniform(0.04,0.10)
            bc=box_cols[rng.randint(0,3)]
            b=pv.Box(bounds=(ix-0.13+gap,ix+0.13-gap,
                             sy+0.12,sy+sd-0.12,sh,sh+ih))
            pl.add_mesh(b,color=bc,show_edges=True,
                        edge_color=hex_rgb('#FF9060'),line_width=0.3,reset_camera=False)

    for (dx,dy) in DOCKS:
        d=pv.Box(bounds=(dx,dx+1.1,dy-0.7,dy+0.7,0,0.6))
        pl.add_mesh(d,color=hex_rgb('#3A2810'),show_edges=True,
                    edge_color=hex_rgb('#E8A018'),line_width=1.0,reset_camera=False)

    for (px,py) in STATIONS:
        s=pv.Box(bounds=(px,px+1.2,py-0.75,py+0.75,0,0.78))
        pl.add_mesh(s,color=hex_rgb('#182E22'),show_edges=True,
                    edge_color=hex_rgb('#48B870'),line_width=1.2,reset_camera=False)

    for (ppx,ppy) in PALLETS:
        pallet=pv.Box(bounds=(ppx,ppx+1.0,ppy,ppy+0.75,0,0.18))
        pl.add_mesh(pallet,color=hex_rgb('#5A4010'),show_edges=True,
                    edge_color=hex_rgb('#8A6020'),line_width=0.5,reset_camera=False)
        rng2=np.random.RandomState(int(ppx*7+ppy*13))
        for pi in range(2):
            pb=pv.Box(bounds=(ppx+0.08,ppx+0.92,ppy+0.08+pi*0.30,
                              ppy+0.30+pi*0.30,0.18,0.40))
            pl.add_mesh(pb,color=hex_rgb('#D4622A'),show_edges=False,reset_camera=False)

    pl.camera.position    = CAM_POS
    pl.camera.focal_point = CAM_FOCUS
    pl.camera.up          = CAM_UP
    pl.camera.view_angle  = 46

build_static()
print("Static scene built.")

try:
    FONT_SM = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",15)
    FONT_LG = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",22)
    FONT_TL = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",19)
    FONT_LB = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",13)
except:
    FONT_SM=FONT_LG=FONT_TL=FONT_LB=ImageFont.load_default()

def draw_legend(draw, x, y):
    items = [
        (ROBOT_HEX[0], 'AMR-1 (Robot)'),
        (ROBOT_HEX[1], 'AMR-2 (Robot)'),
        (ROBOT_HEX[2], 'AMR-3 (Robot)'),
        (HUMAN_HEX[0], 'Worker-1 (Human)'),
        (HUMAN_HEX[1], 'Worker-2 (Human)'),
        (HUMAN_HEX[2], 'Worker-3 (Human)'),
        ('#E74C3C',    'Collision Event'),
        ('#3498DB',    'Safety Zone (ISO 10218)'),
        ('#D4622A',    'Inventory Box'),
        ('#48B870',    'Pick Station'),
        ('#E8A018',    'Loading Dock'),
        ('#5A4010',    'Staging Pallet'),
        ('#1C3A5A',    'Wall / Barrier'),
    ]
    pad=8; rw=14; rh=14; lh=22
    bw=260; bh=len(items)*lh+pad*2+20
    draw.rectangle([x,y,x+bw,y+bh],fill=(5,13,24,220),outline=(42,90,138),width=1)
    draw.text((x+pad,y+pad),'LEGEND',fill=(138,180,212),font=FONT_LB)
    for k,(col,label) in enumerate(items):
        ry2=y+pad+20+k*lh
        r,g,b=hex_rgb(col)
        draw.rectangle([x+pad,ry2,x+pad+rw,ry2+rh],
                       fill=(int(r*255),int(g*255),int(b*255),230))
        draw.text((x+pad+rw+6,ry2),label,fill=(170,212,245),font=FONT_LB)

def overlay(img_arr, frame, state):
    img  = Image.fromarray(img_arr)
    draw = ImageDraw.Draw(img,'RGBA')
    W,H  = img.size
    t    = frame*DT

    draw.rectangle([0,0,W,54],fill=(6,14,26,248))
    draw.text((W//2,27),
        "Warehouse Coordination Demo  —  ORCA / RVO2  |  "
        "Mohammed Al-Khudhair  |  Habitat 3.0 (Puig et al., 2023)",
        fill=(138,180,212),font=FONT_TL,anchor='mm')

    draw.text((W//2,76),"ORCA / RVO2 Multi-Agent Coordination",
              fill=(46,204,113),font=FONT_LG,anchor='mm')

    tot  = sum(state['completed'])
    rate = tot/max(t,0.5)*60.0
    lines=[
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
    bw=340; lh=22
    bx1=W-bw-14; by1=H-len(lines)*lh-30
    draw.rectangle([bx1,by1,bx1+bw,by1+len(lines)*lh+20],
                   fill=(5,13,24,218),outline=(46,204,113),width=2)
    for k,line in enumerate(lines):
        col=(46,204,113) if k==0 else (170,212,245)
        draw.text((bx1+10,by1+10+k*lh),line,fill=col,font=FONT_SM)

    draw_legend(draw, 14, 95)

    prog=frame/FRAMES
    bary=H-14
    draw.line([14,bary,W-14,bary],fill=(21,42,64),width=7)
    draw.line([14,bary,int(14+(W-28)*prog),bary],fill=(58,128,200),width=7)

    return np.array(img)

def render_agents(state):
    for i in range(N_ROBOTS):
        rx,ry=state['robot_pos'][i]
        pl.add_mesh(tmesh(ROBOT_BODY,rx,ry),color=ROBOT_RGB[i],
                    show_edges=False,reset_camera=False,name=f'rb{i}')
        pl.add_mesh(tmesh(ROBOT_CAP,rx,ry),color='white',opacity=0.75,
                    show_edges=False,reset_camera=False,name=f'rc{i}')
        pl.add_mesh(tmesh(ZONE_RING,rx,ry),color=ROBOT_RGB[i],opacity=0.55,
                    show_edges=False,reset_camera=False,name=f'rz{i}')
        vel=state['robot_vel'][i]; spd=np.linalg.norm(vel)
        if spd>0.08:
            vn=vel/spd
            arr=pv.Arrow(start=(rx,ry,0.80),direction=(vn[0],vn[1],0),scale=0.70)
            pl.add_mesh(arr,color='white',show_edges=False,reset_camera=False,name=f'ra{i}')
        else:
            pl.add_mesh(pv.Sphere(center=(-99,-99,0),radius=0.01),color='white',
                        opacity=0.0,show_edges=False,reset_camera=False,name=f'ra{i}')
        lm=pv.PolyData(np.array([[rx,ry,1.32]]))
        lm['labels']=[ROBOT_NAME[i]]
        pl.add_point_labels(lm,'labels',font_size=11,font_family='courier',
                            text_color=ROBOT_HEX[i],point_color='black',
                            point_size=0,always_visible=True,reset_camera=False,
                            name=f'rl{i}',shape_opacity=0)

    for h in range(N_HUMANS):
        hx,hy=state['human_pos'][h]
        pl.add_mesh(tmesh(HUMAN_BODY,hx,hy),color=HUMAN_RGB[h],
                    show_edges=False,reset_camera=False,name=f'hb{h}')
        pl.add_mesh(tmesh(HUMAN_HEAD,hx,hy),color=HUMAN_RGB[h],
                    show_edges=False,reset_camera=False,name=f'hd{h}')
        pl.add_mesh(tmesh(HUMAN_RING,hx,hy),color=HUMAN_RGB[h],opacity=0.55,
                    show_edges=False,reset_camera=False,name=f'hz{h}')
        lm=pv.PolyData(np.array([[hx,hy,1.58]]))
        lm['labels']=[HUMAN_NAME[h]]
        pl.add_point_labels(lm,'labels',font_size=11,font_family='courier',
                            text_color=HUMAN_HEX[h],point_color='black',
                            point_size=0,always_visible=True,reset_camera=False,
                            name=f'hl{h}',shape_opacity=0)

    ap=list(state['human_pos'])+list(state['robot_pos'])
    n=len(ap); k=0
    for i in range(n):
        for j in range(i+1,n):
            if np.linalg.norm(ap[i]-ap[j])<COLLISION_R and k<N_ROBOTS:
                mid=(ap[i]+ap[j])/2
                pl.add_mesh(tmesh(COL_SPHERE,mid[0],mid[1]),color='red',
                            opacity=0.58,show_edges=False,reset_camera=False,
                            name=f'col{k}')
                k+=1
    for kk in range(k,N_ROBOTS):
        pl.add_mesh(pv.Sphere(center=(-99,-99,0),radius=0.01),color='red',
                    opacity=0.0,show_edges=False,reset_camera=False,name=f'col{kk}')

print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps) — TEST RUN...")
print(f"Output -> {OUTPUT_FILE}\n")

render_agents(O)
writer = iio.get_writer(OUTPUT_FILE,fps=FPS,macro_block_size=None)

for frame in range(FRAMES):
    step_orca(O,o_sim)
    render_agents(O)

    raw=pl.screenshot(return_img=True)
    if raw.shape[2]==4: raw=raw[:,:,:3]
    final=overlay(raw,frame,O)
    writer.append_data(np.array(final)[:,:,:3])

    if frame%FPS==0:
        pct=int(frame/FRAMES*100)
        print(f"  {pct:3d}%  t={frame*DT:.1f}s  "
              f"tasks={sum(O['completed'])}  col={O['collisions']}  "
              f"wall={O['wall_hits']}")

writer.close()
print(f"\nTest complete -> {OUTPUT_FILE}")
print(f"Tasks: {sum(O['completed'])} | Collisions: {O['collisions']} | "
      f"Wall events: {O['wall_hits']} | Safety: {O['safety_evts']}")
