"""
Trained Policy Visualisation - Warehouse Multi-Agent Coordination
Honours Project: Deep Reinforcement Learning for Multi-Agent Coordination
in Mixed Human-Robot Warehouse Environments
Mohammed Al-Khudhair - 24826341

Loads the trained CTDE PPO model and records a demo video showing:
  - Trained robot coordination behaviour
  - Live stats panel
  - Training curve inset showing learning progress
  - Comparison overlay vs random policy baseline
"""

import os
import sys
import json
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import imageio.v2 as iio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from warehouse_env import WarehouseEnv
from train_warehouse import ActorNetwork, CriticNetwork, CTDEAgent

TRAINING_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             'training_output')
OUTPUT_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             '..', '..', 'demo_output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_PATH   = os.path.join(TRAINING_DIR, 'model_final.pt')
LOG_PATH     = os.path.join(TRAINING_DIR, 'training_log.json')
OUTPUT_FILE  = os.path.join(OUTPUT_DIR,   'warehouse_demo_trained_v3.mp4')

FPS      = 24
DURATION = 60
FRAMES   = FPS * DURATION
DT       = 1.0 / FPS

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

N_ROBOTS = WarehouseEnv.N_ROBOTS
N_HUMANS = WarehouseEnv.N_HUMANS

ROBOT_COL = '#3498DB'
HUMAN_COL = '#FF6B35'
CARRY_COL = '#D4622A'
BG_COL    = '#07101C'
FLOOR_COL = '#0A1828'
HWCOL     = '#0D2540'

def hex_rgba(h, a=1.0):
    h = h.lstrip('#')
    r,g,b = [int(h[i:i+2],16)/255 for i in (0,2,4)]
    return (r,g,b,a)

# ── Load training log ─────────────────────────────────────────────
def load_log(path):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

# ── Load trained agent ────────────────────────────────────────────
def load_agent():
    agent = CTDEAgent()
    agent.load(MODEL_PATH)
    for actor in agent.actors:
        actor.eval()
    agent.critic.eval()
    return agent

# ── Smooth helper ─────────────────────────────────────────────────
def smooth(vals, w=15):
    if len(vals) < 2:
        return vals
    return [np.mean(vals[max(0,i-w):i+1]) for i in range(len(vals))]

# ── Draw warehouse frame ──────────────────────────────────────────
def draw_warehouse(ax, env, robot_trails, frame_num, ep_stats):
    ax.clear()
    ax.set_facecolor(FLOOR_COL)
    ax.set_xlim(0, env.WW)
    ax.set_ylim(0, env.WH)
    ax.set_aspect('equal')
    ax.axis('off')

    # Highway aisle
    ax.add_patch(mpatches.Rectangle((0,10.5), env.WW, 3.0,
        facecolor=HWCOL, edgecolor='none', zorder=1))
    ax.text(env.WW/2, 12.0, 'HIGHWAY AISLE', ha='center', va='center',
            fontsize=6, color='#1A4070', fontfamily='monospace', zorder=2)

    # Zone backgrounds
    for (zx,zw,zc) in [
        (0,  4.0,'#091828'),(4.0, 6.0,'#07162A'),
        (10.0,8.0,'#081828'),(18.0,7.0,'#071626'),
        (25.0,9.0,'#081828')]:
        ax.add_patch(mpatches.Rectangle((zx,0),zw,env.WH,
            facecolor=zc,edgecolor='none',zorder=1))

    # Zone labels
    for (zx,zw,ztxt) in [
        (0,4,'RECV'),(4,6,'COLD'),(10,8,'BULK'),(18,7,'PICK'),(25,9,'SHIP')]:
        ax.text(zx+zw/2, env.WH-0.6, ztxt, ha='center', va='top',
                fontsize=5, color='#1E4872', fontfamily='monospace', zorder=2)

    # Shelves
    for (sx,sy,sw,sd) in env.shelves:
        ax.add_patch(mpatches.Rectangle((sx,sy),sw,sd,
            facecolor='#122438',edgecolor='#1C3C6A',lw=0.5,zorder=3))
        rng = np.random.RandomState(int(sx*17+sy*11))
        for ix in np.arange(sx+0.22, sx+sw-0.12, 0.52):
            ih  = 0.18 + rng.uniform(0,0.12)
            gap = rng.uniform(0.04,0.10)
            b   = mpatches.Rectangle((ix-0.13+gap, sy+0.10),
                                      0.26-2*gap, sd-0.20,
                  facecolor=CARRY_COL, edgecolor='none', zorder=4)
            ax.add_patch(b)

    # Docks
    for (dx,dy) in [(0.8,4.0),(0.8,12.0),(0.8,20.0)]:
        ax.add_patch(mpatches.FancyBboxPatch((dx,dy-0.8),1.1,1.6,
            boxstyle='round,pad=0.05',
            facecolor='#3A2810',edgecolor='#E8A018',lw=0.8,zorder=3))
        ax.text(dx+0.55,dy,'DOCK',ha='center',va='center',
                fontsize=4.5,color='#E8A018',fontweight='bold',zorder=4)

    # Pick stations
    for (px,py) in env.STATIONS:
        ax.add_patch(mpatches.FancyBboxPatch((px,py-0.8),1.2,1.6,
            boxstyle='round,pad=0.05',
            facecolor='#182E22',edgecolor='#48B870',lw=1.0,zorder=3))
        ax.text(px+0.6,py,'PICK\nSTATION',ha='center',va='center',
                fontsize=4,color='#48B870',fontweight='bold',zorder=4)

    # Zone divider walls
    for xd in [4.0,30.0]:
        ax.plot([xd,xd],[0,9.5],color='#1C3A5A',lw=1.5,zorder=5)
        ax.plot([xd,xd],[14.5,env.WH],color='#1C3A5A',lw=1.5,zorder=5)

    # Junction markers
    CX = env.CX[1:6]
    CY = env.CY
    for cx in CX:
        for cy in CY:
            ax.plot(cx,cy,'+',color='#3A6890',markersize=6,
                    markeredgewidth=0.6,alpha=0.6,zorder=3)

    # Robot trails
    for i in range(N_ROBOTS):
        trail = robot_trails[i]
        n = len(trail)
        if n > 1:
            for k in range(n-1):
                alpha = (k/n)**1.5 * 0.6
                lw    = max(0.4, (k/n)*1.8)
                ax.plot([trail[k][0],trail[k+1][0]],
                        [trail[k][1],trail[k+1][1]],
                        color=ROBOT_COL, alpha=alpha, lw=lw, zorder=5)

    # Safety zones
    for i in range(N_ROBOTS):
        rp = env.robot_pos[i]
        circle = plt.Circle(rp, env.SAFETY_R,
                            fill=True, facecolor=ROBOT_COL, alpha=0.06)
        ax.add_patch(circle)
        circle2 = plt.Circle(rp, env.SAFETY_R,
                             fill=False, edgecolor=ROBOT_COL,
                             ls='--', lw=0.5, alpha=0.3)
        ax.add_patch(circle2)

    # Robots
    for i in range(N_ROBOTS):
        rp  = env.robot_pos[i]
        rv  = env.robot_vel[i]
        spd = np.linalg.norm(rv)

        body = plt.Circle(rp, env.ROBOT_R, color=ROBOT_COL, zorder=7)
        ax.add_patch(body)
        cap  = plt.Circle(rp, env.ROBOT_R*0.40, color='white',
                          alpha=0.7, zorder=8)
        ax.add_patch(cap)

        if env.robot_carrying[i]:
            box = mpatches.Rectangle(
                (rp[0]-0.15, rp[1]-0.15), 0.30, 0.30,
                facecolor=CARRY_COL, edgecolor='white', lw=0.4, zorder=9)
            ax.add_patch(box)

        if spd > 0.05:
            vn = rv / spd
            ax.annotate('', xy=rp+vn*0.45, xytext=rp+vn*env.ROBOT_R,
                        arrowprops=dict(arrowstyle='->',
                                        color='white', lw=0.8), zorder=10)

        ax.text(rp[0], rp[1]+env.ROBOT_R+0.22,
                f'AMR-{i+1}', ha='center', va='bottom',
                fontsize=5.5, color=ROBOT_COL,
                fontfamily='monospace', zorder=10)

    # Humans
    for h in range(N_HUMANS):
        hp = env.human_pos[h]
        body = plt.Circle(hp, env.HUMAN_R, color=HUMAN_COL, zorder=7)
        ax.add_patch(body)
        ax.text(hp[0], hp[1]+env.HUMAN_R+0.22,
                f'W-{h+1}', ha='center', va='bottom',
                fontsize=5.5, color=HUMAN_COL,
                fontfamily='monospace', zorder=10)

    # Stats
    t = frame_num * DT
    carrying = sum(1 for c in env.robot_carrying if c)
    stats = (f"t={t:.1f}s  "
             f"Deliveries: {ep_stats['deliveries']}  "
             f"Collisions: {ep_stats['collisions']}  "
             f"({carrying} carrying)")
    ax.text(env.WW/2, -0.8, stats, ha='center', va='top',
            fontsize=7, color='#AAC4E0', fontfamily='monospace', zorder=10)

# ── Draw training curve inset ─────────────────────────────────────
def draw_curve_inset(ax, log, highlight_step):
    ax.clear()
    ax.set_facecolor('#0D1428')
    for spine in ax.spines.values():
        spine.set_edgecolor('#1A3050')
    ax.tick_params(colors='#8090A8', labelsize=6)

    steps = [m['steps'] for m in log]
    dels  = [m['deliveries'] for m in log]
    sm    = smooth(dels, w=15)

    ax.plot(steps, dels, color='#2ECC71', alpha=0.2, lw=0.6)
    ax.plot(steps, sm,   color='#2ECC71', lw=1.4, label='Deliveries')

    if highlight_step <= max(steps):
        ax.axvline(highlight_step, color='white', lw=0.8,
                   ls='--', alpha=0.5)

    ax.set_xlabel('Timesteps', color='#8090A8', fontsize=6)
    ax.set_title('Training Progress', color='#AAC4E0',
                 fontsize=7, pad=3)
    ax.grid(alpha=0.12, color='#2A4060')

# ── Build frame ───────────────────────────────────────────────────
def build_figure():
    fig = plt.figure(figsize=(16, 9), dpi=100)
    fig.patch.set_facecolor(BG_COL)
    gs  = gridspec.GridSpec(2, 2,
                            left=0.02, right=0.98,
                            top=0.92,  bottom=0.06,
                            wspace=0.08, hspace=0.30,
                            width_ratios=[3,1],
                            height_ratios=[3,1])
    ax_main  = fig.add_subplot(gs[0,0])
    ax_curve = fig.add_subplot(gs[0,1])
    ax_stats = fig.add_subplot(gs[1,:])
    ax_stats.axis('off')
    return fig, ax_main, ax_curve, ax_stats

# ── Stats bar ─────────────────────────────────────────────────────
def draw_stats_bar(ax, ep_stats, frame_num, total_frames,
                   episode_num, total_reward):
    ax.clear()
    ax.set_facecolor('#050D18')
    ax.set_xlim(0,1); ax.set_ylim(0,1)
    ax.axis('off')

    t    = frame_num * DT
    rate = ep_stats['deliveries'] / max(t, 0.1) * 60.0
    txt  = (
        f"Algorithm: CTDE PPO (Trained)   |   "
        f"Episode: {episode_num}   |   "
        f"Sim Time: {t:.1f}s   |   "
        f"Deliveries: {ep_stats['deliveries']}   |   "
        f"Throughput: {rate:.1f}/min   |   "
        f"Collisions: {ep_stats['collisions']}   |   "
        f"Safety Events: {ep_stats['safety_events']}   |   "
        f"Cumulative Reward: {total_reward:.1f}"
    )
    ax.text(0.5, 0.5, txt, ha='center', va='center',
            fontsize=8, color='#AAC4E0', fontfamily='monospace',
            transform=ax.transAxes)

    prog = frame_num / total_frames
    ax.add_patch(mpatches.Rectangle((0,0),1,0.08,
        facecolor='#152A40',edgecolor='none',transform=ax.transAxes))
    ax.add_patch(mpatches.Rectangle((0,0),prog,0.08,
        facecolor='#3A80C8',edgecolor='none',transform=ax.transAxes))

# ── Legend panel ──────────────────────────────────────────────────
def draw_legend(ax_curve, log, highlight_step):
    draw_curve_inset(ax_curve, log, highlight_step)
    handles = [
        mpatches.Patch(color=ROBOT_COL,  label='AMR Robot (trained)'),
        mpatches.Patch(color=HUMAN_COL,  label='Human Worker'),
        mpatches.Patch(color=CARRY_COL,  label='Carrying item'),
        mpatches.Patch(color='#48B870',  label='Pick Station'),
        mpatches.Patch(color='#E8A018',  label='Loading Dock'),
    ]
    ax_curve.legend(handles=handles, loc='lower right',
                    fontsize=5.5, framealpha=0.85,
                    facecolor='#0D1428', edgecolor='#1A3050',
                    labelcolor='#AAC4E0')

# ── Main render loop ──────────────────────────────────────────────
def render_demo():
    print("Loading trained model...")
    agent = load_agent()
    log   = load_log(LOG_PATH)
    if log is None:
        print("WARNING: training_log.json not found. Curve inset disabled.")
        log = [{'steps':0,'deliveries':0,'mean_reward':-200,'collisions':0}]

    env = WarehouseEnv(max_steps=2000)
    obs, info = env.reset(seed=0)

    trail_len   = 80
    robot_trails = [[] for _ in range(N_ROBOTS)]

    ep_stats = {
        'deliveries'  : 0,
        'collisions'  : 0,
        'safety_events': 0,
    }
    episode_num  = 1
    total_reward = 0.0
    highlight_step = len(log) * 500_000 // max(len(log),1)

    fig, ax_main, ax_curve, ax_stats = build_figure()

    try:
        FONT_TL = ImageFont.truetype(
            "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf",18)
    except:
        FONT_TL = ImageFont.load_default()

    print(f"Rendering {FRAMES} frames ({DURATION}s at {FPS}fps)...")
    print(f"Output -> {OUTPUT_FILE}\n")

    writer = iio.get_writer(OUTPUT_FILE, fps=FPS, macro_block_size=None)

    for frame in range(FRAMES):
        # ── Step environment ──────────────────────────────────────
        actions, _, _ = agent.get_actions(obs)
        obs, rewards, terminated, truncated, info = env.step(actions)

        for k in rewards:
            total_reward += rewards[k]

        ep_stats['deliveries']   = info['deliveries']
        ep_stats['collisions']   = info['collisions']
        ep_stats['safety_events']= info['safety_events']

        for i in range(N_ROBOTS):
            robot_trails[i].append(tuple(env.robot_pos[i]))
            if len(robot_trails[i]) > trail_len:
                robot_trails[i].pop(0)

        if terminated or truncated:
            episode_num += 1
            obs, info    = env.reset()
            robot_trails = [[] for _ in range(N_ROBOTS)]
            ep_stats = {'deliveries':0,'collisions':0,'safety_events':0}
            total_reward = 0.0

        # ── Draw main warehouse ───────────────────────────────────
        draw_warehouse(ax_main, env, robot_trails, frame, ep_stats)
        ax_main.set_title(
            'Warehouse Multi-Agent Coordination  -  Trained CTDE PPO Policy\n'
            'Mohammed Al-Khudhair  |  Honours Project  |  '
            'Habitat 3.0 (Puig et al., 2023)',
            color='#AAC4E0', fontsize=9, pad=6, fontfamily='monospace')

        # ── Draw curve inset + legend ─────────────────────────────
        draw_legend(ax_curve, log,
                    min(frame * 2000, int(log[-1]['steps'])))

        # ── Draw stats bar ────────────────────────────────────────
        draw_stats_bar(ax_stats, ep_stats, frame, FRAMES,
                       episode_num, total_reward)

        # ── Capture frame ─────────────────────────────────────────
        fig.canvas.draw()
        w_px, h_px = fig.canvas.get_width_height()
        raw = np.frombuffer(fig.canvas.tostring_rgb(),
                            dtype=np.uint8).reshape(h_px, w_px, 3)

        img  = Image.fromarray(raw)
        draw = ImageDraw.Draw(img, 'RGBA')
        prog = frame / FRAMES
        draw.rectangle([0, img.height-8, int(img.width*prog),
                         img.height], fill=(58,128,200,220))
        writer.append_data(np.array(img))

        if frame % (FPS*5) == 0:
            pct = int(frame/FRAMES*100)
            print(f"  {pct:3d}%  t={frame*DT:.0f}s  "
                  f"ep={episode_num}  "
                  f"del={ep_stats['deliveries']}  "
                  f"col={ep_stats['collisions']}")

    writer.close()
    plt.close(fig)
    env.close()

    print(f"\nDone! -> {OUTPUT_FILE}")
    print(f"Episodes completed : {episode_num-1}")
    print(f"Final deliveries   : {ep_stats['deliveries']}")


if __name__ == '__main__':
    render_demo()
