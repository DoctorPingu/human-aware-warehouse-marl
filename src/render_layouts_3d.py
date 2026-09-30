import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Patch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

from layouts import LAYOUTS
from warehouse_env import WarehouseEnv
from train import EVAL_SEED_BASE

C, S = math.cos(math.radians(30)), 0.5
VIEW = np.array([-1.0, -1.0]) / math.sqrt(2)
LIGHT = np.array([-0.35, -0.94])

H_WALL, H_RACK, H_PICK, H_PD = 2.4, 1.8, 0.9, 0.45
COL = dict(floor="#EDEFF2", wall="#C9CED6", receiving="#8FBEE8", shipping="#F2937A",
           staging="#A5D8C4", aisle="#B9C9DC", rack="#F5C24F", beam="#B8861E",
           charge="#5FC244", pick="#8A7DE0", pd="#2BA8C4", robot="#F2793A",
           human="#3C5A8C", zone="#3C5A8C", rroute="#F2793A", hroute="#3C5A8C")


def P(x, y, z=0.0):
    return ((x - y) * C, (x + y) * S + z)


def shade(col, f):
    r, g, b = to_rgb(col)
    return (min(1, r * f), min(1, g * f), min(1, b * f)) if f <= 1 else \
        tuple(c + (1 - c) * (f - 1) for c in (r, g, b))


def poly3(ax, pts, col, z=0.0, **kw):
    ax.add_patch(Polygon([P(x, y, z) for x, y in pts], closed=True, facecolor=col,
                         edgecolor=kw.pop("ec", col), lw=kw.pop("lw", 0.4), **kw))


def circle_pts(cx, cy, r, n=24):
    return [(cx + r * math.cos(t), cy + r * math.sin(t))
            for t in np.linspace(0, 2 * math.pi, n, endpoint=False)]


def prism(ax, pts, h, col, beams=None, z0=0.0, top_col=None):
    n = len(pts)
    for i in range(n):
        a, b = np.array(pts[i]), np.array(pts[(i + 1) % n])
        e = b - a
        nrm = np.array([e[1], -e[0]]) / (np.hypot(*e) + 1e-12)
        if nrm @ VIEW <= 1e-9:
            continue
        f = 0.72 + 0.22 * max(0.0, nrm @ LIGHT)
        quad = [P(*a, z0), P(*b, z0), P(*b, z0 + h), P(*a, z0 + h)]
        ax.add_patch(Polygon(quad, closed=True, facecolor=shade(col, f),
                             edgecolor=shade(col, f), lw=0.4))
        for zb in (beams or []):
            ax.plot(*zip(P(*a, z0 + zb), P(*b, z0 + zb)), color=COL["beam"], lw=0.5)
    poly3(ax, pts, top_col or shade(col, 1.12), z=z0 + h)


def split_rect(pts, max_len=1.0):
    p = [np.array(q, float) for q in pts]
    if np.hypot(*(p[1] - p[0])) < np.hypot(*(p[2] - p[1])):
        p = p[1:] + p[:1]
    k = max(1, math.ceil(np.hypot(*(p[1] - p[0])) / max_len))
    out = []
    for i in range(k):
        a, b = p[0] + (p[1] - p[0]) * i / k, p[0] + (p[1] - p[0]) * (i + 1) / k
        d, c = p[3] + (p[2] - p[3]) * i / k, p[3] + (p[2] - p[3]) * (i + 1) / k
        out.append([tuple(a), tuple(b), tuple(c), tuple(d)])
    return out


def rect_pts(r):
    x, y, w, d = r
    return [(x, y), (x + w, y), (x + w, y + d), (x, y + d)]


def render(name, ax):
    env = WarehouseEnv(name, "standard", "high")
    env.reset_arrays(seed=EVAL_SEED_BASE)
    lay = env.layout
    ax.set_aspect("equal")
    ax.axis("off")

    poly3(ax, lay.footprint, COL["floor"], ec="#9AA0A8", lw=1.0)
    for r, c in ((lay.receiving, "receiving"), (lay.shipping, "shipping"),
                 (lay.staging, "staging")):
        poly3(ax, rect_pts(r), COL[c])
    for r in lay.aisles:
        poly3(ax, rect_pts(r), COL["aisle"])
    for (px, py) in lay.charge_pads:
        poly3(ax, rect_pts((px - 0.75, py - 0.75, 1.5, 1.5)), COL["charge"], alpha=0.9)
    for route in lay.human_routes:
        r = list(route) + [route[0]]
        ax.plot(*zip(*[P(x, y, 0.01) for x, y in r]), color=COL["hroute"], lw=0.8,
                ls=":", alpha=0.75)
    for route in lay.robot_tasks:
        ax.plot(*zip(*[P(x, y, 0.01) for x, y in route]), color=COL["rroute"], lw=0.9,
                alpha=0.6)
    for h in range(env.N_HUMANS):
        poly3(ax, circle_pts(*env._hpos[h], env.SAFETY_R, 40), "none",
              ec=COL["zone"], lw=0.7, ls="--", alpha=0.7)

    items = []
    fp = lay.footprint
    for i in range(len(fp)):
        a, b = np.array(fp[i]), np.array(fp[(i + 1) % len(fp)])
        e = b - a
        nrm = np.array([e[1], -e[0]]) / np.hypot(*e)
        facing = nrm @ VIEW > 0
        h = 0.15 if facing else H_WALL
        k = max(1, math.ceil(np.hypot(*e)))
        for j in range(k):
            p, q = a + e * j / k, a + e * (j + 1) / k
            t = nrm * 0.12
            pts = [tuple(p), tuple(q), tuple(q + t), tuple(p + t)]
            if not facing:
                pts = [tuple(p + t), tuple(q + t), tuple(q), tuple(p)][::-1]
                pts = [tuple(p), tuple(p + t), tuple(q + t), tuple(q)]
            d = (p + q)[0] / 2 + (p + q)[1] / 2 + (0.3 if not facing else -0.3)
            items.append((d, lambda pts=pts, h=h: prism(ax, _ccw(pts), h, COL["wall"])))
    for shelf in lay.shelves:
        for piece in split_rect(shelf):
            cx, cy = np.mean(piece, axis=0)
            items.append((cx + cy, lambda pc=piece: prism(
                ax, pc, H_RACK, COL["rack"], beams=[0.45, 0.9, 1.35])))
    for (px, py) in lay.pick_stations:
        items.append((px + py, lambda px=px, py=py: prism(
            ax, rect_pts((px - 0.75, py - 0.55, 1.5, 1.1)), H_PICK, COL["pick"])))
    for (px, py) in lay.pd_points:
        items.append((px + py, lambda px=px, py=py: prism(
            ax, rect_pts((px - 0.8, py - 0.8, 1.6, 1.6)), H_PD, COL["pd"])))
    for i in range(env.N_ROBOTS):
        x, y = env._rpos[i]

        def draw_robot(x=x, y=y, i=i):
            prism(ax, circle_pts(x, y, env.ROBOT_R + 0.1, 16), 0.35, COL["robot"])
            ax.text(*P(x, y, 0.36), str(i + 1), ha="center", va="center", fontsize=6,
                    color="white", fontweight="bold")
        items.append((x + y, draw_robot))
    for h in range(env.N_HUMANS):
        x, y = env._hpos[h]

        def draw_human(x=x, y=y):
            prism(ax, circle_pts(x, y, env.HUMAN_R, 14), 1.45, COL["human"])
            prism(ax, circle_pts(x, y, 0.13, 14), 0.24, shade(COL["human"], 1.25), z0=1.5)
        items.append((x + y, draw_human))

    for _, fn in sorted(items, key=lambda t: -t[0]):
        fn()

    xs = [P(x, y)[0] for x, y in fp]
    ys = [P(x, y)[1] for x, y in fp]
    ax.set_xlim(min(xs) - 1, max(xs) + 1)
    ax.set_ylim(min(ys) - 1, max(ys) + H_WALL + 1)
    ax.set_title(f"{lay.label}   ({lay.width:g} x {lay.height:g} m, {len(lay.shelves)} racks, "
                 f"{env.N_ROBOTS} robots, {env.N_HUMANS} humans at high density)", fontsize=10)
    env.close()


def _ccw(pts):
    a = sum(pts[i][0] * pts[(i + 1) % len(pts)][1] - pts[(i + 1) % len(pts)][0] * pts[i][1]
            for i in range(len(pts)))
    return pts if a > 0 else pts[::-1]


def legend_items():
    return [Patch(color=COL["receiving"], label="Receiving"),
            Patch(color=COL["shipping"], label="Shipping"),
            Patch(color=COL["staging"], label="Staging (fleet start)"),
            Patch(color=COL["aisle"], label="Main aisle"),
            Patch(color=COL["rack"], label="Rack (1.8 m)"),
            Patch(color=COL["charge"], label="Charge pad"),
            Patch(color=COL["pick"], label="Pick station"),
            Patch(color=COL["pd"], label="Pick-up / drop-off point"),
            Line2D([], [], color=COL["rroute"], marker="o", lw=0.9, label="Robot + task route"),
            Line2D([], [], color=COL["hroute"], marker="o", ls=":",
                   label="Human + patrol route (0.8 m ISO zone dashed)")]


if __name__ == "__main__":
    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "figures", "layouts_3d")
    os.makedirs(out, exist_ok=True)
    names = list(LAYOUTS)
    for n in names:
        fig, ax = plt.subplots(figsize=(12, 8))
        render(n, ax)
        fig.legend(handles=legend_items(), loc="lower center", ncol=5, fontsize=8, frameon=False)
        fig.tight_layout(rect=(0, 0.07, 1, 1))
        fig.savefig(f"{out}/{n}_3d.png", dpi=150)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(17, 19))
    for ax, n in zip(axes.flat, names):
        render(n, ax)
    fig.legend(handles=legend_items(), loc="lower center", ncol=5, fontsize=9, frameon=False)
    fig.suptitle("The six warehouse layouts in 3D, built from the simulation's own data",
                 fontsize=13)
    fig.tight_layout(rect=(0, 0.03, 1, 0.97))
    fig.savefig(f"{out}/all_layouts_3d.png", dpi=110)
    print(f"Wrote 7 images to {out}")
