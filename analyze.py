from __future__ import annotations

import argparse
import glob
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS: Dict[str, tuple] = {
    "orders_per_hour": ("Orders per hour", True),
    "congestion_prop": ("Congestion time proportion", False),
    "collision_rate_100": ("Collisions per 100 timesteps", False),
    "violations_per_interaction": ("Separation violations per interaction", False),
}
TRAVEL = ("distance_per_delivery", "Metres driven per delivery")
SUPPLEMENTARY = {"distance_per_delivery": ("Metres driven per delivery", False),
                 "min_separation": ("Minimum separation (m)", True),
                 "collisions_rh": ("Robot-human contacts", False),
                 "collisions_rr": ("Robot-robot contacts", False)}

COLORS = {"orca": "#6C8EBF", "heuristic": "#9E9E9E", "mapf": "#9673A6",
          "ppo": "#F2793A", "ppo_v1": "#F2B880", "ppo_v2": "#F2793A",
          "ppo_full": "#F2793A", "ppo_throughput_only": "#E3B23C",
          "ppo_safety_only": "#4F9D69", "ppo_transfer": "#C75B39",
          "ppo_mixed": "#F2793A", "ppo_std_only": "#4F7FBF", "ppo_direct": "#F2793A",
          "ppo_curriculum": "#4F9D69", "ppo_3m": "#F7C4A5", "ppo_6m": "#F49C68",
          "ppo_10m": "#F2793A"}
LABELS = {"orca": "ORCA/RVO", "heuristic": "Heuristic", "mapf": "MAPF",
          "ppo": "CTDE PPO", "ppo_v1": "PPO (v1 reward)", "ppo_v2": "PPO (v2 reward)",
          "ppo_full": "PPO (full reward)", "ppo_throughput_only": "PPO (throughput only)",
          "ppo_safety_only": "PPO (safety only)", "ppo_transfer": "PPO (transferred)",
          "ppo_mixed": "PPO (mixed-demand training)",
          "ppo_std_only": "PPO (standard-demand training)",
          "ppo_direct": "PPO (trained directly)", "ppo_curriculum": "PPO (curriculum)",
          "ppo_3m": "PPO at 3M steps", "ppo_6m": "PPO at 6M steps",
          "ppo_10m": "PPO at 10M steps"}
SHORT = {"small_1_parallel_aisle": "Small 1", "small_2_l_shaped": "Small 2",
         "medium_1_twin_block": "Medium 1", "medium_2_fishbone": "Medium 2",
         "large_1_fulfilment": "Large 1", "large_2_robotic_grid": "Large 2"}
LAYOUT_ORDER = list(SHORT)
PERTURB_ORDER = ["nominal", "speed_x0.8", "speed_x1.2", "avoid_0.30m", "avoid_0.40m"]
PERTURB_LABEL = {"nominal": "Nominal", "speed_x0.8": "Humans 20% slower",
                 "speed_x1.2": "Humans 20% faster", "avoid_0.30m": "Avoidance radius 0.30 m",
                 "avoid_0.40m": "Avoidance radius 0.40 m"}
COND_ORDER = [("standard", "standard"), ("standard", "high"),
              ("peak", "standard"), ("peak", "high")]

B = 5000
RNG = np.random.default_rng(0)


def boot_ci(x: np.ndarray, fn=np.mean, n: int = B) -> tuple:
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return np.nan, np.nan, np.nan
    if np.all(x == x[0]):
        return float(x[0]), float(x[0]), float(x[0])
    idx = RNG.integers(0, x.size, (n, x.size))
    samples = np.apply_along_axis(fn, 1, x[idx]) if fn is not np.mean else x[idx].mean(1)
    return float(fn(x)), float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))


def iqm(x) -> float:
    return float(stats.trim_mean(np.asarray(x, float), 0.25))


def stratified_iqm_ci(groups: List[np.ndarray], n: int = 2000) -> tuple:
    groups = [g[np.isfinite(g)] for g in groups if len(g)]
    point = iqm(np.concatenate(groups))
    boots = np.empty(n)
    for b in range(n):
        boots[b] = iqm(np.concatenate([g[RNG.integers(0, g.size, g.size)] for g in groups]))
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def mannwhitney(ppo: np.ndarray, base: np.ndarray, higher_better: bool) -> tuple:
    ppo, base = ppo[np.isfinite(ppo)], base[np.isfinite(base)]
    if ppo.size == 0 or base.size == 0:
        return np.nan, np.nan
    if np.all(ppo == ppo[0]) and np.all(base == ppo[0]):
        return 1.0, 0.0
    u, p = stats.mannwhitneyu(ppo, base, alternative="two-sided")
    r = 2.0 * u / (ppo.size * base.size) - 1.0
    return float(p), float(r if higher_better else -r)


def holm(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    out = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    pv = p[ok]
    order = np.argsort(pv)
    m = pv.size
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * pv[i]))
        adj[i] = running
    out[ok] = adj
    return out


def condition_table(df: pd.DataFrame, metrics: Dict[str, tuple]) -> pd.DataFrame:
    rows = []
    for (lay, dem, den, algo), g in df.groupby(["layout", "demand", "density", "algo"]):
        row = {"layout": lay, "demand": dem, "density": den, "algo": algo, "n": len(g)}
        for m in metrics:
            mean, lo, hi = boot_ci(g[m].to_numpy())
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = mean, lo, hi
        rows.append(row)
    return pd.DataFrame(rows)


def significance(df: pd.DataFrame, focal: str = "ppo") -> pd.DataFrame:
    rows = []
    baselines = [a for a in df["algo"].unique() if a != focal]
    for (lay, dem, den), g in df.groupby(["layout", "demand", "density"]):
        if focal not in set(g["algo"]):
            continue
        f = g[g["algo"] == focal]
        for base in baselines:
            b = g[g["algo"] == base]
            if b.empty:
                continue
            for m, (_, hib) in METRICS.items():
                p, r = mannwhitney(f[m].to_numpy(float), b[m].to_numpy(float), hib)
                rows.append({"layout": lay, "demand": dem, "density": den,
                             "baseline": base, "metric": m, "p": p, "effect_r": r,
                             "ppo_mean": f[m].mean(), "base_mean": b[m].mean()})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out["p_holm"] = np.nan
    for m in METRICS:
        mask = out["metric"] == m
        out.loc[mask, "p_holm"] = holm(out.loc[mask, "p"].to_numpy())
    out["significant"] = out["p_holm"] < 0.05
    out["ppo_better"] = out["significant"] & (out["effect_r"] > 0)
    out["ppo_worse"] = out["significant"] & (out["effect_r"] < 0)
    return out


UNIT_KEYS = ["suite", "algo", "tag", "train_layout", "layout", "demand", "density",
             "perturb", "unit"]


def to_units(df: pd.DataFrame) -> pd.DataFrame:
    if "unit" not in df.columns:
        out = df.copy()
        out["n_episodes"] = 1
        return out
    keys = [k for k in UNIT_KEYS if k in df.columns]
    num = [c for c in df.columns if c not in keys and pd.api.types.is_numeric_dtype(df[c])
           and c not in ("episode", "env_seed", "seconds")]
    g = df.groupby(keys, dropna=False)
    out = g[num].mean().reset_index()
    out["n_episodes"] = g.size().to_numpy()
    return out


def prob_improvement(df: pd.DataFrame, focal: str, base: str, metric: str,
                     higher_better: bool, n_boot: int = 2000) -> Tuple[float, float, float]:
    cells = []
    for _, g in df.groupby(["layout", "demand", "density"]):
        x = g[g["algo"] == focal][metric].to_numpy(float)
        y = g[g["algo"] == base][metric].to_numpy(float)
        x, y = x[np.isfinite(x)], y[np.isfinite(y)]
        if x.size and y.size:
            cells.append((x, y) if higher_better else (-x, -y))
    if not cells:
        return float("nan"), float("nan"), float("nan")

    def score(pairs):
        return float(np.mean([np.mean((x[:, None] > y[None, :]) + 0.5 * (x[:, None] == y[None, :]))
                              for x, y in pairs]))

    rng = np.random.default_rng(0)
    boots = [score([(rng.choice(x, x.size), rng.choice(y, y.size)) for x, y in cells])
             for _ in range(n_boot)]
    return score(cells), float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for algo, g in df.groupby("algo"):
        row = {"algo": algo}
        for m in METRICS:
            groups = [c[m].to_numpy(float) for _, c in g.groupby(["layout", "demand", "density"])]
            row[m], row[f"{m}_lo"], row[f"{m}_hi"] = stratified_iqm_ci(groups)
        rows.append(row)
    return pd.DataFrame(rows)


def to_markdown(tab: pd.DataFrame, metrics: Dict[str, tuple]) -> str:
    head = "| Layout | Demand | Density | Algorithm | " + " | ".join(
        v[0] for v in metrics.values()) + " |"
    lines = [head, "|" + "---|" * (4 + len(metrics))]
    for _, r in tab.sort_values(["layout", "demand", "density", "algo"]).iterrows():
        cells = [f"{r[m]:.2f} [{r[f'{m}_lo']:.2f}, {r[f'{m}_hi']:.2f}]" for m in metrics]
        lines.append(f"| {SHORT.get(r['layout'], r['layout'])} | {r['demand']} | "
                     f"{r['density']} | {LABELS.get(r['algo'], r['algo'])} | "
                     + " | ".join(cells) + " |")
    return "\n".join(lines)


def _style(ax):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.grid(axis="y", alpha=0.3)


def fig_by_layout(tab: pd.DataFrame, metric: str, algos: List[str], path: str):
    label, _ = {**METRICS, **SUPPLEMENTARY}[metric]
    lays = [l for l in LAYOUT_ORDER if l in set(tab["layout"])]
    fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharey=False)
    width = 0.8 / max(1, len(algos))
    for ax, lay in zip(axes.flat, lays):
        sub = tab[tab["layout"] == lay]
        for k, algo in enumerate(algos):
            ys, lo, hi = [], [], []
            for dem, den in COND_ORDER:
                r = sub[(sub["algo"] == algo) & (sub["demand"] == dem) & (sub["density"] == den)]
                if r.empty:
                    ys.append(np.nan); lo.append(0); hi.append(0); continue
                r = r.iloc[0]
                ys.append(r[metric]); lo.append(r[metric] - r[f"{metric}_lo"])
                hi.append(r[f"{metric}_hi"] - r[metric])
            x = np.arange(len(COND_ORDER)) + (k - (len(algos) - 1) / 2) * width
            ax.bar(x, ys, width, yerr=[lo, hi], capsize=2, color=COLORS.get(algo, "#777"),
                   label=LABELS.get(algo, algo))
        ax.set_title(SHORT[lay])
        ax.set_xticks(range(len(COND_ORDER)))
        ax.set_xticklabels([f"{d}\n{s}" for d, s in COND_ORDER], fontsize=8)
        _style(ax)
    for ax in axes.flat[len(lays):]:
        ax.axis("off")
    axes.flat[0].set_ylabel(label)
    axes.flat[3].set_ylabel(label)
    handles, labs = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="upper center", ncol=len(algos), frameon=False)
    fig.suptitle(f"{label} by layout and condition (demand / density), mean with 95% CI",
                 y=0.93, fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_tradeoff(tab: pd.DataFrame, algos: List[str], path: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for algo in algos:
        s = tab[tab["algo"] == algo]
        ax.scatter(s["orders_per_hour"].to_numpy(float),
                   s["violations_per_interaction"].to_numpy(float), s=36, alpha=0.8,
                   color=COLORS.get(algo, "#777"), label=LABELS.get(algo, algo))
    ax.set_xlabel("Orders per hour  (higher is better)")
    ax.set_ylabel("Separation violations per interaction  (lower is better)")
    ax.set_title("Throughput-safety trade-off, one point per condition\n"
                 "Better is towards the bottom right")
    ax.legend(frameon=False)
    _style(ax)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_aggregate(agg: pd.DataFrame, algos: List[str], path: str):
    fig, axes = plt.subplots(1, len(METRICS), figsize=(16, 3.2))
    for ax, (m, (label, hib)) in zip(axes, METRICS.items()):
        for k, algo in enumerate(algos):
            r = agg[agg["algo"] == algo]
            if r.empty:
                continue
            r = r.iloc[0]
            ax.plot([r[f"{m}_lo"], r[f"{m}_hi"]], [k, k], lw=6, solid_capstyle="butt",
                    color=COLORS.get(algo, "#777"), alpha=0.7)
            ax.plot(r[m], k, "|", color="black", ms=14)
        ax.set_yticks(range(len(algos)))
        ax.set_yticklabels([LABELS.get(a, a) for a in algos] if ax is axes[0] else [])
        ax.set_title(f"{label}\n({'higher' if hib else 'lower'} is better)", fontsize=9)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
    fig.suptitle("Aggregate over all 24 conditions: IQM with 95% stratified bootstrap CI",
                 fontsize=10, y=1.05)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def fig_training(runs_dir: str, tag: str, path: str):
    files = glob.glob(os.path.join(runs_dir, tag, "*", "seed*", "progress.csv"))
    if not files:
        return False
    by_layout: Dict[str, List[pd.DataFrame]] = {}
    for f in files:
        lay = f.split(os.sep)[-3]
        d = pd.read_csv(f)
        by_layout.setdefault(lay, []).append(d)
    keys = [("orders_per_hour", "Orders per hour"),
            ("violations_per_interaction", "Violations per interaction"),
            ("collision_rate_100", "Collisions per 100 steps")]
    lays = [l for l in LAYOUT_ORDER if l in by_layout]
    fig, axes = plt.subplots(len(keys), len(lays), figsize=(max(7.0, 3.2 * len(lays)), 8),
                             squeeze=False)
    for j, lay in enumerate(lays):
        for i, (k, lab) in enumerate(keys):
            ax = axes[i, j]
            curves = []
            for d in by_layout[lay]:
                d = d.dropna(subset=[k])
                if d.empty:
                    continue
                steps = d["step"].to_numpy(float)
                y = d[k].rolling(10, min_periods=1).mean().to_numpy(float)
                ax.plot(steps, y, color=COLORS["ppo"], alpha=0.25, lw=0.8,
                        marker="o" if len(steps) < 20 else None, ms=3)
                curves.append(pd.Series(y, index=steps))
            if curves:
                grid = np.linspace(max(c.index.min() for c in curves),
                                   min(c.index.max() for c in curves), 100)
                ys = np.array([np.interp(grid, c.index, c.to_numpy()) for c in curves])
                ax.plot(grid, ys.mean(0), color=COLORS["ppo"], lw=2)
            if i == 0:
                ax.set_title(SHORT.get(lay, lay))
            if j == 0:
                ax.set_ylabel(lab)
            if i == len(keys) - 1:
                ax.set_xlabel("Environment steps")
            _style(ax)
    fig.suptitle(f"Training curves ({tag}): thin lines are seeds, thick line is the mean",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_transfer(df: pd.DataFrame, metric: str, path: str, ref: pd.Series = None):
    t = df[df["density"] == "standard"]
    t = t[t["algo"].isin(["ppo", "ppo_transfer"])]
    if t.empty:
        return
    piv = t.groupby(["train_layout", "layout"])[metric].mean().unstack()
    lays = [l for l in LAYOUT_ORDER if l in piv.index or l in piv.columns]
    piv = piv.reindex(index=[l for l in lays if l in piv.index],
                      columns=[l for l in lays if l in piv.columns])
    n_train = piv.shape[0]
    if ref is not None and not ref.empty:
        piv.loc["orca_reference"] = [ref.get(c, np.nan) for c in piv.columns]
    label, hib = METRICS[metric]
    fig, ax = plt.subplots(figsize=(7.5, 6))
    im = ax.imshow(piv.to_numpy(float), cmap="viridis" if hib else "viridis_r")
    ax.set_xticks(range(piv.shape[1]))
    ax.set_xticklabels([SHORT[c] for c in piv.columns], rotation=30)
    ax.set_yticks(range(piv.shape[0]))
    ax.set_yticklabels([SHORT.get(c, "ORCA (reference)") for c in piv.index])
    if piv.shape[0] > n_train:
        ax.axhline(n_train - 0.5, color="white", lw=4)
    ax.set_xlabel("Evaluated on")
    ax.set_ylabel("Trained on")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.iat[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}" if v < 10 else f"{v:.0f}",
                        ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(im, ax=ax, label=label)
    ax.set_title(f"Cross-layout transfer: {label}\nDiagonal = in-distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def fig_sensitivity(d: pd.DataFrame, algos: List[str], path: str):
    levels = [p for p in PERTURB_ORDER if p in set(d["perturb"])]
    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 4.2))
    x = np.arange(len(levels))
    for ax, (m, (label, hib)) in zip(axes, METRICS.items()):
        for k, algo in enumerate(algos):
            ys, lo, hi = [], [], []
            for p in levels:
                v, l, h = boot_ci(d[(d["algo"] == algo) & (d["perturb"] == p)][m].to_numpy(float))
                ys.append(v); lo.append(v - l); hi.append(h - v)
            off = (k - (len(algos) - 1) / 2) * 0.08
            ax.errorbar(x + off, ys, yerr=[lo, hi], marker="o", ms=4, capsize=2, lw=1.3,
                        color=COLORS.get(algo, "#777"), label=LABELS.get(algo, algo))
        ax.set_xticks(x)
        ax.set_xticklabels([PERTURB_LABEL[p].replace(" ", "\n", 1) for p in levels], fontsize=7)
        ax.set_title(f"{label}\n({'higher' if hib else 'lower'} is better)", fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=7)
    fig.suptitle("Sensitivity to human behaviour (Small 1 and Large 1, high density), "
                 "mean with 95% CI", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def analyse_sensitivity(df: pd.DataFrame, out: str, lines: List[str]) -> None:
    d = df.copy()
    d["density"] = d["perturb"]
    algos = [a for a in ["orca", "heuristic", "mapf", "ppo"] if a in set(d["algo"])]
    levels = [p for p in PERTURB_ORDER if p in set(d["perturb"])]
    condition_table(d, {k: v for k, v in {**METRICS, **SUPPLEMENTARY}.items()
                        if k in d.columns}).to_csv(
        os.path.join(out, "sensitivity_table.csv"), index=False)
    fig_sensitivity(d, algos, os.path.join(out, "sensitivity.png"))

    lines.append("\n## Suite: sensitivity\n")
    lines.append(f"{int(d['n_episodes'].sum())} episodes in {len(d)} seed units. Human walking speed and avoidance radius varied at "
                 f"evaluation time only; policies were trained with the nominal values. "
                 f"Layouts: {', '.join(SHORT[l] for l in sorted(set(d['layout']), key=LAYOUT_ORDER.index))}; "
                 f"standard demand, high density. Cells: mean over both layouts "
                 f"(absolute change against nominal).\n")
    for m, (label, hib) in METRICS.items():
        lines.append(f"{label} ({'higher' if hib else 'lower'} is better):\n")
        lines.append("| Algorithm | " + " | ".join(PERTURB_LABEL[p] for p in levels) + " |")
        lines.append("|---|" + "---|" * len(levels))
        for a_ in algos:
            base = d[(d["algo"] == a_) & (d["perturb"] == "nominal")][m].mean()
            cells = []
            for p in levels:
                v = d[(d["algo"] == a_) & (d["perturb"] == p)][m].mean()
                cells.append(f"{v:.2f}" if p == "nominal" else f"{v:.2f} ({v - base:+.2f})")
            lines.append(f"| {LABELS.get(a_, a_)} | " + " | ".join(cells) + " |")
        lines.append("")
    if "ppo" in set(d["algo"]):
        sig = significance(d, "ppo")
        sig.to_csv(os.path.join(out, "sensitivity_significance.csv"), index=False)
        lines.append("Does CTDE PPO's position against each baseline hold under each change? "
                     "Conditions (layout x level) where PPO is significantly better / worse / "
                     "not different (Mann-Whitney U, Holm-corrected):\n")
        lines.append("| Metric | Baseline | " + " | ".join(PERTURB_LABEL[p] for p in levels) + " |")
        lines.append("|---|---|" + "---|" * len(levels))
        for m, (label, _) in METRICS.items():
            for base in sorted(sig["baseline"].unique()):
                cells = []
                for p in levels:
                    s = sig[(sig["metric"] == m) & (sig["baseline"] == base) & (sig["density"] == p)]
                    b, w = int(s["ppo_better"].sum()), int(s["ppo_worse"].sum())
                    cells.append(f"{b} / {w} / {len(s) - b - w}")
                lines.append(f"| {label} | {LABELS.get(base, base)} | " + " | ".join(cells) + " |")


def analyse_progress(df: pd.DataFrame, main_df: pd.DataFrame, out: str,
                     lines: List[str], final_steps: int) -> None:
    parts = [df]
    if main_df is not None:
        final = main_df[main_df["algo"] == "ppo"].copy()
        final["algo"] = f"ppo_{final_steps / 1e6:g}m"
        parts += [final, main_df[main_df["algo"] == "orca"]]
    d = pd.concat(parts, ignore_index=True)
    snaps = sorted({x for x in d["algo"] if x.startswith("ppo_") and x.endswith("m")},
                   key=lambda s: float(s[4:-1]))
    order = snaps + (["orca"] if "orca" in set(d["algo"]) else [])
    agg = aggregate(d).set_index("algo").loc[order].reset_index()
    agg.to_csv(os.path.join(out, "progress_aggregate.csv"), index=False)

    fig, axes = plt.subplots(1, len(METRICS), figsize=(4.2 * len(METRICS), 3.8))
    snaps = [x for x in order if x != "orca"]
    xs = [float(s[4:-1]) for s in snaps]
    for ax, (m, (label, hib)) in zip(axes, METRICS.items()):
        r = agg.set_index("algo")
        ys = [r.loc[s, m] for s in snaps]
        lo = [r.loc[s, m] - r.loc[s, f"{m}_lo"] for s in snaps]
        hi = [r.loc[s, f"{m}_hi"] - r.loc[s, m] for s in snaps]
        ax.errorbar(xs, ys, yerr=[lo, hi], marker="o", capsize=3, color=COLORS["ppo"],
                    label="CTDE PPO")
        if "orca" in r.index:
            ax.axhline(r.loc["orca", m], color=COLORS["orca"], ls="--", label="ORCA/RVO")
        ax.set_xticks(xs)
        ax.set_xlabel("Training steps (millions)")
        ax.set_title(f"{label}\n({'higher' if hib else 'lower'} is better)", fontsize=9)
        _style(ax)
    axes[0].legend(fontsize=7)
    fig.suptitle("Learning progress: the same runs evaluated at "
                 + ", ".join(f"{x:g}M" for x in xs) + " steps (IQM over all conditions, 95% CI)",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "progress.png"), dpi=160)
    plt.close(fig)

    lines.append("\n## Suite: progress\n")
    lines.append("The main policies evaluated at snapshots taken during their 10M-step runs. "
                 "The learning rate schedule spans the full run, so a snapshot shows the "
                 "run part-way through, not a policy trained with a smaller budget. "
                 "IQM [95% CI] over all 24 conditions:\n")
    lines.append("| Policy | " + " | ".join(v[0] for v in METRICS.values()) + " |")
    lines.append("|---|" + "---|" * len(METRICS))
    for _, row in agg.iterrows():
        name = (f"PPO at {float(row['algo'][4:-1]):g}M steps"
                if row["algo"].startswith("ppo_") else LABELS.get(row["algo"], row["algo"]))
        lines.append(f"| {name} | " + " | ".join(
            f"{row[m]:.2f} [{row[f'{m}_lo']:.2f}, {row[f'{m}_hi']:.2f}]" for m in METRICS) + " |")


def analyse_comparison(df: pd.DataFrame, suite: str, out: str, focal: str,
                       lines: List[str]) -> None:
    algos = [a for a in ["orca", "heuristic", "mapf", "ppo", "ppo_v1", "ppo_v2",
                         "ppo_full", "ppo_throughput_only", "ppo_safety_only",
                         "ppo_mixed", "ppo_std_only", "ppo_direct", "ppo_curriculum"]
             if a in set(df["algo"])]
    metrics = {k: v for k, v in {**METRICS, **SUPPLEMENTARY}.items() if k in df.columns}
    tab = condition_table(df, metrics)
    tab.to_csv(os.path.join(out, f"{suite}_table.csv"), index=False)
    with open(os.path.join(out, f"{suite}_table.md"), "w") as f:
        f.write(to_markdown(tab, METRICS))
    for m in METRICS:
        fig_by_layout(tab, m, algos, os.path.join(out, f"{suite}_{m}.png"))
    fig_tradeoff(tab, algos, os.path.join(out, f"{suite}_tradeoff.png"))
    agg = aggregate(df)
    agg.to_csv(os.path.join(out, f"{suite}_aggregate.csv"), index=False)
    fig_aggregate(agg, algos, os.path.join(out, f"{suite}_aggregate.png"))

    lines.append(f"\n## Suite: {suite}\n")
    lines.append(f"{int(df['n_episodes'].sum())} episodes averaged into {len(df)} seed units, "
                 f"{df.groupby(['layout','demand','density']).ngroups} conditions, algorithms: "
                 f"{', '.join(LABELS.get(a, a) for a in algos)}. Statistics treat each seed "
                 f"unit (the mean of its episodes in one condition) as one observation.\n")
    lines.append("Aggregate over all conditions (IQM [95% CI]):\n")
    lines.append("| Algorithm | " + " | ".join(v[0] for v in METRICS.values()) + " |")
    lines.append("|---|" + "---|" * len(METRICS))
    for a in algos:
        r = agg[agg["algo"] == a]
        if r.empty:
            continue
        r = r.iloc[0]
        lines.append(f"| {LABELS.get(a, a)} | " + " | ".join(
            f"{r[m]:.2f} [{r[f'{m}_lo']:.2f}, {r[f'{m}_hi']:.2f}]" for m in METRICS) + " |")

    if TRAVEL[0] in df.columns:
        lines.append(f"\n{TRAVEL[1]} (travel efficiency, secondary KPI, lower is better; "
                     f"IQM [95% CI], seed units with no deliveries excluded):\n")
        lines.append("| Algorithm | Metres per delivery | Units excluded |")
        lines.append("|---|---|---|")
        for a_ in algos:
            g = df[df["algo"] == a_]
            groups = [c[TRAVEL[0]].to_numpy(float) for _, c in g.groupby(["layout", "demand", "density"])]
            groups = [x[np.isfinite(x)] for x in groups]
            groups = [x for x in groups if x.size]
            excl = int((~np.isfinite(g[TRAVEL[0]].to_numpy(float))).sum())
            if not groups:
                lines.append(f"| {LABELS.get(a_, a_)} | none (no deliveries) | {excl} |")
                continue
            v, lo, hi = stratified_iqm_ci(groups)
            lines.append(f"| {LABELS.get(a_, a_)} | {v:.1f} [{lo:.1f}, {hi:.1f}] | {excl} |")

    if focal in set(df["algo"]):
        sig = significance(df, focal)
        sig.to_csv(os.path.join(out, f"{suite}_significance.csv"), index=False)
        lines.append(f"\nHead-to-head, {LABELS.get(focal, focal)} vs each baseline "
                     f"(Mann-Whitney U, Holm-corrected, alpha 0.05). "
                     f"Counts are conditions out of those tested:\n")
        lines.append("| Metric | Baseline | Better | Worse | No difference |")
        lines.append("|---|---|---|---|---|")
        for m, (label, _) in METRICS.items():
            for base in sorted(sig["baseline"].unique()):
                s = sig[(sig["metric"] == m) & (sig["baseline"] == base)]
                lines.append(f"| {label} | {LABELS.get(base, base)} | {int(s['ppo_better'].sum())} | "
                             f"{int(s['ppo_worse'].sum())} | "
                             f"{int(len(s) - s['ppo_better'].sum() - s['ppo_worse'].sum())} |")
        lines.append(f"\nProbability of improvement: the chance that a random seed unit of "
                     f"{LABELS.get(focal, focal)} beats a random unit of the baseline in the same "
                     f"condition, averaged over conditions (0.5 means no difference; Agarwal et "
                     f"al., 2021). Mean [95% CI]:\n")
        lines.append("| Baseline | " + " | ".join(v[0] for v in METRICS.values()) + " |")
        lines.append("|---|" + "---|" * len(METRICS))
        for base in [b for b in algos if b != focal]:
            cells = []
            for m, (label, hib) in METRICS.items():
                v, lo, hi = prob_improvement(df, focal, base, m, hib)
                cells.append(f"{v:.2f} [{lo:.2f}, {hi:.2f}]")
            lines.append(f"| {LABELS.get(base, base)} | " + " | ".join(cells) + " |")


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--suite", default="all")
    p.add_argument("--results", default="results")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--final-steps", type=int, default=10_000_000)
    a = p.parse_args(argv)
    out = os.path.join(a.results, "analysis")
    os.makedirs(out, exist_ok=True)
    lines = ["# Results summary\n",
             "Generated by analyze.py. Numbers are means or IQMs with 95% bootstrap "
             "confidence intervals. Copy into the thesis with care: check each figure.\n"]

    suites = (["smoke", "main", "calibration", "ablation", "generalisation", "sensitivity",
               "robot_stress", "demand", "curriculum", "progress"]
              if a.suite == "all" else [a.suite])
    for suite in suites:
        path = os.path.join(a.results, f"{suite}.csv")
        if not os.path.exists(path):
            continue
        raw = pd.read_csv(path)
        df = to_units(raw)
        print(f"Analysing {suite}: {len(raw)} episodes, {len(df)} seed units")
        if suite == "generalisation":
            stress = df[df["density"] == "stress"]
            if not stress.empty:
                analyse_comparison(stress, "generalisation_stress", out, "ppo", lines)
            ref = {}
            mp = os.path.join(a.results, "main.csv")
            if os.path.exists(mp):
                mn = to_units(pd.read_csv(mp))
                mn = mn[(mn["algo"] == "orca") & (mn["demand"] == "standard")
                        & (mn["density"] == "standard")]
                ref = {m: mn.groupby("layout")[m].mean()
                       for m in ["orders_per_hour", "violations_per_interaction"]}
            for m in ["orders_per_hour", "violations_per_interaction"]:
                fig_transfer(df, m, os.path.join(out, f"transfer_{m}.png"), ref.get(m))
        elif suite == "sensitivity":
            analyse_sensitivity(df, out, lines)
        elif suite == "progress":
            mp = os.path.join(a.results, "main.csv")
            analyse_progress(df, to_units(pd.read_csv(mp)) if os.path.exists(mp) else None,
                             out, lines, a.final_steps)
        elif suite == "demand":
            analyse_comparison(df, suite, out, "ppo_std_only", lines)
        elif suite == "curriculum":
            analyse_comparison(df, suite, out, "ppo_curriculum", lines)
        elif suite == "calibration":
            analyse_comparison(df, suite, out, "ppo_v2", lines)
        elif suite == "ablation":
            analyse_comparison(df, suite, out, "ppo_full", lines)
        else:
            analyse_comparison(df, suite, out, "ppo", lines)

    for tag in ["main", "v1", "throughput_only", "safety_only", "std_only",
                "curriculum_stage1", "curriculum_stage2", "curriculum", "test"]:
        if fig_training(a.runs_dir, tag, os.path.join(out, f"training_{tag}.png")):
            print(f"Training curves: {tag}")

    with open(os.path.join(out, "summary.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {out}/summary.md and figures")


if __name__ == "__main__":
    main()
