"""Per-layer distribution of the trained recurrent delays.

Draws one cumulative distribution per hidden layer, with the shared
initialization for reference and a strip below the axis carrying each layer's
mean and standard deviation. Delays are the raw parameter d. The physical lag is
1 + d.

Branches on the delay array's shape, so per-neuron and per-connection delays are
both handled.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import save_fig
from common.recdel import _delays_flat
from common import style   # importing applies the DelRec theme (rcParams)


def _layer_delays_ms(data, li, ms_per_step):
    """Per-seed flat delay arrays (in ms) for one model at layer ``li``. Delays are
    already the physical lag (``1 + d`` steps). The +1 is applied at load time."""
    out = []
    for seed_layers in data["layers_per_seed"]:
        if li < len(seed_layers):
            d = np.asarray(seed_layers[li]["delays"]).reshape(-1).astype(float)
            out.append(d * ms_per_step)
    return out


def _survival_curves(delays_ms_list, grid):
    """Per-seed survival function ``P(delay > x)`` on ``grid``, ``(n_seeds, len)``."""
    curves = []
    for d in delays_ms_list:
        d = np.sort(d)
        curves.append(1.0 - np.searchsorted(d, grid, side="right") / d.size)
    return np.array(curves)


def _mean_sem(curves):
    """Mean and SEM across the seed axis (axis 0). SEM is zeros for one seed."""
    mean = curves.mean(0)
    n = curves.shape[0]
    sem = curves.std(0, ddof=1) / np.sqrt(n) if n > 1 else np.zeros_like(mean)
    return mean, sem


# The per-layer reductions of a converged delay distribution. ``std`` is the spread
# (how wide a temporal window the layer spans) and ``mean`` its location, the
# statistic the cortical-timescale reading is actually about, since a layer can widen
# and shorten at once. ``max`` is the longest delay (= the ring-buffer depth the layer
# needs) and ``skew`` separates "short delays plus a long tail" from a symmetric
# spread. The two are indistinguishable in the std alone.
MOMENTS = ("mean", "std", "max", "skew")
MOMENT_LABELS = {"mean": "Mean delay (timesteps)", "std": "Delay std (timesteps)",
                 "max": "Max delay (timesteps)", "skew": "Delay skewness"}


def _moments(d):
    """The MOMENTS of one layer's flattened delays."""
    m, s = float(np.mean(d)), float(np.std(d))
    return {"mean": m, "std": s, "max": float(np.max(d)),
            "skew": float(np.mean((d - m) ** 3) / (s ** 3 + 1e-12))}


def _per_layer_std_rows(results):
    """List of {model, layer, seed, **MOMENTS} over every (model, seed, layer)."""
    rows = []
    for mkey, data in results.items():
        for seed, seed_layers in zip(data["seeds"], data["layers_per_seed"]):
            for li, lay in enumerate(seed_layers):
                rows.append({"model": mkey, "layer": li + 1, "seed": seed,
                             **_moments(_delays_flat(lay))})
    return rows


def _summary(rows, col="std"):
    """Mean +/- SEM over seeds of one per-layer moment, per (model, layer)."""
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    g = df.groupby(["model", "layer"])[col].agg(["mean", "std", "count"]).reset_index()
    g["sem"] = g["std"] / np.sqrt(g["count"].clip(lower=1))
    return g


def plot_delay_std_vs_depth(rows, out_dir, spec, dataset=""):
    """Per-layer delay std vs depth, one line per family, mean +/- SEM over seeds."""
    summary = _summary(rows)
    if summary.empty:
        print("[delay_depth] no rows to plot.")
        return summary

    layers = sorted(summary["layer"].unique())
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mkey in [m for m in spec.prefixes if m in summary["model"].values]:
        sub = summary[summary["model"] == mkey].set_index("layer")
        means = [sub.loc[l, "mean"] if l in sub.index else np.nan for l in layers]
        sems = [sub.loc[l, "sem"] if l in sub.index else 0.0 for l in layers]
        ax.errorbar(x, means, yerr=sems, capsize=4, marker="o", lw=1.8,
                    color=spec.color(mkey), ls=spec.linestyle(mkey),
                    label=spec.label(mkey))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Recurrent layer (depth)")
    ax.set_ylabel("Delay std (timesteps)")
    ax.set_title(f"{dataset}: delay spread vs depth (mean +/- SEM over seeds)".strip(": "))
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "delay_std_vs_depth.svg"))
    return summary


def plot_delay_moments_vs_depth(rows, out_dir, spec, dataset="", models=None, suffix=""):
    """The four MOMENTS vs depth, one panel each, one line per family (mean +/- SEM)."""
    keys = [m for m in (models if models is not None else spec.prefixes)]
    df = pd.DataFrame(rows)
    df = df[df["model"].isin(keys)]
    if df.empty:
        return
    layers = sorted(df["layer"].unique())
    x = np.arange(len(layers))
    fig, axes = plt.subplots(2, 2, figsize=(9.6, 7.0))
    for ax, stat in zip(axes.ravel(), MOMENTS):
        g = _summary(df.to_dict("records"), col=stat)
        for mkey in [m for m in keys if m in set(g["model"])]:
            sub = g[g["model"] == mkey].set_index("layer")
            means = [sub.loc[l, "mean"] if l in sub.index else np.nan for l in layers]
            sems = [sub.loc[l, "sem"] if l in sub.index else 0.0 for l in layers]
            ax.errorbar(x, means, yerr=sems, capsize=4, marker="o", lw=1.8,
                        color=spec.color(mkey), ls=spec.linestyle(mkey),
                        label=spec.label(mkey))
        ax.set_xticks(x)
        ax.set_xticklabels([f"L{l}" for l in layers])
        ax.set_ylabel(MOMENT_LABELS[stat])
        ax.grid(axis="y", alpha=0.3)
    axes[1][0].set_xlabel("Recurrent layer (depth)")
    axes[1][1].set_xlabel("Recurrent layer (depth)")
    axes[0][0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: converged delay distribution vs depth "
                 f"(mean +/- SEM over seeds)".strip())
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, os.path.join(out_dir, f"delay_moments_vs_depth{suffix}.svg"))


def _n_layers(data):
    return max((len(s) for s in data["layers_per_seed"]), default=0)


def plot_delay_cdf_by_layer(results, out_dir, spec, dataset=""):
    """Per-layer delay CDF, learned families only, one panel per delay type."""
    # Learned keys grouped by delay type, in a stable order.
    learned = [m for m in spec.prefixes if style.condition_of(m) == "learned"]
    panels = [(t, m) for t in ("axonal", "synaptic")
              for m in learned if style.type_of(m) == t and results.get(m, {}).get("seeds")]
    if not panels:
        print("[delay_depth] no learned families for the per-layer CDF plot.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.2),
                             squeeze=False)
    axes = axes[0]
    for ax, (dtype, mkey) in zip(axes, panels):
        data = results[mkey]
        nL = _n_layers(data)
        # Shared timestep grid for this panel: 0 -> max delay over its layers/seeds.
        mx = max((float(np.asarray(lay["delays"]).max())
                  for seed_layers in data["layers_per_seed"] for lay in seed_layers),
                 default=1.0)
        grid = np.linspace(0.0, max(mx, 1.0), 200)
        colors = style.ordinal_colors(nL)   # light (layer 1) -> dark (layer L)
        for li in range(nL):
            dl = _layer_delays_ms(data, li, 1.0)   # ms_per_step=1 -> raw timesteps
            if not dl:
                continue
            mean, sem = _mean_sem(1.0 - _survival_curves(dl, grid))   # CDF = 1 - survival
            ax.plot(grid, mean, color=colors[li], lw=1.9, label=f"Layer {li + 1}")
            ax.fill_between(grid, mean - sem, mean + sem, color=colors[li], alpha=0.22, lw=0)
        ax.set_xlim(0, grid[-1])
        ax.set_ylim(0, 1.02)
        ax.set_xlabel("Recurrent delay (timesteps)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{spec.label(mkey)}")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(alpha=0.3)
    fig.suptitle(f"{dataset}: per-layer delay CDF (learned, mean +/- SEM over seeds)".strip(": "))
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    save_fig(fig, os.path.join(out_dir, "delay_cdf_by_layer.svg"))


def plot_delay_profile_panel(results, rows, out_dir, spec, dataset="", dtype="axonal"):
    """The descriptive result as a single paper panel. Everything is measured in the same
    unit (delay in timesteps) which is what makes the fold legitimate: the CDF's x-axis,
    the mean, the spread and the shift from initialization are all lengths on one axis.
    """
    learned = next((m for m in spec.prefixes if style.type_of(m) == dtype
                    and style.condition_of(m) == "learned"), None)
    fixed = next((m for m in spec.prefixes if style.type_of(m) == dtype
                  and style.condition_of(m) == "fixed"), None)
    if learned is None or not results.get(learned, {}).get("seeds"):
        return
    df = pd.DataFrame(rows)
    nL = _n_layers(results[learned])
    colors = style.ordinal_colors(nL)

    mx = max(float(np.asarray(lay["delays"]).max())
             for k in (learned, fixed) if k and results.get(k, {}).get("seeds")
             for sl in results[k]["layers_per_seed"] for lay in sl)
    grid = np.linspace(0.0, mx, 300)

    def _stat(mkey, layer, col):
        return float(df[(df["model"] == mkey) & (df["layer"] == layer)][col].mean())

    # Arial first: svg.fonttype='none' keeps text as text, so the SVG carries this
    # family list and resolves to Arial wherever it exists (Illustrator, most systems).
    # One step above style.RC_LARGE_TEXT: this is a sub-panel of a larger paper
    # figure and is reduced once more on the page. Arial first, svg.fonttype='none'
    # keeps text as text, so the SVG carries the family list and resolves to real
    # Arial in Illustrator or on any machine that has it.
    rc = {**style.RC_LARGE_TEXT,
          "font.size": 15, "axes.titlesize": 17.5, "axes.labelsize": 16,
          "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
          "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]}
    with plt.rc_context(rc):
        # Two stacked axes sharing x: the CDFs, and a dedicated strip for the
        # mean +/- SD whiskers. A real sub-axis (rather than markers floated below the
        # spine) keeps the whiskers on the same x scale and gives them their own
        # labeled y, which is what makes them read as part of the figure.
        fig, (ax, axs) = plt.subplots(
            2, 1, sharex=True, figsize=(5.6, 6.4),
            gridspec_kw={"height_ratios": [4.0, 1.15], "hspace": 0.11})

        entries = []
        if fixed and results.get(fixed, {}).get("seeds"):
            pooled = [d for li in range(nL) for d in _layer_delays_ms(results[fixed], li, 1.0)]
            entries.append((style.INK_MUTED, "init", (0, (5, 2.5)), 1.6,
                            1.0 - _survival_curves(pooled, grid),
                            float(np.mean([_stat(fixed, l + 1, "mean") for l in range(nL)])),
                            float(np.mean([_stat(fixed, l + 1, "std") for l in range(nL)]))))
        for li in range(nL):
            dl = _layer_delays_ms(results[learned], li, 1.0)
            if dl:
                entries.append((colors[li], f"L{li + 1}", "-", 2.4,
                                1.0 - _survival_curves(dl, grid),
                                _stat(learned, li + 1, "mean"),
                                _stat(learned, li + 1, "std")))

        for col, lab, ls, lw, c, m, sd in entries:
            ax.plot(grid, c.mean(axis=0), color=col, lw=lw, ls=ls,
                    zorder=4 if lab != "init" else 2)
            ax.fill_between(grid, c.mean(0) - c.std(0), c.mean(0) + c.std(0),
                            color=col, alpha=0.20, lw=0,
                            zorder=3 if lab != "init" else 1)
        # The curves separate over the first ~2/3 of the delay range and then converge
        # on 1. Carrying the empty tail to the single largest delay would compress that
        # region for nothing. Crop to the 99.5th percentile, and make the panel taller
        # than wide so the remaining vertical gaps get the resolution.
        allde = np.concatenate([_delays_flat(lay) for k in (learned, fixed)
                                if k and results.get(k, {}).get("seeds")
                                for sl in results[k]["layers_per_seed"] for lay in sl])
        xmax = float(np.ceil(np.percentile(allde, 99.5) / 5.0) * 5.0)
        ax.set_ylim(0, 1.02)
        ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
        ax.set_ylabel("Cumulative fraction of delays")
        ax.grid(alpha=0.22)
        ax.legend(handles=[plt.Line2D([], [], color=c, ls=l, lw=w)
                           for c, _, l, w, *_ in entries],
                  labels=[lab for _, lab, *_ in entries],
                  loc="lower right", fontsize=13, frameon=True, framealpha=0.95,
                  edgecolor=style.GRID, handlelength=2.0, labelspacing=0.4,
                  borderaxespad=0.8, ncol=2, columnspacing=1.2)

        # --- whisker strip --------------------------------------------------------
        for i, (col, lab, _ls, _lw, _c, m, sd) in enumerate(entries):
            y = len(entries) - 1 - i
            axs.errorbar([m], [y], xerr=[[min(sd, m)], [sd]], color=col, marker="o",
                         ms=6.5, capsize=4, lw=2.0, mec=col, mfc=col, zorder=3)
            axs.text(xmax * 0.995, y, f"{m:.1f} $\\pm$ {sd:.1f}", color=col, fontsize=12.5,
                     ha="right", va="center")
        axs.set_yticks(range(len(entries)))
        axs.set_yticklabels([lab for _, lab, *_ in entries][::-1], fontsize=13.5)
        axs.set_ylim(-0.75, len(entries) - 0.25)
        axs.set_xlim(0, xmax)
        axs.set_xlabel("Recurrent delay (timesteps)")
        axs.grid(axis="x", alpha=0.22)
        axs.tick_params(axis="y", length=0)
        for sp in ("left", "bottom"):
            axs.spines[sp].set_visible(sp == "bottom")

        n_seeds = df.loc[df["model"] == learned, "seed"].nunique()
        fig.suptitle("Learned delays lengthen and narrow with depth",
                     fontweight="bold", y=0.995)
        # The x crop is stated, not silent: 0.5% of delays lie beyond it, and that tail
        # is exactly the sparse long-delay subpopulation the descriptive claim rests on.
        # The two SDs in this panel are orthogonal and ~30x apart in magnitude: the
        # band is run-to-run variability of the CDF, the strip is the spread of delays
        # within one network. Naming both "SD" without qualification invites reading
        # the strip whiskers as seed error bars, so each is labeled by what it is over.
        fig.text(0.5, 0.935,
                 f"curves: mean CDF over {n_seeds} seeds, band = $\\pm$ SD across seeds\n"
                 f"strip: mean $\\pm$ SD of the delays within a network "
                 f"(averaged over seeds)",
                 ha="center", va="top", fontsize=11, color=style.INK_SECONDARY)
        # Inside the axes: save_fig writes with bbox_inches="tight", which recomputes
        # the bounding box and undoes any tight_layout rect padding, so a second
        # figure-level line would collide with the axes. The top-left of a CDF is free.
        ax.text(0.025, 0.985, f"x cropped: {xmax:.0f} of {allde.max():.0f} steps "
                f"(p99.5)", transform=ax.transAxes, ha="left", va="top",
                fontsize=10.5, color=style.INK_MUTED)
        axs.set_ylabel("within-network\nmean $\\pm$ SD", fontsize=10.5,
                       color=style.INK_SECONDARY)
        fig.tight_layout(rect=[0, 0, 1, 0.86])
        save_fig(fig, os.path.join(out_dir, f"delay_profile_panel_{dtype}.svg"))


def run_delay_depth_analysis(results, out_dir, spec, dataset=""):
    """Single entry point: the per-layer delay distributions, plus the CSVs."""
    os.makedirs(out_dir, exist_ok=True)
    rows = _per_layer_std_rows(results)
    if not rows:
        print("[delay_depth] no recurrent-delay layers found in results.")
        return
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "delay_depth_std.csv"), index=False)

    plot_delay_cdf_by_layer(results, out_dir, spec, dataset)

    moments = pd.concat([_summary(rows, col=s).assign(stat=s) for s in MOMENTS])
    moments.to_csv(os.path.join(out_dir, "delay_depth_moments_summary.csv"), index=False)
    print(f"\n=== {dataset} delay moments vs depth (mean +/- SEM over seeds) ===")
    for stat in MOMENTS:
        g = _summary(rows, col=stat)
        for mkey in [m for m in spec.prefixes if m in set(g["model"])]:
            sub = g[g["model"] == mkey].sort_values("layer")
            cells = "  ".join(f"L{int(r['layer'])}: {r['mean']:6.2f}+/-{r['sem']:.2f}"
                              for _, r in sub.iterrows())
            print(f"  {stat:5s} {spec.label(mkey):18s} {cells}")

    summary = _summary(rows, col="std")
    if summary is not None and not summary.empty:
        summary.to_csv(os.path.join(out_dir, "delay_depth_std_summary.csv"), index=False)
        print(f"\n=== {dataset} delay std vs depth (mean +/- SEM over seeds) ===")
        for _, r in summary.sort_values(["model", "layer"]).iterrows():
            print(f"  {spec.label(r['model']):16s} layer {int(r['layer'])}: "
                  f"{r['mean']:.2f} +/- {r['sem']:.2f}  (n={int(r['count'])})")
    print(f"\nFigures (SVG) + CSVs written to {out_dir}/")
