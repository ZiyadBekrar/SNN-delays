"""Depth-resolved views of the branch-pruning sweeps.

Re-renders the cached per-run table. Everything is plotted against the pruned
fraction rather than the delay threshold: a fixed threshold removes different
fractions in different families, since their delay distributions differ, so a
comparison at matched threshold is confounded by matched surgery.

Accuracy is also not monotone in the threshold, keeping only the zero-delay
branches can beat keeping a delay-truncated subset, because a partially ablated
loop is neither the trained circuit nor a clean lesion.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style   # importing applies the DelRec theme (rcParams)

# Common pruned-fraction grid every family/layer/seed is interpolated onto, so
# curves and deltas are read at matched surgery rather than matched threshold.
PF_GRID = np.linspace(0.0, 1.0, 51)
_KEY = ["model", "direction", "layer", "model_seed"]


# --------------------------------------------------------------------------- #
# Reductions
# --------------------------------------------------------------------------- #
def _baselines(df):
    """Identity accuracy per (model, direction, layer, seed): the least-pruned row."""
    idx = df.groupby(_KEY)["pruned_frac"].idxmin()
    return df.loc[idx].set_index(_KEY)["acc"]


def _curve_on_grid(sub):
    """One (model, direction, layer, seed) run's accuracy interpolated onto PF_GRID."""
    s = sub.groupby("pruned_frac")["acc"].mean().sort_index()
    pf, acc = s.index.to_numpy(float), s.to_numpy(float)
    out = np.interp(PF_GRID, pf, acc)
    out[(PF_GRID < pf.min()) | (PF_GRID > pf.max())] = np.nan
    return out


def _grid_mean_sem(sub):
    """Mean +/- SEM over seeds of the on-grid accuracy, for one (model, dir, layer)."""
    curves = np.vstack([_curve_on_grid(s) for _, s in sub.groupby("model_seed")])
    n = np.sum(~np.isnan(curves), axis=0)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(curves, axis=0)
        sd = np.nanstd(curves, axis=0, ddof=0)
    sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), 0.0)
    return np.where(n > 0, mean, np.nan), sem


def _type_groups(df, spec):
    """Model keys present in ``df``, grouped by delay type, in ``spec.prefixes`` order."""
    present = set(df["model"].unique())
    groups = {}
    for k in spec.prefixes:
        if k in present:
            groups.setdefault(style.type_of(k), []).append(k)
    return groups


# --------------------------------------------------------------------------- #
# (1) accuracy vs pruned fraction, all layers on one axis
# --------------------------------------------------------------------------- #
def _threshold_mean_sem(sub):
    """Mean +/- SEM over seeds at each threshold, for one (model, direction, layer)."""
    g = sub.groupby("magnitude")["acc"].agg(mean="mean", sd="std", n="count").sort_index()
    return (g.index.to_numpy(float), g["mean"].to_numpy(),
            (g["sd"] / np.sqrt(g["n"].clip(lower=1))).to_numpy())


def plot_layers_overlay(df, out_dir, spec, dataset, direction, dtype, order, x="fraction"):
    """Accuracy vs pruning magnitude, one curve per layer. One panel per condition."""
    conds = [c for c in ("learned", "fixed")
             if any(style.condition_of(m) == c for m in order)]
    layers = sorted(df["layer"].unique())
    colors = style.ordinal_colors(len(layers))   # light (layer 1) -> dark (layer L)
    comp = "<" if direction == "short" else ">"
    xlabel = ("Fraction of the layer's recurrent synapses removed" if x == "fraction"
              else f"Delay threshold d (removed: delay {comp} d, timesteps)")

    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.2),
                             squeeze=False, sharey=True)
    for ax, cond in zip(axes[0], conds):
        mkeys = [m for m in order if style.condition_of(m) == cond]
        for mkey in mkeys:
            for li, layer in enumerate(layers):
                sub = df[(df["model"] == mkey) & (df["layer"] == layer)]
                if sub.empty:
                    continue
                if x == "fraction":
                    xs, mean, sem = PF_GRID, *_grid_mean_sem(sub)
                else:
                    xs, mean, sem = _threshold_mean_sem(sub)
                ax.plot(xs, mean, color=colors[li], lw=1.9, label=f"Layer {layer}")
                ax.fill_between(xs, mean - sem, mean + sem,
                                color=colors[li], alpha=0.22, lw=0)
        if x == "fraction":
            ax.set_xlim(0, 1)
        ax.set_xlabel(xlabel)
        ax.set_title(", ".join(spec.label(m) for m in mkeys))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("Test accuracy (%)")
    fig.suptitle(f"{dataset} {dtype}: {direction}-delay pruning, per layer "
                 f"(mean +/- SEM over seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    suffix = "" if x == "fraction" else "_vs_threshold"
    save_fig(fig, os.path.join(out_dir, f"prune_layers_overlay{suffix}.svg"))


# --------------------------------------------------------------------------- #
# (2) learned minus fixed at matched pruned fraction
# --------------------------------------------------------------------------- #
def plot_matched_fraction_delta(df, out_dir, spec, dataset, direction, dtype, order):
    """Delta accuracy (learned minus fixed) vs pruned fraction, one line per layer."""
    learned = [m for m in order if style.condition_of(m) == "learned"]
    fixed = [m for m in order if style.condition_of(m) == "fixed"]
    if not learned or not fixed:
        return                       # nothing to contrast for this delay type
    ml, mf = learned[0], fixed[0]

    layers = sorted(df["layer"].unique())
    colors = style.ordinal_colors(len(layers))
    rows = []
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for li, layer in enumerate(layers):
        sl = df[(df["model"] == ml) & (df["layer"] == layer)]
        sf = df[(df["model"] == mf) & (df["layer"] == layer)]
        if sl.empty or sf.empty:
            continue
        m_l, s_l = _grid_mean_sem(sl)
        m_f, s_f = _grid_mean_sem(sf)
        delta, err = m_l - m_f, np.sqrt(s_l ** 2 + s_f ** 2)
        ax.plot(PF_GRID, delta, color=colors[li], lw=1.9, label=f"Layer {layer}")
        ax.fill_between(PF_GRID, delta - err, delta + err,
                        color=colors[li], alpha=0.22, lw=0)
        rows += [{"direction": direction, "type": dtype, "layer": layer,
                  "pruned_frac": pf, "delta_acc": d, "err": e}
                 for pf, d, e in zip(PF_GRID, delta, err)]
    ax.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
    ax.set_xlim(0, 1)
    ax.set_xlabel("Fraction of the layer's recurrent synapses removed")
    ax.set_ylabel(f"{spec.label(ml)} - {spec.label(mf)}  (accuracy pts)")
    ax.set_title(f"{dataset} {dtype}: {direction}-delay pruning\n"
                 f"learned advantage at matched surgery")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "matched_fraction_delta.svg"))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "matched_fraction_delta.csv"), index=False)


# --------------------------------------------------------------------------- #
# (3) full ablation + worst case vs depth
# --------------------------------------------------------------------------- #
def _ablation_rows(df):
    """Per (model, layer, seed): the drop at the most-pruned point, and the worst
    drop anywhere on the grid. The former is the unambiguous lesion (the layer's
    recurrence removed outright). The latter bounds the non-monotone middle."""
    base = _baselines(df)
    d = df.copy()
    d["drop"] = d.apply(lambda r: base.loc[(r["model"], r["direction"],
                                            r["layer"], r["model_seed"])] - r["acc"],
                        axis=1)
    full = d.loc[d.groupby(["model", "layer", "model_seed"])["pruned_frac"].idxmax()]
    worst = d.loc[d.groupby(["model", "layer", "model_seed"])["drop"].idxmax()]
    return (full[["model", "layer", "model_seed", "pruned_frac", "drop"]],
            worst[["model", "layer", "model_seed", "direction", "magnitude",
                   "pruned_frac", "drop"]])


def _summarize(sub, col="drop"):
    g = sub.groupby(["model", "layer"])[col].agg(mean="mean", sd="std", n="count")
    g["sem"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    return g.reset_index()


def plot_ablation_vs_depth(df, out_dir, spec, dataset, models=None):
    """Accuracy drop vs depth: full ablation (left) and worst case on the grid (right)."""
    if models is not None:
        df = df[df["model"].isin(models)]
    full, worst = _ablation_rows(df)
    gf, gw = _summarize(full), _summarize(worst)
    gf.to_csv(os.path.join(out_dir, "ablation_vs_depth.csv"), index=False)
    worst.to_csv(os.path.join(out_dir, "worst_case_per_run.csv"), index=False)

    layers = sorted(df["layer"].unique())
    x = np.arange(len(layers))
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.2), sharey=True)
    panels = [(axes[0], gf, f"Full ablation "
                            f"({100 * full['pruned_frac'].min():.0f}-"
                            f"{100 * full['pruned_frac'].max():.0f}% pruned)"),
              (axes[1], gw, "Worst case over the threshold grid")]
    for ax, g, title in panels:
        for mkey in [m for m in spec.prefixes if m in set(g["model"])]:
            sub = g[g["model"] == mkey].set_index("layer")
            means = [sub.loc[l, "mean"] if l in sub.index else np.nan for l in layers]
            sems = [sub.loc[l, "sem"] if l in sub.index else 0.0 for l in layers]
            ax.errorbar(x, means, yerr=sems, capsize=4, marker="o", lw=1.8,
                        color=spec.color(mkey), ls=spec.linestyle(mkey),
                        label=spec.label(mkey))
        ax.set_xticks(x)
        ax.set_xticklabels([f"Layer {l}" for l in layers])
        ax.set_xlabel("Recurrent layer (depth)")
        ax.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("Accuracy drop (pts)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"{dataset}: how much each layer's recurrence is worth "
                 f"(mean +/- SEM over seeds)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, os.path.join(out_dir, "ablation_vs_depth.svg"))
    return gf


# --------------------------------------------------------------------------- #
# (4) Part 1 vs Part 2: delay spread against causal importance
# --------------------------------------------------------------------------- #
def plot_spread_vs_importance(df, out_dir, spec, dataset, spread_csv, models=None):
    """Per-layer converged delay std (Part 1) vs full-ablation drop (Part 2)."""
    if not os.path.exists(spread_csv):
        print(f"[prune_depth] no delay-spread summary at {spread_csv}; "
              f"run analyze_delay_depth.py first (skipping spread_vs_importance).")
        return
    if models is not None:
        df = df[df["model"].isin(models)]
    spread = pd.read_csv(spread_csv).set_index(["model", "layer"])["mean"]
    full, _ = _ablation_rows(df)
    gf = _summarize(full).set_index(["model", "layer"])

    rows = []
    fig, ax = plt.subplots(figsize=(5.6, 4.4))
    for mkey in (models if models is not None else spec.prefixes):
        xs, ys, es, ls_ = [], [], [], []
        for layer in sorted(df["layer"].unique()):
            if (mkey, layer) not in spread.index or (mkey, layer) not in gf.index:
                continue
            xs.append(float(spread.loc[(mkey, layer)]))
            ys.append(float(gf.loc[(mkey, layer), "mean"]))
            es.append(float(gf.loc[(mkey, layer), "sem"]))
            ls_.append(layer)
        if not xs:
            continue
        learned = style.condition_of(mkey) == "learned"
        ax.errorbar(xs, ys, yerr=es, ls=spec.linestyle(mkey), lw=1.2, capsize=3,
                    color=spec.color(mkey), marker=style.TYPE_MARKERS[style.type_of(mkey)],
                    ms=8, mfc=spec.color(mkey) if learned else style.SURFACE,
                    mec=spec.color(mkey), label=spec.label(mkey))
        for xi, yi, li in zip(xs, ys, ls_):
            ax.annotate(str(li), (xi, yi), textcoords="offset points", xytext=(7, 5),
                        fontsize=8, color=style.INK_MUTED)
        rows += [{"model": mkey, "layer": li, "delay_std": xi, "full_ablation_drop": yi}
                 for xi, yi, li in zip(xs, ys, ls_)]
    ax.set_xlabel("Converged delay std of that layer (timesteps)")
    ax.set_ylabel("Accuracy drop when that layer's\nrecurrence is fully ablated (pts)")
    ax.set_title(f"{dataset}: delay spread vs causal importance\n"
                 f"(labels = layer; the line traces depth order, it is not a fit)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "spread_vs_importance.svg"))
    pd.DataFrame(rows).to_csv(
        os.path.join(out_dir, "spread_vs_importance.csv"), index=False)


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def run_prune_depth_plots(rows, out_root, spec, dataset, spread_csv):
    """Write all four depth views from the sweep's per-run rows."""
    df = pd.DataFrame(rows)
    if df.empty:
        print("[prune_depth] no rows to plot.")
        return
    os.makedirs(out_root, exist_ok=True)
    for direction in sorted(df["direction"].unique()):
        dir_rows = df[df["direction"] == direction]
        for dtype, order in _type_groups(dir_rows, spec).items():
            out_dir = os.path.join(out_root, direction, dtype)
            os.makedirs(out_dir, exist_ok=True)
            sub = dir_rows[dir_rows["model"].isin(order)]
            for x in ("fraction", "threshold"):
                plot_layers_overlay(sub, out_dir, spec, dataset, direction, dtype,
                                    order, x=x)
            plot_matched_fraction_delta(sub, out_dir, spec, dataset, direction, dtype, order)
    gf = plot_ablation_vs_depth(df, out_root, spec, dataset)
    plot_spread_vs_importance(df, out_root, spec, dataset, spread_csv)
    # Per-delay-type twins of the two cross-direction figures (the paper's analysis is
    # axonal-only, and 2 families read better than 4).
    for dtype, order in _type_groups(df, spec).items():
        d = os.path.join(out_root, dtype)
        os.makedirs(d, exist_ok=True)
        plot_ablation_vs_depth(df, d, spec, f"{dataset} {dtype}", models=order)
        plot_spread_vs_importance(df, d, spec, f"{dataset} {dtype}", spread_csv,
                                  models=order)

    print(f"\n=== {dataset}: accuracy drop from fully ablating one layer's recurrence ===")
    for _, r in gf.sort_values(["model", "layer"]).iterrows():
        print(f"  {spec.label(r['model']):18s} layer {int(r['layer'])}: "
              f"{r['mean']:6.2f} +/- {r['sem']:.2f} pts")
