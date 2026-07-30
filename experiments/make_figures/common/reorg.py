"""How far the delays moved from their initialization.

Compares trained delays against the shared random draw each run started from,
which is stored per run. The frozen fixed-delay family is an exact zero control.
"""

import os

import numpy as np
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style
from common.recdel import _delays_flat, _seed_trend_band


def _deltas(data):
    """List over seeds of lists over layers of signed Δ = trained − init (flat)."""
    out = []
    for seed_layers, init_layers in zip(data["layers_per_seed"], data["init_per_seed"]):
        seed_d = []
        for lay, init in zip(seed_layers, init_layers):
            seed_d.append(_delays_flat(lay) - np.asarray(init).reshape(-1))
        out.append(seed_d)
    return out


# --------------------------------------------------------------------------- #
def _stats_rows(results):
    rows = []
    for mkey, data in results.items():
        for seed, seed_d in zip(data["seeds"], _deltas(data)):
            for li, dl in enumerate(seed_d):
                a = np.abs(dl)
                rows.append({"model": mkey, "seed": seed, "layer": li + 1,
                             "mean_abs_delta": float(a.mean()),
                             "median_abs_delta": float(np.median(a)),
                             "frac_moved": float((a >= 1).mean()),
                             "mean_signed_delta": float(dl.mean())})
    return rows


def plot_signed_delta_vs_depth(results, out_dir, spec):
    """Mean signed Δdelay per layer vs depth, one line per family (mean ± SEM)."""
    import pandas as pd
    df = pd.DataFrame(_stats_rows(results))
    if df.empty:
        return
    g = (df.groupby(["model", "layer"])["mean_signed_delta"]
         .agg(mean="mean", sd="std", n="count").reset_index())
    g["sem"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    layers = sorted(g["layer"].unique())
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mkey in [m for m in spec.prefixes if m in set(g["model"])]:
        sub = g[g["model"] == mkey].set_index("layer")
        means = [sub.loc[l, "mean"] if l in sub.index else np.nan for l in layers]
        sems = [sub.loc[l, "sem"] if l in sub.index else 0.0 for l in layers]
        ax.errorbar(x, means, yerr=sems, capsize=4, marker="o", lw=1.8,
                    color=spec.color(mkey), ls=spec.linestyle(mkey),
                    label=spec.label(mkey))
    ax.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Recurrent layer (depth)")
    ax.set_ylabel("Mean signed Δ delay from init (timesteps)")
    ax.set_title("Direction of delay reorganization vs depth\n(mean ± SEM over seeds)")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "signed_delta_vs_depth.svg"))


def plot_delta_magnitude_bar(results, out_dir, spec):
    """Mean |Δdelay| per family (mean ± std over seeds, pooled over layers)."""
    fig, ax = plt.subplots(figsize=(5, 4))
    xs, means, stds, labels, colors, hatches = [], [], [], [], [], []
    for i, (mkey, data) in enumerate(results.items()):
        per_seed = [np.abs(np.concatenate(seed_d)).mean() for seed_d in _deltas(data) if seed_d]
        if not per_seed:
            continue
        xs.append(i)
        means.append(np.mean(per_seed))
        stds.append(np.std(per_seed))
        labels.append(f"{spec.label(mkey)}\n(n={len(per_seed)})")
        colors.append(spec.color(mkey))
        hatches.append(spec.hatch(mkey))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.02, f"{m:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean |Δdelay| from init (time steps)")
    ax.set_title("Delay reorganization from initialization (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "delta_magnitude_bar.svg"))


def plot_delta_distributions(results, out_dir, spec):
    """Per layer, overlaid signed-Δ histograms per family (fixed spikes at 0)."""
    n_layers = max((len(d["layers_per_seed"][0]) if d["layers_per_seed"] else 0)
                   for d in results.values())
    if n_layers == 0:
        return
    all_d = [np.concatenate(seed_d) for data in results.values()
             for seed_d in _deltas(data) if seed_d]
    if not all_d:
        return
    lo = int(np.floor(np.concatenate(all_d).min()))
    hi = int(np.ceil(np.concatenate(all_d).max()))
    bins = np.arange(lo - 0.5, hi + 1.5, 1.0)

    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
    axes = axes[0]
    for li in range(n_layers):
        ax = axes[li]
        for mkey, data in results.items():
            pooled = [seed_d[li] for seed_d in _deltas(data) if li < len(seed_d)]
            if not pooled:
                continue
            pooled = np.concatenate(pooled)
            ax.hist(pooled, bins=bins, density=True, alpha=0.5,
                    label=f"{spec.label(mkey)} (μ={pooled.mean():+.1f})",
                    color=spec.color(mkey), hatch=spec.hatch(mkey))
        ax.axvline(0, color=style.INK_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Δdelay from init (time steps)")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {li + 1}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Signed delay shift from initialization (pooled over seeds)")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "delta_distributions.svg"))


def plot_delta_vs_init(results, out_dir, spec):
    """Δdelay vs initial delay: per-seed kernel-smoothed trend + ±std band, one
    figure per layer, all families overlaid (fixed sits on Δ=0)."""
    n_layers = max((len(d["layers_per_seed"][0]) if d["layers_per_seed"] else 0)
                   for d in results.values())
    if n_layers == 0:
        return
    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
    axes = axes[0]
    for li in range(n_layers):
        ax = axes[li]
        for mkey, data in results.items():
            inits, deltas = [], []
            for seed_layers, init_layers in zip(data["layers_per_seed"], data["init_per_seed"]):
                if li >= len(seed_layers):
                    continue
                x = np.asarray(init_layers[li]).reshape(-1)
                y = _delays_flat(seed_layers[li]) - x
                inits.append(x)
                deltas.append(y)
            if len(inits) < 1:
                continue
            grid, _curves, mean, sd = _seed_trend_band(inits, deltas)
            ax.plot(grid, mean, color=spec.color(mkey), ls=spec.linestyle(mkey),
                    label=spec.label(mkey))
            ax.fill_between(grid, mean - sd, mean + sd, color=spec.color(mkey), alpha=0.15, lw=0)
        ax.axhline(0, color=style.INK_MUTED, lw=0.8, ls=":")
        ax.set_xlabel("Initial delay (time steps)")
        ax.set_ylabel("Δdelay from init (time steps)")
        ax.set_title(f"Layer {li + 1}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Delay shift vs initial delay (per-seed trend, mean ± std over seeds)")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "delta_vs_init.svg"))


def run_reorg_analysis(results, out_dir, spec):
    """Write all reorganization figures + the stats CSV."""
    import pandas as pd
    os.makedirs(out_dir, exist_ok=True)
    have = {k: v for k, v in results.items()
            if v.get("layers_per_seed") and v.get("init_per_seed")}
    if not have:
        print("[warn] no runs with init_delays.npz, skipping reorg analysis.")
        return
    pd.DataFrame(_stats_rows(have)).to_csv(os.path.join(out_dir, "reorg_stats.csv"), index=False)
    plot_delta_magnitude_bar(have, out_dir, spec)
    plot_signed_delta_vs_depth(have, out_dir, spec)
    plot_delta_distributions(have, out_dir, spec)
    plot_delta_vs_init(have, out_dir, spec)
