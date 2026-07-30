"""Multi-seed delay and weight statistics, dataset-agnostic.

:func:`run_recdel_analysis` is the single entry point. The caller builds a results
dict keyed by model, holding per-seed accuracies, validation curves, and
``layers_per_seed``, a list per seed of ``{"layer_idx", "delays", "weights"}`` from
:func:`common.utils.load_recurrent_params`. ``delays`` is ``(N,)`` for axonal and
``(N_in, N_out)`` for synaptic, and every function branches on ``delays.ndim`` so both
are handled. Presentation comes from an :class:`common.utils.AnalysisSpec`.
"""

import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.utils import save_fig
from common import style   # importing applies the DelRec theme (rcParams)


# --------------------------------------------------------------------------- #
# Axonal / synaptic pairing helpers
# --------------------------------------------------------------------------- #
def _delays_flat(lay):
    """Delays flattened to 1D (for histograms / KS / summary stats)."""
    return np.asarray(lay["delays"]).reshape(-1)


def _is_synaptic(lay):
    return np.asarray(lay["delays"]).ndim == 2


def _delay_weight_xy(delays, W, direction):
    """(delay, weight-metric) pairs for the delay-vs-weight scatter."""
    delays = np.asarray(delays)
    if delays.ndim == 1:
        y = np.linalg.norm(W, axis=0) if direction == "outgoing" else np.linalg.norm(W, axis=1)
        return delays, y
    return delays.reshape(-1), np.abs(W).reshape(-1)


def _weight_ylabel(is_syn, direction):
    if is_syn:
        return "|synaptic weight|"
    return "‖outgoing weight column‖₂" if direction == "outgoing" else "‖incoming weight row‖₂"


# --------------------------------------------------------------------------- #
# Accuracy
# --------------------------------------------------------------------------- #
def plot_accuracy(results, out_dir, spec, dataset):
    """Bar chart (mean±std accuracy over seeds) and validation learning curves."""
    use_test = any(data.get("test_accs") for data in results.values())
    acc_field = "test_accs" if use_test else "best_accs"
    acc_word = "test" if use_test else "validation (best ckpt)"

    # ---- bar chart of accuracy ----
    fig, ax = plt.subplots(figsize=(5, 4))
    xs, means, stds, labels, colors, hatches = [], [], [], [], [], []
    for i, (mkey, data) in enumerate(results.items()):
        accs = data.get(acc_field) or []
        if not accs:
            continue
        xs.append(i)
        means.append(np.mean(accs))
        stds.append(np.std(accs))
        labels.append(f"{spec.label(mkey)}\n(n={len(accs)})")
        colors.append(spec.color(mkey))
        hatches.append(spec.hatch(mkey))
    bars = ax.bar(xs, means, yerr=stds, capsize=6, color=colors, alpha=0.85)
    for b, h in zip(bars, hatches):
        if h:
            b.set_hatch(h)
    for x, m, s in zip(xs, means, stds):
        ax.text(x, m + s + 0.3, f"{m:.2f}±{s:.2f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"{dataset} {acc_word} accuracy (%)")
    ax.set_title(f"{dataset} {acc_word} accuracy (mean ± std over seeds)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "accuracy_bar.svg"))

    # ---- learning curves (mean ± std band) ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for mkey, data in results.items():
        curves = data["curves"]
        if not curves:
            continue
        L = min(len(c) for c in curves)
        arr = np.stack([c[:L] for c in curves])  # (n_seeds, L)
        mean, std = arr.mean(0), arr.std(0)
        epochs = np.arange(L)
        ax.plot(epochs, mean, label=spec.label(mkey), color=spec.color(mkey),
                ls=spec.linestyle(mkey))
        ax.fill_between(epochs, mean - std, mean + std, color=spec.color(mkey), alpha=0.2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation accuracy (%)")
    ax.set_title(f"{dataset} validation learning curves (mean ± std over seeds)")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "accuracy_curves.svg"))


# --------------------------------------------------------------------------- #
# Delay distributions
# --------------------------------------------------------------------------- #
def plot_delay_distributions(results, out_dir, spec):
    """One subplot per layer. Overlaid normalized histograms across seeds."""
    n_layers = max((len(d["layers_per_seed"][0]) if d["layers_per_seed"] else 0)
                   for d in results.values())
    if n_layers == 0:
        return
    all_d = []
    for data in results.values():
        for seed_layers in data["layers_per_seed"]:
            for lay in seed_layers:
                all_d.append(_delays_flat(lay))
    if not all_d:
        return
    dmax = int(np.ceil(np.concatenate(all_d).max()))
    bins = np.arange(-0.5, dmax + 1.5, 1.0)

    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
    axes = axes[0]
    for li in range(n_layers):
        ax = axes[li]
        for mkey, data in results.items():
            pooled = [_delays_flat(seed_layers[li])
                      for seed_layers in data["layers_per_seed"]
                      if li < len(seed_layers)]
            if not pooled:
                continue
            pooled = np.concatenate(pooled)
            ax.hist(pooled, bins=bins, density=True, alpha=0.5,
                    label=f"{spec.label(mkey)} (μ={pooled.mean():.1f})",
                    color=spec.color(mkey), hatch=spec.hatch(mkey))
        ax.set_xlabel("Recurrent delay (time steps)")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {li + 1}")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.suptitle("Recurrent delay distributions per layer (pooled over seeds)")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "delay_distributions.svg"))


# --------------------------------------------------------------------------- #
# Delay vs. Weight
# --------------------------------------------------------------------------- #
def _pearson_spearman(x, y):
    if len(x) < 3 or np.std(x) == 0 or np.std(y) == 0:
        return np.nan, np.nan
    pear = np.corrcoef(x, y)[0, 1]
    rx = pd.Series(x).rank().to_numpy()
    ry = pd.Series(y).rank().to_numpy()
    spear = np.corrcoef(rx, ry)[0, 1]
    return pear, spear


def _nw_smooth(x, y, grid, h):
    """Nadaraya–Watson Gaussian-kernel regression: a smooth y(x) trend evaluated
    on ``grid``. Returns NaN where no data falls within the kernel bandwidth."""
    diff = grid[:, None] - x[None, :]
    w = np.exp(-0.5 * (diff / h) ** 2)
    wsum = w.sum(axis=1)
    out = (w @ y) / np.where(wsum > 0, wsum, 1.0)
    out[wsum <= 1e-8] = np.nan
    return out


def _seed_trend_band(delays_list, y_list, n_points=60):
    """Per-seed kernel-smoothed trend curves on a shared delay grid + the cross-seed mean/std."""
    gmin = max(float(d.min()) for d in delays_list)
    gmax = min(float(d.max()) for d in delays_list)
    if not gmax > gmin:  # degenerate overlap -> fall back to the global range
        gmin = min(float(d.min()) for d in delays_list)
        gmax = max(float(d.max()) for d in delays_list)
    grid = np.linspace(gmin, gmax, n_points)
    h = max((gmax - gmin) / 20.0, 1.0)
    curves = np.stack([_nw_smooth(d, y, grid, h) for d, y in zip(delays_list, y_list)])
    return grid, curves, np.nanmean(curves, axis=0), np.nanstd(curves, axis=0)


def plot_delay_vs_weight(results, out_dir, spec, direction="outgoing"):
    """Delay vs the per-neuron/per-synapse weight metric, per layer, two figures per model:
    the raw per-seed scatter and the per-seed kernel-smoothed trend (``_seed_trend_band``)
    with the cross-seed mean line + ±std band.
    """
    corr_rows = []
    corr_rows_seed = []
    for mkey, data in results.items():
        seeds = data["seeds"]
        layers_per_seed = data["layers_per_seed"]
        if not layers_per_seed:
            continue
        n_layers = max(len(s) for s in layers_per_seed)
        is_syn = any(_is_synaptic(s[0]) for s in layers_per_seed if s)
        ylabel = _weight_ylabel(is_syn, direction)
        seed_colors = dict(zip(seeds, style.ordinal_colors(len(seeds))))

        fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
        axes = axes[0]
        fig_t, axes_t = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
        axes_t = axes_t[0]
        for li in range(n_layers):
            ax, axt = axes[li], axes_t[li]
            pears, spears = [], []
            d_list, y_list, seed_ids = [], [], []
            for seed, seed_layers in zip(seeds, layers_per_seed):
                if li >= len(seed_layers):
                    continue
                d, wnorm = _delay_weight_xy(seed_layers[li]["delays"],
                                            seed_layers[li]["weights"], direction)
                pear, spear = _pearson_spearman(d, wnorm)
                pears.append(pear)
                spears.append(spear)
                corr_rows_seed.append({"model": mkey, "direction": direction,
                                       "layer": li + 1, "seed": seed,
                                       "pearson_r": pear, "spearman_r": spear, "n": len(d)})
                ax.scatter(d, wnorm, s=8, alpha=0.4, color=seed_colors[seed],
                           label=f"seed {seed} (ρ={spear:.2f})")
                d_list.append(d)
                y_list.append(wnorm)
                seed_ids.append(seed)
            if not spears:
                continue
            pears, spears = np.array(pears), np.array(spears)
            corr_rows.append({
                "model": mkey, "direction": direction, "layer": li + 1,
                "n_seeds": len(spears),
                "pearson_mean": np.nanmean(pears), "pearson_std": np.nanstd(pears),
                "spearman_mean": np.nanmean(spears), "spearman_std": np.nanstd(spears),
            })
            title = (f"Layer {li + 1}\n"
                     f"Pearson r={np.nanmean(pears):.2f}±{np.nanstd(pears):.2f}, "
                     f"Spearman ρ={np.nanmean(spears):.2f}±{np.nanstd(spears):.2f}")

            ax.set_xlabel("Recurrent delay (time steps)")
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)

            # ---- trend panel: per-seed smoothed curves + cross-seed mean±std ----
            grid, curves, mean, std = _seed_trend_band(d_list, y_list)
            for seed, curve in zip(seed_ids, curves):
                axt.plot(grid, curve, color=seed_colors[seed], alpha=0.35, lw=1.0,
                         label=f"seed {seed}")
            axt.plot(grid, mean, color=spec.color(mkey), lw=2.2, label="mean over seeds")
            axt.fill_between(grid, mean - std, mean + std,
                             color=spec.color(mkey), alpha=0.25, label="±1 std")
            axt.set_xlabel("Recurrent delay (time steps)")
            axt.set_ylabel(ylabel)
            axt.set_title(title)
            axt.legend(fontsize=7)
            axt.grid(alpha=0.3)
        fig.suptitle(f"Delay vs. {direction} recurrent weight — {spec.label(mkey)} "
                     f"(per-seed scatter; r = mean ± std over seeds)")
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"delay_vs_weight_{direction}_{mkey}.svg"))

        fig_t.suptitle(f"Delay vs. {direction} recurrent weight — {spec.label(mkey)} "
                       f"(per-seed kernel-smoothed trend; band = mean ± std over seeds)")
        fig_t.tight_layout()
        save_fig(fig_t, os.path.join(out_dir, f"delay_vs_weight_trend_{direction}_{mkey}.svg"))
    return corr_rows, corr_rows_seed


def _model_is_synaptic(data):
    """True if the model's recurrent delays are per-synapse (2D)."""
    return any(_is_synaptic(s[0]) for s in data["layers_per_seed"] if s)


def _plot_trend_combined_group(results, mkeys, out_dir, spec, direction, group):
    """Overlay one delay-type group's kernel-smoothed trends on shared axes."""
    n_layers = max(max(len(s) for s in results[m]["layers_per_seed"]) for m in mkeys)
    ylabel = _weight_ylabel(group == "synaptic", direction)

    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
    axes = axes[0]
    for li in range(n_layers):
        ax = axes[li]
        for mkey in mkeys:
            d_list, y_list, pears, spears = [], [], [], []
            for seed_layers in results[mkey]["layers_per_seed"]:
                if li >= len(seed_layers):
                    continue
                d, wnorm = _delay_weight_xy(seed_layers[li]["delays"],
                                            seed_layers[li]["weights"], direction)
                pear, spear = _pearson_spearman(d, wnorm)
                pears.append(pear)
                spears.append(spear)
                d_list.append(d)
                y_list.append(wnorm)
            if not d_list:
                continue
            grid, _curves, mean, std = _seed_trend_band(d_list, y_list)
            color = spec.color(mkey)
            ax.plot(grid, mean, color=color, lw=2.2, ls=spec.linestyle(mkey),
                    label=(f"{spec.label(mkey)} "
                           f"(r={np.nanmean(pears):.2f}±{np.nanstd(pears):.2f}, "
                           f"ρ={np.nanmean(spears):.2f}±{np.nanstd(spears):.2f})"))
            ax.fill_between(grid, mean - std, mean + std, color=color, alpha=0.22)
        ax.set_xlabel("Recurrent delay (time steps)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"Layer {li + 1}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle(f"Delay vs. {direction} recurrent weight — {group} combined "
                 f"(kernel-smoothed trend; band = mean ± std over seeds)")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, f"delay_vs_weight_trend_combined_{group}_{direction}.svg"))


def plot_delay_vs_weight_trend_combined(results, out_dir, spec, direction="outgoing"):
    """Overlay each delay type's models on shared axes, per layer, so learned vs fixed can be
    compared directly.
    """
    mkeys = [m for m in results if results[m]["layers_per_seed"]]
    if not mkeys:
        return
    groups = {"axonal": [], "synaptic": []}
    for m in mkeys:
        groups["synaptic" if _model_is_synaptic(results[m]) else "axonal"].append(m)
    for group, gmkeys in groups.items():
        if gmkeys:
            _plot_trend_combined_group(results, gmkeys, out_dir, spec, direction, group)


# --------------------------------------------------------------------------- #
# Recurrent weight distribution & spectral radius
# --------------------------------------------------------------------------- #
def plot_weight_spectrum(results, out_dir, spec):
    """Per layer/model: histogram of recurrent weight entries (pooled over seeds)
    and the spectral radius rho(W)=max|eig(W)| (per seed -> mean ± std), plus the
    largest singular value sigma_max(W) (worst-case one-step gain)."""
    spec_rows = []
    for mkey, data in results.items():
        seeds = data["seeds"]
        layers_per_seed = data["layers_per_seed"]
        if not layers_per_seed:
            continue
        n_layers = max(len(s) for s in layers_per_seed)

        fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
        axes = axes[0]
        for li in range(n_layers):
            ax = axes[li]
            pooled_w, radii, svmax = [], [], []
            for seed, seed_layers in zip(seeds, layers_per_seed):
                if li >= len(seed_layers):
                    continue
                W = seed_layers[li]["weights"]
                pooled_w.append(W.reshape(-1))
                eig = np.linalg.eigvals(W)
                radii.append(float(np.max(np.abs(eig))))
                svmax.append(float(np.linalg.svd(W, compute_uv=False)[0]))
            if not radii:
                continue
            pooled_w = np.concatenate(pooled_w)
            radii, svmax = np.array(radii), np.array(svmax)
            spec_rows.append({
                "model": mkey, "layer": li + 1, "n_seeds": len(radii),
                "spectral_radius_mean": radii.mean(), "spectral_radius_std": radii.std(),
                "sigma_max_mean": svmax.mean(), "sigma_max_std": svmax.std(),
                "w_std": float(pooled_w.std()),
            })
            ax.hist(pooled_w, bins=80, density=True, color=spec.color(mkey), alpha=0.8,
                    hatch=spec.hatch(mkey))
            ax.set_xlabel("Recurrent weight value")
            ax.set_ylabel("Density")
            ax.set_title(f"Layer {li + 1}\n"
                         f"ρ(W)={radii.mean():.2f}±{radii.std():.2f}, "
                         f"σmax={svmax.mean():.2f}±{svmax.std():.2f}")
            ax.grid(alpha=0.3)
        fig.suptitle(f"Recurrent weight distribution & spectrum — {spec.label(mkey)}")
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"weight_spectrum_{mkey}.svg"))
    return spec_rows


def plot_weight_spectrum_combined(results, out_dir, spec):
    """Superpose the recurrent weight distributions of every model on shared
    axes, per layer. Writes weight_spectrum_combined.svg. Ρ(W)/σmax
    (mean ± std over seeds) per model are shown in the legend."""
    mkeys = [m for m in results if results[m]["layers_per_seed"]]
    if not mkeys:
        return
    n_layers = max(max(len(s) for s in results[m]["layers_per_seed"]) for m in mkeys)

    fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
    axes = axes[0]
    for li in range(n_layers):
        ax = axes[li]
        pooled, stats = {}, {}
        for mkey in mkeys:
            ws, radii, svmax = [], [], []
            for seed_layers in results[mkey]["layers_per_seed"]:
                if li >= len(seed_layers):
                    continue
                W = seed_layers[li]["weights"]
                ws.append(W.reshape(-1))
                radii.append(float(np.max(np.abs(np.linalg.eigvals(W)))))
                svmax.append(float(np.linalg.svd(W, compute_uv=False)[0]))
            if not ws:
                continue
            pooled[mkey] = np.concatenate(ws)
            stats[mkey] = (np.array(radii), np.array(svmax))
        if not pooled:
            continue
        allw = np.concatenate(list(pooled.values()))
        bins = np.linspace(allw.min(), allw.max(), 80)  # shared bins for a fair overlay
        for mkey in mkeys:
            if mkey not in pooled:
                continue
            radii, svmax = stats[mkey]
            ax.hist(pooled[mkey], bins=bins, density=True, color=spec.color(mkey),
                    alpha=0.5, hatch=spec.hatch(mkey),
                    label=(f"{spec.label(mkey)} "
                           f"(ρ={radii.mean():.2f}±{radii.std():.2f}, "
                           f"σmax={svmax.mean():.2f}±{svmax.std():.2f})"))
        ax.set_xlabel("Recurrent weight value")
        ax.set_ylabel("Density")
        ax.set_title(f"Layer {li + 1}")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
    fig.suptitle("Recurrent weight distribution and spectrum, combined")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "weight_spectrum_combined.svg"))


# --------------------------------------------------------------------------- #
# Signed delay-weight relation
# --------------------------------------------------------------------------- #
def plot_signed_delay_weight(results, out_dir, spec):
    """Relation between delay and the signed recurrent weight."""
    rows = []
    for mkey, data in results.items():
        seeds = data["seeds"]
        layers_per_seed = data["layers_per_seed"]
        if not layers_per_seed:
            continue
        n_layers = max(len(s) for s in layers_per_seed)
        is_syn = any(_is_synaptic(s[0]) for s in layers_per_seed if s)
        seed_colors = dict(zip(seeds, style.ordinal_colors(len(seeds))))

        fig, axes = plt.subplots(1, n_layers, figsize=(4.5 * n_layers, 4), squeeze=False)
        axes = axes[0]
        for li in range(n_layers):
            ax = axes[li]
            spears = []
            for seed, seed_layers in zip(seeds, layers_per_seed):
                if li >= len(seed_layers):
                    continue
                delays = np.asarray(seed_layers[li]["delays"])
                W = seed_layers[li]["weights"]
                if delays.ndim == 1:
                    exc = np.clip(W, 0, None).sum(axis=0)
                    inh = np.clip(-W, 0, None).sum(axis=0)
                    d, yv = delays, exc / (exc + inh + 1e-12)
                else:
                    d, yv = delays.reshape(-1), W.reshape(-1)
                _, spear = _pearson_spearman(d, yv)
                spears.append(spear)
                rows.append({"model": mkey, "layer": li + 1, "seed": seed,
                             "spearman_delay_vs_signed": spear})
                ax.scatter(d, yv, s=8, alpha=0.4, color=seed_colors[seed],
                           label=f"seed {seed} (ρ={spear:.2f})")
            if not spears:
                continue
            spears = np.array(spears)
            ax.axhline(0.5 if not is_syn else 0.0, color="gray", ls="--", lw=0.8)
            ax.set_xlabel("Recurrent delay (time steps)")
            ax.set_ylabel("Excitatory fraction of outgoing mass" if not is_syn
                          else "Synaptic weight (signed)")
            ax.set_title(f"Layer {li + 1}\nSpearman ρ={np.nanmean(spears):.2f}±{np.nanstd(spears):.2f}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
        kind = "excitatory/inhibitory outgoing balance" if not is_syn else "signed synaptic weight"
        fig.suptitle(f"Delay vs. {kind} — {spec.label(mkey)}")
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"signed_delay_weight_{mkey}.svg"))
    return rows


# --------------------------------------------------------------------------- #
# Cross-seed delay distribution stability (KS distance)
# --------------------------------------------------------------------------- #
def _ks_2samp(a, b):
    """Two-sample Kolmogorov–Smirnov statistic (max CDF gap), scipy-free."""
    a = np.sort(a)
    b = np.sort(b)
    grid = np.concatenate([a, b])
    cdf_a = np.searchsorted(a, grid, side="right") / len(a)
    cdf_b = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(cdf_a - cdf_b)))


def cross_seed_stability(results, out_dir, spec):
    """Mean pairwise KS distance between seed delay-histograms, per layer/model
    (lower = more stable)."""
    rows = []
    for mkey, data in results.items():
        layers_per_seed = data["layers_per_seed"]
        if len(layers_per_seed) < 2:
            continue
        n_layers = max(len(s) for s in layers_per_seed)
        for li in range(n_layers):
            delays = [_delays_flat(s[li]) for s in layers_per_seed if li < len(s)]
            ks_vals = [_ks_2samp(delays[i], delays[j])
                       for i in range(len(delays)) for j in range(i + 1, len(delays))]
            if ks_vals:
                rows.append({"model": mkey, "layer": li + 1, "n_pairs": len(ks_vals),
                             "ks_mean": float(np.mean(ks_vals)),
                             "ks_std": float(np.std(ks_vals)),
                             "ks_max": float(np.max(ks_vals))})

    df = pd.DataFrame(rows)
    if not df.empty:
        within = df[df["model"].isin(results.keys())]
        layers = sorted(within["layer"].unique())
        mkeys = [m for m in results if m in within["model"].values]
        x = np.arange(len(layers))
        w = 0.8 / max(len(mkeys), 1)
        fig, ax = plt.subplots(figsize=(6, 4))
        for i, mkey in enumerate(mkeys):
            sub = within[within["model"] == mkey].set_index("layer")
            means = [sub.loc[l, "ks_mean"] if l in sub.index else 0 for l in layers]
            stds = [sub.loc[l, "ks_std"] if l in sub.index else 0 for l in layers]
            ax.bar(x + i * w, means, w, yerr=stds, capsize=4,
                   label=spec.label(mkey), color=spec.color(mkey), hatch=spec.hatch(mkey))
        ax.set_xticks(x + w * (len(mkeys) - 1) / 2)
        ax.set_xticklabels([f"Layer {l}" for l in layers])
        ax.set_ylabel("Mean pairwise KS distance (across seeds)")
        ax.set_title("Cross-seed delay-distribution stability (lower = more stable)")
        ax.legend()
        ax.grid(axis="y", alpha=0.3)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, "cross_seed_ks.svg"))
    return rows


# --------------------------------------------------------------------------- #
# Orchestrator: every figure + CSV from a built ``trained_models`` dict
# --------------------------------------------------------------------------- #
def run_recdel_analysis(results, out_dir, spec, dataset):
    """Produce every recdel figure + summary CSV from a built ``trained_models`` dict."""
    os.makedirs(out_dir, exist_ok=True)

    plot_accuracy(results, out_dir, spec, dataset)
    plot_delay_distributions(results, out_dir, spec)
    corr_out, corr_out_seed = plot_delay_vs_weight(results, out_dir, spec, "outgoing")
    corr_in, corr_in_seed = plot_delay_vs_weight(results, out_dir, spec, "incoming")
    plot_delay_vs_weight_trend_combined(results, out_dir, spec, "outgoing")
    plot_delay_vs_weight_trend_combined(results, out_dir, spec, "incoming")
    corr_rows = corr_out + corr_in
    corr_rows_seed = corr_out_seed + corr_in_seed
    spec_rows = plot_weight_spectrum(results, out_dir, spec)
    plot_weight_spectrum_combined(results, out_dir, spec)
    signed_rows = plot_signed_delay_weight(results, out_dir, spec)
    ks_rows = cross_seed_stability(results, out_dir, spec)

    # ---- accuracy summary ----
    use_test = any(data.get("test_accs") for data in results.values())
    acc_word = "TEST" if use_test else "VALIDATION (best ckpt)"
    print(f"\n=== {dataset} {acc_word} accuracy summary (mean ± std over seeds) ===")
    for mkey, data in results.items():
        accs = data["test_accs"] if use_test else data["best_accs"]
        if accs:
            sd = np.std(accs, ddof=1) if len(accs) > 1 else 0.0
            print(f"  {spec.label(mkey):20s}: {np.mean(accs):.2f} ± {sd:.2f} "
                  f"(n={len(accs)})  per-seed={[round(a, 2) for a in accs]}")

    # ---- per-layer delay statistics ----
    delay_stat_rows = []
    for mkey, data in results.items():
        if not data["layers_per_seed"]:
            continue
        n_layers = max(len(s) for s in data["layers_per_seed"])
        for li in range(n_layers):
            pooled = np.concatenate([_delays_flat(s[li]) for s in data["layers_per_seed"]
                                     if li < len(s)])
            delay_stat_rows.append({
                "model": mkey, "layer": li + 1,
                "mean": pooled.mean(), "std": pooled.std(),
                "median": np.median(pooled), "max": pooled.max(),
                "frac_zero": float(np.mean(pooled < 0.5)),
            })
    delay_stats = pd.DataFrame(delay_stat_rows)
    delay_stats.to_csv(os.path.join(out_dir, "delay_stats.csv"), index=False)
    if not delay_stats.empty:
        print("\n=== Per-layer recurrent delay statistics (pooled over seeds) ===")
        print(delay_stats.to_string(index=False))

    if corr_rows:
        pd.DataFrame(corr_rows).to_csv(os.path.join(out_dir, "delay_weight_corr.csv"), index=False)
        pd.DataFrame(corr_rows_seed).to_csv(
            os.path.join(out_dir, "delay_weight_corr_per_seed.csv"), index=False)
        print("\n=== Delay vs. weight correlation (mean ± std over seeds) ===")
        for r in corr_rows:
            print(f"  {spec.label(r['model']):20s} {r['direction']:8s} layer {r['layer']}: "
                  f"Spearman ρ = {r['spearman_mean']:.3f} ± {r['spearman_std']:.3f}, "
                  f"Pearson r = {r['pearson_mean']:.3f} ± {r['pearson_std']:.3f} "
                  f"(n_seeds={r['n_seeds']})")

    if spec_rows:
        pd.DataFrame(spec_rows).to_csv(os.path.join(out_dir, "weight_spectrum.csv"), index=False)
        print("\n=== Recurrent weight spectrum (mean ± std over seeds) ===")
        for r in spec_rows:
            print(f"  {spec.label(r['model']):20s} layer {r['layer']}: "
                  f"ρ(W) = {r['spectral_radius_mean']:.3f} ± {r['spectral_radius_std']:.3f}, "
                  f"σmax = {r['sigma_max_mean']:.3f} ± {r['sigma_max_std']:.3f}, "
                  f"w_std = {r['w_std']:.4f}")

    if signed_rows:
        signed = pd.DataFrame(signed_rows)
        signed.to_csv(os.path.join(out_dir, "signed_delay_weight.csv"), index=False)
        agg = (signed.groupby(["model", "layer"])["spearman_delay_vs_signed"]
               .agg(["mean", "std", "count"]).reset_index())
        print("\n=== Delay vs. signed weight (mean ± std over seeds) ===")
        for _, r in agg.iterrows():
            print(f"  {spec.label(r['model']):20s} layer {int(r['layer'])}: "
                  f"Spearman ρ = {r['mean']:.3f} ± {r['std']:.3f} (n_seeds={int(r['count'])})")

    if ks_rows:
        pd.DataFrame(ks_rows).to_csv(os.path.join(out_dir, "cross_seed_ks.csv"), index=False)
        print("\n=== Cross-seed delay-distribution stability (mean pairwise KS, lower = stabler) ===")
        for r in ks_rows:
            print(f"  {spec.label(r['model']):20s} layer {r['layer']}: "
                  f"KS = {r['ks_mean']:.3f} ± {r['ks_std']:.3f} "
                  f"(max {r['ks_max']:.3f}, n_pairs={r['n_pairs']})")

    print(f"\nFigures (SVG) + CSVs written to {out_dir}/")
