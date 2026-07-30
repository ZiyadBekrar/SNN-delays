"""Layer-resolved permutation ablation of the trained delays.

Shuffles a fraction of one layer's delays across its neurons, leaving every
weight, the connectivity and the layer's whole delay distribution untouched. Only
the assignment of delays to neurons is destroyed, which is precisely what delay
learning creates, so a drop cannot be blamed on removed capacity, the confound
in any pruning lesion.

The fixed-delay family is an exact null: its delays are an i.i.d. draw, so
permuting them is a no-op in distribution and whatever degradation it shows is
the measurement floor.

Outputs:  figures/SSC/permute_delays_depth/
Usage:    python experiments/make_figures/SSC/permute_delays_depth_sweep.py [--plots-only]
"""

import argparse
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from common.delay_perturb import (evaluate_delay_perturbed, permute_op, realized_dose,
                                  PERMUTE_SELECTIONS, OPERATORS)
from common.delay_perturb import _recdel_modules
from common.perturb import (aggregate, plot_perf_vs_magnitude,
                            plot_degradation_vs_magnitude)
from common.utils import save_fig
from common.style import type_of
from common import style
import runs as ssc
from common import paths

from delrec.utils import calc_metric_SSC

# ======================= TUNE THE SWEEP HERE ============================== #
PERMUTE_FRACS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0]
# Which entries get shuffled (see common.delay_perturb.PERMUTE_SELECTIONS). Quantile
# bands, not absolute delay thresholds: at a fixed threshold the learned and fixed
# families would have different proportions of branches above it, so the comparison
# would be confounded by unequal amounts of surgery, exactly the flaw the pruning
# sweep has to correct for post hoc. At a fixed quantile they are matched by
# construction. Frac in {0, 1} is selection-independent and is only evaluated once.
SELECTIONS = list(PERMUTE_SELECTIONS)
# Independent permutation realizations per fraction (frac=0 is the identity, so one
# realization there is enough. The rest give a SEM over permutation draws).
N_PERTURB_SEEDS = 3
# Fraction used for the degradation-vs-layer summary figure (must be in the grid).
SUMMARY_FRAC = 0.75
# Which model families to sweep, as keys of config.MODEL_PREFIXES. None = all
# four. Set e.g. ["ax_learned", "ax_fixed"] to run the axonal pair only. The
# figures are already split per delay type, so a restricted run simply yields
# fewer subtrees. Overridable with --families.
FAMILIES = ["ax_learned", "ax_fixed"]
MODEL_SEEDS = None
CKPT = "best.pth"
FORWARD_VERSION = "triton_exact"
# ========================================================================= #

MAG_LABEL = OPERATORS["permute"][1]


def _mag_label(selection=None):
    """X-axis label naming which delays the fraction refers to."""
    return {
        "long":   "Shuffled fraction q  (the longest q of the layer's delays)",
        "short":  "Shuffled fraction q  (the shortest q of the layer's delays)",
        "random": "Shuffled fraction q  (a random q of the layer's delays)",
    }.get(selection, "Shuffled fraction q  (which delays: see legend)")



def _collect_rows():
    """Sweep every (family, seed, layer, fraction, permutation seed)."""
    rows = []
    # The SSC sparse .h5 decode dominates a test pass. The inputs are identical across
    # every family/seed/lesion, so decode once and reuse the materialised batches.
    cached_loader = None
    for mkey in (FAMILIES or list(ssc.MODEL_PREFIXES)):
        run_dirs = ssc.resolve_run_dirs(mkey, ".", MODEL_SEEDS)
        if not run_dirs:
            print(f"[warn] no runs for {mkey}")
            continue
        for seed in sorted(run_dirs):
            model, config, device, kernel = ssc.build_test_model(
                mkey, run_dirs[seed], CKPT, forward_version_override=FORWARD_VERSION)
            if cached_loader is None:
                print("Caching SSC test set (one-time decode)...", flush=True)
                cached_loader = list(ssc.get_test_loader(config))
            loader = cached_loader
            n_layers = len(_recdel_modules(model))
            print(f"\n{ssc.SPEC.label(mkey)} | seed {seed} | {n_layers} recurrent "
                  f"layers | kernel={kernel}")
            for layer_index in range(n_layers):
              for selection in SELECTIONS:
                for frac in PERMUTE_FRACS:
                    # frac 0 (identity) and frac 1 (whole layer) are the same lesion
                    # for every selection, at frac 1 all three shuffle the entire
                    # layer, so they differ only in which random draw is realized, not
                    # in distribution. Evaluate them once, under the first selection.
                    if frac in (0.0, 1.0) and selection != SELECTIONS[0]:
                        continue
                    n_rep = 1 if frac == 0.0 else N_PERTURB_SEEDS
                    for prep in range(n_rep):
                        pseed = 1000 * prep + 7
                        acc = evaluate_delay_perturbed(
                            loader, model, device, calc_metric_SSC,
                            permute_op(frac, selection), seed=pseed,
                            layer_index=layer_index)
                        dose = realized_dose(model, permute_op(frac, selection),
                                             pseed, device, layer_index)
                        rows.append({"model": mkey, "layer": layer_index + 1,
                                     "selection": selection, "magnitude": frac,
                                     "model_seed": seed, "perturb_seed": prep,
                                     "dose": dose, "acc": acc})
                        print(f"   {mkey:11s} s{seed} L{layer_index+1} "
                              f"{selection:6s} frac={frac:<4g} "
                              f"draw {prep+1}/{n_rep} (rng {pseed:>4d}): "
                              f"acc={acc:6.2f}%  dose={dose:5.2f}", flush=True)
                    if n_rep > 1:
                        accs = [r["acc"] for r in rows[-n_rep:]]
                        doses = [r["dose"] for r in rows[-n_rep:]]
                        print(f"   {'':11s}  \u2514 mean over {n_rep} draws: "
                              f"acc={np.mean(accs):6.2f} +/-{np.std(accs):.2f}%  "
                              f"dose={np.mean(doses):5.2f} steps", flush=True)
    return rows


def _type_groups(rows):
    """Model keys present in ``rows``, grouped by delay type, in MODEL_PREFIXES order."""
    present = {r["model"] for r in rows}
    groups = {}
    for k in ssc.MODEL_PREFIXES:
        if k in present:
            groups.setdefault(type_of(k), []).append(k)
    return groups


def _plot_layers_overlay(rows, out_dir, order, dtype):
    """All layers on one axis: accuracy vs permuted fraction, panel per condition."""
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    conds = [c for c in ("learned", "fixed")
             if any(style.condition_of(m) == c for m in order)]
    layers = sorted(df["layer"].unique())
    colors = style.ordinal_colors(len(layers))
    fig, axes = plt.subplots(1, len(conds), figsize=(5.2 * len(conds), 4.2),
                             squeeze=False, sharey=True)
    for ax, cond in zip(axes[0], conds):
        mkeys = [m for m in order if style.condition_of(m) == cond]
        for mkey in mkeys:
            for li, layer in enumerate(layers):
                sub = df[(df["model"] == mkey) & (df["layer"] == layer)]
                g = sub.groupby("magnitude")["acc"].agg(
                    m="mean", sd="std", n="count").sort_index()
                sem = (g["sd"] / np.sqrt(g["n"].clip(lower=1))).fillna(0.0)
                ax.plot(g.index, g["m"], color=colors[li], lw=1.9, marker="o", ms=4,
                        label=f"Layer {layer}")
                ax.fill_between(g.index, g["m"] - sem, g["m"] + sem,
                                color=colors[li], alpha=0.22, lw=0)
                # frac 1 is the shared whole-layer limit of all three selections,
                # open marker, so it is not mistaken for a band-restricted point.
                if 1.0 in g.index:
                    ax.plot([1.0], [g.loc[1.0, "m"]], marker="o", ms=8, mfc=style.SURFACE,
                            mec=colors[li], mew=1.6, ls="none")
        ax.set_xlabel(_mag_label(df["selection"].mode()[0]))
        ax.set_title(", ".join(ssc.SPEC.label(m) for m in mkeys))
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="lower left")
    axes[0][0].set_ylabel("Test accuracy (%)")
    fig.suptitle(f"SSC {dtype}: delay permutation per layer (mean +/- SEM)\n"
                 f"open marker at 1.0 = whole layer shuffled (all bands coincide)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, os.path.join(out_dir, "permute_layers_overlay.svg"))


def _plot_degradation_vs_layer(rows, out_dir, order, dtype):
    """Headline: accuracy drop at SUMMARY_FRAC vs layer, learned vs fixed."""
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    base = (df[df["magnitude"] == 0.0]
            .groupby(["model", "layer", "model_seed"])["acc"].mean())
    cur = df[df["magnitude"] == SUMMARY_FRAC].copy()
    cur["drop"] = cur.apply(
        lambda r: base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"], axis=1)
    # Average the permutation draws within a model seed first, so the SEM is over
    # trained models (the unit of replication), not over permutation realizations.
    per_seed = cur.groupby(["model", "layer", "model_seed"])["drop"].mean().reset_index()
    g = (per_seed.groupby(["model", "layer"])["drop"]
         .agg(mean="mean", sd="std", n="count").reset_index())
    g["sem"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    g.to_csv(os.path.join(out_dir, "permute_degradation_vs_layer.csv"), index=False)

    layers = sorted(g["layer"].unique())
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mkey in [m for m in order if m in set(g["model"])]:
        sub = g[g["model"] == mkey].set_index("layer")
        means = [sub.loc[l, "mean"] if l in sub.index else np.nan for l in layers]
        sems = [sub.loc[l, "sem"] if l in sub.index else 0.0 for l in layers]
        ax.errorbar(x, means, yerr=sems, capsize=4, marker="o", lw=1.8,
                    color=ssc.SPEC.color(mkey), ls=ssc.SPEC.linestyle(mkey),
                    label=ssc.SPEC.label(mkey))
    ax.set_xticks(x)
    ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Recurrent layer (depth)")
    ax.set_ylabel(f"Accuracy drop (pts) — {SUMMARY_FRAC:g} of delays permuted")
    ax.set_title(f"SSC {dtype}: cost of scrambling the delay→neuron assignment")
    ax.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "permute_degradation_vs_layer.svg"))


# Selection is a nominal variable here (not ordered), and within one delay-type figure
# color is free. The type is already fixed by the panel. Gray = the unbiased
# baseline, so a colored line above it means that band is more structured than average.
SELECTION_STYLE = {"random": (style.INK_MUTED, "o"),
                   "long": (style.BLUE, "^"),
                   "short": (style.ORANGE, "v")}


def _rows_for_selection(rows, selection):
    """Rows of one selection, plus the frac-0 and frac-1 anchors it shares. Both anchors
    genuinely belong to every selection's curve: at frac 0 nothing is shuffled, and at frac
    1 the top-q and bottom-q bands have both grown to the whole layer, so all three
    selections coincide there.
    """
    shared = [dict(r, selection=selection) for r in rows
              if r["magnitude"] in (0.0, 1.0)]
    return [r for r in rows if r["selection"] == selection
            and r["magnitude"] not in (0.0, 1.0)] + shared


def _plot_full_shuffle_vs_layer(rows, out_dir, order, dtype):
    """Accuracy drop from shuffling a layer's delays ENTIRELY, vs depth."""
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    base = (df[df["magnitude"] == 0.0]
            .groupby(["model", "layer", "model_seed"])["acc"].mean())
    cur = df[df["magnitude"] == 1.0].copy()
    if cur.empty:
        return
    cur["drop"] = cur.apply(
        lambda r: base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"], axis=1)
    per_seed = cur.groupby(["model", "layer", "model_seed"])["drop"].mean().reset_index()
    g = (per_seed.groupby(["model", "layer"])["drop"]
         .agg(mean="mean", sd="std", n="count").reset_index())
    g["sem"] = g["sd"] / np.sqrt(g["n"].clip(lower=1))
    g.to_csv(os.path.join(out_dir, "permute_full_shuffle_vs_layer.csv"), index=False)

    layers = sorted(g["layer"].unique())
    x = np.arange(len(layers))
    fig, ax = plt.subplots(figsize=(6, 4.2))
    for mkey in [m for m in order if m in set(g["model"])]:
        sub = g[g["model"] == mkey].set_index("layer")
        ax.errorbar(x, [sub.loc[l, "mean"] for l in layers],
                    yerr=[sub.loc[l, "sem"] for l in layers], capsize=4, marker="o",
                    lw=1.8, color=ssc.SPEC.color(mkey), ls=ssc.SPEC.linestyle(mkey),
                    label=ssc.SPEC.label(mkey))
    ax.set_xticks(x); ax.set_xticklabels([f"Layer {l}" for l in layers])
    ax.set_xlabel("Recurrent layer (depth)")
    ax.set_ylabel("Accuracy drop (pts)")
    ax.set_title(f"SSC {dtype}: cost of fully scrambling one layer's\ndelay-to-neuron assignment (mean +/- SEM over seeds)")
    ax.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "permute_full_shuffle_vs_layer.svg"))


def _plot_selection_compare(rows, out_dir, order, dtype):
    """Which part of the delay distribution is deliberately placed?"""
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    base = (df[df["magnitude"] == 0.0]
            .groupby(["model", "layer", "model_seed"])["acc"].mean())
    df = df[df["magnitude"] > 0.0].copy()
    df["drop"] = df.apply(
        lambda r: base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"], axis=1)

    conds = [c for c in ("learned", "fixed")
             if any(style.condition_of(m) == c for m in order)]
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(len(conds), len(layers), squeeze=False, sharey=True,
                             figsize=(3.6 * len(layers), 3.4 * len(conds)))
    for ri, cond in enumerate(conds):
        mkeys = [m for m in order if style.condition_of(m) == cond]
        for ci, layer in enumerate(layers):
            ax = axes[ri][ci]
            for sel, (col, mk) in SELECTION_STYLE.items():
                sub = df[(df["model"].isin(mkeys)) & (df["layer"] == layer)]
                sub = pd.DataFrame(_rows_for_selection(sub.to_dict("records"), sel))
                if sub.empty:
                    continue
                # Average permutation draws within a model seed first, so the SEM is
                # over trained models, the unit of replication.
                per_seed = (sub.groupby(["magnitude", "model_seed"])["drop"]
                            .mean().reset_index())
                g = (per_seed.groupby("magnitude")["drop"]
                     .agg(m="mean", sd="std", n="count").sort_index())
                sem = (g["sd"] / np.sqrt(g["n"].clip(lower=1))).fillna(0.0)
                ax.plot(g.index, g["m"], color=col, marker=mk, ms=5, lw=1.8, label=sel)
                ax.fill_between(g.index, g["m"] - sem, g["m"] + sem,
                                color=col, alpha=0.18, lw=0)
                # At frac 1 every band has grown to the whole layer, so the three
                # curves necessarily meet on one shared measurement. Drawn hollow so
                # the convergence is not misread as three agreeing estimates.
                if 1.0 in g.index:
                    ax.plot([1.0], [g.loc[1.0, "m"]], marker=mk, ms=9, mfc=style.SURFACE,
                            mec=col, mew=1.5, ls="none")
            ax.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
            ax.grid(alpha=0.3)
            if ri == 0:
                ax.set_title(f"Layer {layer}")
            if ri == len(conds) - 1:
                ax.set_xlabel(_mag_label())
            if ci == 0:
                ax.set_ylabel(f"{cond}\naccuracy drop (pts)")
    axes[0][0].legend(fontsize=8, title="shuffled band", title_fontsize=8)
    fig.suptitle(f"SSC {dtype}: which delays carry the assignment structure?\n"
                 f"hollow marker at 1.0 = whole layer (bands coincide; one shared measurement)")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    save_fig(fig, os.path.join(out_dir, "permute_selection_compare.svg"))


def _plot_dose_response(rows, out_dir, order, dtype):
    """Accuracy drop against the realized dose (mean |delta delay| actually induced)."""
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    if "dose" not in df.columns:
        return
    base = (df[df["magnitude"] == 0.0]
            .groupby(["model", "layer", "model_seed"])["acc"].mean())
    df = df[df["magnitude"] > 0.0].copy()
    df["drop"] = df.apply(
        lambda r: base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"], axis=1)
    learned = [m for m in order if style.condition_of(m) == "learned"]
    layers = sorted(df["layer"].unique())
    fig, axes = plt.subplots(1, len(layers), squeeze=False, sharey=True,
                             figsize=(3.8 * len(layers), 3.6))
    for ci, layer in enumerate(layers):
        ax = axes[0][ci]
        for sel, (col, mk) in SELECTION_STYLE.items():
            sub = df[(df["model"].isin(learned)) & (df["layer"] == layer)]
            sub = pd.DataFrame(_rows_for_selection(sub.to_dict("records"), sel))
            if sub.empty:
                continue
            g = sub.groupby("magnitude")[["dose", "drop"]].mean().sort_values("dose")
            ax.plot(g["dose"], g["drop"], color=col, marker=mk, ms=5, lw=1.8, label=sel)
            if 1.0 in set(sub["magnitude"]):
                pt = g.loc[sub.groupby("magnitude")[["dose", "drop"]].mean().index == 1.0]
                ax.plot(pt["dose"], pt["drop"], marker=mk, ms=9, mfc=style.SURFACE,
                        mec=col, mew=1.5, ls="none")
        ax.set_title(f"Layer {layer}")
        ax.set_xlabel("Realized dose: mean |$\\Delta$ delay| (steps)")
        ax.grid(alpha=0.3)
    axes[0][0].set_ylabel("Accuracy drop (pts)")
    axes[0][0].legend(fontsize=8, title="shuffled band", title_fontsize=8)
    fig.suptitle(f"SSC {dtype}: drop vs realized delay movement (learned)\n"
                 f"hollow marker = whole-layer shuffle (bands coincide)")
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    save_fig(fig, os.path.join(out_dir, "permute_dose_response.svg"))


def _band_decomposition_rows(rows, order):
    """Per (model, layer, seed): the exact long/short partition and the joint shuffle.
    ``long(q)`` shuffles the top-q quantile of a layer's delays and ``short(1-q)`` the
    bottom (1-q): disjoint sets whose union is the whole layer.
    """
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    base = (df[df["magnitude"] == 0.0]
            .groupby(["model", "layer", "model_seed"])["acc"].mean())
    df = df[df["magnitude"] > 0].copy()
    df["drop"] = df.apply(
        lambda r: base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"], axis=1)
    s = df.groupby(["model", "layer", "selection", "magnitude", "model_seed"])["drop"].mean()

    out = []
    qs = sorted({q for q in df["magnitude"].unique()
                 if 0 < q < 1 and round(1 - q, 4) in set(df["magnitude"].unique())})
    for (mk, ly), _ in df.groupby(["model", "layer"]):
        try:
            full = s.loc[(mk, ly, "random", 1.0)]
        except KeyError:
            continue
        for q in qs:
            try:
                a = s.loc[(mk, ly, "long", q)]
                b = s.loc[(mk, ly, "short", round(1 - q, 4))]
            except KeyError:
                continue
            for seed in a.index.intersection(b.index).intersection(full.index):
                out.append(dict(model=mk, layer=ly, q=q, seed=seed,
                                long_only=a[seed], short_only=b[seed],
                                parts=a[seed] + b[seed], full=full[seed],
                                cross=1.0 - (a[seed] + b[seed]) / full[seed]))
    return pd.DataFrame(out)


def _msem(g, col):
    a = g.groupby(level=0)[col].agg(m="mean", sd="std", n="count")
    return a["m"], a["sd"] / np.sqrt(a["n"].clip(lower=1))


def plot_band_decomposition(rows, out_dir, order, dtype, q_main=0.5):
    """Is any single delay band necessary, or only the combination of all of them?"""
    d = _band_decomposition_rows(rows, order)
    if d.empty:
        return
    d.to_csv(os.path.join(out_dir, "permute_band_decomposition.csv"), index=False)

    conds = [c for c in ("learned", "fixed")
             if any(style.condition_of(m) == c for m in order)]
    layers = sorted(d["layer"].unique())
    BARS = [("long_only", f"long band only (top {q_main:g})", style.BLUE, None),
            ("short_only", f"short band only (bottom {1 - q_main:g})", style.ORANGE, None),
            ("parts", "both, separately\n(every delay moved, within its band)",
             style.INK_MUTED, "///"),
            ("full", "all delays shuffled together\n(bands mixed)", style.INK_PRIMARY, None)]

    fig, axes = plt.subplots(len(conds), 2, squeeze=False,
                             figsize=(11.2, 4.0 * len(conds)),
                             gridspec_kw={"width_ratios": [1.35, 1]})
    for ri, cond in enumerate(conds):
        mkeys = [m for m in order if style.condition_of(m) == cond]
        sub = d[(d["model"].isin(mkeys)) & (d["q"] == q_main)]
        ax = axes[ri][0]
        x = np.arange(len(layers)); w = 0.2
        for bi, (col, lab, c, hatch) in enumerate(BARS):
            g = sub.set_index("layer")
            m, e = _msem(g, col)
            ax.bar(x + (bi - 1.5) * w, [m.get(l, np.nan) for l in layers], w,
                   yerr=[e.get(l, 0.0) for l in layers], capsize=3, color=c,
                   hatch=hatch, edgecolor="white" if hatch else c, linewidth=0.8,
                   label=lab if ri == 0 else None)
        ax.set_xticks(x); ax.set_xticklabels([f"Layer {l}" for l in layers])
        ax.set_ylabel(f"{cond}\naccuracy drop (pts)")
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(top=ax.get_ylim()[1] * 1.42)   # headroom so the legend clears the bars
        if ri == 0:
            ax.legend(fontsize=7.5, loc="upper left", framealpha=0.95, ncol=2)

        ax2 = axes[ri][1]
        colors = style.ordinal_colors(len(layers))
        for li, layer in enumerate(layers):
            g = d[(d["model"].isin(mkeys)) & (d["layer"] == layer)].set_index("q")
            m, e = _msem(g, "cross")
            ax2.errorbar(m.index, 100 * m.values, yerr=100 * e.values, capsize=3,
                         marker="o", lw=1.8, color=colors[li], label=f"Layer {layer}")
        ax2.set_ylim(0, 100)
        ax2.axhline(0.0, color=style.BASELINE, lw=0.9, ls=":")
        ax2.set_ylabel("% of damage requiring\ncross-band mixing")
        ax2.grid(alpha=0.3)
        if ri == 0:
            ax2.legend(fontsize=8)
        if ri == len(conds) - 1:
            axes[ri][0].set_xlabel("Recurrent layer (depth)")
            ax2.set_xlabel("Split point q   (long = top q, short = bottom 1-q)")
    fig.suptitle(f"SSC {dtype}: no single delay band is necessary --\n"
                 f"the damage lies in the combination across bands")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    save_fig(fig, os.path.join(out_dir, "permute_band_decomposition.svg"))


# One step above style.RC_LARGE_TEXT: this figure is a sub-panel of a larger paper
# figure, so it is reduced once more on the page and its text has to survive that.
# Sizes only. Every color, spine and grid convention still comes from style.RC.
PANEL_RC = {**style.RC_LARGE_TEXT,
            "font.size": 15, "axes.titlesize": 17.5, "axes.labelsize": 16,
            "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
            # Arial first, matching common.delay_depth's profile panel: the two are
            # panels of one paper figure and must not resolve to different fonts.
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]}


def _selection_title(selection):
    """Panel title naming which delays move and what the move is."""
    band = {"long": "Longest", "short": "Shortest", "random": "Random"}.get(
        selection, selection.capitalize())
    return f"{band} q of delays\nshuffled between neurons"


LAYER_MARKERS = ["o", "s", "^", "D"]   # redundant channel: depth is color AND shape


def _degradation_frame(rows, order):
    """Relative degradation (%) per row, against each run's own clean accuracy."""
    df = pd.DataFrame([r for r in rows if r["model"] in order])
    base = (df[df["magnitude"] == 0.0]
            .groupby(["model", "layer", "model_seed"])["acc"].mean())
    df = df.copy()
    df["degr"] = df.apply(
        lambda r: 100.0 * (base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"])
        / base.loc[(r["model"], r["layer"], r["model_seed"])], axis=1)
    return df


def _degradation_ylim(rows, order):
    """One y-limit across every selection, so the long/short panels are comparable."""
    df = _degradation_frame(rows, order)
    top = df.groupby(["model", "layer", "selection", "magnitude"])["degr"].mean().max()
    return (0.0, float(top) * 1.12)


def _plot_degradation_layers_overlay(rows, out_dir, order, dtype, selection, ylim=None):
    """All layers and both conditions on one square axis, sized as a paper sub-panel.
    Readability rests on three choices: one translucent fill per layer (the
    learned-vs-fixed excess, SEM is drawn as error bars, since three layers x two
    conditions x a band is nine overlapping purples).
    """
    from matplotlib.ticker import MaxNLocator

    df = _degradation_frame(rows, order)
    learned = [m for m in order if style.condition_of(m) == "learned"]
    fixed = [m for m in order if style.condition_of(m) == "fixed"]
    layers = sorted(df["layer"].unique())
    colors = style.ordinal_colors(len(layers))

    with plt.rc_context(style.RC_PAPER):
        fig, ax = plt.subplots(figsize=(3.2, 2.5), constrained_layout=True)
        for li, layer in enumerate(layers):
            mk = LAYER_MARKERS[li % len(LAYER_MARKERS)]
            curves = {}
            for cond, mkeys in (("learned", learned), ("fixed", fixed)):
                if not mkeys:
                    continue
                sub = df[(df["model"].isin(mkeys)) & (df["layer"] == layer)]
                g = sub.groupby("magnitude")["degr"].agg(
                    m="mean", sd="std", n="count").sort_index()
                sem = (g["sd"] / np.sqrt(g["n"].clip(lower=1))).fillna(0.0)
                solid = cond == "learned"
                ax.errorbar(g.index, g["m"], yerr=sem, color=colors[li],
                            ls="-" if solid else (0, (4, 1.8)),
                            lw=1.5 if solid else 1.1, marker=mk, ms=4.0,
                            mfc=colors[li] if solid else style.SURFACE,
                            mec=colors[li], mew=1.0, elinewidth=0.7, capsize=1.8,
                            zorder=3, label=cond if li == 0 else None)
                curves[cond] = g["m"]
            if len(curves) == 2:
                idx = curves["learned"].index.intersection(curves["fixed"].index)
                ax.fill_between(idx, curves["fixed"][idx], curves["learned"][idx],
                                color=colors[li], alpha=0.16, lw=0, zorder=1)
            if "learned" in curves:
                ax.annotate(f"L{layer}",
                            (curves["learned"].index[-1], curves["learned"].iloc[-1]),
                            textcoords="offset points", xytext=(4, -1),
                            color=colors[li], fontsize=8, fontweight="bold",
                            va="center")

        if ylim:
            ax.set_ylim(*ylim)
        ax.set_xlim(-0.04, 1.16)
        ax.set_xticks([0.0, 0.5, 1.0])
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))   # 8 pt ticks: fewer, larger
        ax.axhline(0.0, color=style.BASELINE, lw=0.8, ls=":", zorder=0)
        ax.set_xlabel("Shuffled fraction q")
        ax.set_ylabel("Relative degradation (%)")
        ax.set_title(_selection_title(selection), fontweight="bold",
                     fontsize=8.5, pad=3)
        ax.grid(alpha=0.22)
        ax.legend(loc="upper left", frameon=False, handlelength=1.6,
                  borderpad=0.15, labelspacing=0.3, handletextpad=0.5)
        save_fig(fig, os.path.join(out_dir, "permute_degradation_layers_overlay.svg"))


def _ablation_reference(order, out_root_cfg):
    """Relative accuracy loss (%) from removing each layer's recurrence outright."""
    path = os.path.join(out_root_cfg, "prune_delays_depth", "prune_depth_per_run.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path)
    d = d[(d["direction"] == "short") & (d["model"].isin(order))]
    if d.empty:
        return None
    base = d[d["magnitude"] == d["magnitude"].min()].set_index(
        ["model", "layer", "model_seed"])["acc"]
    full = d.loc[d.groupby(["model", "layer", "model_seed"])["pruned_frac"].idxmax()].copy()
    full["rel"] = full.apply(
        lambda r: 100.0 * (base.loc[(r["model"], r["layer"], r["model_seed"])] - r["acc"])
        / base.loc[(r["model"], r["layer"], r["model_seed"])], axis=1)
    return full.groupby("layer")["rel"].mean().to_dict()


def _plot_degradation_layer_panels(rows, out_dir, order, dtype, selection, ylim=None,
                                   out_root_cfg=None):
    """Per-layer facets, sized to drop into a parent figure as one sub-panel. Two lines per
    panel is the readability floor: nothing overlaps, so the learned-vs-fixed gap stays
    legible.
    """
    df = _degradation_frame(rows, order)
    learned = [m for m in order if style.condition_of(m) == "learned"]
    fixed = [m for m in order if style.condition_of(m) == "fixed"]
    layers = sorted(df["layer"].unique())
    colors = style.ordinal_colors(len(layers))
    ablation = _ablation_reference(order, out_root_cfg) if out_root_cfg else None

    with plt.rc_context(PANEL_RC):
        # Tall panels: the SEM band spans ~1-3% of each panel's y range, so its
        # physical height is what makes it readable, at 5.2 in it is ~1.5-4 mm.
        fig, axes = plt.subplots(1, len(layers), squeeze=False, sharey=False,
                                 figsize=(3.5 * len(layers), 5.2))
        for li, layer in enumerate(layers):
            ax = axes[0][li]
            mk = LAYER_MARKERS[li % len(LAYER_MARKERS)]
            curves = {}
            for cond, mkeys in (("learned", learned), ("fixed", fixed)):
                if not mkeys:
                    continue
                sub = df[(df["model"].isin(mkeys)) & (df["layer"] == layer)]
                g = sub.groupby("magnitude")["degr"].agg(
                    m="mean", sd="std", n="count").sort_index()
                sem = (g["sd"] / np.sqrt(g["n"].clip(lower=1))).fillna(0.0)
                solid = cond == "learned"
                # SEM as a band, not as caps: at this reproduction size error-bar caps
                # vanish under the markers, whereas a band scales with the panel's own
                # zoom and stays visible. With only two curves per panel the shading
                # channel is free, so it carries uncertainty and nothing else, the
                # learned-vs-fixed difference is simply the gap between the two lines.
                ax.plot(g.index, g["m"], color=colors[li],
                        ls="-" if solid else (0, (4, 1.8)),
                        lw=2.1 if solid else 1.6, marker=mk, ms=5.2,
                        mfc=colors[li] if solid else style.SURFACE,
                        mec=colors[li], mew=1.4, zorder=3,
                        label=cond if li == 0 else None)
                ax.fill_between(g.index, g["m"] - sem, g["m"] + sem, color=colors[li],
                                alpha=0.38 if solid else 0.24, lw=0, zorder=2)
                curves[cond] = g["m"]
            if ablation and layer in ablation:
                y = ablation[layer]
                # Identified in the subtitle rather than labeled three times.
                ax.axhline(y, color=style.INK_SECONDARY, lw=1.3, ls=(0, (6, 3)),
                           zorder=2)
                ax.set_ylim(top=max(ax.get_ylim()[1], y * 1.15))
            ax.axhline(0.0, color=style.BASELINE, lw=0.8, ls=":", zorder=0)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks([0.0, 0.5, 1.0])
            ax.set_title(f"L{layer}", color=colors[li], fontweight="bold", pad=4)
            ax.grid(alpha=0.22)
            ax.set_ylabel("Relative degradation (%)")
            if li == 0:
                ax.legend(loc="upper left", frameon=False, handlelength=1.7,
                          borderpad=0.2, labelspacing=0.3)
        fig.supxlabel("Shuffled fraction q", color=style.INK_SECONDARY)
        fig.suptitle(_selection_title(selection).replace("\n", " "),
                     fontweight="bold", y=0.99)
        n_seeds = df["model_seed"].nunique()
        n_draws = df.loc[df["magnitude"] > 0, "perturb_seed"].nunique()
        fig.text(0.5, 0.895,
                 f"line = mean over {n_seeds} seeds x {n_draws} shuffles, "
                 f"band = +/- SEM;  dashed = that layer's recurrence removed entirely",
                 ha="center", va="top", fontsize=11.5, color=style.INK_SECONDARY)
        fig.tight_layout(rect=[0, 0.04, 1, 0.885])
        save_fig(fig, os.path.join(out_dir, "permute_degradation_layer_panels.svg"))


def run_plots(rows, out_root):
    for dtype, order in _type_groups(rows).items():
        type_rows = [r for r in rows if r["model"] in order]
        # Cross-selection figures first: they are the point of the sweep.
        dir_root = os.path.join(out_root, dtype)
        os.makedirs(dir_root, exist_ok=True)
        deg_ylim = _degradation_ylim(type_rows, order)
        for selection in sorted({r["selection"] for r in type_rows}):
            sel_rows = _rows_for_selection(type_rows, selection)
            sel_root = os.path.join(out_root, dtype, selection)
            os.makedirs(sel_root, exist_ok=True)
            for layer in sorted({r["layer"] for r in sel_rows}):
                out_dir = os.path.join(sel_root, f"layer{layer}")
                os.makedirs(out_dir, exist_ok=True)
                layer_rows = [r for r in sel_rows if r["layer"] == layer]
                summary = aggregate(layer_rows, ["model", "magnitude"])
                summary.to_csv(os.path.join(out_dir, "permute_summary.csv"), index=False)
                title = f"SSC {dtype} L{layer} — permute {selection} delays"
            _plot_degradation_layer_panels(sel_rows, sel_root, order, dtype,
                                           selection, ylim=deg_ylim,
                                           out_root_cfg=ssc.OUT_ROOT)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-root",
                    default=os.path.join(ssc.OUT_ROOT, "permute_delays_depth"))
    ap.add_argument("--families", nargs="+", default=None,
                    choices=list(ssc.MODEL_PREFIXES),
                    help="Restrict to these families (default: FAMILIES, else all).")
    ap.add_argument("--seeds", type=int, nargs="+", default=None)
    ap.add_argument("--plots-only", action="store_true",
                    help="Skip evaluation, reload permute_depth_per_run.csv and re-render.")
    args = ap.parse_args()
    global MODEL_SEEDS, FAMILIES
    if args.seeds is not None:
        MODEL_SEEDS = args.seeds
    if args.families is not None:
        FAMILIES = args.families
    os.makedirs(args.out_root, exist_ok=True)

    per_run_csv = os.path.join(args.out_root, "permute_depth_per_run.csv")
    if args.plots_only:
        if not os.path.exists(per_run_csv):
            # fall back to the measurement recorded with the repository, so this
            # figure can be redrawn without a GPU or the dataset
            shipped = paths.measurements("SSC", "permute_delays_depth",
                                         "permute_depth_per_run.csv")
            if os.path.exists(shipped):
                print(f"  using the recorded measurement: {shipped}")
                per_run_csv = shipped
            else:
                print(f"[error] --plots-only but no cached results at {per_run_csv}")
                return
        rows = pd.read_csv(per_run_csv).to_dict("records")
        if FAMILIES:
            rows = [r for r in rows if r["model"] in FAMILIES]
    else:
        rows = _collect_rows()
        if not rows:
            print("[error] no results.")
            return
        # Merge with any previous run rather than overwrite: sweeping one delay type
        # now and the other later must not discard the first result. Rows for the
        # families just swept are replaced. Every other family is carried over.
        swept = {r["model"] for r in rows}
        if os.path.exists(per_run_csv):
            prev = pd.read_csv(per_run_csv)
            kept = prev[~prev["model"].isin(swept)]
            if not kept.empty:
                print(f"[merge] carrying over {len(kept)} cached rows for "
                      f"{sorted(kept['model'].unique())}")
                rows = kept.to_dict("records") + rows
        pd.DataFrame(rows).to_csv(per_run_csv, index=False)
        if FAMILIES:
            rows = [r for r in rows if r["model"] in FAMILIES]
    run_plots(rows, args.out_root)
    print(f"\nFigures (SVG) + CSVs written to {args.out_root}/")


if __name__ == "__main__":
    main()
