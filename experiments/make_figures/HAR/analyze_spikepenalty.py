"""The accuracy/energy trade-off under a firing-rate penalty.

Reads the spike-penalty sweep, in which a penalty on the hidden spike trains
drives the firing rate down. Because the penalty moves every family's firing rate
almost identically, matched penalty is close to matched energy, which is what
makes an accuracy-versus-firing-rate frontier meaningful.

The sweep also trains a delay-free RSNN. It sits far below every delay family and
stretches the axes, so each figure is drawn on model subsets rather than on all
five at once.

Outputs:  figures/HAR/lam/spike_penalty/
Usage:    python experiments/make_figures/HAR/analyze_spikepenalty.py [--lams 0 0.1 1]
"""

import argparse
import glob
import json
import os
import sys


# Make ``experiments/make_figures/`` importable so ``common`` / ``config`` resolve.
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import common.utils as cu
from common import paths
from common import style as st
import runs as har


SWEEP_ROOT = paths.runs("HAR/spike_penalty_sweep")
CKPT = "best.pth"  # the checkpoint behind final_test.json

# Shared accuracy-threshold grid for the per-threshold figures (energy_ratio and
# energy_ratio_curve, and their weight-decay twins in analyze_weightdecay): 80% → 83%
# in 0.5 steps. Every curve is drawn on this one axis. A model that cannot reach a
# threshold simply has no point there.
THRESHOLD_START, THRESHOLD_STEP, THRESHOLD_STOP = 80.0, 0.5, 83.0

# Whether the per-threshold firing rate is interpolated between the two straddling λs
# (default) or snapped to the sparsest sampled λ that reaches the threshold
# (``--no-interp``). Same trade-off as in ``analyze_weightdecay``: interpolating keeps
# the fixed/learned spike ratio off the λ grid, snapping reports only firing rates a
# trained run actually achieved. See ``cu.iso_crossing``.
INTERPOLATE = True

# Accuracy floor of the hero frontier figure (``energy_frontier``): the paper panel is
# cropped to the region the delay families compete in. A family whose whole curve lies
# below it is not drawn at all (the delay-free vanilla RSNN, ~12 points down), and with
# it goes its iso-accuracy arrow, that comparison lives in ``energy_ratio_all_vanilla``
# and the ``synaptic_learned_vs_vanilla`` variant's other figures. Twin of
# ``analyze_weightdecay.ACC_FLOOR``.
ACC_FLOOR = 80.0
# In-panel tag size under RC_LARGE_TEXT (matches its legend size). The helpers' default
# of 8 pt is sized for RC and disappears in a paper panel.
PANEL_ANNOT_FS = 12

# Paper-panel style, same one step above style.RC_LARGE_TEXT as common.delay_depth's
# profile panel and the depth-sweep panels: this figure is a sub-panel of a larger paper
# figure and is reduced once more on the page, so its text must be oversized relative to
# the axes. Arial first, svg.fonttype='none' keeps the text as text, so the SVG carries
# the family list and resolves to real Arial in Illustrator. Sizes only. Every color,
# spine and grid convention still comes from style.RC.
PANEL_RC = {**st.RC_LARGE_TEXT,
            "font.size": 15, "axes.titlesize": 17.5, "axes.labelsize": 16,
            "legend.fontsize": 13, "xtick.labelsize": 14, "ytick.labelsize": 14,
            "font.sans-serif": ["Arial", "Helvetica", "Nimbus Sans", "DejaVu Sans"]}
FIGSIZE_PANEL = (5.6, 4.6)   # same canvas as the per-type hero frontier panel

# model key -> delrec.networks class name. The four delay families share their keys with
# har.MODEL_INFO. ``vanilla`` is the extra family this sweep trains and is kept out of
# har.MODEL_INFO on purpose (no recurrent delays -> the delay views are degenerate).
FAMILIES = {
    "ax_learned":  "SNN_recurrent_delays",
    "ax_fixed":    "SNN_fixed_recurrent_delays",
    "syn_learned": "SNN_synaptic_recurrent_delays",
    "syn_fixed":   "SNN_fixed_synaptic_recurrent_delays",
    "vanilla":     "SNN_vanilla_recurrent",
}
VANILLA = "vanilla"
# type -> (learned key, fixed key): the two learned-vs-fixed comparisons.
TYPES = {"axonal": ("ax_learned", "ax_fixed"),
         "synaptic": ("syn_learned", "syn_fixed")}
LEARNED_KEYS = [l for l, _ in TYPES.values()]

LABELS = dict(har.SPEC.labels, **{VANILLA: "Vanilla RSNN (no delays)"})


def _color(mkey):
    """Color = delay type (style grammar). The delay-free control is gray."""
    return st.INK_MUTED if mkey == VANILLA else st.TYPE_COLORS[st.type_of(mkey)]


def _ls(mkey):
    """Line style = condition (learned solid / fixed dashed). Vanilla dotted."""
    return ":" if mkey == VANILLA else st.CONDITION_LS[st.condition_of(mkey)]


def _marker(mkey):
    """Marker = delay type (style grammar). The delay-free control is a triangle."""
    if mkey == VANILLA:
        return "^"
    return st.TYPE_MARKERS[st.type_of(mkey)]


def _run_style(mkey):
    """``(marker, facecolor)`` for the individual runs scattered behind a Pareto frontier."""
    if mkey == VANILLA:
        return "^", "none"
    if st.condition_of(mkey) == "fixed":
        return "s", "none"
    return "o", _color(mkey)


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _parse_lam(dirname):
    """'lam0.3' -> 0.3 . 'lam0' -> 0.0."""
    return float(dirname[3:])


def discover_lam_runs(sweep_root, model_class, seeds):
    """{lam: {seed: run_dir}} for one swept model class."""
    out = {}
    for lam_dir in sorted(glob.glob(os.path.join(sweep_root, model_class, "lam*"))):
        if not os.path.isdir(lam_dir):
            continue
        lam = _parse_lam(os.path.basename(lam_dir))
        per_seed = {seed: os.path.join(lam_dir, f"seed{seed}") for seed in seeds
                    if os.path.isdir(os.path.join(lam_dir, f"seed{seed}"))}
        if per_seed:
            out[lam] = per_seed
    return dict(sorted(out.items()))


def _read_final_test(run_dir):
    """acc / firing_rate / spike_cost of the tested model (None if not finished)."""
    path = os.path.join(run_dir, "final_test.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        d = json.load(f)
    return {"acc": float(d["acc"]), "firing_rate": float(d["firing_rate"]),
            "spike_cost": float(d["spike_cost"])}


def _pooled_delays(run_dir, ckpt):
    """All recurrent delays of a run pooled across layers -> 1-D array (or None)."""
    layers = cu.load_recurrent_params(run_dir, ckpt)
    if not layers:
        return None
    return np.concatenate([l["delays"].reshape(-1) for l in layers])


def build_rows(sweep_root, families, seeds, ckpt, lams=None):
    """One row per run: metrics from final_test.json + pooled delay stats from the
    checkpoint. Returns the row list (each row keeps its raw ``delays`` array for the
    histograms)."""
    rows = []
    for mkey in families:
        runs = discover_lam_runs(sweep_root, FAMILIES[mkey], seeds)
        if not runs:
            print(f"[warn] no runs for {mkey} under "
                  f"{os.path.join(sweep_root, FAMILIES[mkey])}")
            continue
        n = 0
        for lam, per_seed in runs.items():
            if lams is not None and lam not in lams:
                continue
            for seed, rd in per_seed.items():
                res = _read_final_test(rd)
                if res is None:
                    print(f"[warn] skipping {rd} (no final_test.json)")
                    continue
                delays = _pooled_delays(rd, ckpt)
                rows.append({
                    "model": mkey, "type": st.type_of(mkey) if mkey != VANILLA else "none",
                    "condition": st.condition_of(mkey) if mkey != VANILLA else "vanilla",
                    "lam": lam, "seed": seed, **res,
                    "delay_mean": float(delays.mean()) if delays is not None else np.nan,
                    "delay_std": float(delays.std()) if delays is not None else np.nan,
                    "delay_max": float(delays.max()) if delays is not None else np.nan,
                    "delays": delays,
                })
                n += 1
        print(f"  {LABELS.get(mkey, mkey):26s} {n:3d} runs over "
              f"{len({r['lam'] for r in rows if r['model'] == mkey})} λ values")
    return rows


# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _agg(df, mkey, field):
    """Sorted λs and per-λ (mean, SEM) of ``field`` across seeds, for one family."""
    sub = df[df["model"] == mkey]
    lams = np.array(sorted(sub["lam"].unique()))
    mu, sem = [], []
    for lam in lams:
        v = sub.loc[sub["lam"] == lam, field].to_numpy(dtype=float)
        mu.append(v.mean())
        sem.append(v.std(ddof=0) / max(np.sqrt(len(v)), 1))
    return lams, np.array(mu), np.array(sem)


def _frame_to_curves(ax, thresholds, ys, pad=0.06):
    """Set the axes to the curves' extent (curve ± band) with a fixed relative pad,
    rather than letting matplotlib autoscale. Twin of
    ``analyze_weightdecay._frame_to_curves``. Anything outside is clipped."""
    if not ys:
        return
    x0, x1 = float(min(thresholds)), float(max(thresholds))
    y0, y1 = float(min(ys)), float(max(ys))
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or max(abs(y0), 1.0)
    ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
    ax.set_ylim(y0 - pad * dy, y1 + pad * dy)


def _curve(df, mkey, agg="mean"):
    """One family's frontier as ``(lams, acc, acc_sem, firing, firing_sem)``. Both axes are
    per-seed quantities (the firing rate as much as the accuracy) so how the seeds of a λ
    are collapsed is a real choice: ``agg='mean'`` the λ's seed mean on both axes, with the
    SEM over seeds on each (the firing-rate SEM is what makes the energy axis' own seed
    spread visible).
    """
    sub = df[df["model"] == mkey]
    if agg.startswith("pareto"):
        s = sub.sort_values("firing_rate", kind="stable")
        run = np.maximum.accumulate(s["acc"].to_numpy())
        if agg == "pareto_interp":  # keep only the runs that beat every cheaper one
            keep = run > np.r_[-np.inf, run[:-1]]
            s, run = s[keep], run[keep]
        z = np.zeros(len(s))
        return s["lam"].to_numpy(), run, z, s["firing_rate"].to_numpy(), z
    if agg == "best":
        best = sub.loc[sub.groupby("lam")["acc"].idxmax()].sort_values("lam")
        z = np.zeros(len(best))
        return (best["lam"].to_numpy(), best["acc"].to_numpy(), z,
                best["firing_rate"].to_numpy(), z)
    lams, acc, sem = _agg(df, mkey, "acc")
    _, fir, fsem = _agg(df, mkey, "firing_rate")
    return lams, acc, sem, fir, fsem


def _min_firing_at(df, mkey, target, agg="mean"):
    """The cheapest firing rate at which ``mkey``'s frontier reaches ``target`` accuracy,
    walking up from the sparse end, under the same seed aggregation as the curve it
    annotates (see ``_curve``). The energy analog of
    ``analyze_weightdecay._min_budget_at``: accuracy is non-monotonic in λ (a mild penalty
    helps), so we threshold the per-λ values rather than assuming the last λ is the
    cheapest that works.
    """
    _, acc, _, fir, _ = _curve(df, mkey, agg)
    return cu.iso_crossing(fir, acc, target,
                           interpolate=INTERPOLATE and agg != "pareto")


def _seed_firing_crossings(df, mkey, target):
    """Per-seed min firing rate reaching ``target``, as an array over seeds (or None). Each
    seed was trained at every λ of the sweep, so it is an independent replicate of the
    whole energy sweep: this reads the crossing off each seed's own (firing rate, accuracy)
    frontier.
    """
    sub = df[df["model"] == mkey]
    out = []
    for _, g in sub.groupby("seed"):
        c = cu.iso_crossing(g["firing_rate"].to_numpy(dtype=float),
                            g["acc"].to_numpy(dtype=float), target,
                            interpolate=INTERPOLATE)
        if c is None:
            return None
        out.append(c)
    return np.array(out, dtype=float) if out else None


def _firing_crossing_stats(df, mkey, target):
    """(mean, sem) over seeds of ``_seed_firing_crossings``, or None. SEM convention as
    everywhere else in this module: population std / sqrt(n)."""
    c = _seed_firing_crossings(df, mkey, target)
    if c is None:
        return None
    return float(c.mean()), float(c.std(ddof=0) / np.sqrt(len(c)))


def _types_with(df, need_fixed=True):
    """The delay types actually loaded, so ``--families`` subsets don't leave empty
    panels (or ask for a learned-vs-fixed comparison that has no fixed runs)."""
    have = set(df["model"])
    return {t: (l, f) for t, (l, f) in TYPES.items()
            if l in have and (f in have or not need_fixed)}


# --------------------------------------------------------------------------- #
# λ-axis helpers (log scale, but λ=0 has no log position -> place it as a labeled
# tick a decade below the smallest positive λ). Mirrors analyze_weightdecay._wd_x.
# --------------------------------------------------------------------------- #
def _lam_x(lams):
    lams = np.asarray(lams, dtype=float)
    pos = lams[lams > 0]
    floor = (pos.min() / 3.0) if pos.size else 1e-2
    return np.where(lams > 0, lams, floor)


def _set_lam_axis(ax, lams):
    ax.set_xscale("log")
    lams = np.asarray(sorted(set(np.asarray(lams, dtype=float))))
    ax.set_xticks(_lam_x(lams))
    ax.set_xticklabels([("0" if l == 0 else f"{l:g}") for l in lams])
    ax.set_xlabel("spike penalty λ")


def _plot_vs_lam(df, models, field, ylabel, title, out_path, normalize=False):
    """Shared body of the three "<metric> vs λ" line plots: one errorbar line per
    family (color = delay type, style = condition), mean ± SEM over seeds. Drawn as a
    paper panel on the same convention as ``analyze_weightdecay.plot_compression_ratio_all``
    square axes, RC_LARGE_TEXT, open markers for the fixed control."""
    with plt.rc_context(st.RC_LARGE_TEXT):
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        all_lams = set()
        for mkey in models:
            if mkey not in set(df["model"]):
                continue
            lams, mu, sem = _agg(df, mkey, field)
            if normalize:  # express as % of this family's own λ=0 value
                base = mu[list(lams).index(0.0)] if 0.0 in lams else mu[0]
                mu, sem = 100.0 * mu / base, 100.0 * sem / base
            all_lams |= set(lams)
            ax.errorbar(_lam_x(lams), mu, yerr=sem, ls=_ls(mkey), marker=_marker(mkey),
                        mfc=st.SURFACE if st.condition_of(mkey) == "fixed" else None,
                        color=_color(mkey), capsize=2,
                        label=LABELS.get(mkey, mkey))
        if not all_lams:
            plt.close(fig)
            return
        _set_lam_axis(ax, sorted(all_lams))
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False)
        fig.tight_layout()
        cu.save_fig(fig, out_path)


# --------------------------------------------------------------------------- #
# Plots. The λ axis (what the penalty does)
# --------------------------------------------------------------------------- #

def plot_firing_vs_lam(df, models, out_dir):
    """Firing rate vs λ. The curves nearly coincide across families, so λ sets the
    spike budget largely independently of the delay condition, which is what makes
    the learned-vs-fixed comparison at matched λ a matched-energy comparison."""
    # Label + title wrapped to two lines, as in compression_ratio_all: at panel text
    # size the one-line forms overrun the square axes and collide with each other.
    _plot_vs_lam(df, models, "firing_rate",
                 "hidden firing rate\n(spikes/neuron/step)",
                 "HAR: firing rate vs spike penalty\n— λ sets the budget for every family",
                 os.path.join(out_dir, "firing_vs_lam.svg"))



# --------------------------------------------------------------------------- #
# Plots, the energy axis (the frontier)
# --------------------------------------------------------------------------- #
def _panels_w(types):
    """Width of a one-panel-per-delay-type figure. A lone panel still needs room for
    the suptitle (the single-type model subsets would otherwise come out cramped)."""
    return max(4.75 * len(types), 6.2)


def _annotate_lam(ax, x, y, lam, side, fontsize=8, dy=-13):
    """Tag a point of a frontier with its λ, offset away from the plot edge. ``dy`` puts
    the tag below (default) or above the point, at the λ=0 end of a cropped panel the
    curves stack, and below is where the other family's line runs."""
    dx = -5 if side == "right" else 5
    ax.annotate(f"λ={lam:g}", (x, y), textcoords="offset points", xytext=(dx, dy),
                ha=("right" if side == "right" else "left"), fontsize=fontsize,
                color=st.INK_SECONDARY)


def _iso_arrow(ax, df, learned, ref, fontsize=8, agg="mean"):
    """Anchor an iso-accuracy arrow at ``ref``'s best accuracy (its ceiling): how many fewer
    spikes ``learned`` needs to match it, measured under the same seed aggregation as the
    curves it sits on (``agg``, see ``_curve``).
    """
    have = set(df["model"])
    if learned not in have or ref not in have:
        return
    target = max(_curve(df, ref, agg)[1])
    lb = _min_firing_at(df, learned, target, agg)
    rb = _min_firing_at(df, ref, target, agg)
    if lb is None or rb is None or rb <= lb:
        return
    ax.axhline(target, ls=":", color=st.INK_MUTED, lw=0.8, alpha=0.6)
    ax.annotate("", xy=(lb, target), xytext=(rb, target),
                arrowprops=dict(arrowstyle="<->", color=st.INK_SECONDARY, lw=1.4))
    ax.text((lb + rb) / 2, target - 0.6, f"{rb / max(lb, 1e-9):.1f}× fewer spikes",
            ha="center", va="top", fontsize=fontsize, color=st.INK_SECONDARY,
            bbox=dict(facecolor=st.SURFACE, edgecolor="none", pad=1.5))


def plot_energy_frontier(df, out_dir, agg="mean"):
    """HERO, the energy/accuracy frontier: test accuracy vs firing rate, with λ traversed
    along each curve (λ=0 at the right / most spikes, λ=max at the left). One figure per
    delay type (``energy_frontier_axonal.svg`` / ``_synaptic.svg``, never the two on one
    figure): learned (solid ●) vs fixed (dashed ■) of that type.
    """
    from matplotlib.lines import Line2D

    types = _types_with(df, need_fixed=False)
    if not types:
        return
    pareto = agg.startswith("pareto")
    for dtype, (learned, fixed) in types.items():
        with plt.rc_context(st.RC_LARGE_TEXT):
            fig, ax = plt.subplots(figsize=(5.6, 4.6))
            drawn, handles = set(), []
            for mkey in (fixed, learned, VANILLA):
                if mkey not in set(df["model"]):
                    continue
                lams, acc, sem, fir, fsem = _curve(df, mkey, agg)
                if acc.max() < ACC_FLOOR:  # entirely below the crop, vanilla
                    continue
                drawn.add(mkey)
                if pareto:
                    # Every run of the family, keyed by shape + fill (see _run_style):
                    # faint for the dominated ones, solid for the frontier runs the
                    # envelope actually rests on.
                    mk, face = _run_style(mkey)
                    sub = df[df["model"] == mkey]
                    _, fro, _, xfro, _ = _curve(df, mkey, "pareto_interp")
                    ax.scatter(sub["firing_rate"], sub["acc"], s=16, marker=mk,
                               facecolors=face, edgecolors=_color(mkey), linewidths=0.8,
                               alpha=0.35, zorder=1)
                    ax.scatter(xfro, fro, s=45, marker=mk, facecolors=face,
                               edgecolors=_color(mkey), linewidths=1.2, zorder=4)
                    ax.plot(fir, acc, ls=_ls(mkey), color=_color(mkey), lw=1.9, zorder=3,
                            drawstyle="steps-post" if agg == "pareto" else "default")
                    handles.append(Line2D([], [], color=_color(mkey), ls=_ls(mkey), lw=1.9,
                                          marker=mk, mfc=face, mec=_color(mkey), ms=6,
                                          label=LABELS.get(mkey, mkey)))
                else:
                    ax.fill_between(fir, acc - sem, acc + sem, color=_color(mkey),
                                    alpha=0.10 if mkey != learned else 0.18, lw=0)
                    ax.errorbar(fir, acc, xerr=fsem, fmt="none", ecolor=_color(mkey),
                                elinewidth=1.0, alpha=0.5)
                    ax.plot(fir, acc, ls=_ls(mkey), marker=_marker(mkey),
                            mfc="none" if st.condition_of(mkey) == "fixed" else None,
                            color=_color(mkey), lw=1.9, ms=5,
                            label=LABELS.get(mkey, mkey))
                if mkey == learned and not pareto:  # anchor λ on the frontier itself
                    # λ=0 sits at the right end (most spikes), λ=max at the left, but
                    # tag the λ extremes that are still inside the crop, else the
                    # left tag lands under the panel with its point.
                    vis = np.flatnonzero(acc >= ACC_FLOOR)
                    _annotate_lam(ax, fir[vis[0]], acc[vis[0]], lams[vis[0]],
                                  side="right", fontsize=PANEL_ANNOT_FS, dy=9)
                    _annotate_lam(ax, fir[vis[-1]], acc[vis[-1]], lams[vis[-1]],
                                  side="left", fontsize=PANEL_ANNOT_FS)
            if not drawn:
                plt.close(fig)
                continue

            for ref in (fixed, VANILLA):  # an arrow only against a reference on the panel
                if ref in drawn:
                    _iso_arrow(ax, df, learned, ref, fontsize=PANEL_ANNOT_FS, agg=agg)
            ax.set_ylim(bottom=ACC_FLOOR)
            ax.set_xlabel("hidden firing rate (spikes/neuron/step)")
            ax.set_ylabel("test accuracy (%)")
            # The variant tag goes on the second line: appended to the first it overruns
            # the panel width at RC_LARGE_TEXT.
            note = {"best": "(best seed) ", "pareto": "(Pareto) ",
                    "pareto_interp": "(Pareto, interpolated) "}.get(agg, "")
            ax.set_title(f"HAR {dtype}: accuracy vs spike budget\n"
                         + note + "— learned needs fewer spikes", pad=8)
            ax.set_box_aspect(1)
            ax.grid(alpha=0.25, lw=0.5)
            # The frontier + its run cloud fill the corners differently per panel, so
            # the envelope figures let matplotlib place the legend. Their handles are
            # proxies carrying the runs' own marker, so a scattered point is
            # attributable to its condition.
            ax.legend(handles=handles or None, frameon=False,
                      loc="best" if pareto else "lower right", handlelength=1.8)
            fig.tight_layout()
            tag_file = "" if agg == "mean" else f"_{agg}"
            cu.save_fig(fig, os.path.join(
                out_dir, f"energy_frontier_{dtype}{tag_file}.svg"))







def plot_energy_ratio_all(df, models, out_dir, start=80.0, step=0.5, stop=THRESHOLD_STOP):
    """The per-type ``energy_ratio`` panels combined on one axes, every model head to head,
    which works because they share the firing-rate unit.
    """
    thresholds = np.arange(start, stop + 1e-9, step)
    with plt.rc_context(st.RC_LARGE_TEXT):
        # Panel-sized like energy_ratio's own panels, only wider: up to five curves and
        # their legend share one axes.
        fig, ax = plt.subplots(figsize=(5.6, 4.6))
        drawn = False
        curve_y = []
        for mkey in models:
            if mkey not in set(df["model"]):
                continue
            # As in energy_ratio, a curve stops where one of its seeds can no longer
            # reach the threshold at any λ, and vanilla usually never enters at all.
            pts = [(t, v) for t, v in ((t, _firing_crossing_stats(df, mkey, t))
                                       for t in thresholds) if v is not None]
            if not pts:
                continue
            drawn = True
            ts = np.array([p[0] for p in pts])
            mu = np.array([p[1][0] for p in pts])
            sem = np.array([p[1][1] for p in pts])
            ax.fill_between(ts, mu - sem, mu + sem, color=_color(mkey), alpha=0.15,
                            lw=0, zorder=2)
            ax.plot(ts, mu, ls=_ls(mkey), marker=_marker(mkey),
                    mfc=st.SURFACE if st.condition_of(mkey) == "fixed" else None,
                    color=_color(mkey), label=LABELS.get(mkey, mkey), zorder=3)
            curve_y += list(mu - sem) + list(mu + sem)
        if not drawn:
            plt.close(fig)
            return
        _frame_to_curves(ax, thresholds, curve_y)
        ax.set_xlabel("accuracy threshold (%)")
        ax.set_ylabel("min firing rate to reach it\n(spikes/neuron/step)")
        ax.set_title("HAR: spike budget needed per accuracy\n— learned delays need fewer")
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        cu.save_fig(fig, os.path.join(out_dir, "energy_ratio_all.svg"))



# --------------------------------------------------------------------------- #
# Plots. The delay axis (what the penalty does to the delays)
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# The figures the vanilla RSNN appears in, rendered once per model subset (see
# VARIANTS): the delay-free control is a floor ~12 accuracy points below every delay
# family, so it dominates the axes. Dropping it re-scales the same figure onto the
# region where the delay families actually differ. Keeping it against a single delay
# model isolates the delays-vs-no-delays claim.
# --------------------------------------------------------------------------- #
def plot_paper_panels(df, models, out_dir):
    """The accuracy/energy panels, for one model subset."""
    plot_firing_vs_lam(df, models, out_dir)          # firing rate vs the penalty
    plot_energy_ratio_all(df, models, out_dir)       # cheapest firing rate per accuracy
    plot_energy_frontier(df, out_dir, agg="mean")    # the accuracy/firing frontier


# subfolder -> the model subset it re-renders those figures on.
VARIANTS = {
    "no_vanilla": [m for m in FAMILIES if m != VANILLA],   # the four delay families
    # The synaptic type in full (learned vs its fixed control) against vanilla, so the
    # panel carries both spike gains: learned-vs-fixed and learned-vs-vanilla.
    "synaptic_learned_vs_vanilla": ["syn_learned", "syn_fixed", VANILLA],
}


# --------------------------------------------------------------------------- #
# Console summary
# --------------------------------------------------------------------------- #
def print_sweet_spot(df, models):
    """Per family: the λ with the best mean accuracy, and what it costs in spikes
    relative to λ=0, the "free lunch" region of the penalty."""
    print("\n---------- best-accuracy λ per family (vs its own λ=0) ----------")
    print(f"{'family':26s} {'λ*':>5s} {'acc':>7s} {'Δacc':>7s} {'firing':>8s} {'spike saving':>13s}")
    for mkey in models:
        if mkey not in set(df["model"]):
            continue
        lams, acc, _ = _agg(df, mkey, "acc")
        _, fir, _ = _agg(df, mkey, "firing_rate")
        i = int(np.argmax(acc))
        b = list(lams).index(0.0) if 0.0 in lams else 0
        print(f"{LABELS.get(mkey, mkey):26s} {lams[i]:>5g} {acc[i]:7.2f} "
              f"{acc[i] - acc[b]:+7.2f} {fir[i]:8.4f} {fir[b] / max(fir[i], 1e-9):12.2f}×")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", default=SWEEP_ROOT)
    ap.add_argument("--families", nargs="+", default=list(FAMILIES), choices=list(FAMILIES))
    ap.add_argument("--seeds", type=int, nargs="+", default=har.SEEDS)
    ap.add_argument("--lams", type=float, nargs="+", default=None,
                    help="Restrict to these spike-penalty values. Default: all found.")
    ap.add_argument("--ckpt", default=CKPT,
                    help="Checkpoint to read delays from (matches final_test.json).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/lam/spike_penalty"
                         " (+ '_snapped' with --no-interp).")
    ap.add_argument("--no-interp", action="store_true",
                    help="Report the sparsest sampled λ reaching each threshold instead "
                         "of interpolating the crossing (see INTERPOLATE).")
    args = ap.parse_args()

    global INTERPOLATE
    INTERPOLATE = not args.no_interp
    if args.out_dir is None:
        args.out_dir = os.path.join(har.OUT_ROOT, "lam",
                                    "spike_penalty" + ("" if INTERPOLATE else "_snapped"))

    print(f"Sweep root: {args.sweep_root}")
    rows = build_rows(args.sweep_root, args.families, args.seeds, args.ckpt, args.lams)
    if not rows:
        print("\nNo runs found, nothing to plot.")
        return
    df = pd.DataFrame([{k: v for k, v in r.items() if k != "delays"} for r in rows])

    os.makedirs(args.out_dir, exist_ok=True)
    models = [m for m in FAMILIES if m in set(df["model"])]  # keep the canonical order


    # The vanilla RSNN is ~12 pts below every delay family, so it stretches the axes and
    # squashes the region where the delay families differ. Re-render every figure it
    # appears in on two model subsets: the delay families alone (the comparison at full
    # resolution), and the best delay model against vanilla alone (the delays-vs-no-delays
    # statement on its own). Same figures, same code, only the model subset changes.
    for name, subset in VARIANTS.items():
        sub_models = [m for m in models if m in subset]
        if len(sub_models) < 2:
            continue
        sub_dir = os.path.join(args.out_dir, name)
        os.makedirs(sub_dir, exist_ok=True)
        plot_paper_panels(df[df["model"].isin(sub_models)], sub_models, sub_dir)
        print(f"  variant '{name}': {', '.join(LABELS.get(m, m) for m in sub_models)}")

    df.to_csv(os.path.join(args.out_dir, "spike_penalty_summary.csv"), index=False)
    # SEM with ddof=0 (pandas' .sem() uses ddof=1) so the CSV matches the error bars.
    sem = lambda v: v.std(ddof=0) / max(np.sqrt(len(v)), 1)
    fields = ["acc", "firing_rate", "spike_cost", "delay_mean", "delay_std", "delay_max"]
    agg = (df.groupby(["model", "type", "condition", "lam"])
             .agg(**{f"{f}_{n}": (f, a) for f in fields
                     for n, a in (("mean", "mean"), ("sem", sem))})
             .reset_index())
    agg.to_csv(os.path.join(args.out_dir, "spike_penalty_agg.csv"), index=False)

    print_sweet_spot(df, models)
    print(f"\nFigures (SVG) + CSVs written to {args.out_dir}/")


if __name__ == "__main__":
    main()
