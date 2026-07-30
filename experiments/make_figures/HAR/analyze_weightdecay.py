"""Compressing learned delays with weight decay on the delay parameters.

Reads the weight-decay sweep and, for each delay type, the already-trained
fixed-delay baseline it is compared against. Fixed delays are never updated, so
they cannot be compressed this way and are shrunk by a smaller initialization
instead. That separate sweep is what the frontier plots read as their control.

Outputs:  figures/HAR/wd/weight_decay/
Usage:    python experiments/make_figures/HAR/analyze_weightdecay.py [--families axonal]
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


SWEEP_ROOT = paths.runs("HAR/weight_decay_sweep")
FIXED_COMP_ROOT = paths.runs("HAR/fixed_compression_sweep")  # the fixed-delay small-scale sweep

# Accuracy-threshold grid for the per-threshold figures (compression_ratio and
# compression_ratio_curve), kept equal to analyze_spikepenalty's THRESHOLD_* so its
# energy twins share the same x-axis. Curves stop early where a condition cannot reach
# a threshold at all, synaptic fixed peaks at ~82.8%, so it never reaches 83.
THRESHOLD_START, THRESHOLD_STEP, THRESHOLD_STOP = 80.0, 0.5, 83.0

# Whether the per-threshold budget is interpolated between the two straddling configs
# (default) or snapped to the shallowest sampled config that reaches the threshold
# (``--no-interp``). Interpolating keeps the fixed/learned ratio off the sweep grid.
# Snapping reports only depths a trained configuration actually achieved. See
# ``cu.iso_crossing``.
INTERPOLATE = True

# Accuracy floor of the hero frontier figure (``compression_frontier``): the paper panel
# is cropped to the region every condition competes in. The low-budget tail runs down to
# ~74% and squashes the plateau where learned and fixed actually separate. Curves are
# clipped there, not dropped. Twin of ``analyze_spikepenalty.ACC_FLOOR``.
ACC_FLOOR = 80.0
# In-panel tag size under RC_LARGE_TEXT (matches its legend size). The module default of
# 8 pt is sized for RC and disappears in a paper panel.
PANEL_ANNOT_FS = 12

# Whether the compression pools also include the unconstrained delay_std_init sweep
# (learned and fixed runs trained at std 3..23 with no compression pressure), or only
# the two sweeps that deliberately shorten the delays: weight decay for the learned
# families, small-scale initializations for the fixed ones (``--compression-only``).
# The restricted version answers "how far can each condition be compressed by the
# mechanism available to it", with no help from runs that were never compressed.
INCLUDE_STD_SWEEP = True

# family -> (swept learned model class, fixed-baseline model key in har.MODEL_INFO)
FAMILIES = {
    "axonal":   ("SNN_recurrent_delays",           "ax_fixed"),
    "synaptic": ("SNN_synaptic_recurrent_delays",  "syn_fixed"),
}
FIXED_STD = 8  # the fixed-delay baselines to compare against (matches the sweep's delay_std_init)

# family -> learned-delay key in har.MODEL_INFO for the delay_std_init sweep
# (same model class as the wd sweep, but trained WITHOUT weight decay across a
# range of init stds). Used to compare "with weight decay" vs "without".
LEARNED_STD_KEY = {"axonal": "ax_learned", "synaptic": "syn_learned"}
# Only overlay the no-weight-decay runs up to this init std, larger inits (13/18/23)
# have much bigger delays and stretch the axes, hiding the small-delay region where
# the weight-decay route lives.
NOWD_MAX_STD = 8


# --------------------------------------------------------------------------- #
# Data loading
# --------------------------------------------------------------------------- #
def _pooled_delays(run_dir, ckpt):
    """All recurrent delays of a run pooled across layers -> 1-D array (or None)."""
    layers = cu.load_recurrent_params(run_dir, ckpt)
    if not layers:
        return None
    return np.concatenate([l["delays"].reshape(-1) for l in layers])


def _parse_wd(dirname):
    """'wd0.001' -> 0.001 . 'wd0' -> 0.0."""
    return float(dirname[2:])


def discover_wd_runs(sweep_root, model_class, seeds):
    """{wd: {seed: run_dir}} for a swept model class."""
    out = {}
    for wd_dir in sorted(glob.glob(os.path.join(sweep_root, model_class, "wd*"))):
        if not os.path.isdir(wd_dir):
            continue
        wd = _parse_wd(os.path.basename(wd_dir))
        per_seed = {}
        for seed in seeds:
            rd = os.path.join(wd_dir, f"seed{seed}")
            if os.path.isdir(rd):
                per_seed[seed] = rd
        if per_seed:
            out[wd] = per_seed
    return dict(sorted(out.items()))


def build_family_results(family, sweep_root, exp_dir, seeds, ckpt,
                         fixed_comp_root=FIXED_COMP_ROOT):
    """Return a dict with the swept runs + the fixed baseline for one family."""
    model_class, fixed_key = FAMILIES[family]
    wd_runs = discover_wd_runs(sweep_root, model_class, seeds)

    wd_rows = []
    for wd, per_seed in wd_runs.items():
        for seed, rd in per_seed.items():
            delays = _pooled_delays(rd, ckpt)
            acc = _read_acc(rd)
            if delays is None or acc is None:
                print(f"[warn] skipping {rd} (missing delays/acc)")
                continue
            wd_rows.append({"wd": wd, "seed": seed, "acc": acc,
                            "mean": float(delays.mean()), "std": float(delays.std()),
                            "delays": delays})

    fixed_rows = []
    fixed_dirs = har.resolve_run_dirs(fixed_key, FIXED_STD, exp_dir, seeds)
    for seed, rd in fixed_dirs.items():
        delays = _pooled_delays(rd, ckpt)
        acc = har.read_test_accuracy(rd)
        if delays is None or acc is None:
            continue
        fixed_rows.append({"seed": seed, "acc": acc,
                           "mean": float(delays.mean()), "std": float(delays.std()),
                           "delays": delays})

    # Learned models trained WITHOUT weight decay, over the delay_std_init sweep
    # (a second route to different delay distributions), for the comparison overlay.
    nowd_rows = []
    lkey = LEARNED_STD_KEY.get(family)
    if lkey is not None:
        for std in har.STDS:
            if std > NOWD_MAX_STD:
                continue
            for seed, rd in har.resolve_run_dirs(lkey, std, exp_dir, seeds).items():
                delays = _pooled_delays(rd, ckpt)
                acc = har.read_test_accuracy(rd)
                if delays is None or acc is None:
                    continue
                nowd_rows.append({"std_init": std, "seed": seed, "acc": acc,
                                  "mean": float(delays.mean()), "std": float(delays.std()),
                                  "delays": delays})

    # Fixed-delay small-scale (compression) sweep, the frozen-delay runs at small
    # init stds (fixed_comp_root/<fixed_class>/std<X>/seed<N>). These extend the
    # fixed baseline down into the small-delay regime the wd route explores.
    fixed_comp_rows = []
    fixed_class = har.MODEL_PREFIXES[fixed_key]
    for std, per_seed in discover_std_runs(fixed_comp_root, fixed_class, seeds).items():
        for seed, rd in per_seed.items():
            delays = _pooled_delays(rd, ckpt)
            acc = _read_acc(rd)
            if delays is None or acc is None:
                continue
            fixed_comp_rows.append({"std_init": std, "seed": seed, "acc": acc,
                                    "mean": float(delays.mean()), "std": float(delays.std()),
                                    "delays": delays})

    return {"wd_rows": wd_rows, "fixed_rows": fixed_rows, "nowd_rows": nowd_rows,
            "fixed_comp_rows": fixed_comp_rows}


def _read_acc(run_dir):
    path = os.path.join(run_dir, "final_test.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return float(json.load(f).get("acc"))


# --------------------------------------------------------------------------- #
# Learned-vs-fixed compression comparison (accuracy vs delay budget)
# --------------------------------------------------------------------------- #
def _delay_stats(run_dir, ckpt):
    """Pooled-delay mean/std/max + the raw pooled delays for a run (or None).
    ``maxd`` = ring-buffer depth (= on-chip memory / worst-case reaction latency). ``delays`` is kept for distribution-level (ECDF) views."""
    d = _pooled_delays(run_dir, ckpt)
    if d is None:
        return None
    return {"mean": float(d.mean()), "std": float(d.std()), "maxd": float(d.max()),
            "delays": d}


def discover_std_runs(root, model_class, seeds):
    """{std: {seed: run_dir}} for a std<X>/seed<N> sweep under root/model_class."""
    out = {}
    for sd in sorted(glob.glob(os.path.join(root, model_class, "std*"))):
        if not os.path.isdir(sd):
            continue
        try:
            std = float(os.path.basename(sd)[3:])
        except ValueError:
            continue
        per = {seed: os.path.join(sd, f"seed{seed}") for seed in seeds
               if os.path.isdir(os.path.join(sd, f"seed{seed}"))}
        if per:
            out[std] = per
    return dict(sorted(out.items()))


def build_compression_comparison(family, sweep_root, fixed_comp_root, exp_dir, seeds, ckpt):
    """Learned vs fixed compression rows (each: mean/std/maxd/acc/cfg)."""
    learned_class = FAMILIES[family][0]
    fixed_class = har.MODEL_PREFIXES[FAMILIES[family][1]]
    lkey, fkey = LEARNED_STD_KEY[family], FAMILIES[family][1]
    learned, fixed = [], []

    def add(dst, run_dir, cfg, seed, acc):
        s = _delay_stats(run_dir, ckpt)
        if s is None or acc is None:
            return
        s["acc"], s["cfg"], s["seed"] = acc, cfg, seed
        dst.append(s)

    # learned: weight-decay route (std8) ...
    for wd, per in discover_wd_runs(sweep_root, learned_class, seeds).items():
        for seed, rd in per.items():
            add(learned, rd, ("wd", wd), seed, _read_acc(rd))
    # ... + learned delay_std_init sweep (no weight decay), all stds
    if INCLUDE_STD_SWEEP:
        for std in har.STDS:
            for seed, rd in har.resolve_run_dirs(lkey, std, exp_dir, seeds).items():
                add(learned, rd, ("std", std), seed, har.read_test_accuracy(rd))

    # fixed: small-scale compression sweep ...
    for std, per in discover_std_runs(fixed_comp_root, fixed_class, seeds).items():
        for seed, rd in per.items():
            add(fixed, rd, ("std", std), seed, _read_acc(rd))
    # ... + existing fixed delay_std_init sweep, all stds
    if INCLUDE_STD_SWEEP:
        for std in har.STDS:
            for seed, rd in har.resolve_run_dirs(fkey, std, exp_dir, seeds).items():
                add(fixed, rd, ("std", std), seed, har.read_test_accuracy(rd))

    return {"learned": learned, "fixed": fixed}


def _agg_by_cfg(rows, stat, agg="mean"):
    """Per-config (cfg) summary of the compression rows, sorted by budget. Returns ``(budget,
    acc, acc_sem, budget_sem)``.
    """
    if agg.startswith("pareto"):
        xs = np.array([r[stat] for r in rows], dtype=float)
        ys = np.array([r["acc"] for r in rows], dtype=float)
        order = np.argsort(xs, kind="stable")
        xs, ys = xs[order], np.maximum.accumulate(ys[order])
        if agg == "pareto_interp":  # keep only the runs that beat every cheaper one
            keep = ys > np.r_[-np.inf, ys[:-1]]
            xs, ys = xs[keep], ys[keep]
        return xs, ys, np.zeros_like(xs), np.zeros_like(xs)

    byc = {}
    for r in rows:
        byc.setdefault(r["cfg"], []).append(r)
    xs, ys, es, xes = [], [], [], []
    for rs in byc.values():
        accs = np.array([r["acc"] for r in rs])
        stats = np.array([r[stat] for r in rs], dtype=float)
        if agg == "best":
            i = int(np.argmax(accs))
            xs.append(stats[i]), ys.append(accs[i]), es.append(0.0), xes.append(0.0)
        else:
            n = max(np.sqrt(len(accs)), 1)
            xs.append(stats.mean()), ys.append(accs.mean())
            es.append(accs.std(ddof=0) / n), xes.append(stats.std(ddof=0) / n)
    order = np.argsort(xs)
    return (np.array(xs)[order], np.array(ys)[order], np.array(es)[order],
            np.array(xes)[order])


def _min_budget_at(rows, stat, target, agg="mean"):
    """Smallest per-config ``stat`` whose accuracy still reaches ``target`` (or None), under
    the same seed aggregation as the curve it annotates (see ``_agg_by_cfg``).
    """
    xs, ys, *_ = _agg_by_cfg(rows, stat, agg)
    return cu.iso_crossing(xs, ys, target,
                           interpolate=INTERPOLATE and agg != "pareto")


def plot_compression_frontier(family, comp, out_dir, agg="mean"):
    """Hero compression figure, test accuracy vs delay budget, three elements: * learned +
    compression (weight-decay sweep, solid ●), the frontier. * fixed + compression
    (fixed small-init sweep + fixed std sweep, dashed ■) the control (best any fixed-delay
    model does at that budget).
    """
    from matplotlib.lines import Line2D

    learned_wd = [r for r in comp["learned"] if r["cfg"][0] == "wd"]
    fixed = comp["fixed"]
    if not learned_wd or not fixed:
        return
    color = st.TYPE_COLORS[family]
    # Anchor the iso-accuracy arrow at the fixed model's best accuracy (its ceiling):
    # how much smaller a budget the learned model needs to match the best any fixed
    # model achieves. (acc is per-config and independent of the x-stat.)
    target = max(_agg_by_cfg(fixed, "mean", agg)[1])
    anchor = [r for r in learned_wd if r["cfg"] == ("wd", 0.0)]
    pareto = agg.startswith("pareto")
    one_run = agg in ("best", "pareto", "pareto_interp")  # drawn points are single runs

    panels = [("mean", "mean delay (time steps)", "mean delay", "mean"),
              ("maxd", "buffer depth = max delay (time steps)", "buffer depth", "max")]
    for stat, xlabel, tag, suffix in panels:
        with plt.rc_context(st.RC_LARGE_TEXT):
            fig, ax = plt.subplots(figsize=(5.6, 4.6))
            handles = []

            def draw(rows_c, cond, marker, mfc, label, band_alpha):
                """One condition's curve: the Pareto frontier over its pooled runs (with every run
                scattered underneath, faint when dominated, solid when on the frontier), or the
                per-config curve with its ±SEM on both axes.
                """
                x, y, e, ex = _agg_by_cfg(rows_c, stat, agg)
                if pareto:
                    fx, fy, *_ = _agg_by_cfg(rows_c, stat, "pareto_interp")
                    ax.scatter([r[stat] for r in rows_c], [r["acc"] for r in rows_c],
                               s=16, marker=marker, facecolors=mfc or color,
                               edgecolors=color, linewidths=0.8, alpha=0.35, zorder=1)
                    ax.scatter(fx, fy, s=45, marker=marker, facecolors=mfc or color,
                               edgecolors=color, linewidths=1.2, zorder=4)
                    ax.plot(x, y, ls=st.CONDITION_LS[cond], color=color, lw=1.9, zorder=3,
                            drawstyle="steps-post" if agg == "pareto" else "default")
                    handles.append(Line2D([], [], color=color, ls=st.CONDITION_LS[cond],
                                          lw=1.9, marker=marker, mfc=mfc or color,
                                          mec=color, ms=6, label=label))
                    return
                ax.fill_between(x, y - e, y + e, color=color, alpha=band_alpha, lw=0)
                ax.errorbar(x, y, xerr=ex, fmt="none", ecolor=color, elinewidth=1.0,
                            alpha=0.5)
                ax.plot(x, y, ls=st.CONDITION_LS[cond], marker=marker, mfc=mfc,
                        color=color, lw=1.9, ms=5, label=label, zorder=3)

            draw(fixed, "fixed", "s", "none", "fixed + compression", 0.12)
            draw(learned_wd, "learned", "o", None, "learned + compression", 0.20)
            if anchor:
                # The uncompressed anchor follows the same rule as the curves: its
                # best-accuracy run where they are single runs, the seed mean otherwise.
                a = max(anchor, key=lambda r: r["acc"])
                ax_x = a[stat] if one_run else np.mean([r[stat] for r in anchor])
                ax.scatter([ax_x],
                           [a["acc"] if one_run else np.mean([r["acc"] for r in anchor])],
                           marker="*", s=280, color=color, edgecolor=st.INK_MUTED,
                           linewidths=0.6, zorder=5, label="learned, uncompressed")
                # Cut the x-axis just past the uncompressed-learned model, the fixed
                # curve runs much further out and otherwise squashes the frontier.
                ax.set_xlim(right=ax_x * 1.18)
            lb = _min_budget_at(learned_wd, stat, target, agg)
            fb = _min_budget_at(fixed, stat, target, agg)
            if lb is not None and fb is not None and fb > lb:
                ax.axhline(target, ls=":", color=st.INK_MUTED, lw=0.8, alpha=0.6)
                ax.annotate("", xy=(lb, target), xytext=(fb, target),
                            arrowprops=dict(arrowstyle="<->", color=st.INK_SECONDARY, lw=1.4))
                ax.text((lb + fb) / 2, target + 0.2, f"{fb / max(lb, 1e-9):.1f}× smaller",
                        ha="center", va="bottom", fontsize=PANEL_ANNOT_FS,
                        color=st.INK_SECONDARY)
            # Crop the low-accuracy tail (both curves keep running below, clipped).
            ax.set_ylim(bottom=ACC_FLOOR)
            ax.set_xlabel(xlabel)
            ax.set_ylabel("test accuracy (%)")
            note = {"best": ", best seed", "pareto": ", Pareto",
                    "pareto_interp": ", Pareto interpolated"}.get(agg, "")
            ax.set_title(f"HAR {family}: accuracy vs delay budget\n({tag}{note})", pad=8)
            ax.set_box_aspect(1)
            ax.grid(alpha=0.25, lw=0.5)
            # The frontier + its run cloud fill the corners differently per panel, so the
            # envelope figures let matplotlib place the legend, and hand it the proxy
            # handles that carry the runs' own marker.
            if pareto and anchor:
                handles.append(Line2D([], [], ls="", marker="*", mfc=color,
                                      mec=st.INK_MUTED, ms=12,
                                      label="learned, uncompressed"))
            ax.legend(handles=handles or None, frameon=False,
                      loc="best" if pareto else "lower right", handlelength=1.8)
            fig.tight_layout()
            tag_file = "" if agg == "mean" else f"_{agg}"
            cu.save_fig(fig, os.path.join(
                out_dir, f"compression_frontier_{suffix}{tag_file}.svg"))



# --------------------------------------------------------------------------- #
# Distribution-level compression views (learned vs fixed at matched accuracy).
#
# Summary stats (mean/std/max) reduce a whole delay distribution to one scalar, so
# the "learnable delays compress the distribution more for equal accuracy" claim,
# which is really about the accuracy-vs-cost frontier and the delay-distribution
# shape, can't be read off them. These views make it explicit: the iso-accuracy
# delay ECDF (whole distribution), the accuracy-vs-buffer-depth frontier with the
# iso-accuracy gap, and the buffer depth each condition needs per accuracy threshold.
# --------------------------------------------------------------------------- #
def _cfg_label(cfg):
    """('wd', 0.03) -> 'wd0.03' . ('std', 5.0) -> 'std5'."""
    return f"{cfg[0]}{cfg[1]:g}"


def _agg_configs(rows):
    """Aggregate compression rows per config -> list (sorted by mean buffer depth)
    of {cfg, acc (mean over seeds), maxd (mean), delays (pooled over seeds)}."""
    byc = {}
    for r in rows:
        byc.setdefault(r["cfg"], []).append(r)
    out = []
    for cfg, rs in byc.items():
        out.append({"cfg": cfg,
                    "acc": float(np.mean([r["acc"] for r in rs])),
                    "maxd": float(np.mean([r["maxd"] for r in rs])),
                    "delays": np.concatenate([r["delays"] for r in rs])})
    return sorted(out, key=lambda r: r["maxd"])


def _plateau_target(comp, margin=0.3):
    """Iso-accuracy target: just below the lower of the two conditions' best accuracies."""
    best = [max(_agg_by_cfg(comp[c], "mean")[1]) for c in ("learned", "fixed")]
    return min(best) - margin


def _min_buffer_at(cfgs, target):
    """Most-compressed config (smallest mean buffer depth) still reaching >= target accuracy,
    or None.
    """
    ok = [c for c in cfgs if c["acc"] >= target]
    return min(ok, key=lambda c: c["maxd"]) if ok else None


def _buffer_crossing_at(cfgs, target):
    """Interpolated buffer depth at which ``cfgs`` reaches ``target`` accuracy, or None
    the scalar analogue of ``_min_buffer_at``, for the ratio figures, where snapping
    to a sampled config would quantize the ratio onto the sweep grid."""
    return cu.iso_crossing([c["maxd"] for c in cfgs], [c["acc"] for c in cfgs], target,
                           interpolate=INTERPOLATE)


def _seed_buffer_crossings(rows, target):
    """Per-seed min buffer depth reaching ``target``, as an array over seeds (or None). Each
    seed was trained at every config of the sweep, so it is an independent replicate of
    the whole compression experiment: this reads the crossing off each seed's own (buffer
    depth, accuracy) curve.
    """
    byseed = {}
    for r in rows:
        byseed.setdefault(r["seed"], []).append(r)
    out = []
    for rs in byseed.values():
        c = cu.iso_crossing([r["maxd"] for r in rs], [r["acc"] for r in rs], target,
                            interpolate=INTERPOLATE)
        if c is None:
            return None
        out.append(c)
    return np.array(out, dtype=float) if out else None


def _buffer_crossing_stats(rows, target):
    """(mean, sem) over seeds of ``_seed_buffer_crossings``, or None. SEM convention as
    everywhere else in this module: population std / sqrt(n)."""
    c = _seed_buffer_crossings(rows, target)
    if c is None:
        return None
    return float(c.mean()), float(c.std(ddof=0) / np.sqrt(len(c)))


def _ecdf(x):
    xs = np.sort(np.asarray(x, dtype=float))
    return xs, np.arange(1, len(xs) + 1) / len(xs)




def _frame_to_curves(ax, thresholds, ys, pad=0.06):
    """Set the axes to the curves' extent (curve ± band) with a fixed relative pad,
    rather than letting matplotlib autoscale. Anything outside is clipped."""
    if not ys:
        return
    x0, x1 = float(min(thresholds)), float(max(thresholds))
    y0, y1 = float(min(ys)), float(max(ys))
    dx, dy = (x1 - x0) or 1.0, (y1 - y0) or max(abs(y0), 1.0)
    ax.set_xlim(x0 - pad * dx, x1 + pad * dx)
    ax.set_ylim(y0 - pad * dy, y1 + pad * dy)


def _draw_compression_curves(ax, family, comp, thresholds, prefix=""):
    """Draw one family's learned + fixed min-buffer-depth curves onto ``ax`` and report the
    y-extent drawn (curve and band, so the caller can frame to it).
    """
    if not comp["learned"] or not comp["fixed"]:
        return []
    drawn = []
    for rows, cond in [(comp["learned"], "learned"), (comp["fixed"], "fixed")]:
        pts = [(t, s) for t, s in ((t, _buffer_crossing_stats(rows, t)) for t in thresholds)
               if s is not None]
        if not pts:
            continue
        ts = np.array([p[0] for p in pts])
        mu = np.array([p[1][0] for p in pts])
        sem = np.array([p[1][1] for p in pts])
        drawn += list(mu - sem) + list(mu + sem)
        ax.fill_between(ts, mu - sem, mu + sem, color=st.TYPE_COLORS[family],
                        alpha=0.15, lw=0, zorder=2)
        ax.plot(ts, mu, ls=st.CONDITION_LS[cond],
                marker=st.TYPE_MARKERS[family], mfc=st.SURFACE if cond == "fixed" else None,
                color=st.TYPE_COLORS[family], label=f"{prefix}{cond}", zorder=3)
    return drawn


def plot_compression_ratio_all(comps, out_dir, start=THRESHOLD_START,
                               step=THRESHOLD_STEP, stop=THRESHOLD_STOP):
    """The per-family ``compression_ratio`` curves combined on one axes, both delay
    types head to head, which works because they share the buffer-depth unit. Same
    overlay convention as ``accuracy_vs_wd`` / ``acc_vs_delay_*``: color = delay type,
    dashed + open marker = the fixed control."""
    with plt.rc_context(st.RC_LARGE_TEXT):
        # Panel-sized like the per-family compression_ratio, only wider: four curves and
        # a four-entry legend share one axes.
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        thresholds = np.arange(start, stop + 1e-9, step)
        drawn = [_draw_compression_curves(ax, family, comp, thresholds, prefix=f"{family} ")
                 for family, comp in comps.items()]
        if not any(drawn):
            plt.close(fig)
            return
        _frame_to_curves(ax, thresholds, [y for ys in drawn for y in ys])
        ax.set_xlabel("accuracy threshold (%)")
        ax.set_ylabel("min buffer depth to reach it\n(max recurrent delay, time steps)")
        ax.set_title("HAR: buffer depth needed per accuracy\n— learned needs less")
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, loc="upper left")
        fig.tight_layout()
        cu.save_fig(fig, os.path.join(out_dir, "compression_ratio_all.svg"))


def plot_stats_vs_wd_all(results, out_dir):
    """The per-family ``stats_vs_wd`` curves combined on one axes, both delay types head to
    head, which works because they share the time-step unit (the same reasoning that puts
    both on one buffer-depth axis in ``plot_compression_ratio_all``).
    """
    wds_all = sorted({r["wd"] for res in results.values() for r in res["wd_rows"]})
    if not wds_all:
        return
    pos = {wd: i for i, wd in enumerate(wds_all)}
    with plt.rc_context(st.RC_LARGE_TEXT):
        # Panel-sized like compression_ratio_all: four curves + a four-entry legend.
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        for family, res in results.items():
            wd_rows, fixed_rows = res["wd_rows"], res["fixed_rows"]
            if not wd_rows:
                continue
            for field, ls in [("mean", "-"), ("std", "--")]:
                wds, mu, sem = _agg_by_wd(wd_rows, field)
                ax.errorbar([pos[w] for w in wds], mu, yerr=sem, ls=ls,
                            marker=st.TYPE_MARKERS[family],
                            mfc=st.SURFACE if field == "std" else None,
                            color=st.TYPE_COLORS[family], capsize=2,
                            label=f"{family} {field}")
                fb, _ = _fixed_band(fixed_rows, field)
                if fb is not None:
                    ax.axhline(fb, ls=ls, color=st.INK_MUTED, lw=1.0, alpha=0.8)
        ax.set_xticks(list(pos.values()))
        ax.set_xticklabels([("0" if w == 0 else f"{w:g}") for w in wds_all],
                           rotation=45, ha="right")
        ax.set_xlabel("weight decay on delays")
        ax.set_ylabel("delay (time steps)")
        ax.set_title("HAR: delay statistics vs weight decay\n"
                     "(gray lines = fixed-delay baseline)")
        ax.set_box_aspect(1)
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(frameon=False, loc="upper right")
        fig.tight_layout()
        cu.save_fig(fig, os.path.join(out_dir, "stats_vs_wd_all.svg"))




# --------------------------------------------------------------------------- #
# Aggregation helpers
# --------------------------------------------------------------------------- #
def _agg_by_wd(wd_rows, field):
    """Sorted wds and per-wd (mean, sem) of `field` across seeds."""
    wds = sorted({r["wd"] for r in wd_rows})
    mu, sem = [], []
    for wd in wds:
        vals = np.array([r[field] for r in wd_rows if r["wd"] == wd], dtype=float)
        mu.append(vals.mean())
        sem.append(vals.std(ddof=0) / max(np.sqrt(len(vals)), 1))
    return np.array(wds), np.array(mu), np.array(sem)


def _fixed_band(fixed_rows, field):
    """(mean, sem) of `field` over the fixed baseline seeds, or (None, None)."""
    if not fixed_rows:
        return None, None
    vals = np.array([r[field] for r in fixed_rows], dtype=float)
    return float(vals.mean()), float(vals.std(ddof=0) / max(np.sqrt(len(vals)), 1))


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #





# --------------------------------------------------------------------------- #
# wd-axis helpers (log scale, but wd=0 has no log position -> place it as a
# labeled tick just left of the smallest positive wd)
# --------------------------------------------------------------------------- #
def _wd_x(wds):
    """Map wd values to x positions: positive wds at their value, wd=0 at a
    decade below the smallest positive wd (so it sits on the log axis)."""
    wds = np.asarray(wds, dtype=float)
    pos = wds[wds > 0]
    floor = (pos.min() / 10.0) if pos.size else 1e-4
    return np.where(wds > 0, wds, floor)


def _set_wd_axis(ax, wds):
    ax.set_xscale("log")
    wds = np.asarray(sorted(set(np.asarray(wds, dtype=float))))
    x = _wd_x(wds)
    ax.set_xticks(x)
    ax.set_xticklabels([("0" if w == 0 else f"{w:g}") for w in wds], rotation=0)
    ax.set_xlabel("weight decay on delays")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", default=SWEEP_ROOT)
    ap.add_argument("--fixed-comp-root", default=FIXED_COMP_ROOT,
                    help="Root of the fixed-delay small-scale (compression) sweep.")
    ap.add_argument("--exp-dir", default=".",
                    help="Root the fixed-baseline paths are resolved against.")
    ap.add_argument("--families", nargs="+", default=list(FAMILIES),
                    choices=list(FAMILIES))
    ap.add_argument("--seeds", type=int, nargs="+", default=har.SEEDS)
    ap.add_argument("--ckpt", default="best.pth",
                    help="Checkpoint to read delays from (matches final_test.json).")
    ap.add_argument("--out-dir", default=None,
                    help="Default: figures/HAR/wd/weight_decay"
                         " (+ '_snapped' with --no-interp).")
    ap.add_argument("--no-interp", action="store_true",
                    help="Report the shallowest sampled config reaching each threshold "
                         "instead of interpolating the crossing (see INTERPOLATE).")
    ap.add_argument("--compression-only", action="store_true",
                    help="Drop the unconstrained delay_std_init sweep from the pools, "
                         "keeping weight decay (learned) and small-scale inits (fixed).")
    args = ap.parse_args()

    global INTERPOLATE, INCLUDE_STD_SWEEP
    INTERPOLATE = not args.no_interp
    INCLUDE_STD_SWEEP = not args.compression_only
    if args.out_dir is None:
        args.out_dir = os.path.join(
            har.OUT_ROOT, "wd",
            "weight_decay" + ("" if INCLUDE_STD_SWEEP else "_compressiononly")
            + ("" if INTERPOLATE else "_snapped"))

    results = {}
    summary = []
    comp_rows = []
    comps = {}   # family -> compression comparison, for the combined figure
    for family in args.families:
        print(f"\n########## family = {family} ##########")
        res = build_family_results(family, args.sweep_root, args.exp_dir,
                                   args.seeds, args.ckpt)
        n_wd = len({r["wd"] for r in res["wd_rows"]})
        print(f"  {len(res['wd_rows'])} learned runs over {n_wd} wd values; "
              f"{len(res['fixed_rows'])} fixed-baseline runs")
        if not res["wd_rows"]:
            print(f"  [warn] no sweep runs found under "
                  f"{os.path.join(args.sweep_root, FAMILIES[family][0])}")
            continue
        results[family] = res

        out_dir = os.path.join(args.out_dir, family)
        os.makedirs(out_dir, exist_ok=True)

        # Learned-vs-fixed compression comparison (acc vs max/mean/std delay).
        comp = build_compression_comparison(family, args.sweep_root, args.fixed_comp_root,
                                             args.exp_dir, args.seeds, args.ckpt)
        print(f"  compression: {len(comp['learned'])} learned + {len(comp['fixed'])} fixed runs")
        # The seed aggregations of the hero frontier (see _agg_by_cfg): the seed mean
        # (± SEM on both axes), the per-config best seed, and the Pareto envelope over
        # every run, as a staircase and as its frontier runs joined by straight lines.
        # Increasingly a deployment claim, increasingly optimistic.
        # accuracy against the buffer depth the model needs
        plot_compression_frontier(family, comp, out_dir, agg="mean")
        comps[family] = comp
        for cond in ("learned", "fixed"):
            for r in comp[cond]:
                comp_rows.append({"family": family, "condition": cond,
                                  "cfg_type": r["cfg"][0], "cfg_value": r["cfg"][1],
                                  "seed": r["seed"],
                                  "delay_mean": r["mean"], "delay_std": r["std"],
                                  "delay_max": r["maxd"], "test_acc": r["acc"]})

        for r in res["wd_rows"]:
            summary.append({"family": family, "condition": "learned", "wd": r["wd"],
                            "seed": r["seed"], "delay_mean": r["mean"],
                            "delay_std": r["std"], "test_acc": r["acc"]})
        for r in res["fixed_rows"]:
            summary.append({"family": family, "condition": "fixed", "wd": np.nan,
                            "seed": r["seed"], "delay_mean": r["mean"],
                            "delay_std": r["std"], "test_acc": r["acc"]})

    if not results:
        print("\nNo sweep runs found, nothing to plot.")
        return

    os.makedirs(args.out_dir, exist_ok=True)
    plot_compression_ratio_all(comps, args.out_dir)
    plot_stats_vs_wd_all(results, args.out_dir)

    pd.DataFrame(summary).to_csv(
        os.path.join(args.out_dir, "weight_decay_summary.csv"), index=False)
    if comp_rows:
        pd.DataFrame(comp_rows).to_csv(
            os.path.join(args.out_dir, "compression_comparison.csv"), index=False)
    print(f"\nSummary CSV: {os.path.join(args.out_dir, 'weight_decay_summary.csv')}")


if __name__ == "__main__":
    main()
