"""The Mackey-Glass sweep: forecasting error across chaos and horizon.

Reads the (family x tau x horizon x seed) grid of runs and reports test NMSE,
absolute, against tau, and as the reduction the full model achieves over each
baseline. Everything comes from each run's stored predictions and metrics, so
this needs neither a GPU nor the dataset. The series is generated in memory
during training and never stored.

Outputs:  figures/MG/sweep_report/
Usage:    python experiments/make_figures/MG/analyze_sweep.py [--taus 17 30]
"""

import argparse
import csv
import os
import sys

os.environ.setdefault("MPLBACKEND", "Agg")
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path[:0] = [_ROOT, os.path.join(_ROOT, "src"), os.path.join(_ROOT, "experiments", "make_figures")]

import numpy as np
import matplotlib.pyplot as plt

from common import style
from common.utils import save_fig
import runs as mg

# Metrics plotted on the "performance" figures: (key, axis label, better direction).
# The paper reports test-set NMSE throughout (see the paper's Methods), so that is what these
# figures draw. The sweep records train/val and R2 as well, they are in each run's
# final_test.json and in summary_report.csv if you want them.
METRICS = [("nmse", "NMSE")]
SPLITS = ["test"]

# The representative window the paper illustrates.
PAPER_TAU, PAPER_H = 40, 20

# Per-family markers, color alone can't separate the two learned families (both purple).
FAMILY_MARKERS = {
    "learned_delays":       "o",
    "learned_no_annealing": "D",
    "fixed_delays":         "s",
    "no_delays":            "X",
}


# --------------------------------------------------------------------------- #
# Loading + aggregation
# --------------------------------------------------------------------------- #
def compute_metrics(preds, tgts):
    """Regression metrics for one split (the trainer's own function, same numbers)."""
    from delrec.training.mg import compute_metrics as _cm
    return _cm(preds, tgts)


def load_grid(sweep_root, families, taus, horizons, seeds):
    """{(tau, H, family): [per-seed {split: {'preds','tgts'}}]}, the raw predictions."""
    grid = {}
    for family in families:
        for tau in taus:
            for H in horizons:
                for seed in seeds:
                    preds = mg.read_predictions(
                        mg.run_dir(family, tau, H, seed, sweep_root))
                    if preds is not None:
                        grid.setdefault((tau, H, family), []).append(preds)
    return grid


def aggregate(grid):
    """{(tau, H, family, split): {metric: (mean, sem)}} over seeds."""
    agg = {}
    for (tau, H, family), per_seed in grid.items():
        for split in SPLITS:
            vals = [compute_metrics(e[split]["preds"], e[split]["tgts"])
                    for e in per_seed if split in e]
            if not vals:
                continue
            agg[(tau, H, family, split)] = {
                m: (float(np.mean([v[m] for v in vals])),
                    float(np.std([v[m] for v in vals]) / np.sqrt(len(vals)))
                    if len(vals) > 1 else 0.0)
                for m, _ in METRICS
            }
    return agg


def per_seed_improvements(grid, tau, H, baseline, split):
    """Per-seed NMSE reduction of the reference family over `baseline`, in percent:
    (nmse_baseline, nmse_reference) / nmse_baseline * 100. Seeds are paired by index
    (the sweep runs the same seed list for every family)."""
    ref = grid.get((tau, H, mg.REFERENCE))
    base = grid.get((tau, H, baseline))
    if not ref or not base:
        return np.array([])
    out = []
    for r, b in zip(ref, base):
        if split not in r or split not in b:
            continue
        nr = compute_metrics(r[split]["preds"], r[split]["tgts"])["nmse"]
        nb = compute_metrics(b[split]["preds"], b[split]["tgts"])["nmse"]
        out.append((nb - nr) / nb * 100 if nb > 0 else 0.0)
    return np.array(out)


def _sem(vals):
    return float(np.std(vals) / np.sqrt(len(vals))) if len(vals) > 1 else 0.0


# --------------------------------------------------------------------------- #
# Plots
# --------------------------------------------------------------------------- #

def plot_bars(agg, out_dir, families, taus, horizons, suffix=""):
    """One bar per family, panelled over the (tau, H) grid. One figure per split × metric."""
    for split in SPLITS:
        for metric, ylabel in METRICS:
            fig, axes = plt.subplots(len(taus), len(horizons), squeeze=False,
                                     figsize=(2.0 * len(horizons) + 2, 1.7 * len(taus) + 1))

            for i, tau in enumerate(taus):
                for j, H in enumerate(horizons):
                    ax = axes[i, j]
                    for k, family in enumerate(families):
                        key = (tau, H, family, split)
                        m, s = agg[key][metric] if key in agg else (0.0, 0.0)
                        ax.bar(k, m, 0.45, yerr=s, capsize=2,
                               color=mg.SPEC.colors[family],
                               hatch=mg.SPEC.hatches[family],
                               edgecolor="white", linewidth=0.4,
                               label=mg.SPEC.labels[family] if (i == 0 and j == 0) else None)
                    ax.set_xticks([])
                    ax.set_xlim(-0.7, len(families) - 0.3)
                    if i == len(taus) - 1:
                        ax.set_xlabel(f"H = {H}")
                    if j == 0:
                        ax.set_ylabel(ylabel)
                    if j == len(horizons) - 1:
                        ax.yaxis.set_label_position("right")
                        ax.set_ylabel(f"τ = {tau:g}", rotation=0, labelpad=30, va="center")
                    ax.grid(axis="y", alpha=0.4, ls="--")

            handles, labels = axes[0, 0].get_legend_handles_labels()
            fig.legend(handles, labels, loc="lower center", ncol=len(families))
            fig.suptitle(f"{ylabel} — {split}  (mean ± SEM over seeds)", fontsize=11)
            fig.tight_layout(rect=[0, 0.08, 1, 1])
            save_fig(fig, os.path.join(out_dir, f"bars_{split}_{metric}{suffix}.svg"))




def plot_trend_by_H(agg, out_dir, families, taus, horizons, suffix=""):
    """Metric vs τ, one small panel per H, one line per family (±SEM band): how each family
    degrades as the system delay τ grows. A compact, single-split cousin of perf_vs_tau."""
    for split in SPLITS:
        for metric, mlabel in METRICS:
            fig, axes = plt.subplots(1, len(horizons), squeeze=False, sharey=False,
                                     figsize=(2.7 * len(horizons) + 1, 3.2))
            for j, H in enumerate(horizons):
                ax = axes[0, j]
                for fam in families:
                    xs, ms, ss = [], [], []
                    for tau in taus:
                        key = (tau, H, fam, split)
                        if key not in agg:
                            continue
                        m, s = agg[key][metric]
                        xs.append(tau); ms.append(m); ss.append(s)
                    if not xs:
                        continue
                    ms, ss = np.array(ms), np.array(ss)
                    ax.plot(xs, ms, marker=FAMILY_MARKERS.get(fam, "o"), ms=4, lw=1.6,
                            color=mg.SPEC.colors[fam], ls=mg.SPEC.linestyles[fam],
                            label=mg.SPEC.labels[fam])
                    ax.fill_between(xs, ms - ss, ms + ss, color=mg.SPEC.colors[fam],
                                    alpha=0.12, lw=0)
                ax.set_title(f"H = {H}", fontsize=9)
                ax.set_xlabel("τ")
                ax.grid(alpha=0.4, ls="--")
                ax.set_xticks(taus, [f"{t:g}" for t in taus])
                if j == 0:
                    ax.set_ylabel(mlabel)
                if j == len(horizons) - 1:
                    ax.legend(fontsize=7)
            fig.suptitle(f"{mlabel} vs τ — {split}  (mean ± SEM over seeds)", fontsize=10)
            fig.tight_layout()
            save_fig(fig, os.path.join(out_dir, f"trend_{split}_{metric}{suffix}.svg"))




def plot_improvement_heatmap(grid, out_dir, families, taus, horizons):
    """The same NMSE reduction over the (tau, H) plane, one panel per baseline."""
    baselines = [f for f in families if f != mg.REFERENCE]
    if not baselines:
        return

    for split in SPLITS:
        fig, axes = plt.subplots(1, len(baselines), squeeze=False,
                                 figsize=(3.4 * len(baselines) + 1.5,
                                          0.7 * len(horizons) + 2.0))

        mats = {}
        for baseline in baselines:
            mat = np.full((len(horizons), len(taus)), np.nan)
            sem = np.full_like(mat, np.nan)
            for i, H in enumerate(horizons):
                for j, tau in enumerate(taus):
                    imps = per_seed_improvements(grid, tau, H, baseline, split)
                    if len(imps):
                        mat[i, j] = imps.mean()
                        sem[i, j] = _sem(imps)
            mats[baseline] = (mat, sem)

        if all(not np.isfinite(m).any() for m, _ in mats.values()):
            plt.close(fig)
            return

        for ax, baseline in zip(axes[0], baselines):
            mat, sem = mats[baseline]
            fin = mat[np.isfinite(mat)]
            if fin.size == 0:
                ax.set_visible(False)
                continue
            # Each panel normalized to its own reduction range, the huge "vs no delays"
            # panel no longer flattens the small "vs fixed" one. Symmetric, so 0 stays gray.
            vmax = float(np.abs(fin).max()) or 1.0
            im = ax.imshow(mat, cmap=style.DIVERGING, aspect="auto", vmin=-vmax, vmax=vmax)
            ax.set_xticks(range(len(taus)), [f"{t:g}" for t in taus])
            ax.set_yticks(range(len(horizons)), [str(h) for h in horizons])
            ax.set_xlabel("τ")
            ax.set_ylabel("H")
            for i in range(len(horizons)):
                for j in range(len(taus)):
                    if np.isfinite(mat[i, j]):
                        # White ink on the saturated ends of the ramp, dark on the pale middle.
                        light = abs(mat[i, j]) / vmax > 0.55 if vmax > 0 else False
                        ax.text(j, i, f"{mat[i, j]:+.1f}%\n±{sem[i, j]:.1f}",
                                ha="center", va="center", fontsize=7,
                                color="white" if light else style.INK_PRIMARY)
            ax.set_title(f"vs {mg.SPEC.labels[baseline]}", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8, label="NMSE reduction (%)")

        fig.suptitle(f"{mg.SPEC.labels[mg.REFERENCE]}: NMSE reduction — {split}",
                     fontsize=11)
        fig.tight_layout()
        save_fig(fig, os.path.join(out_dir, f"improvement_heatmap_{split}.svg"))


def _delay_wout_xy(delays, W):
    """(delay, weight-metric) pairs, branching on shape like ``common.recdel``."""
    delays, W = np.asarray(delays), np.asarray(W)
    if delays.ndim == 1:
        if delays.shape[0] != W.shape[1]:
            return None
        return delays, np.abs(W).mean(axis=0)
    if delays.shape != W.shape:
        return None
    return delays.reshape(-1), np.abs(W).reshape(-1)



def plot_prediction_window(grid, out_dir, families, taus, horizons, window=100):
    """The test window where the learned family beats the fixed one the most."""
    split = "test"
    if mg.REFERENCE not in families or "fixed_delays" not in families:
        return

    for tau in taus:
        for H in horizons:
            ref = grid.get((tau, H, mg.REFERENCE))
            cmp_ = grid.get((tau, H, "fixed_delays"))
            if not ref or not cmp_:
                continue

            ref_arrs = [(e[split]["preds"].ravel(), e[split]["tgts"].ravel())
                        for e in ref if split in e]
            cmp_arrs = [(e[split]["preds"].ravel(), e[split]["tgts"].ravel())
                        for e in cmp_ if split in e]
            if not ref_arrs or not cmp_arrs:
                continue

            T = min(min(len(p) for p, _ in ref_arrs), min(len(p) for p, _ in cmp_arrs))
            W = min(window, T)
            se_ref = np.mean([(p[:T] - t[:T]) ** 2 for p, t in ref_arrs], axis=0)
            se_cmp = np.mean([(p[:T] - t[:T]) ** 2 for p, t in cmp_arrs], axis=0)
            diff = se_cmp - se_ref  # positive where learned is better

            cum = np.cumsum(np.concatenate([[0.0], diff]))
            start = int(np.argmax(cum[W:] - cum[:T - W + 1]))
            sl = slice(start, start + W)
            xs = np.arange(start, start + W)

            fig, ax = plt.subplots(figsize=(9, 3.6))

            tgts = np.array([t[sl] for _, t in ref_arrs])
            ax.plot(xs, tgts.mean(0), color=style.INK_PRIMARY, lw=1.8,
                    label="Ground truth", zorder=10)

            for family in families:
                entries = grid.get((tau, H, family))
                if not entries:
                    continue
                pw = np.array([e[split]["preds"].ravel()[sl] for e in entries
                               if split in e and len(e[split]["preds"]) >= start + W])
                tw = np.array([e[split]["tgts"].ravel()[sl] for e in entries
                               if split in e and len(e[split]["tgts"]) >= start + W])
                if not len(pw):
                    continue
                nmses = [compute_metrics(pw[i], tw[i])["nmse"] for i in range(len(pw))]
                mean_p = pw.mean(0)
                sem_p = pw.std(0) / np.sqrt(len(pw)) if len(pw) > 1 else np.zeros_like(mean_p)
                ax.plot(xs, mean_p, lw=1.4, color=mg.SPEC.colors[family],
                        ls=mg.SPEC.linestyles[family],
                        label=f"{mg.SPEC.labels[family]}  "
                              f"(NMSE={np.mean(nmses):.3f}±{_sem(nmses):.3f})")
                ax.fill_between(xs, mean_p - sem_p, mean_p + sem_p,
                                color=mg.SPEC.colors[family], alpha=0.12, lw=0)

            ax.set_xlabel("Test-set sample")
            ax.set_ylabel("Value (z-scored)")
            ax.set_title(f"τ={tau:g}, H={H} — window maximizing SE(fixed) − SE(learned)",
                         fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(alpha=0.4, ls="--")
            fig.tight_layout()
            save_fig(fig, os.path.join(out_dir,
                                       f"pred_window_tau{tau:g}_H{H}.svg"))


# --------------------------------------------------------------------------- #
# Summary table
# --------------------------------------------------------------------------- #
def write_summary(agg, out_dir, families, taus, horizons):
    rows = []
    for tau in taus:
        for H in horizons:
            for family in families:
                row = {"tau": tau, "H": H, "family": family}
                for split in SPLITS:
                    key = (tau, H, family, split)
                    if key not in agg:
                        continue
                    for metric, _ in METRICS:
                        m, s = agg[key][metric]
                        row[f"{split}_{metric}_mean"] = f"{m:.6f}"
                        row[f"{split}_{metric}_sem"] = f"{s:.6f}"
                rows.append(row)
    if not rows:
        return

    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "summary_report.csv")
    fields = list({k: None for r in rows for k in r})
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"  summary_report.csv")

    print("\n" + "=" * 78)
    print("TEST SPLIT: NMSE (mean ± SEM over seeds)")
    print("=" * 78)
    print(f"{'τ':>5} {'H':>4}  " + "".join(f"{mg.SPEC.labels[f][:22]:>24}" for f in families))
    for tau in taus:
        for H in horizons:
            line = f"{tau:>5g} {H:>4}  "
            for family in families:
                key = (tau, H, family, "test")
                if key in agg:
                    m, s = agg[key]["nmse"]
                    line += f"{m:>16.4f}±{s:<7.4f}"
                else:
                    line += f"{'—':>24}"
            print(line)


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sweep-root", default=mg.SWEEP_ROOT,
                    help=f"Sweep output root (default {mg.SWEEP_ROOT}).")
    ap.add_argument("--out-dir", default=os.path.join(mg.OUT_ROOT, "sweep_report"))
    ap.add_argument("--families", nargs="+", default=None, help="Restrict to these families.")
    ap.add_argument("--taus", type=float, nargs="+", default=None, help="Restrict to these τ.")
    ap.add_argument("--horizons", type=int, nargs="+", default=None, help="Restrict to these H.")
    ap.add_argument("--seeds", type=int, nargs="+", default=None, help="Restrict to these seeds.")
    ap.add_argument("--window", type=int, default=100,
                    help="Length of the prediction-trace window (default 100 samples).")
    args = ap.parse_args()

    families, taus, horizons, seeds = mg.discover_axes(args.sweep_root)
    families = [f for f in families if args.families is None or f in args.families]
    taus = [t for t in taus if args.taus is None or t in args.taus]
    horizons = [h for h in horizons if args.horizons is None or h in args.horizons]
    seeds = [s for s in seeds if args.seeds is None or s in args.seeds]

    if not families or not taus or not horizons:
        raise SystemExit(f"No completed runs found under {args.sweep_root}")

    print(f"MG sweep report → {args.out_dir}")
    print(f"  families={families}\n  taus={taus}\n  horizons={horizons}\n  seeds={seeds}")

    grid = load_grid(args.sweep_root, families, taus, horizons, seeds)
    agg = aggregate(grid)
    print(f"  {len(grid)} (τ, H, family) cells, "
          f"{sum(len(v) for v in grid.values())} runs\n")

    # absolute test NMSE for every (tau, H) cell
    plot_bars(agg, args.out_dir, families, taus, horizons)
    # test NMSE against tau, at each horizon
    plot_trend_by_H(agg, args.out_dir, families, taus, horizons)
    # NMSE reduction of the full model over each baseline
    plot_improvement_heatmap(grid, args.out_dir, families, taus, horizons)
    # predicted trajectories on one window
    plot_prediction_window(grid, args.out_dir, families, [PAPER_TAU], [PAPER_H],
                           window=args.window)
    write_summary(agg, args.out_dir, families, taus, horizons)

    print(f"\nAll figures saved to {args.out_dir}/")


if __name__ == "__main__":
    main()
