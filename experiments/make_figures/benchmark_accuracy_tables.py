"""Benchmark accuracies and the significance of the learned-vs-fixed difference.

Shows:    our accuracies against the published literature on SSC and PS-MNIST and
          on AL and HAR. Learned versus fixed recurrent delays on all four
          benchmarks, and a two-sided Welch t-test on each of the eight
          learned-vs-fixed differences.
Inputs:   trained_models/{SSC,PSMNIST,AL,HAR}/           [measurements]
Outputs:  figures/tables/*.{md,csv}
Usage:    python experiments/make_figures/benchmark_accuracy_tables.py

Our numbers are read from each run's recorded final metrics. The competing methods
are quoted from their publications and carried here as data. HAR reports the best
delay-initialization width per family, so its cell is a maximum over that sweep of
the mean over seeds.

Welch's test (unequal variances) rather than Student's: the two families are
independently seeded and there is no reason to assume equal variance.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path[:0] = [str(HERE.parent), str(HERE.parent / "src"), str(HERE)]

from common import paths  # noqa: E402

def _std(v):
    """Spread across seeds, as the paper reports it.

    Sample standard deviation (``ddof=1``): the seeds are a sample, and the quantity of
    interest is the spread of the underlying distribution, not of the five runs that
    happened to be drawn. Note this is not numpy's default.
    """
    return np.std(np.asarray(v, dtype=float), ddof=1) if len(v) > 1 else 0.0


def _md(df):
    """Markdown table, without pulling in `tabulate` for four small tables."""
    cols = [str(c) for c in df.columns]
    rows = [[("" if pd.isna(v) else str(v)) for v in r] for r in df.itertuples(index=False)]
    w = [max(len(cols[i]), *(len(r[i]) for r in rows)) if rows else len(cols[i])
         for i in range(len(cols))]
    line = lambda cells: "| " + " | ".join(c.ljust(w[i]) for i, c in enumerate(cells)) + " |"
    return "\n".join([line(cols), "|" + "|".join("-" * (n + 2) for n in w) + "|",
                       *(line(r) for r in rows)])


FAMILIES = ["ax_fixed", "ax_learned", "syn_fixed", "syn_learned"]
LABEL = {"ax_fixed": ("Axonal (ax.)", "Fixed"), "ax_learned": ("Axonal (ax.)", "Learned"),
         "syn_fixed": ("Synaptic (syn.)", "Fixed"), "syn_learned": ("Synaptic (syn.)", "Learned")}
DATASETS = ["SSC", "PSMNIST", "AL", "HAR"]

# Published comparisons, quoted from the cited papers (Tables 1 and 2).
LITERATURE_1 = [
    # dataset, model, neuron, recurrent, delays, params, seeds, acc
    ("SSC", "Adaptive RSNN [6]", "AdLIF", "yes", "no", "0.78M", 1, "74.20"),
    ("SSC", "EventProp [22]", "LIF", "no", "Ff.", "~5M", 8, "76.1+/-1.0"),
    ("SSC", "Bullet trains [31]", "LIF", "no", "Ff.", "n/r", 5, "77.35+/-0.23"),
    ("SSC", "RadLIF [7]", "RadLIF", "yes", "no", "3.9M", 1, "77.40"),
    ("SSC", "cAdLIF [15]", "cAdLIF", "no", "no", "0.35M", 1, "77.50"),
    ("SSC", "d-cAdLIF [15]", "cAdLIF", "no", "Ff.", "0.7M", 10, "80.23+/-0.07"),
    ("SSC", "SE-adLIF [8]", "SE-adLIF", "yes", "no", "1.6M", 20, "80.44+/-0.26"),
    ("SSC", "DCLS [14]", "LIF", "no", "Ff. (syn.)", "2.5M", 5, "80.69+/-0.21"),
    ("SSC", "ASRC-SNN [21]", "LIF", "yes", "Rec.", "0.37M", 1, "81.54"),
    ("SSC", "SiLIF [32]", "SiLIF (SSM)", "yes", "no", "0.35M", 10, "82.03+/-0.25"),
    ("PSMNIST", "GLIF [33]", "GLIF", "yes", "no", "0.15M", 1, "90.47"),
    ("PSMNIST", "Adaptive RSNN [6]", "AdLIF", "yes", "no", "0.15M", 1, "94.30"),
    ("PSMNIST", "BRF [34]", "RF", "yes", "no", "69k", 1, "95.20"),
    ("PSMNIST", "ASRC-SNN [21]", "LIF", "yes", "Rec.", "0.15M", 1, "95.77"),
]
LITERATURE_2 = [
    # model, neuron, delays, AL, HAR
    ("RSNN", "LIF", "no", "56.50", "77.32"),
    ("Spike-Driven Transformer [35]", "LIF", "no", "58.00", "71.07"),
    ("CE-LIF [36]", "CE-LIF", "no", "60.62", "80.83"),
    ("GSN [37]", "LIF", "no", "67.22", "82.31"),
    ("Spiking TCN [38]", "LIF", "no", "69.88", "82.41"),
    ("LTC [39]", "LTC", "no", "79.04", "82.03"),
    ("Gated Spiking Unit [40]", "LIF", "no", "80.42", "89.16"),
    ("Binary S4D [40]", "LIF (SSM)", "no", "81.44", "89.37"),
]
# The DelRec rows of Tables 1-2 report the best delay configuration per dataset.
BEST_CONFIG = {"SSC": ("ax_learned", "Rec. (ax.)", "0.37M"),
               "PSMNIST": ("syn_learned", "Rec. (syn.)", "0.25M"),
               "AL": ("ax_learned", "Rec. (ax.)", "~100k"),
               "HAR": ("ax_learned", "Rec. (ax.)", "~100k")}


def _accs_flat(dataset, mkey):
    """{seed: accuracy} for a family whose runs form a flat seed grid."""
    import importlib
    cfg = importlib.import_module(f"{dataset}.runs")
    out = {}
    for seed, run_dir in sorted(cfg.resolve_run_dirs(mkey).items()):
        fj = Path(run_dir) / "final_test.json"
        if fj.exists():
            # AL is reported with its delays as trained (see the paper's Methods), i.e. "acc".
            out[seed] = float(json.loads(fj.read_text())["acc"])
    return out


def _accs_har(mkey):
    """HAR reports the best delay_std_init per family, so take the max over the
    sweep of the mean over seeds."""
    from HAR import runs as har
    best, best_mean = {}, -np.inf
    for std in har.STDS:
        runs = har.resolve_run_dirs(mkey, std)
        accs = {s: har.read_test_accuracy(d) for s, d in sorted(runs.items())}
        accs = {s: float(a) for s, a in accs.items() if a is not None}
        if len(accs) >= 2 and np.mean(list(accs.values())) > best_mean:
            best, best_mean = accs, float(np.mean(list(accs.values())))
    return best


def collect():
    """{dataset: {family: {seed: acc}}}."""
    out = {}
    for ds in DATASETS:
        out[ds] = {f: (_accs_har(f) if ds == "HAR" else _accs_flat(ds, f)) for f in FAMILIES}
    return out


def table_learned_vs_fixed(data, out_dir):
    rows = []
    for f in FAMILIES:
        dtype, learning = LABEL[f]
        row = {"Delay type": dtype, "Learning": learning}
        for ds in DATASETS:
            v = list(data[ds][f].values())
            row[ds] = f"{np.mean(v):.2f}+/-{_std(v):.2f}" if v else "no"
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "learned_vs_fixed.csv", index=False)
    (out_dir / "learned_vs_fixed.md").write_text(
        "# Learned and fixed recurrent delays on all four benchmarks\n\n"
        "Test accuracy [%], mean +/- std over 5 seeds (3 on HAR).\n\n"
        + _md(df) + "\n")
    return df


def table_significance(data, out_dir):
    """Two-sided Welch t-test on each learned-vs-fixed pair."""
    rows = []
    for ds in DATASETS:
        for dtype, fixed, learned in [("Axonal", "ax_fixed", "ax_learned"),
                                      ("Synaptic", "syn_fixed", "syn_learned")]:
            a = np.array(list(data[ds][learned].values()), dtype=float)
            b = np.array(list(data[ds][fixed].values()), dtype=float)
            if len(a) < 2 or len(b) < 2:
                continue
            t, p = stats.ttest_ind(a, b, equal_var=False)   # Welch
            rows.append({"Dataset": ds, "Delay type": dtype,
                         "d acc. [pts]": round(float(a.mean() - b.mean()), 2),
                         "t": round(float(t), 3), "p": float(p), "n": len(a)})
    df = pd.DataFrame(rows)
    df["p"] = df["p"].map(lambda x: f"{x:.4f}" if x >= 1e-4 else f"{x:.1e}")
    df.to_csv(out_dir / "significance.csv", index=False)
    (out_dir / "significance.md").write_text(
        "# Significance of the learned-versus-fixed difference\n\n"
        "Difference in mean accuracy (learned - fixed), in accuracy points, and the\n"
        "two-sided Welch t-test p-value for unequal variances.\n\n"
        + _md(df) + "\n")
    return df


def tables_literature(data, out_dir):
    def ours(ds):
        f, delays, params = BEST_CONFIG[ds]
        v = list(data[ds][f].values())
        return delays, params, (f"{np.mean(v):.2f}+/-{_std(v):.2f}" if v else "no"), len(v)

    r1 = []
    for ds in ("SSC", "PSMNIST"):
        for row in [r for r in LITERATURE_1 if r[0] == ds]:
            r1.append(dict(zip(["Dataset", "Model", "Neuron", "Rec.", "Delays",
                                "Param", "Seeds", "Test Acc. [%]"], row)))
        delays, params, acc, n = ours(ds)
        r1.append({"Dataset": ds, "Model": "DelRec (ours)", "Neuron": "LIF", "Rec.": "yes",
                   "Delays": delays, "Param": params, "Seeds": n, "Test Acc. [%]": acc})
    df1 = pd.DataFrame(r1)
    df1.to_csv(out_dir / "literature_ssc_psmnist.csv", index=False)
    (out_dir / "literature_ssc_psmnist.md").write_text(
        "# Comparison with the literature on SSC and PS-MNIST\n\n"
        + _md(df1) + "\n")

    r2 = [dict(zip(["Model", "Neuron", "Delays", "AL", "HAR"], row)) for row in LITERATURE_2]
    r2.append({"Model": "DelRec (ours)", "Neuron": "LIF", "Delays": "Rec. (ax.)",
               "AL": ours("AL")[2], "HAR": ours("HAR")[2]})
    df2 = pd.DataFrame(r2)
    df2.to_csv(out_dir / "literature_al_har.csv", index=False)
    (out_dir / "literature_al_har.md").write_text(
        "# Comparison with the literature on AL and HAR\n\n"
        "Baselines are single runs from [30], same splits and protocol, all models\n"
        "use ~100k parameters.\n\n" + _md(df2) + "\n")
    return df1, df2


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default=os.path.join(paths.FIGURES_ROOT, "tables"))
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    data = collect()
    for ds in DATASETS:
        got = {f: len(data[ds][f]) for f in FAMILIES}
        print(f"  {ds:8s} seeds per family: {got}")

    print("\n" + "=" * 72)
    df3 = table_learned_vs_fixed(data, out_dir)
    print("Learned vs fixed delays\n" + df3.to_string(index=False))
    df7 = table_significance(data, out_dir)
    print("\nSignificance (Welch t-test)\n" + df7.to_string(index=False))
    tables_literature(data, out_dir)
    print(f"\nwrote learned_vs_fixed, significance and literature_* .{{md,csv}} -> {out_dir}")


if __name__ == "__main__":
    main()
