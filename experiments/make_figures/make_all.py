"""Regenerate the figures and tables of the paper.

Usage:
    python experiments/make_figures/make_all.py --measurements-only
    python experiments/make_figures/make_all.py --all     # also the one needing the weights
    python experiments/make_figures/make_all.py --list    # what each figure needs

Outputs land under figures/, or under $DELREC_FIGURES. What each figure needs:

    [measurements]  nothing but this repository
    [weights]       also `python fetch_checkpoints.py --dataset ssc`

This repository records measurements of the 691 trained models, not the models. A figure
that reads measurements needs no download: the accuracies, trained delays and per-epoch
curves were all recorded at training time, and the analyses that must run a model forward
have their outputs recorded too, under figures/measurements/. Only running a model forward
yourself needs the weights.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from _run import run

# (script, requirement, what it shows)
FIGURES = [
    ("benchmark_accuracy_tables.py",     "measurements", "benchmark accuracies and their significance"),
    ("mackey_glass_forecasting.py",      "measurements", "Mackey-Glass forecasting"),
    ("memory_and_energy_efficiency.py",  "measurements", "buffer depth and spike cost"),
    ("shared_delay_per_layer.py",        "measurements", "one delay per layer vs per neuron/connection"),
    ("delay_discretization_har.py",      "measurements", "integer delays on HAR"),
    ("delay_discretization_al.py",       "measurements", "integer delays on AL"),
    ("kernel_cost.py",                   "measurements", "kernel time and memory"),
    ("delay_structure.py",               "measurements", "delay distributions and the permutation ablation"),
    ("robustness_to_perturbations.py",   "measurements", "robustness to degraded input"),
    ("gradient_flow_during_training.py", "measurements", "gradient reach over training"),
    ("gradient_maps.py",                 "weights",      "the full selectivity-map catalog"),
]

MEASUREMENTS_ONLY = {"measurements"}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    # --file-only is the former name, kept so existing commands still work.
    g.add_argument("--measurements-only", "--file-only", dest="measurements_only",
                   action="store_true",
                   help="only figures that need nothing but this repository")
    g.add_argument("--all", action="store_true",
                   help="also the figures needing fetched weights")
    g.add_argument("--list", action="store_true", help="print the table above and exit")
    args = ap.parse_args()

    if args.list:
        print(f"{'script':36s} {'needs':14s} shows")
        for script, req, shows in FIGURES:
            print(f"{script:36s} [{req:12s}] {shows}")
        return 0

    todo = [f for f in FIGURES if args.all or f[1] in MEASUREMENTS_ONLY]
    print(f"Running {len(todo)} of {len(FIGURES)} figure scripts\n")

    failed = []
    for script, req, shows in todo:
        print(f"\n{'=' * 70}\n{shows}   [{req}]\n{'=' * 70}")
        try:
            run(HERE / script)
        except Exception as exc:                       # keep going, report at the end
            print(f"  FAILED: {exc}")
            failed.append((script, shows))

    print(f"\n{'=' * 70}")
    print(f"{len(todo) - len(failed)}/{len(todo)} succeeded")
    for script, shows in failed:
        print(f"  FAILED  {script}  ({shows})")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
