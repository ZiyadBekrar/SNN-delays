"""Learned recurrent delays on the Mackey-Glass chaotic system.

Shows:    how much the NMSE improves over each baseline (no annealing, fixed
          delays, no delays) across the chaos parameter tau and the prediction
          horizon H. The absolute error against tau, and one predicted trajectory.
Inputs:   trained_models/MG/mackey_glass_sweep/          [measurements]
Outputs:  figures/MG/sweep_report/
Usage:    python experiments/make_figures/mackey_glass_forecasting.py

Needs neither a GPU nor the dataset: every number comes from each run's stored
predictions, and the series itself is generated in memory during training rather
than stored.
"""
from _run import main

if __name__ == "__main__":
    main("experiments/make_figures/MG/analyze_sweep.py")
