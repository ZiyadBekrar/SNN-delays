"""What projecting the delays onto the integer grid costs on HAR.

Shows:    learned delays trained and tested rounded to integers at every epoch,
          against the same models kept fractional, and two alternative
          discretizations (straight-through estimator, stochastic rounding)
          against the nearest-integer rounding used throughout.
Inputs:   trained_models/HAR/{std_init_sweep,std_init_sweep_noround,
          rounding_alternatives_sweep}/           [measurements]
Outputs:  figures/HAR/std/{round_compare,rounding_alternatives}/
Usage:    python experiments/make_figures/delay_discretization_har.py

Integer delays are what neuromorphic hardware implements, so the projection is
applied at every epoch and at test time. Neither alternative improves on nearest
rounding, and the per-epoch projection appears to act as a mild regularizer.
"""
from _run import main

if __name__ == "__main__":
    main("experiments/make_figures/HAR/analyze_rounding.py",
         "experiments/make_figures/HAR/analyze_rounding_alternatives.py")
