"""What projecting the delays onto the integer grid costs on AL.

Shows:    AL test accuracy for learned axonal and synaptic delays under three
          regimes, kept fractional throughout, rounded at test time only, and
          rounded at every epoch during training.
Inputs:   trained_models/AL/ and trained_models/AL/rounding_sweep/   [measurements]
Outputs:  figures/AL/round_compare/
Usage:    python experiments/make_figures/delay_discretization_al.py

AL is the exception among the benchmarks: it trains without rounding, so rounding
its delays at test time alone is expensive. That gap is a train/test mismatch
rather than an intrinsic cost of integer delays, training under the constraint
recovers most of it.
"""
from _run import main

if __name__ == "__main__":
    main("experiments/make_figures/AL/analyze_rounding.py")
