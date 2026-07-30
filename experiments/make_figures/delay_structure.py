"""The structure delay learning builds, and whether the network relies on it.

Shows:    the per-layer distribution of the trained delays, against the shared
          initialization, and a permutation ablation that shuffles a fraction of
          one layer's delays across neurons, with the weights frozen.
Inputs:   trained_models/SSC/ and figures/measurements/SSC/    [measurements]
          (recomputing the ablation needs the checkpoints, the dataset and a GPU)
Outputs:  figures/SSC/delay_depth/ and figures/SSC/permute_delays_depth/
Usage:    python experiments/make_figures/delay_structure.py

The permutation leaves each layer's marginal delay distribution, its weights and
its total recurrent drive untouched, only the assignment of delays to neurons is
destroyed, so a drop cannot be blamed on removed capacity. The fixed-delay
family is the exact null: permuting an i.i.d. draw is a no-op in distribution.
"""
from _run import main
# --plots-only redraws the ablation from the recorded per-run table. Drop it (and
# supply a GPU and the SSC dataset) to recompute it from the checkpoints.

if __name__ == "__main__":
    main(("experiments/make_figures/SSC/analyze_delay_depth.py",),
         ("experiments/make_figures/SSC/permute_delays_depth_sweep.py", "--plots-only"))
