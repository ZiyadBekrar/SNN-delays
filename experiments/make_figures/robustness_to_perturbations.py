"""How learned and fixed delays hold up when the input is degraded.

Shows:    the accuracy difference between learned and fixed delays under temporal
          subsampling, random deletion of input samples, and Gaussian jitter on
          the input sampling times, as a function of perturbation strength and
          of the delay spread the model reached.
Inputs:   figures/measurements/HAR/std/{subsample,deletion,jitter}/   [measurements]
          (recomputing instead needs the checkpoints, the HAR dataset and a GPU)
Outputs:  figures/HAR/std/{subsample,deletion,jitter}/
Usage:    python experiments/make_figures/robustness_to_perturbations.py

Nothing is retrained: only the test inputs are perturbed. Learned and fixed models
are compared at matched post-training delay spread rather than at matched
initialization, because training moves the delays, equal initial width would
mean comparing different delay distributions. Each panel therefore interpolates
both families onto one common measured-spread axis before differencing.
"""
from _run import main
# --plots-only redraws from the recorded measurements. Drop it (and supply a GPU
# and the HAR dataset) to recompute the sweeps from the checkpoints.

if __name__ == "__main__":
    main(("experiments/make_figures/HAR/subsample_sweep.py", "--sweep", "std", "--plots-only"),
         ("experiments/make_figures/HAR/deletion_sweep.py", "--sweep", "std", "--plots-only"),
         ("experiments/make_figures/HAR/jitter_sweep.py", "--sweep", "std", "--plots-only"))
