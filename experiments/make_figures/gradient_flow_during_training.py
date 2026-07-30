"""How far back in time error gradients survive, over the course of training.

Shows:    the norm of the gradient reaching each past input step, as a function of
          temporal depth and training epoch, for a delay-free RSNN, fixed random
          axonal delays, and learned axonal delays.
Inputs:   figures/measurements/SSC/gradflow_epochs/       [measurements]
          (re-measuring instead retrains three models: GPU + dataset, hours each)
Outputs:  figures/SSC/gradflow_epochs/
Usage:    python experiments/make_figures/gradient_flow_during_training.py

Unlike every other script here this one trains: the probe is a per-epoch
measurement, so the three models are retrained from a shared initialization while
the gradient reach is logged after each epoch. Budget several hours per model.
"""
from _run import main
# --plots-only redraws from the recorded per-epoch matrices. Drop it (and supply a
# GPU and the SSC dataset) to retrain the three models and re-measure.

if __name__ == "__main__":
    main(("experiments/make_figures/SSC/grad_flow_train.py", "--plots-only"))
