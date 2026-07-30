"""What delay learning costs in buffer depth and in spikes.

Shows:    accuracy and the delay spread reached, against the width the delays were
          initialized with. How weight decay on the delay parameters compresses
          them, and the shortest buffer that still reaches a given accuracy. The
          firing rate under a spike-count penalty, and the lowest firing rate that
          still reaches a given accuracy.
Inputs:   trained_models/HAR/{std_init_sweep,weight_decay_sweep,
          fixed_compression_sweep,spike_penalty_sweep}/   [measurements]
Outputs:  figures/HAR/{std,wd,lam}/
Usage:    python experiments/make_figures/memory_and_energy_efficiency.py

File-only: accuracy and firing rate come from each run's recorded metrics, the
delays from the stored per-layer snapshots. The compression panels need the
fixed-delay sweep as well. Fixed delays are never updated, so they cannot be
compressed by weight decay and are shrunk by initializing them smaller instead.
"""
from _run import main

if __name__ == "__main__":
    main(("experiments/make_figures/HAR/plot_final_acc_compare.py", "--sweep", "std"),
         ("experiments/make_figures/HAR/analyze_weightdecay.py",),
         ("experiments/make_figures/HAR/analyze_spikepenalty.py",))
