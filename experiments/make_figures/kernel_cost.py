"""What a delayed recurrent layer costs in time and memory.

Shows:    per-batch time and peak memory of a full forward+backward, for the
          PyTorch implementation, the fused Triton kernel, and a delay-free RSNN
          of the same architecture, on real batches from each benchmark, and for
          a single layer with one factor swept at a time.
Inputs:   trained_models/profiling/       [measurements to plot. GPU to re-measure]
Outputs:  figures/profiling/
Usage:    python experiments/make_figures/kernel_cost.py            # replot recorded runs
          python experiments/profiling/profile_kernels.py           # re-measure (GPU)
          python experiments/profiling/profile_kernels_datasets.py  # re-measure (GPU)
          python experiments/profiling/profile_kernels_bigN.py      # fill the widest layers

Re-measuring is hardware-specific. The recorded numbers come from one machine. The
backward kernel's shared-memory tiling is what forces a reduced weight tile at the
largest layer widths, hence the third script.
"""
from _run import main

if __name__ == "__main__":
    main("experiments/profiling/plot_profiling_panels.py")
