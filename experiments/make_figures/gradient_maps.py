"""The spatiotemporal selectivity that delay learning produces.

Sensitivity of a readout neuron to each input channel and time step, for an untrained
model, feedforward delays, learned recurrent delays and fixed random recurrent delays,
plus the energy profile of those maps. Evaluated on an all-zero input, so no dataset is
needed and what it shows is intrinsic temporal structure rather than a stimulus
response. A plain run produces the whole catalog, every class and seed, which is what
the paper points here for. Narrow it with ``--classes``.

Outputs:  figures/SSC/gradmaps/                              [GPU + checkpoints]
Usage:    python experiments/make_figures/gradient_maps.py
"""
from _run import main

if __name__ == "__main__":
    main("experiments/make_figures/SSC/analyze_gradmaps.py")
