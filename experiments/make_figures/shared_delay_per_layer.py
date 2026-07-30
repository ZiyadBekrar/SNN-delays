"""One delay shared by a whole layer, against per-neuron and per-connection delays.

Shows:    accuracy against the initialization width for a single delay parameter
          broadcast to every recurrent connection of a layer, next to the four
          heterogeneous families, and how far that shared delay moves in training.
Inputs:   trained_models/HAR/std_init_sweep/              [measurements]
Outputs:  figures/HAR/common/
Usage:    python experiments/make_figures/shared_delay_per_layer.py

The shared-delay model carries 2 delay parameters on HAR instead of 304 (one per
neuron) or 47,360 (one per connection), and never reaches the heterogeneous
configurations, which is the argument for parameterizing delays per neuron or
per connection despite the extra parameters.
"""
from _run import main

if __name__ == "__main__":
    main("experiments/make_figures/HAR/analyze_commondelay.py")
