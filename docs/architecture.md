# Architecture

The shape of the code. File tree in the [README](../README.md), commands in
[training.md](training.md) and [reproducing.md](reproducing.md).

`src/delrec` is the library, `experiments` is everything you run against it.

## Library

The delay layer is the core. `delay_layers.py` holds one `nn.Module` per delay configuration and
dispatches `forward` to a backend at run time: a pure-PyTorch scan that runs anywhere and is the
reference, or a fused Triton kernel that is faster and numerically equivalent.

| class | `recurrent_delays` |
|---|---|
| `axonal_recdel` | `(N,)`, one per presynaptic neuron |
| `synaptic_recdel` | `(N, N)`, one per connection |
| `common_recdel` | `(1,)`, broadcast to the whole layer |
| `vanilla_recurrent` | none, the delay-free control |

`networks.py` stacks those layers into the benchmark architectures and ends in a plain linear
readout, which the trainer reduces over time: summed on SSC and PS-MNIST, averaged on HAR and
AL, leaky-integrated on Mackey-Glass. `networks_shd.py` is a separate stack for the SHD
appendix, and is the one architecture whose readout is a non-spiking leaky-integrate neuron.
`datasets/` loads the six benchmarks. `training/` holds one train and test loop per benchmark,
each optimizing weights and delays with separate optimizers.

## Experiments

`train.py` runs one model on one benchmark and `sweeps/` runs many. Both write a self-contained
run directory under `exp/`, which is gitignored, holding the config, the per-epoch metrics, the
final metrics and the delays before and after training. The curated runs that ship with the
repository, and that the analyses read, are the same directories under `trained_models/`.

`make_figures/` reads those directories and writes `figures/`. `common/` is dataset-agnostic.
`<BENCHMARK>/runs.py` says where that benchmark's runs live and how to rebuild a model from one.