# Reproducing the results

```bash
python experiments/make_figures/make_all.py --measurements-only   # no download needed
python experiments/make_figures/make_all.py --all                 # also the one needing weights
python experiments/make_figures/make_all.py --list                # what each script needs
```

Output goes to `figures/`, over the committed versions. Set `DELREC_FIGURES` to a scratch
directory to write elsewhere and leave those untouched.

## What the repository records

The repository records measurements of the 691 trained models rather than the models themselves.
The weights are 385 MB in total. The measurements taken from them are a few kilobytes per run,
so the measurements are committed and the weights are not. There are two kinds, and together
they are what every figure below reads.

### Per-run measurements

In each `trained_models/<run>/` directory, recorded at training time:

```
final_test.json      the final test metrics
final_delays.npz     the trained delays
init_delays.npz      the delays at initialization
train_res.csv        per-epoch training accuracy and loss
val_res.csv          per-epoch validation accuracy and loss
config.json          the hyperparameters this run overrode
```

The SSC, PS-MNIST and AL runs predate parts of that set: none of the SSC or PS-MNIST runs carry
`init_delays.npz`, nor do 20 of the 30 AL runs, and the three SSC feedforward-delay baselines
have neither `final_test.json` nor `final_delays.npz`. See [checkpoints.md](checkpoints.md).

### Analysis measurements

In `figures/measurements/`. Some questions cannot be answered from recorded numbers and need a
model run forward, such as what jittering the input does to accuracy. Those need weights, a
dataset and a GPU, so their output is recorded here instead and the plotting redraws from it.

The weights, `best.pth`, are the one thing left out, at 385 MB against 63 MB for everything
else. They live on Zenodo, see [checkpoints.md](checkpoints.md), and are needed only to run a
model forward.

## One script per result

| shows | script | needs |
|---|---|---|
| benchmark accuracies against the literature, learned against fixed delays, and the significance of that difference | `benchmark_accuracy_tables.py` | measurements |
| learned delays on the Mackey-Glass chaotic system, across chaos and prediction horizon | `mackey_glass_forecasting.py` | measurements |
| what delay learning costs in buffer depth and in spikes | `memory_and_energy_efficiency.py` | measurements |
| one delay shared by a whole layer, against per-neuron and per-connection delays | `shared_delay_per_layer.py` | measurements |
| what integer delays cost on HAR, and two alternative discretizations | `delay_discretization_har.py` | measurements |
| what integer delays cost on AL | `delay_discretization_al.py` | measurements |
| time and memory of a delayed recurrent layer against a delay-free one | `kernel_cost.py` | measurements |
| the delay structure training builds, and whether the network relies on it | `delay_structure.py` | measurements |
| robustness to subsampled, deleted and jittered input | `robustness_to_perturbations.py` | measurements |
| how far back in time gradients survive, over training | `gradient_flow_during_training.py` | measurements |
| the catalog of input-gradient selectivity maps | `gradient_maps.py` | weights |

Paths are relative to `experiments/make_figures/`. `weights` also needs
`python fetch_checkpoints.py --dataset ssc` and a GPU, though no dataset: the maps are produced
by driving the model with an all-zero input.

The selectivity maps are emitted as a full catalog, one map per class, seed and model over all
35 SSC classes and 5 seeds, plus the energy profiles raw and normalized, because the paper
states that the maps for every class and seed are available here. Narrow it with `--classes`.

## Make your own measurements

The scripts above are thin front ends over the per-benchmark modules in
`experiments/make_figures/<BENCHMARK>/`, which accept their own flags. The three that redraw
from `figures/measurements/` do so through `--plots-only`. Drop that flag to measure again.
What that costs is not the same for all three.

Two of them re-evaluate the trained models, so they need the fetched weights, the benchmark's
dataset and a GPU:

| module | measurement it retakes |
|---|---|
| `HAR/{subsample,deletion,jitter}_sweep.py` | accuracy under degraded input |
| `SSC/permute_delays_depth_sweep.py` | accuracy under permuted delays |

The third retrains. `SSC/grad_flow_train.py` probes the gradient after every epoch, so it trains
the learned-delay, fixed-delay and delay-free models from one shared initialization rather than
loading a checkpoint. It needs the SSC dataset and a GPU but no fetched weights, and at roughly
5.5 h per model it is about 16 h of training. Pass `--epochs` to shorten it for a smoke test.

[training.md](training.md) lists which sweep produces the runs each figure reads, and
[architecture.md](architecture.md) covers how the analysis code is organized.
