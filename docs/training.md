# Training

## One model

```bash
python experiments/train.py --dataset ssc --model SNN_recurrent_delays --seeds 0,1,2,3,4
python experiments/train.py --dataset al --smoke                 # check the pipeline runs
```

| flag | |
|---|---|
| `--dataset` | `ssc`, `psmnist`, `har`, `al`, `shd` |
| `--model` | a class from `src/delrec/networks.py`, or from `networks_shd.py` on SHD, default axonal learned delays |
| `--seeds` | comma-separated, one run per seed |
| `--kernel`, `--synaptic-kernel` | pin a forward kernel, default is the fastest available |
| `--epochs`, `--out` | override the config |
| `--smoke` | 2 epochs over a handful of batches, to check the pipeline runs |

Mackey-Glass has no single-run entry point: its runs come from the sweep below.

Each run writes one directory under `exp/<BENCHMARK>/<Model>/perf/seed<N>_<timestamp>/`, which
is gitignored, so a re-run never touches the curated runs in `trained_models/`.

```
config.json          the full effective configuration
train_res.csv        per-epoch training accuracy and loss
val_res.csv          per-epoch validation accuracy and loss
best.pth  last.pth   checkpoints, best is selected on validation
init_delays.npz      the delays at initialization
final_delays.npz     the delays after training
final_test.json      the final test metrics
perm.pt              the pixel permutation, PS-MNIST only
```

## Model families

| class | delay type | delays |
|---|---|---|
| `SNN_recurrent_delays` | axonal | learned |
| `SNN_fixed_recurrent_delays` | axonal | frozen at a random initialization |
| `SNN_synaptic_recurrent_delays` | synaptic | learned |
| `SNN_fixed_synaptic_recurrent_delays` | synaptic | frozen at a random initialization |
| `SNN_common_recurrent_delays` | one per layer | learned |
| `SNN_vanilla_recurrent` | none | the delay-free control |

A fixed family differs from its learned counterpart in freezing the delays and pinning the
spread to zero, so the pair isolates the contribution of learning the delays from that of having
them. The frozen delays are also rounded to integers at construction, which matters on AL, the
one benchmark whose learned delays are reported unrounded. Feedforward-delay variants live in
the same module, for the SSC baselines.

SHD has its own zoo, `src/delrec/networks_shd.py`, holding only the axonal learned, axonal fixed
and vanilla classes. The synaptic and common families are `networks.py` only.

## Hyperparameters

`configs/perf_<BENCHMARK>.py`, one plain class per benchmark, and the source for the paper's
hyperparameter tables. Edit those rather than `experiments/train.py`.

Configs hold defaults as class attributes and are instantiated then mutated, so `vars(config)`
sees only what a run overrode. `experiments/train.py` merges the class attributes back in before
writing `config.json`, which makes that file a complete record of a new run. The runs shipped
under `trained_models/` predate that merge and record only their overrides, so read their
defaults off the config class instead.

## Sweeps

The multi-run experiments behind the HAR, AL and Mackey-Glass figures. Each `parallel_` launcher
runs one process per run across the available GPUs, giving each slot its own Triton cache, and
each run holds the same files as a single training run.

A sweep writes to `exp/<root>/`, which is gitignored. The curated equivalents that ship with the
repository, and that the analyses actually read, are at `trained_models/<root>/`. The roots below
name both.

| sweep | root | feeds |
|---|---|---|
| `parallel_sweep_HAR_delaystd.py` | `HAR/std_init_sweep` | the HAR accuracy column, robustness, efficiency |
| `parallel_sweep_HAR_commondelay.py` | `HAR/std_init_sweep` | `shared_delay_per_layer.py` |
| `parallel_sweep_HAR_weightdecay.py` | `HAR/weight_decay_sweep`, `HAR/fixed_compression_sweep` | `memory_and_energy_efficiency.py` |
| `parallel_sweep_HAR_spikepenalty.py` | `HAR/spike_penalty_sweep` | `memory_and_energy_efficiency.py` |
| `parallel_sweep_HAR_noround.py` | `HAR/std_init_sweep_noround` | `delay_discretization_har.py` |
| `parallel_sweep_HAR_rounding_alternatives.py` | `HAR/rounding_alternatives_sweep` | `delay_discretization_har.py` |
| `parallel_sweep_AL_rounding.py` | `AL/rounding_sweep` | `delay_discretization_al.py` |
| `parallel_sweep_MG.py` | `MG/mackey_glass_sweep` | `mackey_glass_forecasting.py` |

Every sweep but `parallel_sweep_AL_rounding.py` also has a sequential `sweep_*.py` form.

The weight-decay launcher has two axes and the buffer-depth analysis needs both:

```bash
HAR_SWEEP_AXIS=wd  python experiments/sweeps/parallel_sweep_HAR_weightdecay.py
HAR_SWEEP_AXIS=std python experiments/sweeps/parallel_sweep_HAR_weightdecay.py
```

`wd` compresses learned delays by penalizing the delay parameters, `std` compresses fixed delays
by initializing them smaller, since frozen delays are never updated and cannot be compressed the
first way.
