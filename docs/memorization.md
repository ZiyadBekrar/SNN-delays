# Spike memorization

Run the memorization SNN configured in `configs/perf_MEM.py` (`model` there selects
the network class, e.g. a synaptic-feedforward-delay or axonal-recurrent-delay
network; see `src/delrec/networks.py` for the full list):

```bash
.venv/bin/python experiments/train_mem.py
```

The default temporal task has 128 fixed samples, 16 input neurons, 32 time
steps, four independent random classes, and one hidden layer of 64 neurons.
Every input neuron spikes exactly once. The spatial alternative repeats one
random current vector throughout the sequence. Both use the supplied
`SpikeMemorization` generation procedure. There is no validation or test split.

Examples for comparing configurations:

```bash
.venv/bin/python experiments/train_mem.py --task-type spatial
.venv/bin/python experiments/train_mem.py --hidden-layers 64,64 --num-samples 256
.venv/bin/python experiments/train_mem.py --model SNN_vanilla_recurrent
.venv/bin/python experiments/train_mem.py --model SNN_synaptic_feedforward_delays --seed 1
.venv/bin/python experiments/train_mem.py --epochs 300 --readout last
```

`--seed` controls model initialization and batch order; `--dataset-seed` controls
the fixed inputs and random labels independently. Keep dataset settings and the
dataset seed identical when comparing architectures. Other hyperparameters,
including delay ranges and learning rates, can be edited in `perf_MEM.py`.
`--device` accepts `auto`, `cpu`, `cuda`, or `mps`; auto selects CUDA when available,
otherwise CPU. The CPU default uses one thread for these small tensor operations.

Training minimizes cross entropy of time-averaged output logits by default.
`--readout sum` and `--readout last` select other reductions. The mean readout
measures random-pattern memorization using evidence throughout the sequence;
it does not require withholding the answer until a separate recall period.

Both feedforward positions and recurrent delays are optimized. Gaussian DCLS
widths are scheduled from `siginit` to 0.23 over the first half of training.
Recurrent smoothing uses the configured sigma schedule. Delays are clamped after
each update and remain fractional; measurements preserve the current smoothing
and positions, without rounding or a separate inference-time transformation.

Each run writes to a unique `exp/MEM/<model>/...` directory, or `--out PATH`:

- `config.json`: effective settings and device.
- `dataset.pt`: the exact input tensors and random labels.
- `train_res.csv`: epoch, loss, accuracy in percent, and online optimization loss.
  Loss/accuracy are recomputed on the entire training set after each epoch;
  online loss is the sample-weighted loss observed during that epoch's updates.
- `final_train.json`: final training loss, accuracy, correct count and model size.
- `last.pth`: final model, optimizer/scheduler, settings, metrics, and recurrent
  sigma values (which are not part of the model state dict).
- `training_summary.png` and `.pdf`: loss versus epoch and final training accuracy.

The final checkpoint is used directly; there is no validation-based selection.
Neuron and dropout states are reset between batches. Training accuracy quantifies
memorization of these samples, not generalization to new random labels.

## Delay parametrization x pathway (axonal / synaptic / hybrid, feedforward / recurrent)

```bash
.venv/bin/python experiments/compare_mem_delays.py
```

This trains six single-pathway models with the same topology from `perf_MEM.py`,
crossing delay parametrization with pathway:

|            | Feedforward only                     | Recurrent only                  |
|------------|---------------------------------------|----------------------------------|
| axonal     | `SNN_axonal_feedforward_delays`       | `SNN_axonal_recurrent_delays`    |
| synaptic   | `SNN_synaptic_feedforward_delays`     | `SNN_synaptic_recurrent_delays`  |
| hybrid     | `SNN_hybrid_feedforward_delays`       | `SNN_recurrent_hybrid_delays`    |

- axonal: one learned delay per source neuron, shared by its connections.
- synaptic: one learned delay per connection.
- hybrid: the axonal model's learned per-source delay plus a fixed random
  per-connection integer offset, drawn once and never trained.

All six require `kernel_count=1` and apply recurrence to every hidden layer
(feedforward-pathway models never have recurrence to begin with; recurrent-pathway
models get it forced on regardless of `no_recurrence_in_last_layer`), so weights
and delays line up one-to-one across the comparison.

All six are seeded from one shared reference (`generate_matched_parameters` in
`compare_mem_delays.py`): the same feedforward weights/biases, feedforward axonal
delays, recurrent weights/biases and recurrent axonal delays are injected into
every model, using the same dataset, minibatch order and settings. Within each
pathway, axonal and synaptic initial outputs are checked for numerical
equivalence before training (synaptic delays start as broadcast copies of the
axonal ones, then learn independently). The hybrids share the same weights,
biases and base axonal delays, but their fixed offsets change the initial
outputs. Gaussian widths use the same schedule throughout.

The neuron counts and connectivity match across a pathway's three models, but
parameter counts differ by design: axonal learns one delay per source neuron,
synaptic one per connection, and hybrid keeps the axonal count while adding a
fixed (non-trainable) offset per connection. `comparison.json` reports the exact
counts for each run's topology.

Each model has its own complete run directory under
`exp/MEM/delay_comparison/<run>/`. The parent contains `comparison.json` and
`delay_comparison.png` / `.pdf`, showing six loss curves, six accuracy curves,
and six final training accuracy bars (colour = parametrization, line style =
pathway). All measurements use the training set. This is a single-seed
comparison, not an estimate of variation across seeds.

Supported overrides include `--epochs`, `--seed`, `--dataset-seed`,
`--num-samples`, `--task-type`, `--hidden-layers`, `--device cpu|cuda`, and `--out`.
For example:

```bash
.venv/bin/python experiments/compare_mem_delays.py --task-type spatial --seed 1
```

### Fixed random synaptic delays in the hybrids

The effective physical delay is `d[i,j] = axonal[j] + offset[i,j]`, on whichever
pathway the hybrid model covers (`SNN_hybrid_feedforward_delays` for feedforward,
`SNN_recurrent_hybrid_delays` for recurrent). Recurrence also retains its usual
mandatory one-step feedback lag. Offsets are drawn once **before training**,
allowing the learned weights and axonal delays to adapt to them. Adding random
offsets after training would instead measure robustness to a timing perturbation.

`hybrid_max_synaptic_delay = 4` bounds the integer offsets to `[0, 4]`.
`hybrid_delay_seed = 123` controls the draw independently of the dataset and
weight seeds. `hybrid_offset_distribution` selects the sampling law within that
range: `'uniform'` (default, every integer equally likely), `'gaussian'`
(`Normal(hybrid_max_synaptic_delay / 2, hybrid_offset_sigma)`, rounded and
clamped into range), or `'triangular'` (peaked at `hybrid_max_synaptic_delay / 2`,
falling off linearly to 0 at both ends). `hybrid_offset_sigma` is required, and
only used, when the distribution is `'gaussian'`. Override any of these on
`compare_mem_delays.py` with `--hybrid-max-synaptic-delay`, `--hybrid-delay-seed`,
`--hybrid-offset-distribution`, and `--hybrid-offset-sigma`. Offsets remain fixed
throughout training and evaluation and are saved as buffers in `last.pth`; they
are not optimizer parameters.

Each hybrid learns exactly as many delays as its paired axonal model (one per
source neuron), and additionally stores one fixed offset per connection (the
`comparison.json`/`final_*_delay_parameters` and `fixed_synaptic_offsets` fields
report the exact counts for a given topology). It uses synaptic computations to
apply those distinct offsets, so equal learned parameter counts do not imply
equal runtime or storage costs.

DCLS positions run in the opposite direction from transmission delays. The
feedforward hybrid's implementation subtracts fixed offsets from those positions
and extends the feedforward kernel window to accommodate the extra lag without
clipping it; the original learnable axonal position range is retained. The
recurrent hybrid's axonal delays are clamped nonnegative; their offsets are
added afterwards. Gaussian filtering remains the same type of smoothing,
evaluated on the extended window where applicable. The hybrids' longer effective
delays are part of this experimental condition; the experiment does not match
maximum effective delays across models.

The comparison reads the current `perf_MEM.py`, so reruns may differ from older
results if neuron parameters have been edited. The recurrent-pathway models
apply recurrence to every hidden layer, even when `no_recurrence_in_last_layer`
is set, so they stay comparable to their feedforward-pathway counterparts.
