# Spike memorization

Run the combined recurrent/feedforward delay SNN using `configs/perf_MEM.py`:

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
.venv/bin/python experiments/train_mem.py --model SNN_feedforward_delays --seed 1
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

## Axonal, synaptic and hybrid delays on both pathways

```bash
.venv/bin/python experiments/compare_mem_delays.py
```

This trains three models with the same topology from `perf_MEM.py`:

- `SNN_axonal_recurrent_and_feedforward_delays`: a depthwise unit-weight,
  bias-free delay filter before each feedforward Linear, plus one recurrent
  delay per source neuron.
- `SNN_synaptic_recurrent_and_feedforward_delays`: dense DCLS feedforward
  projections and synaptic recurrent layers, with one delay per connection.
- `SNN_hybrid_recurrent_and_feedforward_delays`: one learned axonal delay per
  source neuron plus fixed random delays per connection, on both pathways.

All three use the same hidden order: delayed projection, dropout, recurrent LIF,
spike recorder, optional batch normalization. All three include delays in the output
projection by default. These paired variants require `kernel_count=1` and
apply recurrence to every hidden layer, matching the original combined model.

The comparison uses the same dataset, minibatch order, settings, initial
connection weights and biases. Synaptic delays start as broadcast copies of
axonal delays and then learn independently. Axonal and synaptic initial outputs are checked for
numerical equivalence before training. The hybrid shares their weights, biases
and base axonal delays, but its fixed offsets change the initial outputs. Gaussian widths use the same schedule.
The neuron counts and connectivity match, but parameter counts differ by design:
with the default 16 → 64 → 4 topology, the axonal model has 80 feedforward and
64 recurrent delay parameters; the synaptic model has 1,280 feedforward and
4,096 recurrent delay parameters.

Each model has its own complete run directory under
`exp/MEM/delay_comparison/<run>/`. The parent contains `comparison.json` and
`delay_comparison.png` / `.pdf`, showing three loss curves, three accuracy curves,
and three final training accuracy bars. All measurements use the training set.
This is a single-seed comparison, not an estimate of variation across seeds.

Supported overrides include `--epochs`, `--seed`, `--dataset-seed`,
`--num-samples`, `--task-type`, `--hidden-layers`, `--device cpu|cuda`, and `--out`.
For example:

```bash
.venv/bin/python experiments/compare_mem_delays.py --task-type spatial --seed 1
```

### Fixed random synaptic delays in the hybrid

The effective physical delay is `d[i,j] = axonal[j] + offset[i,j]` on both
feedforward and recurrent pathways. Recurrence also retains its usual mandatory
one-step feedback lag. Offsets are drawn once **before training**, allowing the
learned weights and axonal delays to adapt to them. Adding random offsets after
training would instead measure robustness to a timing perturbation.

`hybrid_max_synaptic_delay = 4` samples integer offsets uniformly from 0 through
4 time steps. `hybrid_delay_seed = 123` controls this draw independently of the
dataset and weight seeds. Override them with `--hybrid-max-synaptic-delay` and
`--hybrid-delay-seed`. Offsets remain fixed throughout training and evaluation
and are saved as buffers in `last.pth`; they are not optimizer parameters.

The hybrid learns 80 feedforward and 64 recurrent delays for the default topology,
just like the axonal model, and stores 5,376 additional fixed offsets. It uses
synaptic computations to apply those distinct offsets, so equal learned parameter
counts do not imply equal runtime or storage costs.

DCLS positions run in the opposite direction from transmission delays. The
implementation subtracts fixed offsets from those positions and extends the
feedforward kernel window to accommodate the extra lag without clipping it.
The original learnable axonal position range is retained. Recurrent axonal
parameters are clamped nonnegative; their offsets are added afterwards.
Gaussian filtering remains the same type of smoothing, evaluated on the extended
window. The hybrid's longer effective delays are part of this experimental
condition; the experiment does not match maximum effective delays across models.

The comparison reads the current `perf_MEM.py`, so reruns may differ from older
results if neuron parameters have been edited. All three combined architectures
apply recurrence to every hidden layer, even when `no_recurrence_in_last_layer`
is set; this preserves their existing topology.
