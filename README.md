# DelRec

Reference implementation of *DelRec: learning delays in recurrent spiking neural networks*.

[![arXiv](https://img.shields.io/badge/arXiv-2509.24852-b31b1b.svg)](https://arxiv.org/abs/2509.24852)
[![Checkpoints](https://img.shields.io/badge/checkpoints-10.5281%2Fzenodo.21704323-1682d4.svg)](https://doi.org/10.5281/zenodo.21704323)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

## Method

A spike emitted by neuron *j* at time *t* reaches neuron *i* at *t* + 1 + d<sub>j</sub>. DelRec
learns d jointly with the synaptic weights, by surrogate gradient learning and backpropagation
through time.

An index into a buffer has no derivative. So d is relaxed to a real number, and each scheduled
spike is spread over neighboring integer time steps by a triangle function centered on 1 + d,
of half-width 1 + sigma. That function is differentiable in d, so the delay receives a gradient
of its own.

At sigma = 0 the triangle is a linear interpolation between the two closest integers. Where a
wider spread helps, sigma is annealed to 0 over training. 

| | delay parameter | shape |
|---|---|---|
| Axonal | one per presynaptic neuron, shared by its outgoing connections | `(N,)` |
| Synaptic | one per connection | `(N, N)` |
| Hybrid | axonal, plus a fixed per-connection integer offset drawn once and never trained | `(N,)` learned, `(N, N)` fixed |

Axonal and synaptic are either learned jointly with the weights, or frozen at a rounded random
initialization; the frozen models are the control that isolates delay learning. Hybrid keeps the
axonal parameter count but recovers per-connection resolution: the effective delay of connection
(i, j) is the learned axonal value d<sub>j</sub> plus a fixed offset delta<sub>ij</sub>, so it
generalizes the axonal model exactly when every offset is zero. All experiments use leaky
integrate-and-fire neurons.

## TOI Task Experimentation

Feedforward-only delay-parametrization comparison:

```bash
python experiments/compare_mem_feedforward.py
```

Both feedforward and recurrent comparisons:

```bash
python experiments/compare_mem_delays.py
```

All setup is done in `configs/perf_MEM.py`. The most important parameters to tune:

| parameter | meaning |
|---|---|
| `hidden_layers` | sets the network architecture |
| `epochs` | number of training epochs |
| `num_samples` | default: 256 |
| `input_size` | default: 8 |
| `time_window` | default: 32 |
| `output_size` | default: 4 |
| `input_gain` | default: 1.0 |
| `batch_size` | default: 64 |

## Reproducing HAR Table 8

To reproduce the experiments used for Table 8 of the technical report, run:

```bash
python experiments/scripts/run_HAR_table8.py
```

The script launches the six HAR configurations (feedforward and recurrent, each with axonal,
hybrid, and synaptic delays) for seeds 0, 1, and 2 over 100 epochs. It uses commands equivalent
to:

```bash
python experiments/train.py --dataset har \
  --model SNN_recurrent_hybrid_delays \
  --seeds 0,1,2 --epochs 100
```

Each seed is executed separately so that all outputs can be stored safely under one timestamped
directory in `exp/HAR/table8/`. The script also covers the newly implemented hybrid-delay model
classes. To inspect the 18 commands without starting training, use:

After all runs have completed, execute the first two cells of
`notebooks/compare_HAR_recurrent.ipynb`. They automatically select the latest Table 8 run,
aggregate the three seeds with the population standard deviation, and export the reconstructed
table to `figures/har_comparison/table6_HAR.csv`.

### `tests/` folder

- `check_hybrid_kernel.py` - verifies the hybrid delay networks before trusting a GPU run: on CUDA
  it runs the recurrent hybrid's fused Triton path and the pure-PyTorch reference side by side and
  asserts the forward output and every parameter gradient match.
  This is what `kaggle/run.py` runs before every GPU comparison.
- `test_hybrid_delay_invariance.py` - pytest unit tests asserting the hybrid delay classes
  generalize their paired axonal models: at `hybrid_offset_scale == 0` a hybrid model must be
  byte-for-byte identical to the axonal one (same `state_dict`, same init, same forward); at
  `> 0` it must still build, keep the axonal parameter count, and produce a finite forward.

## Citation

```bibtex
@article{queant2025delrec,
  title         = {DelRec: learning delays in recurrent spiking neural networks},
  author        = {Queant, Alexandre and Ran{\c c}on, Ulysse and Cottereau, Benoit R
                   and Masquelier, Timoth{\'e}e},
  journal       = {arXiv preprint arXiv:2509.24852},
  year          = {2025},
  eprint        = {2509.24852},
  archivePrefix = {arXiv},
  doi           = {10.48550/arXiv.2509.24852}
}
```

Machine-readable metadata is in [CITATION.cff](CITATION.cff). To cite the trained weights
themselves, use [10.5281/zenodo.21704323](https://doi.org/10.5281/zenodo.21704323), which
resolves to the latest version of the deposit.

## Acknowledgements

Built on [SpikingJelly](https://github.com/fangwei123456/spikingjelly). Supported by the French
Defense Innovation Agency under grant 2023 65 0082 and by the Agence Nationale de la Recherche
under grant ANR-20-CE45-0005 BRAIN-Net.

## License

MIT, see [LICENSE](LICENSE).
