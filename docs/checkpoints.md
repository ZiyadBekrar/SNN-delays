# Trained model checkpoints

The weights ship as five archives that unpack into the run directories already in `trained_models/`, so a run
looks the same whether or not its weights are present.

They are on Zenodo at [10.5281/zenodo.21704323](https://doi.org/10.5281/zenodo.21704323),
which resolves to the latest version of the deposit. Use it both to cite the weights and to
browse them. `fetch_checkpoints.py` downloads the exact version that
[checksums.sha256](checksums.sha256) describes, and refuses to unpack an archive that does not
match, since a checksum is only meaningful against one version's bytes.

```bash
python fetch_checkpoints.py --list           # what is available and what you already have
python fetch_checkpoints.py --dataset ssc
python fetch_checkpoints.py --all
```

| archive | runs | download | contents |
|---|---|---|---|
| `delrec-checkpoints-ssc.tar.gz` | 23 | 93 MB | 4 delay families over 5 seeds, plus 3 feedforward-delay baselines |
| `delrec-checkpoints-har.tar.gz` | 426 | 171 MB | the delay-init, weight-decay, fixed-compression, spike-penalty, no-rounding and rounding-alternative sweeps |
| `delrec-checkpoints-psmnist.tar.gz` | 20 | 37 MB | 4 delay families over 5 seeds |
| `delrec-checkpoints-al.tar.gz` | 30 | 12 MB | 4 delay families over 5 seeds, plus the rounding sweep |
| `delrec-checkpoints-mg.tar.gz` | 192 | 12 MB | 4 families over 4 tau, 4 horizons and 3 seeds |

SHD checkpoints are not distributed.

## Contents

```python
{"epoch": int, "acc": float,            # validation accuracy at that epoch
 "model": state_dict, "optim": [...], "sched": [...], "rng": {...}}
```

Both optimizer states are kept, since weights and delays are optimized separately at different
learning rates. `best.pth` is the epoch with the best validation accuracy and is where every
reported number comes from. `last.pth` is not distributed, as no analysis reads it. `sigma`, the
delay spread, is a plain attribute rather than a buffer, so it is absent from the state dict and
evaluation code must set it to zero. Everything in `experiments/make_figures/` does.