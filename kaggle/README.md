# Push-triggered Kaggle GPU run

Every push to `ZiyadBekrar/SNN-delays` makes GitHub Actions
(`.github/workflows/kaggle-gpu.yml`) upload `kaggle/` as the Kaggle script kernel
[`ziyadcs/snn-delays-gpu`](https://www.kaggle.com/code/ziyadcs/snn-delays-gpu)
and run it on a GPU. The kernel clones the repo at the pushed commit and runs
`experiments/compare_mem_delays.py --device cuda`. Artifacts
(`comparison.json`, plots, per-model `final_train.json`, checkpoints) are
downloaded and attached to the workflow run as the **kaggle-output** artifact.

## One-time setup

1. Kaggle: https://www.kaggle.com/settings -> **Create New Token** -> downloads
   `kaggle.json` = `{"username": "...", "key": "..."}`.
2. GitHub repo -> **Settings -> Secrets and variables -> Actions -> New repository secret**:
   - `KAGGLE_USERNAME` = `ziyadcs`
   - `KAGGLE_KEY` = the `key` value from `kaggle.json`
3. Push. First run creates the kernel; later pushes add versions.

## Notes / limits

- **GPU quota** ~30 h/week per account, **one GPU session at a time**, ~9 h max
  per run. Rapid pushes queue on Kaggle; `concurrency` in the workflow only
  cancels the *waiting* Action, not a kernel already running.
- Needs a **phone-verified** Kaggle account (for `enable_internet`). Done.
- The kernel installs `DCLS`, `spikingjelly` and two small extras; it uses
  Kaggle's preinstalled CUDA `torch` and never reinstalls it.
- These models are tiny (64 hidden units, 256 samples); the GPU is not
  necessarily faster than CPU here, but the run is short either way.
- Run it by hand: `cd kaggle && kaggle kernels push -p .`
  (with `run.py`'s `COMMIT` left as the placeholder it uses the default branch).
