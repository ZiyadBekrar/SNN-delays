# Push-triggered Kaggle GPU run

Every push to `ZiyadBekrar/SNN-delays` makes GitHub Actions
(`.github/workflows/kaggle-gpu.yml`) upload `kaggle/` as the Kaggle script kernel
[`ziyadcs/snn-delays-gpu`](https://www.kaggle.com/code/ziyadcs/snn-delays-gpu)
and run it on a GPU. The kernel clones the repo at the pushed commit and runs
`experiments/compare_mem_delays.py --device cuda`.

## One-time setup

1. Kaggle: https://www.kaggle.com/settings -> API -> **Create New Token**. Kaggle
   now issues a single access token that looks like `KGAT_625ffa74...` (the old
   `kaggle.json` with `username` + 32-char `key` is gone). Copy the **whole
   string**, `KGAT_` prefix included.
2. GitHub repo -> **Settings -> Secrets and variables -> Actions -> New repository secret**
   (a *Repository* secret on `ZiyadBekrar/SNN-delays` itself, not an
   Environment/Dependabot secret):
   - `KAGGLE_API_TOKEN` = the full `KGAT_...` string
3. Push. First run creates the kernel; later pushes add versions.

Auth failures:

- *"Authentication required to call the Kaggle API"* - `KAGGLE_API_TOKEN` is
  missing or misnamed; an unset `${{ secrets.X }}` silently expands to `""`. The
  **Configure Kaggle auth** step now fails early with an explicit message.
- *"401 - Unauthorized"* - the secret exists but the value is wrong: a **stale
  token** (clicking *Create New Token* invalidates every earlier one), or a quote
  / trailing newline pasted into the secret. The **Verify Kaggle auth** step
  isolates this; **Configure Kaggle auth** prints the token length and prefix
  (must start `KGAT_`).

Run it locally instead:

```bash
pip install --upgrade kaggle
export KAGGLE_API_TOKEN=KGAT_...          # or: mkdir -p ~/.kaggle && printf %s "$KAGGLE_API_TOKEN" > ~/.kaggle/access_token
cd kaggle && kaggle kernels push -p .
```

## Where the results are

Three places, same data:

### 1. GitHub Actions run (start here)

Repo -> **Actions** tab -> the **kaggle-gpu** run for your commit.

- **Wait for the kernel to finish** step: live `queued` / `running` / `complete`
  polling.
- **Show results** step: prints the full kernel log and the contents of
  `comparison.json` inline in the step output.
- **kaggle-output** artifact at the bottom of the run summary: a downloadable zip
  of everything the kernel wrote to `/kaggle/working`:

  ```
  kaggle-output/
    snn-delays-gpu.log                     # full stdout/stderr of the run
    delay_comparison/
      comparison.json                      # the 9-model metrics block
      delay_comparison.png / .pdf          # the 3x3 comparison figure
      ff_axonal/ ff_synaptic/ ... (9 dirs) # one per model:
        final_train.json                   #   final loss/accuracy/params
        train_res.csv                      #   per-epoch loss & accuracy
        training_summary.png / .pdf
        last.pth                           #   model + optimizer checkpoint
        config.json, dataset.pt
  ```

  The job fails (red X) if the kernel errors or times out; the log and artifact
  are still uploaded (`if: always()`).

### 2. The Kaggle kernel page

<https://www.kaggle.com/code/ziyadcs/snn-delays-gpu> -> **Output** and **Logs**
tabs. Each push is a new **Version** (top-right dropdown) with its own output and
log. This is the source of truth if the Action's download step failed.

### 3. From your machine

```bash
kaggle kernels status ziyadcs/snn-delays-gpu            # queued | running | complete | error
kaggle kernels output ziyadcs/snn-delays-gpu -p ./out   # same files as the zip above
```

## Notes / limits

- **GPU quota** ~30 h/week per account, **one GPU session at a time**, ~9 h max
  per run. Rapid pushes queue on Kaggle; `concurrency` in the workflow only
  cancels the *waiting* Action, not a kernel already running.
- Needs a **phone-verified** Kaggle account (for `enable_internet`). Done.
- The kernel installs `DCLS`, `spikingjelly` and two small extras; it uses
  Kaggle's preinstalled CUDA `torch` and never reinstalls it.
- These models are tiny (64 hidden units, 256 samples); the GPU is not
  necessarily faster than CPU here, but the run is short either way.
- Run by hand (see "One-time setup" for local auth): `cd kaggle && kaggle kernels push -p .`
  - with `run.py`'s `COMMIT` left as the placeholder it uses the default branch.
