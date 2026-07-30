# Installation

## The package

```bash
git clone https://github.com/alexmaxad/DelRec.git && cd DelRec
pip install -r requirements.txt
```

Python 3.10 or newer. `requirements.txt` pins the environment that produced every number in
this repository, including the spikingjelly commit, which is not the version on PyPI.

`delrec` itself needs no install: every entry point prepends the repository root and `src/` to
`sys.path`. Run everything from the repository root, since the figure scripts resolve
`trained_models/`, `figures/` and `figures/measurements/` relative to it.

Triton is optional. Without it the delay layers fall back to the pure-PyTorch reference scan,
which is numerically identical and slower.

Check the install by rebuilding every figure and table, which needs no GPU, no dataset and no
download:

```bash
python experiments/make_figures/make_all.py --measurements-only
```

## Datasets

Only for training, or for recomputing an analysis from the checkpoints. Reproducing the
figures needs none of them.

Create `Datasets/` at the repository root. Each loader reads its path from the matching module
in `configs/`.

### SSC, Spiking Speech Commands, 35 classes

Download `ssc_train.h5`, `ssc_valid.h5` and `ssc_test.h5` from
[zenkelab.org/datasets](https://zenkelab.org/datasets) and place them in `Datasets/SSC/`. The
loader bins the 700 input channels to 140 and time to 250 steps of 5.6 ms.

### PS-MNIST, permuted sequential MNIST

Downloaded from torchvision into `Datasets/PSMNIST/` on first use. Each run draws its own pixel
permutation and saves it as `perm.pt` in the run directory. Evaluating a checkpoint under a
different permutation is meaningless, so the analysis code always loads the run's own file.

### HAR, WISDM smartwatch activity, 18 classes

Download the [WISDM 2019 dataset](https://archive.ics.uci.edu/dataset/507/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset)
and extract it so the watch gyroscope files sit at
`Datasets/HAR/wisdm-dataset/raw/watch/gyro/*.txt`. Nothing is cached: every run re-reads those
files and re-windows the signal in memory.

### AL, Autonomous Localization

Synthetic. Generated on first use and cached under `Datasets/AL/`. Nothing to download.

### Mackey-Glass

Nothing to download and nothing cached. The series is integrated in memory on every run from
the parameters in `configs/perf_MG.py`.

### SHD, Spiking Heidelberg Digits

Create `Datasets/SHD/`. The loader downloads and preprocesses the data on first use. Used only
by the appendix result.
