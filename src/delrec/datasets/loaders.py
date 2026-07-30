"""Dataloaders for every benchmark. ``load_dataset(config)`` dispatches on
``config.dataset`` and returns (train, valid, test).

Two things to know. SSC streams from HDF5 through a stateful iterator that must be
``reset()`` between epochs. AL and HAR have no validation split, so their test
loader is returned as the validation loader too, matching the protocol of [30],
which is why those benchmarks report the best checkpoint over training rather than
a held-out selection.
"""

import torch 
import numpy as np
import h5py
import os

from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils.data import random_split
from torch.distributions.binomial import Binomial
from typing import Callable, Optional

from delrec.datasets.shd_downloader import SpikingHeidelbergDigits
from delrec.datasets.wisdm import WISDM
from delrec.datasets.autonomous_localization import AL
from delrec.datasets.mackey_glass import generate_mackey_glass, MackeyGlassDataset
from spikingjelly.datasets import pad_sequence_collate
from sklearn.model_selection import train_test_split

from delrec.utils import seed_everything

def load_dataset(config):
    if config.dataset == 'SHD':
        return SHD_dataloaders(config)
    elif config.dataset == 'SSC':
       return SSC_dataloaders(config)
    elif config.dataset == 'PSMNIST':
        return PS_MNIST_dataloaders(config)
    elif config.dataset == 'HAR':
        return HAR_dataloaders(config)
    elif config.dataset == 'AL':
        return AL_dataloaders(config)
    elif config.dataset == 'MG':
        return MG_dataloaders(config)
    else:
        raise ValueError(f"Dataset {config.dataset} is not supported.")


def MG_dataloaders(config):
    """Mackey-Glass dataloaders (synthetic, generated in memory on every run)."""
    seed_everything(config.seed, is_cuda=True)

    series = generate_mackey_glass(
        total=config.mg_total, tau=config.mg_tau, dt=config.mg_dt,
        beta=config.mg_beta, gamma=config.mg_gamma, n=config.mg_n,
        x0=config.mg_x0, discard=config.mg_discard,
    )

    n_train = int(config.train_frac * len(series))
    n_val = int(config.val_frac * len(series))

    train_series = series[:n_train]
    val_series = series[n_train: n_train + n_val]
    test_series = series[n_train + n_val:]

    mu, sigma = train_series.mean(), train_series.std()
    train_series = (train_series - mu) / sigma
    val_series = (val_series - mu) / sigma
    test_series = (test_series - mu) / sigma

    W, H = config.window_size, config.prediction_horizon
    train_dataset = MackeyGlassDataset(train_series, W, H)
    valid_dataset = MackeyGlassDataset(val_series, W, H)
    test_dataset = MackeyGlassDataset(test_series, W, H)

    kwargs = {'num_workers': config.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
    valid_loader = DataLoader(valid_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    return train_loader, valid_loader, test_loader


def AL_dataloaders(config):
    """AL (autonomous localization) synthetic dataloaders."""
    seed_everything(config.seed, is_cuda=True)

    train_dataset = AL(root=config.datasets_path, subset="train", num_data=config.al_train_num,
                       seq_length=config.time_window, capacity=config.capacity)
    test_dataset = AL(root=config.datasets_path, subset="test", num_data=config.al_test_num,
                      seq_length=config.time_window, capacity=config.capacity)

    kwargs = {'num_workers': config.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_loader



def HAR_dataloaders(config):
    """WISDM Human Activity Recognition dataloaders."""
    seed_everything(config.seed, is_cuda=True)

    x_train_path = os.path.join(config.datasets_path, "x_train.npy")
    y_train_path = os.path.join(config.datasets_path, "y_train.npy")
    x_test_path = os.path.join(config.datasets_path, "x_test.npy")
    y_test_path = os.path.join(config.datasets_path, "y_test.npy")

    if not (os.path.exists(x_train_path) and os.path.exists(y_train_path)
            and os.path.exists(x_test_path) and os.path.exists(y_test_path)):
        data = WISDM(config.datasets_path).dataloading(config.window_size, config.overlap)
        X_ = data[..., 3:6]   # sensor data (X, Y, Z)
        Y_ = data[:, 0, 1]    # activity label
        X_train, X_test, Y_train, Y_test = train_test_split(X_, Y_, test_size=0.2, random_state=42)
        np.save(x_train_path, X_train.numpy())
        np.save(y_train_path, Y_train.numpy())
        np.save(x_test_path, X_test.numpy())
        np.save(y_test_path, Y_test.numpy())
    else:
        X_train = np.load(x_train_path)
        Y_train = np.load(y_train_path)
        X_test = np.load(x_test_path)
        Y_test = np.load(y_test_path)

    train_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_train).float(), torch.tensor(Y_train, dtype=torch.long))
    test_dataset = torch.utils.data.TensorDataset(
        torch.tensor(X_test).float(), torch.tensor(Y_test, dtype=torch.long))

    kwargs = {'num_workers': config.num_workers, 'pin_memory': True}
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, **kwargs)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, shuffle=False, **kwargs)

    return train_loader, test_loader, test_loader

def SHD_dataloaders(config):
    seed_everything(config.seed,is_cuda=True)
  
    train_dataset = BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=True, data_type='frame', duration=config.time_step)
    test_dataset= BinnedSpikingHeidelbergDigits(config.datasets_path, config.n_bins, train=False, data_type='frame', duration=config.time_step)

    train_dataset, valid_dataset = random_split(train_dataset, [0.8, 0.2])
  
    if config.use_augmentations:
        train_dataset = SHDTripleAugDataset(train_dataset, shift_max=config.shift_max, thin_p=config.thin_p, jitter_in_blend=config.jitter_in_blend)
      
    if (config.mask_T_stepsi is not None and config.mask_T_steps > 0) or (config.masked_proba is not None and config.masked_proba > 0.0):
        train_dataset = SHDMaskDataset(train_dataset, mask_T_steps=config.mask_T_steps, masked_proba=config.masked_proba)
        valid_dataset = SHDMaskDataset(valid_dataset, mask_T_steps=config.mask_T_steps, masked_proba=config.masked_proba)
        test_dataset = SHDMaskDataset(test_dataset, mask_T_steps=config.mask_T_steps, masked_proba=config.masked_proba)

    train_loader = DataLoader(train_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size)
    test_loader = DataLoader(test_dataset, collate_fn=pad_sequence_collate, batch_size=config.batch_size, num_workers=4)

    return train_loader, valid_loader, test_loader

class BinnedSpikingHeidelbergDigits(SpikingHeidelbergDigits):
    def __init__(
            self,
            root: str,
            n_bins: int,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
        self.n_bins = n_bins

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            binned_len = frames.shape[1]//self.n_bins
            binned_frames = np.zeros((frames.shape[0], binned_len))
            for i in range(binned_len):
                binned_frames[:,i] = frames[:, self.n_bins*i : self.n_bins*(i+1)].sum(axis=1)

            if self.transform is not None:
                binned_frames = self.transform(binned_frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return binned_frames, label
        
class SHDTripleAugDataset(Dataset):
    def __init__(self,
                 base_ds_or_subset,
                 shift_max: int = 40,
                 thin_p: float = 0.5,
                 jitter_in_blend: bool = False):
        super().__init__()
        self.base = base_ds_or_subset
        self.shift_max = int(shift_max)
        self.thin_p = float(thin_p)
        self.jitter_in_blend = bool(jitter_in_blend)
        
        self.is_subset = hasattr(self.base, "indices") and hasattr(self.base, "dataset")
        if self.is_subset:
            self.indices = list(self.base.indices)
        else:
            self.indices = list(range(len(self.base)))
        self.n = len(self.indices)

        self.labels = np.empty(self.n, dtype=np.int64)
        for pos in range(self.n):
            _, y = self._get_item_pos(pos)
            self.labels[pos] = int(y)

        self.class_pos = {}
        for pos, y in enumerate(self.labels):
            self.class_pos.setdefault(int(y), []).append(pos)
            
    def _get_item_pos(self, pos: int):
        if self.is_subset:
            return self.base[pos]                   
        else:
            return self.base[self.indices[pos]] 

    @staticmethod
    def _time_shift_per_neuron(x: torch.Tensor, shift_max: int) -> torch.Tensor:
        if shift_max <= 0:
            return x
        T, N = x.shape
        out = x.new_zeros(T, N)
        shifts = torch.randint(-shift_max, shift_max + 1, (N,), device=x.device)
        for n in range(N):
            s = int(shifts[n])
            if s > 0:
                # delay by s
                L = T - s
                if L > 0:
                    out[s:s+L, n] = x[:L, n]
            elif s < 0:
                # advance by |s|
                s2 = -s
                L = T - s2
                if L > 0:
                    out[:L, n] = x[s2:s2+L, n]
            else:
                out[:, n] = x[:, n]
        return out

    @staticmethod
    def _com_time(x: torch.Tensor) -> float:
        T = x.shape[0]
        mass_t = x.sum(dim=1)
        denom = mass_t.sum()
        if denom <= 0:
            return 0.5 * (T - 1)
        t = torch.arange(T, device=x.device, dtype=x.dtype)
        return float((t * mass_t).sum() / denom)

    @staticmethod
    def _thin_binomial(x: torch.Tensor, p: float) -> torch.Tensor:
        xi = x.clamp_min(0).round()
        if xi.numel() == 0:
            return xi
        dist = Binomial(total_count=xi, probs=torch.tensor(p, device=xi.device))
        return dist.sample()
    
    @staticmethod
    def _align_and_pad_pair(x: torch.Tensor, xb: torch.Tensor, align: int) -> tuple[torch.Tensor, torch.Tensor]:
        T1, N = x.shape
        T2, N2 = xb.shape
        assert N == N2, "Neuron/channel dimension mismatch."

        pad_x_left  = max(0, -align)   # if align<0, push x to the right
        pad_xb_left = max(0,  align)   # if align>0, push xb to the right

        T_out = max(T1 + pad_x_left, T2 + pad_xb_left)

        x_pad  = x.new_zeros(T_out, N)
        xb_pad = xb.new_zeros(T_out, N)

        x_pad[ pad_x_left : pad_x_left + T1, : ] = x
        xb_pad[pad_xb_left: pad_xb_left + T2, : ] = xb

        return x_pad, xb_pad

    def _sample_same_class_partner(self, y: int, avoid_pos: int) -> Optional[int]:
        pool = self.class_pos.get(int(y), [])
        if len(pool) <= 1:
            return None
        j = avoid_pos
        while j == avoid_pos:
            j = pool[np.random.randint(0, len(pool))]
        if int(self.labels[j]) != int(y):
            return None
        return j

    def __len__(self) -> int:
        return 3 * self.n

    def __getitem__(self, idx: int):
        
        if idx < self.n: 
            x, y = self._get_item_pos(idx)            # (T, N), y
            return torch.as_tensor(x, dtype=torch.float32), int(y)

        if idx < 2 * self.n:
            pos = idx - self.n
            x, y = self._get_item_pos(pos)
            x = torch.as_tensor(x, dtype=torch.float32)
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        pos = idx - 2 * self.n
        x, y = self._get_item_pos(pos)
        x = torch.as_tensor(x, dtype=torch.float32)

        partner_pos = self._sample_same_class_partner(int(y), pos)
        if partner_pos is None:
            # no partner ->  shift-only augmentation
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        xb, yb = self._get_item_pos(partner_pos)
        xb = torch.as_tensor(xb, dtype=torch.float32)
        if int(yb) != int(y):
            if self.shift_max > 0:
                x = self._time_shift_per_neuron(x, self.shift_max)
            return x, int(y)

        align = int(round(self._com_time(x) - self._com_time(xb)))
        x, xb = self._align_and_pad_pair(x, xb, align)

        if self.jitter_in_blend and self.shift_max > 0:
            x  = self._time_shift_per_neuron(x,  self.shift_max)
            xb = self._time_shift_per_neuron(xb, self.shift_max)

        x = self._thin_binomial(x,  self.thin_p) + self._thin_binomial(xb, 1- self.thin_p)
        return x, int(y)
    
def mask_time_steps_tensor(x: torch.Tensor, mask_T_steps: int) -> torch.Tensor:
    """x: (T, N) tensor (time first)."""
    if mask_T_steps <= 0:
        return x

    assert x.dim() == 2, "Expected (T, N) tensor."
    T, N = x.shape
    step = mask_T_steps + 1

    keep = torch.zeros(T, dtype=torch.bool, device=x.device)
    keep[::step] = True  # keep t = 0, step, 2*step, ...

    x_masked = x.clone()
    x_masked[~keep, :] = 0
    return x_masked

def random_mask_spikes_tensor(x: torch.Tensor, masked_proba: float) -> torch.Tensor:
    """
    x: (T, N) or any shape tensor of spike counts.
    Each bin is independently set to 0 with probability `masked_proba`.
    """
    if masked_proba <= 0.0:
        return x
    if masked_proba >= 1.0:
        return torch.zeros_like(x)

    keep_mask = (torch.rand_like(x) > masked_proba).to(x.dtype)
    return x * keep_mask

class SHDMaskDataset(Dataset):
    """Wraps any (x, y) dataset where x is (T, N) or np.array of shape (T, N), and applies:
    deterministic temporal subsampling via mask_T_steps, and/or random spike masking via
    masked_proba.
    """
    def __init__(self, base_ds, mask_T_steps: int = 0, masked_proba: float = 0.0):
        super().__init__()
        self.base = base_ds
        self.mask_T_steps = int(mask_T_steps)
        self.masked_proba = float(masked_proba)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        x, y = self.base[idx]

        x = torch.as_tensor(x, dtype=torch.float32)

        if self.mask_T_steps > 0:
            x = mask_time_steps_tensor(x, self.mask_T_steps)

        if self.masked_proba > 0.0:
            x = random_mask_spikes_tensor(x, self.masked_proba)

        return x, int(y)
        
def SSC_dataloaders(config):
    seed_everything(config.seed,is_cuda=True)
    T = config.time_window
    max_time = 1.4
    
    root_path = config.datasets_path
    dataset = config.dataset
    
    train_file = h5py.File(os.path.join(root_path, dataset.lower()+'_train.h5'), 'r')
    valid_file = h5py.File(os.path.join(root_path, dataset.lower()+'_valid.h5'), 'r')
    test_file = h5py.File(os.path.join(root_path, dataset.lower()+'_test.h5'), 'r')

    x_train = train_file['spikes']
    y_train = train_file['labels']
    x_valid = valid_file['spikes']
    y_valid = valid_file['labels']
    x_test = test_file['spikes']
    y_test = test_file['labels']
    
    train_loader = SSC_SpikeIterator(x_train, y_train, config.batch_size, T, 700, max_time, n_bins = config.n_bins, shuffle=True)
    valid_loader = SSC_SpikeIterator(x_valid, y_valid, config.batch_size, T, 700, max_time, n_bins = config.n_bins, shuffle=True)
    test_loader = SSC_SpikeIterator(x_test, y_test, config.batch_size, T, 700, max_time, n_bins = config.n_bins, shuffle=False)
    
    return train_loader, valid_loader, test_loader
    
class SSC_SpikeIterator:
    def __init__(self, X, y, batch_size, nb_steps, nb_units, max_time, n_bins = 1,shuffle=True, device='cuda:0', indices=None, label_map=None):
        self.batch_size = batch_size
        self.nb_steps = nb_steps
        self.nb_units = nb_units
        self.shuffle = shuffle
        self.labels_ = np.array(y, dtype=np.float32)
        self.num_samples = len(self.labels_)
        self.number_of_batches = np.ceil(self.num_samples / self.batch_size)
        self.sample_index = np.arange(len(self.labels_))
        
        self.firing_times = X['times']
        self.units_fired = X['units']
        self.time_bins = np.linspace(0, max_time, num=nb_steps)

        self.n_bins = n_bins

        # Digitising the spike times into time-bins and the fired-unit indices is
        # pass-invariant (it depends only on the fixed spikes + bins), yet the old
        # code redid it (via a per-sample Python .extend loop) on every batch of
        # every epoch, which was ~98% of the loader cost. Precompute both once here.
        # __next__ then just concatenates them. Bit-identical to the old output.
        self.digitized_times = [np.digitize(t, self.time_bins).astype(np.int64)
                                for t in self.firing_times]
        self.units_int = [np.asarray(u, dtype=np.int64) for u in self.units_fired]

        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.reset()

    def reset(self):
        if self.shuffle:
            np.random.shuffle(self.sample_index)
        self.counter = 0

    def __iter__(self):
        return self

    def __len__(self):
        return int(self.number_of_batches)

    def __next__(self):
        if self.counter < self.number_of_batches:
            batch_index = self.sample_index[
                          self.batch_size * self.counter:min(self.batch_size * (self.counter + 1), self.num_samples)]

            # Vectorised COO assembly from the precomputed per-sample (time-bin, unit)
            # arrays, replaces the old per-sample Python .extend loop (the bottleneck).
            cols_t = [self.digitized_times[idx] for idx in batch_index]
            cols_u = [self.units_int[idx] for idx in batch_index]
            counts = [c.size for c in cols_t]
            bc = np.repeat(np.arange(len(batch_index)), counts)
            t_all = np.concatenate(cols_t) if cols_t else np.zeros(0, dtype=np.int64)
            u_all = np.concatenate(cols_u) if cols_u else np.zeros(0, dtype=np.int64)

            i = torch.from_numpy(np.stack([bc, t_all, u_all])).to(self.device)
            v = torch.ones(t_all.size, device=self.device)
            X_batch = torch.sparse_coo_tensor(
                i, v, (len(batch_index), self.nb_steps, self.nb_units)).to_dense()

            ###############################################################
            # Bin channels: reshape the unit axis (700) into (binned_len, n_bins) and
            # sum, identical to the old per-channel loop (remainder channels dropped).
            binned_len = self.nb_units // self.n_bins
            X_batch = (X_batch[:, :, :binned_len * self.n_bins]
                       .reshape(len(batch_index), self.nb_steps, binned_len, self.n_bins)
                       .sum(-1))
            ###############################################################
            y_batch = torch.tensor(self.labels_[batch_index], device=self.device).long()

            self.counter += 1
            return X_batch, y_batch

        else:
            raise StopIteration

def PS_MNIST_dataloaders(config):
    seed_everything(config.seed,is_cuda=True)
    config.time_window = 784
    config.input_dim = 1
    config.output_dim = 10
    is_cuda = True
    kwargs = {'num_workers': config.num_workers, 'pin_memory': True} if is_cuda else {}
    dataset_train = datasets.MNIST(config.datasets_path, train=True, download=True,
                                    transform=transforms.ToTensor())
    
    train_size = int(0.83 * len(dataset_train))  # 50,000
    val_size = len(dataset_train) - train_size   # 10,000
    dataset_train, dataset_val = random_split(
        dataset_train, [train_size, val_size],
        generator=torch.Generator().manual_seed(config.seed)
    )
    
    train_loader = DataLoader(
        dataset_train, batch_size=config.batch_size, shuffle=True, **kwargs
    )
    val_loader = DataLoader(
        dataset_val, batch_size=config.batch_size, shuffle=False, **kwargs
    )
    test_loader = DataLoader(
        datasets.MNIST(config.datasets_path, train=False, transform=transforms.ToTensor()),
        batch_size=config.batch_size, shuffle=False, **kwargs
    )
    
    return train_loader, val_loader, test_loader