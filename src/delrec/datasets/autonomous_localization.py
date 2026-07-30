"""
AL (Autonomous Localization) synthetic dataset.

Faithful port of neuroseqbench's AL loader
(https://github.com/liyc5929/neuroseqbench/blob/main/src/neuroseqbench/utils/dataset/autonomous_localization.py).

The task is fully synthetic. For each sample, a length-``seq_length`` action sequence is drawn from a
4-way multinomial over {no-op, move-forward, turn-left, turn-right}, where the turn probabilities are
``capacity / seq_length`` each (so ``capacity`` controls the expected number of turns). The input is
the one-hot encoding of the action with the no-op channel dropped -> ``(seq_length, 3)`` float. An
agent with a 4-way heading walks the sequence. The label is the final x-coordinate clamped to {0, 1}
-> binary classification (2 classes). Data is generated once and cached to HDF5 under
``<root>/AL/<subset>_<seq_length>_<capacity>/``.

Only ``compute`` differs from the reference: it is reimplemented with vectorized torch ops that
produce labels bit-identical to neuroseqbench's per-sample Python double loop (cumulative heading is a
ring homomorphism, so mod-4 can be deferred to the end). Everything else, sampling, channel layout,
cache layout, matches the reference exactly.
"""
import os

import h5py
import tqdm
import torch
import numpy as np
from typing import Union
from torch.utils.data import Dataset
import torch.nn.functional as F


class AL(Dataset):
    def __init__(self,
        root,
        subset                           = "train",
        num_data                         = 50000,
        seq_length                       = 400,
        capacity                         = 20,
        device: Union[str, torch.device] = "cpu",
    ):
        saved_data_file = f"AL"
        os.makedirs(os.path.join(root, saved_data_file), exist_ok=True)
        preprocessed_data_root = os.path.join(root, saved_data_file, f"{subset}_{seq_length}_{capacity}")
        if os.path.exists(preprocessed_data_root):
            h5file = h5py.File(f"{preprocessed_data_root}/preprocessed_data_num({num_data})_seqlen({seq_length}).h5", "r")
            input_iter = h5file["inputs"]
            label_iter = h5file["labels"]
            self.inputs = []
            self.labels = []
            for i in tqdm.tqdm(range(len(label_iter))):
                self.inputs.append(torch.tensor(input_iter[i], dtype=torch.float32).to(device))
                self.labels.append(torch.tensor(label_iter[i], dtype=torch.int64).to(device))
        else:
            probabilities = [0.5-(capacity/seq_length), 0.5-(capacity/seq_length), capacity/seq_length, capacity/seq_length]
            distribution = torch.multinomial(torch.tensor(probabilities), num_data * seq_length, replacement=True)
            action = distribution.view(num_data, seq_length)
            X = F.one_hot(action, num_classes=4)[..., 1:].float()
            Y = self.compute(action)
            Y = Y.view(-1)
            self.inputs = []
            self.labels = []
            os.mkdir(preprocessed_data_root)
            for i in tqdm.tqdm(range(len(Y))):
                data_input, data_label = X[i], Y[i].long()
                self.inputs.append(data_input.to(device))
                self.labels.append(data_label.to(device))
            with h5py.File(f"{preprocessed_data_root}/preprocessed_data_num({num_data})_seqlen({seq_length}).h5", "w") as fp:
                saved_inputs = fp.create_dataset("inputs", (len(self.inputs), *self.inputs[0].shape), dtype=np.float32)
                saved_labels = fp.create_dataset("labels", (len(self.labels), *self.labels[0].shape), dtype=np.int64)
                for i in tqdm.tqdm(range(len(self.labels))):
                    saved_inputs[i] = self.inputs[i].cpu()
                    saved_labels[i] = self.labels[i].cpu()

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, n):
        return (self.inputs[n], self.labels[n])

    def compute(self, action):
        # Vectorized equivalent of neuroseqbench's per-sample heading/position simulation.
        # act: 0 no-op, 1 move forward, 2 turn left (h-=1), 3 turn right (h+=1). H kept mod 4.
        # Only the final x-coordinate matters (clamped to {0,1}). X changes by +1 when moving
        # forward with h==1 and by -1 with h==3.
        num_data, seq_length = action.shape
        delta = (action == 3).long() - (action == 2).long()          # heading increment per step
        heading_after = torch.cumsum(delta, dim=1)                   # heading after each step (pre-mod)
        heading_before = (heading_after - delta) % 4                 # heading at the moment of the step
        x_step = (action == 1).long() * ((heading_before == 1).long() - (heading_before == 3).long())
        x = x_step.sum(dim=1)                                        # final x-coordinate
        Y = x.clamp(0, 1).view(num_data, 1).float()
        return Y
