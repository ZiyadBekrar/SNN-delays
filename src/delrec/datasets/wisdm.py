"""
WISDM Human Activity Recognition dataset.

Faithful port of neuroseqbench's WISDM loader
(https://github.com/liyc5929/neuroseqbench/blob/main/src/neuroseqbench/utils/dataset/human_activities_recognition.py).
Reference: Gary Weiss, "WISDM Smartphone and Smartwatch Activity and Biometrics Dataset", 2019.

The raw WISDM 2019 archive must be downloaded and extracted manually (it is not
auto-downloaded). The watch gyroscope ``.txt`` streams are expected under:

    <data_path>/wisdm-dataset/raw/watch/gyro/

Each ``.txt`` line is ``ID,Activity,TimeStamp,X,Y,Z;`` (trailing ``;`` is treated as a
comment). The 18 activity letters are mapped to integer labels 0..17 via ``activity_dic``,
X/Y/Z are z-score normalized per file, and ``dataloading`` cuts the stream into overlapping
windows, dropping windows that straddle two activities.
"""
import os
import pandas as pd
import numpy as np
import torch


class WISDM:
    """Class to design a WISDM Dataset."""

    def __init__(self, data_path):
        """Load every watch/gyro .txt file, normalize, and stack into one big tensor."""
        self.column_names = ["ID", "Activity", "TimeStamp", "X", "Y", "Z"]
        self.acitvity_names = ["Walking", "Jogging", "Stairs", "Sitting", "Standing", "Typing",
                               "Brush teeth", "Eat Soup", "Eat chips", "Eat Pasta", "Drinking",
                               "Eat Sandwich", "Kicking", "Catch", "Dribblilg", "Writing",
                               "Clapping", "Fold Clothes"]

        self.activity_dic = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7,
                             "I": 8, "J": 9, "K": 10, "L": 11, "M": 12, "O": 13, "P": 14,
                             "Q": 15, "R": 16, "S": 17}
        self.activity_dic_inv = {item: element for element, item in self.activity_dic.items()}

        self.folder = os.path.join(data_path, "wisdm-dataset", "raw", "watch", "gyro")
        if not os.path.isdir(self.folder):
            raise FileNotFoundError(
                f"WISDM gyro folder not found at '{self.folder}'. Download the WISDM 2019 "
                f"dataset and extract it so that the watch gyroscope .txt files live under "
                f"<datasets_path>/wisdm-dataset/raw/watch/gyro/."
            )
        self.filelist = [txt for txt in os.listdir(self.folder) if txt[-4:] == ".txt"]
        self.data_tensor = []
        self.data_tensor_raw = []

        self.__create_tensor()

    def __create_tensor(self):
        """Combine all text files into one big tensor (raw and normalized)."""
        for txt in self.filelist:
            self.data = pd.read_csv(os.path.join(self.folder, txt), header=None,
                                    names=self.column_names, comment=";")

            # Replace the letter activity codes with integer labels 0..17
            self.data["Activity"] = self.data["Activity"].map(self.activity_dic)

            if self.data_tensor_raw == []:
                self.data_tensor_raw = torch.tensor(self.data.values).float()
            else:
                self.data_tensor_raw = torch.cat((self.data_tensor_raw, torch.tensor(self.data.values).float()))

            self.__normalize_feature()

            if self.data_tensor == []:
                self.data_tensor = torch.tensor(self.data.values).float()
            else:
                self.data_tensor = torch.cat((self.data_tensor, torch.tensor(self.data.values).float()))

    def __normalize_feature(self):
        """Z-score normalize each sensor axis."""
        for dim in ["X", "Y", "Z"]:
            mue = np.mean(self.data[dim])
            sigma = np.std(self.data[dim])
            self.data[dim] = (self.data[dim] - mue) / sigma

    def slid_win(self, data, window_size, step_size):
        """Sliding window. Drop windows that span two activities."""
        output = data.unfold(0, window_size, step_size).transpose(1, 2)
        mask = torch.ones(output.shape[0], dtype=torch.bool)
        for i in range(output.shape[0]):
            if output[i, 0, 1] != output[i, -1, 1]:
                mask[i] = False
        output = output[mask]
        return output

    def dataloading(self, window_size, overlap):
        """Return the windowed tensor (cols: ID, Activity, TimeStamp, X, Y, Z)."""
        return self.slid_win(self.data_tensor, window_size, overlap)
