import glob
import os
from typing import Tuple

import numpy as np
import torch
from iopath.common.file_io import PathManager
from pytorch3d import io
from pytorch3d.structures import Pointclouds
from torch.utils.data import Dataset
from functools import lru_cache
from math import sqrt


def lm2flat_index(l: int, m: int) -> int:
    return l * (l + 1) - m


def flat2lm_index(i: int) -> Tuple[int, int]:
    l = int(sqrt(i))
    m = l * (l + 1) - i
    return l, m


class OctantDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(self, data_dir: str, sub_sample: int = None):
        self.data_dir = data_dir
        self.ply_files = np.array(glob.glob(os.path.join(data_dir, "*.ply")))
        if sub_sample is not None:
            self.ply_files = np.random.choice(self.ply_files, sub_sample)

    def __len__(self):
        return len(self.ply_files)

    @lru_cache(maxsize=100000)
    def load_ply(self, file: str) -> Tuple[Pointclouds, torch.Tensor]:
        _, data = io.ply_io._load_ply_raw(file, path_manager=PathManager())
        pos, color = data["vertex"]
        sh_coefficients = data["sh_coefficients"]

        sh_coef = torch.empty((len(sh_coefficients), 3), requires_grad=False)

        for (l, m, values) in sh_coefficients:
            sh_coef[lm2flat_index(l, m)] = torch.tensor(values)

        coords = torch.from_numpy(pos)
        color = torch.from_numpy(color)

        before = len(coords)
        nan_mask = coords.isnan().any(-1)
        coords = coords[~nan_mask]
        color = color[~nan_mask]
        if before - len(coords) > 0:
            print(f"warning: {file} contains nan position")

        if len(coords) == 0:
            coords = torch.zeros((len(color), 3))
            print(f"warning: {file} has no coords")

        pc = Pointclouds(coords.unsqueeze(0), features=color.unsqueeze(0),)
        return (pc, sh_coef)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        return self.load_ply(self.ply_files[idx])

