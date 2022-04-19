import glob
import os
from typing import Tuple

import torch
from iopath.common.file_io import PathManager
from pytorch3d import io
from pytorch3d.structures import Pointclouds
from torch.utils.data import Dataset


def lm2flat_index(l: int, m: int) -> int:
    return l * (l + 1) - m


class OctantDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(
        self, data_dir: str,
    ):
        self.data_dir = data_dir
        self.ply_files = glob.glob(os.path.join(data_dir, "*.ply"))

    def __len__(self):
        return len(self.ply_files)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        _, data = io.ply_io._load_ply_raw(
            self.ply_files[idx], path_manager=PathManager()
        )
        pos, color = data["vertex"]
        sh_coefficients = data["sh_coefficients"]

        sh_coef = torch.empty((len(sh_coefficients), 3), requires_grad=False)

        for (l, m, values) in sh_coefficients:
            sh_coef[lm2flat_index(l, m)] = torch.from_numpy(values)

        pc = Pointclouds(
            torch.from_numpy(pos).unsqueeze(0),
            features=torch.from_numpy(color).unsqueeze(0),
        )
        return (pc, sh_coef)

