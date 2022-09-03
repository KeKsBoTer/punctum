from functools import lru_cache
import glob
import os
from typing import Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

from tqdm import tqdm

from .pointclouds import Pointclouds, collate_batched

from .sh import lm2flat_index


class OctantDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(
        self, data_dir: str, sub_sample: int = None, selected_samples: list[str] = None, with_alpha : bool = False
    ):
        self.data_dir = data_dir
        self.with_alpha = with_alpha
        if selected_samples is not None:
            self.ply_files = np.array(selected_samples)
        else:
            self.ply_files = np.array(glob.glob(os.path.join(data_dir, "*.ply")))
            if len(self.ply_files) == 0:
                raise FileNotFoundError(f"no ply files found in '{data_dir}'")
        if sub_sample is not None:
            self.ply_files = np.random.choice(self.ply_files, sub_sample)

        self.load_fn = self.load_ply

    def __len__(self):
        return len(self.ply_files)

    def filename(self, index: int) -> str:
        return self.ply_files[index]

    @lru_cache(maxsize=100000)
    def load_ply(self, file: str) -> Tuple[Pointclouds, torch.Tensor]:
        plydata = PlyData.read(file)
        vertex_data = unpack_data(
            plydata["vertex"].data, ["x", "y", "z", "red", "green", "blue"]
        )
        coords = vertex_data[:, :3]
        color = vertex_data[:, 3:] / 255

        sh_coef = torch.empty(
            (plydata["sh_coefficients"].count, 4 if self.with_alpha else 3), requires_grad=False
        )

        for (l, m, values) in plydata["sh_coefficients"].data:
            sh_coef[lm2flat_index(l, m)] = torch.tensor(values)[:sh_coef.shape[-1]]

        before = len(coords)
        nan_mask = coords.isnan().any(-1)
        coords = coords[~nan_mask]
        color = color[~nan_mask]
        if before - len(coords) > 0:
            print(f"warning: {file} contains nan position")

        if len(coords) == 0:
            coords = torch.zeros((len(color), 3))
            print(f"warning: {file} has no coords")

        pc = Pointclouds(coords, features=color, file_names=[file])
        return (pc, sh_coef)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        return self.load_fn(self.ply_files[idx])


def unpack_data(data, field_names):
    return torch.from_numpy(np.stack([data[key] for key in field_names]).T)


class OctantBatch:
    def __init__(self, pcs, coefs):
        self.pcs = pcs
        self.coefs = coefs

    def pin_memory(self):
        self.pcs = self.pcs.pin_memory()
        self.coefs = self.coefs.pin_memory()
        return self


def collate_batched_point_clouds(
    batch: list[tuple[Pointclouds, torch.Tensor]]
) -> OctantBatch:
    coefs = torch.stack([x[1] for x in batch])
    pcs = collate_batched([pc for pc, _ in batch])
    return OctantBatch(pcs, coefs)


class CamerasDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.ply_files = np.array(glob.glob(os.path.join(data_dir, "*.ply")))

        self.load_fn = self.load_ply

    def __len__(self):
        return len(self.ply_files)

    def filename(self, index: int) -> str:
        return self.ply_files[index]

    def load_ply(self, file: str) -> Tuple[Pointclouds, torch.Tensor]:
        plydata = PlyData.read(file)

        fields = ["x", "y", "z", "red", "green", "blue"]
        print( plydata["camera"].dtype)
        if "alpha" in plydata["camera"].data:
            fields += ["alpha"]

        camera_data = unpack_data(
            plydata["camera"].data, fields
        )
        pos = camera_data[:, :3]
        colors = camera_data[:, 3:] / 255

        sh_coef = None
        if "sh_coefficients" in plydata:
            sh_coef = torch.empty(
                (plydata["sh_coefficients"].count, len(fields)-3), requires_grad=False
            )

            for (l, m, values) in plydata["sh_coefficients"].data:
                sh_coef[lm2flat_index(l, m)] = torch.tensor(values)

        return (pos, colors, file)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        return self.load_fn(self.ply_files[idx])
