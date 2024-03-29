from functools import lru_cache
import glob
import os
from typing import Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset

import numpy as np

from .pointclouds import Pointclouds, collate_batched

from .sh import lm2flat_index


class OctantDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(
        self, data_dir: str, sub_sample: int = None,with_alpha : bool = False,load_cam_colors:bool = False
    ):
        """loads a dataset from files

        Args:
            data_dir (str): directory containing ply files or a txt file listing the ply files locations
            sub_sample (int, optional): size of a random subsample of the dataset. Defaults to None.
            with_alpha (bool, optional): load alpha channel. Defaults to False.
            load_cam_colors (bool, optional): load average camera colors. Defaults to False.

        Raises:
            FileNotFoundError: directory does not contain ply files
        """
        self.with_alpha = with_alpha
        self.load_cam_colors = load_cam_colors
        if data_dir.endswith(".txt"):
            self.ply_files = np.genfromtxt(data_dir,dtype='str')
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

    def cam_colors(self,plydata:PlyData):
        fields = ["x", "y", "z", "red", "green", "blue"]
        if self.with_alpha:
            fields += ["alpha"]

        camera_data = unpack_data(
            plydata["camera"].data, fields
        )
        colors = camera_data[:, 3:] / 255.
        return colors

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

        if self.load_cam_colors:
            cam_colors = self.cam_colors(plydata)
            return (pc, sh_coef,cam_colors)
        else:
            return (pc, sh_coef)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        return self.load_fn(self.ply_files[idx])


def unpack_data(data, field_names):
    return torch.from_numpy(np.stack([data[key] for key in field_names]).T)


class OctantBatch:
    def __init__(self, pcs:torch.Tensor, coefs:torch.Tensor,cam_colors:torch.Tensor = None):
        self.pcs = pcs
        self.coefs = coefs
        self.cam_colors = cam_colors

    def pin_memory(self):
        self.pcs = self.pcs.pin_memory()
        self.coefs = self.coefs.pin_memory()
        if self.cam_colors is not None:
            self.cam_colors = self.cam_colors.pin_memory()
        return self


def collate_batched_point_clouds(
    batch: list[tuple[Pointclouds, torch.Tensor]]
) -> OctantBatch:
    coefs = torch.stack([x[1] for x in batch])
    pcs = collate_batched([x[0] for x in batch])
    cam_colors = None
    if len(batch[0])==3:
        cam_colors = torch.stack([x[2] for x in batch])
    return OctantBatch(pcs, coefs,cam_colors)