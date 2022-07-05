import glob
import os
from typing import Tuple

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import Dataset
from functools import lru_cache

from tqdm import tqdm

from .pointclouds import Pointclouds

from .sh import lm2flat_index


class OctantDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(
        self,
        data_dir: str,
        sub_sample: int = None,
        preload: bool = True,
        selected_samples: torch.LongTensor = None,
    ):
        self.data_dir = data_dir
        self.ply_files = np.array(glob.glob(os.path.join(data_dir, "*.ply")))
        if selected_samples is not None:
            self.ply_files = self.ply_files[selected_samples.numpy()]
        if len(self.ply_files) == 0:
            raise FileNotFoundError(f"no ply files found in '{data_dir}'")
        self.preload = preload
        if sub_sample is not None:
            self.ply_files = np.random.choice(self.ply_files, sub_sample)

        self.load_fn = self.load_ply

        # load all into ram
        if preload:
            for i in tqdm(range(len(self.ply_files)), desc="loading dataset"):
                self.__getitem__(i)

            # self.ply_files = [f for i, f in enumerate(self.ply_files) if stds[i] > 0.01]

    def __len__(self):
        return len(self.ply_files)

    def filename(self, index: int) -> str:
        return self.ply_files[index]

    @lru_cache(maxsize=None)
    def load_ply(self, file: str) -> Tuple[Pointclouds, torch.Tensor]:
        plydata = PlyData.read(file)
        vertex_data = unpack_data(plydata["vertex"].data,["x","y","z","red","green","blue","alpha"])
        coords = vertex_data[:,:3]
        color = vertex_data[:,3:]/255

        sh_coef = torch.empty((plydata["sh_coefficients"].count, 4), requires_grad=False)

        for (l, m, values) in plydata["sh_coefficients"].data:
            sh_coef[lm2flat_index(l, m)] = torch.tensor(values) 

        before = len(coords)
        nan_mask = coords.isnan().any(-1)
        coords = coords[~nan_mask]
        color = color[~nan_mask]
        if before - len(coords) > 0:
            print(f"warning: {file} contains nan position")

        if len(coords) == 0:
            coords = torch.zeros((len(color), 3))
            print(f"warning: {file} has no coords")

        pc = Pointclouds(coords, features=color,file_names=[file])
        return (pc, sh_coef)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        return self.load_fn(self.ply_files[idx])


def unpack_data(data, field_names):
    return torch.from_numpy(np.stack([data[key] for key in field_names]).T)


class CamerasDataset(Dataset):
    """ Dataset containing pointclouds and their respective SH coefficients"""

    def __init__(
        self, data_dir: str, preload: bool = True,
    ):
        self.data_dir = data_dir
        self.ply_files = np.array(glob.glob(os.path.join(data_dir, "*.ply")))

        self.load_fn = self.load_ply
        if preload:
            for i in tqdm(range(len(self.ply_files)), desc="loading dataset"):
                self.__getitem__(i)

    def __len__(self):
        return len(self.ply_files)

    def filename(self, index: int) -> str:
        return self.ply_files[index]

    @lru_cache(maxsize=None)
    def load_ply(self, file: str) -> Tuple[Pointclouds, torch.Tensor]:
        plydata = PlyData.read(file)
        vertex_data = unpack_data(plydata["vertex"].data,["x","y","z","red","green","blue","alpha"])
        vertex_pos = vertex_data[:,:3]
        vertex_color = vertex_data[:,3:]/255


        camera_data = unpack_data(plydata["camera"].data,["x","y","z","red","green","blue","alpha"])
        pos = camera_data[:,:3]
        colors = camera_data[:,3:]/255


        sh_coef = None
        if "sh_coefficients" in plydata:
            sh_coef = torch.empty((plydata["sh_coefficients"].count, 4), requires_grad=False)

            for (l, m, values) in plydata["sh_coefficients"].data:
                sh_coef[lm2flat_index(l, m)] = torch.tensor(values) 

        return (pos, colors, vertex_pos, vertex_color, sh_coef)

    def __getitem__(self, idx: int) -> Tuple[Pointclouds, torch.Tensor]:
        return self.load_fn(self.ply_files[idx])
