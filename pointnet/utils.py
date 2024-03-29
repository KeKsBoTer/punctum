import numpy as np
import torch
from plyfile import PlyData

from .sh import to_spherical


def camera_positions(filename: str,cartesian:bool = True) -> torch.Tensor:
    """loads camera positions from ply file

    Args:
        filename (str): ply file containing the camera positions
        cartesian (bool, optional): cartesian coordinates if true, else spherical. Defaults to True.

    Returns:
        torch.Tensor: camera positions as spherical or cartesian coordinates
    """
    plydata = PlyData.read(filename)
    vertex_data = plydata["vertex"].data

    def unpack_data(data, field_names):
        return torch.from_numpy(np.stack([data[key] for key in field_names]).T)

    cameras = unpack_data(vertex_data, ["x", "y", "z"])

    if cartesian:
        return to_spherical(cameras)
    else:
        return cameras
