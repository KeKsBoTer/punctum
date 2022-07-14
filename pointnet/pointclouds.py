from tkinter.messagebox import NO
from typing import List
from typing_extensions import Self
import torch


class Pointclouds:
    def __init__(
        self, points: torch.Tensor, features: torch.Tensor, file_names: str, batch_index: torch.Tensor = None
    ):
        assert (
            points.shape[0] == features.shape[0]
        ), "number of points and features do not match"
        self.points = points
        self.features = features
        if batch_index is None:
            self.batch_index = torch.zeros(len(points),dtype=torch.long)
        else:
            self.batch_index = batch_index
        self.file_names = file_names

    def points_packed(self) -> torch.Tensor:
        return self.points

    def features_packed(self) -> torch.Tensor:
        return self.features

    def packed_to_cloud_idx(self) -> torch.Tensor:
        return self.batch_index


def collate_batched(pointsclouds: List[Pointclouds]) -> Pointclouds:
    assert all(len(pc.file_names)==1 for pc in pointsclouds), "only pointclouds containing only one pointcloud are supported"
    points = torch.cat([pc.points for pc in pointsclouds])
    features = torch.cat([pc.features for pc in pointsclouds])
    file_names = [pc.file_names[0] for pc in pointsclouds]
    batch_index = torch.cat([torch.zeros_like(pc.batch_index)+i for i,pc in enumerate(pointsclouds)])
    return Pointclouds(points,features,file_names,batch_index)