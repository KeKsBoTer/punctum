""" PointNet implementation.
Based on https://github.com/fxia22/pointnet.pytorch
"""

import torch
from torch import nn
import torch.nn.functional as F


def scatter_reduce(x: torch.Tensor, batch: torch.Tensor,reduce="amax") -> torch.Tensor:
    batch_idx = batch.unsqueeze(-1).repeat(1, x.shape[1])
    target: torch.Tensor = torch.zeros(
        (batch_idx.max() + 1, x.shape[1])
    ,device=x.device)

    return target.scatter_reduce(0, batch_idx, x, reduce=reduce)


class FeatureNet(nn.Module):
    """ Global feature extraction network used in PointNet"""

    def __init__(
        self, batch_norm: bool = False, layer_sizes=[32, 128, 256],
    ):
        super(FeatureNet, self).__init__()
        # self.tnet = TNet()
        # rfft_dim = 16
        # self.gfft = GaussianFourierFeatureTransform(
        #    3 + color_channels, mapping_size=rfft_dim
        # )
        s1, s2, s3 = layer_sizes
        self.conv1 = torch.nn.Conv1d(3 + 3, s1, 1, bias=not batch_norm)
        self.conv2 = torch.nn.Conv1d(s1, s2, 1, bias=not batch_norm)
        self.conv3 = torch.nn.Conv1d(s2, s3, 1, bias=not batch_norm)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(s1)
            self.bn2 = nn.BatchNorm1d(s2)
            self.bn3 = nn.BatchNorm1d(s3)

    def forward(
        self, points: torch.Tensor, color: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """ calculates global features

        Args:
            points  (torch.Tensor[N,3]): batched points
            color   (torch.Tensor[N,3]): point colors
            batch   (torch.Tensor[N]): tensor indicating the batch number of each point

        Returns:
            torch.Tensor[B,1024]: global features
        """
        # x = points.transpose(1, 0)
        # trans = self.tnet(x, batch)
        # x = torch.bmm(trans[batch], x.T.unsqueeze(-1)).squeeze(-1)

        x: torch.Tensor = torch.cat([points, color], dim=-1)
        # x: Tensor = self.gfft(x)
        x = x.permute(1, 0).unsqueeze(0)
        if self.batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

        x = x.squeeze(0).T
        x_max = scatter_reduce(x, batch)
        return x_max

def to_spherical(coords: torch.Tensor) -> torch.Tensor:
    """Cartesian to spherical coordinate conversion.
    Args:
        cords: [N,3] cartesian coordinates
    """

    x = torch.empty_like(coords)
    x[:, 2] = coords.norm(p=2, dim=-1)
    x[:, 0] = (coords[:, 2] / x[:, 2]).acos()
    x[:, 1] = torch.atan2(coords[:, 1], coords[:, 0]) + torch.pi
    return x


class PointNet(nn.Module):
    """_PointNet used to predict SH coefficients """

    def __init__(
        self,
        l: int = 4,
        color_channels: int = 3,
        batch_norm: bool = False,
        use_dropout: bool = False,
        use_spherical: bool = False,
        coef_mean: torch.Tensor = None,
        coef_std: torch.Tensor = None,
        layer_sizes=[128, 64],
    ):
        super(PointNet, self).__init__()
        self.feat = FeatureNet(batch_norm)
        s1, s2 = layer_sizes
        self.fc1 = nn.Linear(256, s1, bias=not batch_norm)
        self.fc2 = nn.Linear(s1, s2, bias=not batch_norm)
        self.fc3 = nn.Linear(s2, (l + 1) ** 2 * color_channels)
        self.l = l
        self.color_channels = color_channels
        self.use_spherical = use_spherical
        self.use_dropout = use_dropout
        if use_dropout != False:
            self.dropout = nn.Dropout(p=use_dropout)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(s1)
            self.bn2 = nn.BatchNorm1d(s2)

        if coef_mean is None:
            coef_mean = torch.zeros(((l + 1) ** 2, color_channels))
        if coef_std is None:
            coef_std = torch.ones(((l + 1) ** 2, color_channels))

        self.coef_mean = nn.parameter.Parameter(coef_mean, requires_grad=False)
        self.coef_std = nn.parameter.Parameter(coef_std, requires_grad=False)

    def forward(
        self, points: torch.Tensor, color: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """ calculate coefficients

        Args:
            points (torch.Tensor[N,3]): vertex positions
            color (torch.Tensor[N,C]): colors c = self.color_channels
            batch (torch.Tensor[N]): tensor indicating the batch number of each point

        Returns:
            torch.Tensor[B,K,C]: coefficients
        """

        if self.use_spherical:
            points = to_spherical(points)
        x = self.feat(points, color, batch)
        if self.batch_norm:
            x = F.relu(self.bn1(self.fc1(x)))
            if self.use_dropout != False:
                x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            else:
                x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            if self.use_dropout != False:
                x = F.relu(self.dropout(self.fc2(x)))
            else:
                x = F.relu(self.fc2(x))

        x: torch.Tensor = self.fc3(x)
        x = x.reshape(-1, (self.l + 1) ** 2, self.color_channels)

        x = x * self.coef_std + self.coef_mean

        return x
