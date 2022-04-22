""" PointNet implementation.
Based on https://github.com/fxia22/pointnet.pytorch
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch_scatter import scatter
from utils import to_dense_batch


class TNet(nn.Module):
    """ Transformation Net. Predicts a 3x3 transformation maxtrix based on a set of points."""

    def __init__(self, batch_norm: bool = False):
        super(TNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 9)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)
            self.bn4 = nn.BatchNorm1d(512)
            self.bn5 = nn.BatchNorm1d(256)

        self.iden = nn.parameter.Parameter(torch.eye(3), requires_grad=False)

    def forward(self, x: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
        """_Predicts a 3x3 transformation matrix.

        Args:
            x       (torch.Tensor[3,N]): batched points
            batch   (torch.Tensor[N]): tensor indicating the batch number of each point

        Returns:
            (torch.Tensor[B,3,3]): transformation matrix
        """

        x = x.unsqueeze(0)

        if self.batch_norm:
            raise NotImplementedError("batchnorm needs heterogeneous implementation")
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.relu(self.bn3(self.conv3(x)))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))

        x = x.squeeze(0)
        x = scatter(x.T, batch, dim=0, reduce="max")

        if self.batch_norm:
            x = F.relu(self.bn4(self.fc1(x)))
            x = F.relu(self.bn5(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            x = F.relu(self.fc2(x))
        x = self.fc3(x)

        x = x.view(-1, 3, 3)
        x = x + self.iden
        return x


class FeatureNet(nn.Module):
    """ Global feature extraction network used in PointNet"""

    def __init__(self, batch_norm: bool = False):
        super(FeatureNet, self).__init__()
        self.tnet = TNet()
        self.conv1 = torch.nn.Conv1d(6, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)

        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(64)
            self.bn2 = nn.BatchNorm1d(128)
            self.bn3 = nn.BatchNorm1d(1024)

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
        x = points.transpose(1, 0)
        trans = self.tnet(x, batch)
        x, mask = to_dense_batch(x.T, batch)
        # TODO converting to dense and then back is inefficient
        # maybe scatter matrix according to batch id
        x = torch.bmm(x, trans)
        x = x[mask]
        x = torch.cat([x, color], dim=-1).transpose(1, 0).unsqueeze(0)
        if self.batch_norm:
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.relu(self.bn2(self.conv2(x)))
            x = self.bn3(self.conv3(x))
        else:
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = self.conv3(x)

        x = scatter(x.squeeze(0).T, batch, dim=0, reduce="max")

        return x


class PointNet(nn.Module):
    """_PointNet used to predict SH coefficients """

    def __init__(self, k: int = 3, batch_norm: bool = False, use_dropout: bool = False):
        super(PointNet, self).__init__()
        self.feat = FeatureNet(batch_norm)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * 3)
        self.use_dropout = use_dropout
        if use_dropout:
            self.dropout = nn.Dropout(p=0.3)
        self.batch_norm = batch_norm
        if batch_norm:
            self.bn1 = nn.BatchNorm1d(512)
            self.bn2 = nn.BatchNorm1d(256)

    def forward(
        self, points: torch.Tensor, color: torch.Tensor, batch: torch.Tensor
    ) -> torch.Tensor:
        """ calculate coefficients

        Args:
            points (torch.Tensor[N,3]): vertex positions
            color (torch.Tensor[N,3]): colors
            batch (torch.Tensor[N]): tensor indicating the batch number of each point

        Returns:
            torch.Tensor[B,K,3]: coefficients
        """
        x = self.feat(points, color, batch)
        if self.batch_norm:
            x = F.relu(self.bn1(self.fc1(x)))
            if self.use_dropout:
                x = F.relu(self.bn2(self.dropout(self.fc2(x))))
            else:
                x = F.relu(self.bn2(self.fc2(x)))
        else:
            x = F.relu(self.fc1(x))
            if self.use_dropout:
                x = F.relu(self.dropout(self.fc2(x)))
            else:
                x = F.relu(self.fc2(x))

        x: torch.Tensor = self.fc3(x)
        return x.reshape(-1, x.shape[1] // 3, 3)
