import argparse
import logging

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import DataLoader

from tqdm import tqdm

from pointnet.dataset import OctantDataset, collate_batched_point_clouds, lm2flat_index
from pointnet.model import PointNet
from pointnet.sh import to_spherical
from pointnet.loss import l2_loss, camera_loss, CoefNormalizer

# torch.backends.cudnn.benchmark = True


def camera_positions(filename: str) -> torch.Tensor:
    plydata = PlyData.read(filename)
    vertex_data = plydata["vertex"].data

    def unpack_data(data, field_names):
        return torch.from_numpy(np.stack([data[key] for key in field_names]).T)

    cameras = unpack_data(vertex_data, ["x", "y", "z"])
    return to_spherical(cameras)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train pointnet")
    parser.add_argument("model", type=str, help="model weights")
    parser.add_argument("dataset", type=str, help="folder containing the dataset")

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=512,
        help="batch size used for training",
    )

    parser.add_argument(
        "--l",
        dest="l_max",
        type=int,
        default=5,
        help="maximum coef. degree to train model with",
    )

    args = parser.parse_args()
    logger = logging.getLogger()

    batch_size = args.batch_size
    ds_path = args.dataset
    l_max = args.l_max
    model_weights = args.model

    logging.basicConfig(level=logging.INFO)
    logger.info(f"loading dataset {ds_path}")
    ds = OctantDataset(ds_path)

    data_loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_batched_point_clouds,
        pin_memory=True,
        drop_last=False,
        prefetch_factor=10,
        pin_memory_device="cuda:0",
    )

    coefs_samples = torch.stack(
        [
            ds[i][1][: lm2flat_index(l_max, l_max) + 1]
            for i in torch.randint(0, len(ds), (10000,))
        ]
    )
    model = PointNet(l_max, 4, batch_norm=True, use_spherical=False,).cuda()
    model.eval()
    model.load_state_dict(torch.load(model_weights))

    coef_transform = CoefNormalizer(model.coef_mean.clone(), model.coef_std.clone())

    cameras_pos = camera_positions("sphere.ply")

    loss_fn = camera_loss(l_max, cameras_pos)
    coef_loss_fn = l2_loss()

    for i, (batch) in tqdm(enumerate(data_loader), total=len(data_loader),):
        pcs, coefs = batch.pcs, batch.coefs

        target_coefs = coefs[:, : lm2flat_index(l_max, l_max) + 1].cuda()
        pred_coefs = model(pcs.unpack())

        c_loss = loss_fn(pred_coefs, target_coefs)

        target_coefs_n = coef_transform.normalize(target_coefs)
        pred_coefs_n = coef_transform.normalize(pred_coefs)
        coef_loss = coef_loss_fn(pred_coefs_n, target_coefs_n)

        val_loss = c_loss + coef_loss
