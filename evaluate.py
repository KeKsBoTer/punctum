import argparse
import csv
import logging
from os import path

import numpy as np
import torch
from plyfile import PlyData
from torch.utils.data import DataLoader

from tqdm import tqdm

from pointnet.dataset import OctantDataset, collate_batched_point_clouds, lm2flat_index
from pointnet.model import PointNet
from pointnet.sh import to_spherical
from pointnet.metrics import camera_color


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
    ds = OctantDataset(ds_path,load_cam_colors=True)

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

    color_channels = 3
    model = PointNet(l_max, color_channels, batch_norm=True, use_spherical=False,).cuda()
    model.eval()
    model.load_state_dict(torch.load(model_weights))

    cameras_pos = camera_positions("sphere.ply")

    loss_fn = camera_color(l_max, cameras_pos)
    
    loss_values = []
    cam_color_variance = []

    for i, (batch) in tqdm(enumerate(data_loader), total=len(data_loader),):
        pcs, coefs,cam_colors = batch.pcs, batch.coefs,batch.cam_colors

        target_coefs = coefs[:, : lm2flat_index(l_max, l_max) + 1].cuda()

        pos, color, batch =pcs.unpack()
        pred_coefs = model(pos, color, batch)

        c_loss:torch.Tensor = loss_fn(pred_coefs, target_coefs)
        loss_values.extend(c_loss.tolist())

        color_variance:torch.Tensor = cam_colors.var(1).max(1).values
        cam_color_variance.extend(color_variance.tolist())

    loss_values = torch.Tensor(loss_values)
    cam_color_variance = torch.Tensor(cam_color_variance)

    num_steps = 10
    max_v = cam_color_variance.max()
    min_v = cam_color_variance.min()
    steps = [min_v+(max_v.max()-min_v)*i*1/num_steps for i in range(num_steps)]
    
    out_file = path.join(path.dirname(ds_path),"evaluation.csv")
    with open(out_file, 'w') as csvfile: 
        writer = csv.writer(csvfile)
        writer.writerow(["dataset","model_weights"]+[f">= {s:.4f}" for s in steps])
        writer.writerow([ds_path,model_weights]+[
            loss_values[cam_color_variance >=s].mean().item() for s in steps
        ])

    logger.info(f"saved results to '{out_file}'")