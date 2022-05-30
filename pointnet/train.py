import argparse
import math
from asyncio.log import logger
from ntpath import join
from typing import List, Tuple
import numpy as np
import torch
import torch.utils.data as data
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from plyfile import PlyData
from pytorch3d.structures import Pointclouds
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset import OctantDataset, lm2flat_index
from .model import PointNet
from sh import calc_sh, to_spherical

torch.backends.cudnn.benchmark = True


def collate_batched_point_clouds(batch: List[Tuple[Pointclouds, torch.Tensor]]):
    coefs = torch.stack([x[1] for x in batch])
    pcs = Pointclouds(
        points=[x[0].points_packed() for x in batch],
        features=[x[0].features_packed() for x in batch],
    )
    return (pcs, coefs)


def random_batches(ds_size: int, batch_size: int) -> torch.LongTensor:
    r = torch.randperm(ds_size)
    batches = torch.arange(ds_size)[r].chunk(math.ceil(ds_size / batch_size))
    return batches


def unpack_pointclouds(pcs: Pointclouds):
    vertices = pcs.points_packed().cuda()
    color = (pcs.features_packed()[:, :3]).cuda().float() / 255.0
    batch = pcs.packed_to_cloud_idx().cuda()
    return vertices, color, batch


def weighted_l2_loss(weights: torch.Tensor):
    def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        l2 = diff.square().sum(-1)
        return (l2 * weights.to(prediction.device)).mean()

    return loss


def camera_loss(l_max: int, positions: torch.Tensor):

    y = calc_sh(l_max, positions)

    def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_r = y.to(prediction.device)
        pred_img = y_r @ prediction
        target_img = y_r @ target
        diff = pred_img - target_img
        l2 = diff.norm(2, -1).mean(-1)
        return l2.mean()

    return loss


def camera_positions(filename: str) -> torch.Tensor:
    plydata = PlyData.read(filename)
    vertex_data = plydata["vertex"].data

    def unpack_data(data, field_names):
        return torch.from_numpy(np.stack([data[key] for key in field_names]).T)

    cameras = unpack_data(vertex_data, ["x", "y", "z"])
    return to_spherical(cameras)


class CoefNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return x  # (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x  # x * self.std + self.mean

    def plot(self) -> Figure:
        fig, ax = plt.subplots()
        x = torch.arange(len(self.mean))
        ax.plot(x, self.std[:, 0].cpu(), label="red")
        ax.plot(x, self.std[:, 1].cpu(), label="green")
        ax.plot(x, self.std[:, 2].cpu(), label="blue")
        ax.set_xlabel("coefs.")  #
        ax.legend()
        return fig


def plot_samples(
    prediction_coefs: torch.Tensor,
    target_coefs: torch.Tensor,
    pc: Pointclouds,
    l_max: int,
    num_samples: int = 8,
    res: int = 100,
):
    x = torch.arange(0, 1, 1 / res)
    grid_x, grid_y = torch.meshgrid(x * torch.pi, x * 2 * torch.pi, indexing="ij")
    coords = torch.stack((grid_x.flatten(), grid_y.flatten())).T

    y = calc_sh(l_max, coords)

    rand_sample = torch.randint(0, len(prediction_coefs), (num_samples,))

    fig, axes = plt.subplots(num_samples, 4, figsize=(8, num_samples * 2))
    fig.tight_layout()

    targets = torch.empty((len(rand_sample), res, res, 3))
    predictions = torch.empty((len(rand_sample), res, res, 3))
    errors = torch.empty((len(rand_sample), res, res))

    for i, s in enumerate(rand_sample):
        targets[i] = (y @ target_coefs[s].cpu()).reshape(res, res, -1).clip(0, 1)
        predictions[i] = (
            (y @ prediction_coefs[s].cpu()).reshape(res, res, -1).clip(0, 1)
        )
        errors[i] = (predictions[i] - targets[i]).square().mean(-1)

    for i, s in enumerate(rand_sample):
        ax1, ax2, ax3, ax4 = axes[i]

        ax3.set_title("avg color")
        ax3.set_axis_off()
        avg = pc.features_list()[s].float().mean(0).reshape(1, 1, 4) / 255.0
        ax3.imshow(avg)

        ax1.set_title("target")
        ax1.set_axis_off()
        ax1.imshow(targets[i])

        ax2.set_title("prediction")
        ax2.set_axis_off()
        ax2.imshow(predictions[i])

        ax4.set_title("error")
        ax4.set_axis_off()
        ax4.imshow(errors[i], vmin=errors.min(), vmax=errors.max(), cmap="winter")

    return fig


def plot_coefs(
    pred: torch.Tensor, targets: torch.Tensor
) -> Tuple[Figure, Figure, Figure]:

    fig1, ax = plt.subplots()

    ax.set_ylabel("value")
    ax.set_xlabel("coef.")
    ax.plot((pred.cpu()).norm(2, -1).mean(0), label="prediction")
    ax.plot((targets.cpu()).norm(2, -1).mean(0), label="target")
    ax.legend()
    ax.set_yscale("log")

    fig2, ax = plt.subplots()

    ax.set_ylabel("error")
    ax.set_xlabel("coef.")
    error = (target_coefs - pred).abs().mean(-1).mean(0)
    ax.plot(error.cpu(), label="error")
    ax.legend()

    fig3, ax = plt.subplots()

    ax.set_ylabel("error (relative)")
    ax.set_xlabel("coef.")
    error = ((target_coefs - pred) / (target_coefs + 1e-8)).abs().mean(-1).mean(0)
    ax.plot(error.cpu(), label="error")
    ax.legend()
    ax.set_yscale("log")

    return fig1, fig2, fig3


def l2_loss():
    def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        l2 = diff.square().sum(-1)
        return l2.mean()

    return loss


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train pointnet")
    parser.add_argument("name", type=str, help="experiment name")
    parser.add_argument("dataset", type=str, help="folder containing the dataset")

    parser.add_argument(
        "--batch-size",
        dest="batch_size",
        type=int,
        default=4096,
        help="batch size used for training",
    )

    parser.add_argument(
        "--l",
        dest="l_max",
        type=int,
        default=5,
        help="maximum coef. degree to train model with",
    )
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs",
    )

    args = parser.parse_args()

    batch_size = args.batch_size
    ds_path = args.dataset
    l_max = args.l_max
    epochs = args.epochs
    experiment_name = args.name

    logger.setLevel("DEBUG")

    logger.info(f"loading dataset {ds_path}")
    ds = OctantDataset(ds_path, preload=True)

    num_train = int(0.8 * len(ds))

    ds_train, ds_val = data.random_split(ds, [num_train, len(ds) - num_train])

    logger.info(f"train: {len(ds_train)}, validation: {len(ds_val)}")

    train_batches = random_batches(len(ds_train), batch_size)
    val_batches = random_batches(len(ds_val), batch_size)

    model = PointNet(
        (l_max + 1) ** 2, batch_norm=False, use_dropout=False, use_spherical=False
    ).cuda()

    coefs = torch.stack([c[: lm2flat_index(l_max, -l_max) + 1] for _, c in ds_train])

    coef_transform = CoefNormalizer(coefs.mean(0).cuda(), coefs.std(0).cuda())

    cameras_pos = camera_positions("sphere.ply")

    loss_fn = camera_loss(l_max, cameras_pos)
    coef_loss_fn = l2_loss()

    writer_train = SummaryWriter(f"logs/{experiment_name}/train")
    writer_val = SummaryWriter(f"logs/{experiment_name}/val")

    writer_train.add_graph(model, unpack_pointclouds(ds_train[0][0]))
    writer_train.add_figure("coefficients/std", coef_transform.plot(), 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = OneCycleLR(
        optimizer, max_lr=1e-3, steps_per_epoch=len(train_batches), epochs=epochs
    )

    train_batches_pre = [
        collate_batched_point_clouds([ds_train[i] for i in s]) for s in train_batches
    ]
    val_batches_pre = [
        collate_batched_point_clouds([ds_val[i] for i in s]) for s in val_batches
    ]

    step = 0
    for epoch in tqdm(range(epochs), desc="epoch"):

        save_plot = epoch % 10 == 0

        for i, batch_indices in enumerate(train_batches):

            pcs, coefs = train_batches_pre[i]
            model.train()
            optimizer.zero_grad()

            vertices, color, batch = unpack_pointclouds(pcs)
            target_coefs = coefs[:, : lm2flat_index(l_max, -l_max) + 1].cuda()

            pred_coefs = model(vertices, color, batch)
            target_coefs_n = coef_transform.normalize(target_coefs)
            c_loss: torch.Tensor = loss_fn(pred_coefs, target_coefs_n)
            coef_loss: torch.Tensor = coef_loss_fn(pred_coefs, target_coefs_n)
            train_loss = c_loss + coef_loss
            train_loss.backward()

            optimizer.step()
            scheduler.step()

            writer_train.add_scalar(
                "loss/camera", train_loss.item(), step, new_style=True
            )
            writer_train.add_scalar("loss/coef", coef_loss.item(), step, new_style=True)
            writer_train.add_scalar(
                "loss/total", train_loss.item(), step, new_style=True
            )

            writer_train.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], step, new_style=True,
            )

            if save_plot:
                fig = plot_samples(
                    coef_transform.denormalize(pred_coefs.detach()),
                    target_coefs.detach(),
                    pcs,
                    l_max,
                )
                writer_train.add_figure("samples", fig, step)
                fig1, fig2, fig3 = plot_coefs(
                    coef_transform.denormalize(pred_coefs.detach()),
                    target_coefs.detach(),
                )
                writer_train.add_figure("coefficients/distribution", fig1, step)
                writer_train.add_figure("coefficients/error", fig2, step)
                writer_train.add_figure("coefficients/error_relative", fig3, step)

            # do validation batch every 4th train batch
            if i % 8 == 0:
                model.eval()
                rand_sample = torch.randint(0, len(val_batches), (1,)).item()
                # pcs, coefs = collate_batched_point_clouds(
                #     [ds_val[i] for i in val_batches[rand_sample]]
                # )

                pcs, coefs = val_batches_pre[rand_sample]

                vertices, color, batch = unpack_pointclouds(pcs)
                target_coefs = coefs[:, : lm2flat_index(l_max, -l_max) + 1].cuda()
                pred_coefs = model(vertices, color, batch)
                target_coefs_n = coef_transform.normalize(target_coefs)

                c_loss = loss_fn(pred_coefs, target_coefs_n)
                coef_loss = coef_loss_fn(pred_coefs, target_coefs_n)
                val_loss = c_loss + coef_loss

                writer_val.add_scalar(
                    "loss/camera", c_loss.item(), step, new_style=True
                )

                writer_val.add_scalar(
                    "loss/coef", coef_loss.item(), step, new_style=True
                )

                writer_val.add_scalar(
                    "loss/total", val_loss.item(), step, new_style=True
                )

                if save_plot:
                    fig = plot_samples(
                        coef_transform.denormalize(pred_coefs.detach()),
                        target_coefs.detach(),
                        pcs,
                        l_max,
                    )
                    writer_val.add_figure("samples", fig, step)
                    fig1, fig2, fig3 = plot_coefs(
                        coef_transform.denormalize(pred_coefs.detach()),
                        target_coefs.detach(),
                    )
                    writer_val.add_figure("coefficients/distribution", fig1, step)
                    writer_val.add_figure("coefficients/error", fig2, step)
                    writer_val.add_figure("coefficients/error_relative", fig3, step)

            step += 1
            save_plot = False

    writer_train.close()
    writer_val.close()

    torch.save(model, join("logs", experiment_name, "model.pt"))

