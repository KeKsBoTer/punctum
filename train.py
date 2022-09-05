import argparse
import logging
from os import path
from typing import Tuple

import torch
import numpy as np
import torch.utils.data as data
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pointnet.dataset import OctantDataset, collate_batched_point_clouds, lm2flat_index
from pointnet.loss import camera_loss, l2_loss
from pointnet.model import PointNet
from pointnet.pointclouds import Pointclouds
from pointnet.sh import calc_sh
from pointnet.utils import camera_positions

class CoefNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.std[self.std.abs() < 1e-6] = 0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def plot(self) -> Figure:
        fig, ax = plt.subplots()
        x = torch.arange(len(self.mean))
        ax.plot(x, self.std[:, 0].cpu(), label="red")
        ax.plot(x, self.std[:, 1].cpu(), label="green")
        ax.plot(x, self.std[:, 2].cpu(), label="blue")
        if self.std.shape[1] == 4:
            ax.plot(x, self.std[:, 3].cpu(), label="alpha")
        ax.set_xlabel("coefs.")  #
        ax.legend()
        return fig

def stds(colors:torch.Tensor,batch_idx:torch.Tensor):
    stds = torch.stack([colors[batch_idx==b].std(0).max() for b in range(batch_idx.max().item()+1)])
    stds[stds.isnan() | (stds==0)]=1e-3
    return stds

def plot_samples(
    prediction_coefs: torch.Tensor,
    target_coefs: torch.Tensor,
    pc: Pointclouds,
    l_max: int,
    num_samples: int = 8,
    res: int = 100,
):
    color_channels = target_coefs.shape[-1]
    alpha = color_channels == 4
    x = torch.arange(0, 1, 1 / res)
    grid_x, grid_y = torch.meshgrid(x * torch.pi, x * 2 * torch.pi, indexing="ij")
    coords = torch.stack((grid_x.flatten(), grid_y.flatten())).T

    y = calc_sh(l_max, coords)

    rand_sample = torch.randint(0, len(prediction_coefs), (num_samples,))

    cols = 5 if alpha else 3
    fig, axes = plt.subplots(num_samples, cols, figsize=(cols * 2, num_samples * 2))
    fig.tight_layout()


    targets = torch.empty((len(rand_sample), res, res, color_channels))
    predictions = torch.empty((len(rand_sample), res, res, color_channels))
    errors = torch.empty((len(rand_sample), res, res))

    for i, s in enumerate(rand_sample):
        targets[i] = (y @ target_coefs[s].cpu()).reshape(res, res, -1).clip(0, 1)
        predictions[i] = (
            (y @ prediction_coefs[s].cpu()).reshape(res, res, -1).clip(0, 1)
        )
        errors[i] = (predictions[i] - targets[i]).square().mean(-1)

    for i, s in enumerate(rand_sample):
        ax1, ax2, ax3 = axes[i][:3]

        name = pc.file_names[s].split("/")[-1]
        ax1.set_title(f"target ({name})")
        ax1.set_axis_off()
        ax1.imshow(targets[i][:, :, :3])

        ax2.set_title("prediction")
        ax2.set_axis_off()
        ax2.imshow(predictions[i][:, :, :3])

        ax3.set_title("error")
        ax3.set_axis_off()
        ax3.imshow(errors[i], vmin=errors.min(), vmax=errors.max(), cmap="winter")

        if alpha:
            ax4, ax5 = axes[i][3:]
            ax4.set_title("alpha target.")
            ax4.set_axis_off()
            ax4.imshow(targets[i][:, :, 3], vmin=0, vmax=1)

            ax5.set_title("alpha pred.")
            ax5.set_axis_off()
            ax5.imshow(predictions[i][:, :, 3], vmin=0, vmax=1)

    return fig


def plot_coefs(
    pred: torch.Tensor, targets: torch.Tensor
) -> Tuple[Figure, Figure, Figure]:
    alpha = target_coefs.shape[-1] == 4

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
    error = (target_coefs - pred).abs().mean(0)
    ax.plot(error[:, 0].cpu(), label="red")
    ax.plot(error[:, 1].cpu(), label="green")
    ax.plot(error[:, 2].cpu(), label="blue")
    if alpha:
        ax.plot(error[:, 3].cpu(), label="alpha")
    ax.legend()

    fig3, ax = plt.subplots()

    ax.set_ylabel("error (relative)")
    ax.set_xlabel("coef.")
    error = ((target_coefs - pred) / (target_coefs + 1e-8)).abs().mean(0)
    ax.plot(error[:, 0].cpu(), label="red")
    ax.plot(error[:, 1].cpu(), label="green")
    ax.plot(error[:, 2].cpu(), label="blue")
    if alpha:
        ax.plot(error[:, 3].cpu(), label="alpha")
    ax.legend()
    ax.set_yscale("log")

    return fig1, fig2, fig3


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Train pointnet")
    parser.add_argument("name", type=str, help="experiment name")
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
    parser.add_argument(
        "--epochs", type=int, default=100, help="number of training epochs",
    )

    parser.add_argument(
        "--checkpoint", type=str, default=None, help="checkpoint to load",
    )

    args = parser.parse_args()
    logger = logging.getLogger()

    batch_size = args.batch_size
    ds_path = args.dataset
    l_max = args.l_max
    epochs = args.epochs
    experiment_name = args.name

    num_color_channels = 3

    sample_weighting = False

    logging.basicConfig(level=logging.INFO)
    logger.info(f"loading dataset {ds_path}")

    best = None
    if path.exists(f"{ds_path}/best.txt"):
        with open(f"{ds_path}/best.txt", "r") as f:
            best = [fn.strip() for fn in f.readlines()]
            best = best[: len(best) // 8]

    ds = OctantDataset(ds_path, selected_samples=best)

    num_train = int(0.8 * len(ds))

    ds_train, ds_val = data.random_split(ds, [num_train, len(ds) - num_train])

    logger.info(f"train: {len(ds_train)}, validation: {len(ds_val)}")

    data_loader_train = DataLoader(
        ds_train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_batched_point_clouds,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=10,
        pin_memory_device="cuda:0",
    )

    data_loader_val = DataLoader(
        ds_val,
        batch_size=batch_size,
        shuffle=True,
        num_workers=12,
        collate_fn=collate_batched_point_clouds,
        pin_memory=False,
        drop_last=False,
        prefetch_factor=10,
        pin_memory_device="cuda:0",
    )

    coefs_samples = torch.stack(
        [
            ds_train[i][1][: lm2flat_index(l_max, l_max) + 1]
            for i in torch.randint(0, len(ds_train), (10000,))
        ]
    )

    coef_transform = CoefNormalizer(
        coefs_samples.mean(0).cuda(), coefs_samples.std(0).cuda()
    )

    model = PointNet(
        l_max,
        num_color_channels,
        batch_norm=True,
        use_dropout=0.4,
        use_spherical=False,
        coef_mean=coef_transform.mean,
        coef_std=coef_transform.std,
    ).cuda()

    if args.checkpoint is not None:
        model.load_state_dict(torch.load(args.checkpoint))

    cameras_pos = camera_positions("sphere.ply")

    loss_fn = camera_loss(l_max, cameras_pos)
    coef_loss_fn = l2_loss()

    writer_train = SummaryWriter(f"logs/{experiment_name}/train")
    writer_val = SummaryWriter(f"logs/{experiment_name}/val")

    with open(f"logs/{experiment_name}/train.txt", "w") as f:
        f.writelines([f"{l}\n" for l in ds.ply_files[ds_train.indices]])
    with open(f"logs/{experiment_name}/validation.txt", "w") as f:
        f.writelines([f"{l}\n" for l in ds.ply_files[ds_val.indices]])

    sample = next(iter(data_loader_train))

    writer_train.add_graph(model, sample.pcs.unpack())
    writer_train.add_figure("coefficients/std", coef_transform.plot(), 0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    scheduler = OneCycleLR(
        optimizer,
        max_lr=1e-3,
        steps_per_epoch=len(data_loader_train),
        epochs=epochs,
        pct_start=0.1,
    )

    coef_loss_weight = 5e-5

    step = 0
    for epoch in range(epochs):

        val_sampler = iter(data_loader_val)

        for i, (batch) in tqdm(
            enumerate(data_loader_train),
            total=len(data_loader_train),
            desc=f"epoch {epoch}/{epochs}",
        ):

            pcs, coefs = batch.pcs, batch.coefs

            save_plot = step % 100 == 0

            model.train()
            optimizer.zero_grad()

            target_coefs = coefs[:, : lm2flat_index(l_max, l_max) + 1].cuda()

            vertices, color, batch =pcs.unpack()
            pred_coefs = model(vertices, color, batch)
            sample_weight = stds(color, batch) if sample_weighting else None
            c_loss = loss_fn(pred_coefs, target_coefs,sample_weight)

            pred_coefs_n = coef_transform.normalize(pred_coefs)
            target_coefs_n = coef_transform.normalize(target_coefs)
            coef_loss = coef_loss_fn(pred_coefs_n, target_coefs_n)

            train_loss = c_loss + coef_loss * coef_loss_weight
            train_loss.backward()

            optimizer.step()
            scheduler.step()

            writer_train.add_scalar("loss/camera", c_loss.item(), step, new_style=True)
            writer_train.add_scalar("loss/coef", coef_loss.item(), step, new_style=True)
            writer_train.add_scalar(
                "loss/total", train_loss.item(), step, new_style=True
            )

            writer_train.add_scalar(
                "learning_rate", optimizer.param_groups[0]["lr"], step, new_style=True,
            )

            if save_plot:
                fig = plot_samples(
                    pred_coefs.detach(), target_coefs.detach(), pcs, l_max,
                )
                writer_train.add_figure("samples", fig, step)
                fig1, fig2, fig3 = plot_coefs(
                    pred_coefs.detach(), target_coefs.detach(),
                )
                writer_train.add_figure("coefficients/distribution", fig1, step)
                writer_train.add_figure("coefficients/error", fig2, step)
                writer_train.add_figure("coefficients/error_relative", fig3, step)

            # do validation batch every 4th train batch
            if i % 8 == 0:
                model.eval()

                val_batch = next(val_sampler)
                pcs, coefs = val_batch.pcs, val_batch.coefs

                target_coefs = coefs[:, : lm2flat_index(l_max, l_max) + 1].cuda()

                vertices, color, batch =pcs.unpack()
                pred_coefs = model(vertices, color, batch)

                sample_weight = stds(color, batch) if sample_weighting else None
                c_loss = loss_fn(pred_coefs, target_coefs,sample_weight)

                target_coefs_n = coef_transform.normalize(target_coefs)
                pred_coefs_n = coef_transform.normalize(pred_coefs)
                coef_loss = coef_loss_fn(pred_coefs_n, target_coefs_n)

                val_loss = c_loss + coef_loss * coef_loss_weight

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
                        pred_coefs.detach(), target_coefs.detach(), pcs, l_max,
                    )
                    writer_val.add_figure("samples", fig, step)
                    fig1, fig2, fig3 = plot_coefs(
                        pred_coefs.detach(), target_coefs.detach(),
                    )
                    writer_val.add_figure("coefficients/distribution", fig1, step)
                    writer_val.add_figure("coefficients/error", fig2, step)
                    writer_val.add_figure("coefficients/error_relative", fig3, step)

            step += 1
        if epoch != 0 and epoch % 50 == 0:
            torch.save(
                model.state_dict(), f"logs/{experiment_name}/model_weights_{epoch}.pt"
            )

    writer_train.close()
    writer_val.close()
    model.eval()

    torch.save(model.state_dict(), f"logs/{experiment_name}/model_weights.pt")

    traced_model = torch.jit.trace_module(
        model.cpu(),
        {
            "forward": [
                torch.rand((1, 3)),
                torch.rand((1, num_color_channels)),
                torch.zeros(1, dtype=torch.long),
            ]
        },
    )

    torch.jit.save(traced_model, f"logs/{experiment_name}/traced_model_cpu.pt")

    traced_model = torch.jit.trace_module(
        model.cuda(),
        {
            "forward": [
                torch.rand((1, 3)).cuda(),
                torch.rand((1, num_color_channels)).cuda(),
                torch.zeros(1, dtype=torch.long).cuda(),
            ]
        },
    )

    torch.jit.save(traced_model, f"logs/{experiment_name}/traced_model_gpu.pt")

