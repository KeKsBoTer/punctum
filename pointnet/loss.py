from .sh import calc_sh
import torch
from matplotlib import pyplot as plt

def l2_loss():
    def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        diff = prediction - target
        l2 = diff.square().sum(-1)
        return l2.mean()

    return loss

def camera_loss(l_max: int, positions: torch.Tensor, random_offset: bool = True):

    num_rand = 64

    rnd_pos = positions
    if random_offset:
        offsets = torch.randn((num_rand, *positions.shape)) * 0.1
        offsets[:, :, 1] *= 2
        rnd_pos = positions + offsets * 0.5


    y = calc_sh(l_max, rnd_pos.flatten(0, 1))
    y = y.reshape((num_rand, positions.shape[0], (l_max + 1) ** 2))

    def loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        r_i = torch.randint(0, len(y), (1,)).item()
        y_r = y.to(prediction.device)[r_i]
        pred_img = y_r @ prediction
        target_img = y_r @ target
        diff = pred_img - target_img
        l2 = diff.norm(2, -1).mean(-1)
        return l2.mean()

    return loss


class CoefNormalizer:
    def __init__(self, mean: torch.Tensor, std: torch.Tensor):
        self.mean = mean
        self.std = std
        self.std[self.std.abs() < 1e-6] = 0

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.std + self.mean

    def plot(self) -> plt.Figure:
        fig, ax = plt.subplots()
        x = torch.arange(len(self.mean))
        ax.plot(x, self.std[:, 0].cpu(), label="red")
        ax.plot(x, self.std[:, 1].cpu(), label="green")
        ax.plot(x, self.std[:, 2].cpu(), label="blue")
        ax.plot(x, self.std[:, 3].cpu(), label="alpha")
        ax.set_xlabel("coefs.")  #
        ax.legend()
        return fig
