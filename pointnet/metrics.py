import torch
from pointnet.sh import calc_sh

def camera_color(l_max: int, positions: torch.Tensor, random_offset: bool = True):

    y = calc_sh(l_max, positions.flatten(0, 1))
    y = y.reshape((positions.shape[0], (l_max + 1) ** 2))

    def metric(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        y_r = y.to(prediction.device)
        pred_img = y_r @ prediction
        target_img = y_r @ target
        diff = pred_img - target_img
        l2 = diff.norm(2, -1).mean(-1)
        return l2.mean()

    return metric
