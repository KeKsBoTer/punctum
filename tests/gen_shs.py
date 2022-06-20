import sys

sys.path.append("../pointnet")

from PIL import Image
import torch

from sh import calc_sh, flat2lm_index

res = 256

x = torch.arange(0, 1, 1 / res) * torch.pi
y = torch.arange(0, 1, 1 / res) * torch.pi * 2
coords = torch.stack(torch.meshgrid(x, y, indexing="ij"), dim=-1).reshape(-1, 2)

l = 10
shs = calc_sh(l, coords).reshape((res, res, (l + 1) ** 2)).permute(2, 0, 1)
for i, sh in enumerate(shs):
    l, m = flat2lm_index(i)
    sh = (sh.clip(-1, 1) + 1) / 2
    img = sh.unsqueeze(-1).repeat(1, 1, 4)
    img[:, :, 3] = 1
    img_byte = (img * 255).byte()
    Image.fromarray(img_byte.numpy()).save(f"shs/l_{l}_m_{m}.png")

