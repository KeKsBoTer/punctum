import torch
from plyfile import PlyData, PlyElement
import numpy as np
import glob
from os.path import join, basename
import os
from pointnet.sh import to_spherical, calc_coeficients, lm2flat_index
from tqdm import tqdm
import torch.multiprocessing as mp
import time

def export_sh(args):
    filename, out_folder, l_max,export_ply = args
    plydata = PlyData.read(filename)
    vertex_data = plydata["camera"].data

    def unpack_data(data, field_names):
        return torch.from_numpy(np.stack([data[key] for key in field_names]).T)

    alpha = len(vertex_data.dtype)==3+4

    cameras = unpack_data(vertex_data, ["x", "y", "z"]).to("cuda")
    color_channels =   ["red", "green", "blue", "alpha"] if alpha else  ["red", "green", "blue"]
    perceived_colors = (
        unpack_data(vertex_data,color_channels).float().to("cuda")
        / 255.0
    )

    # avg_color = perceived_colors.mean(dim=0)
    cameras_spherical = to_spherical(cameras)
    coefs = calc_coeficients(l_max, cameras_spherical, perceived_colors)  # - avg_color)

    if export_ply:
        coef_data = []
        for l in range(l_max + 1):
            for m, sh in enumerate(
                coefs.cpu()[lm2flat_index(l, -l) : lm2flat_index(l, l) + 1]
            ):
                coef_data.append((l, -l + m, list(sh.numpy())))

        ply_sh_data = np.array(
            coef_data, dtype=[("l", "u1"), ("m", "i1"), ("coefficients", "f4", (len(color_channels),))]
        )

        sh_elm = PlyElement.describe(ply_sh_data, "sh_coefficients")

        PlyData([sh_elm, plydata["camera"], plydata["vertex"]]).write(
            join(out_folder, basename(filename))
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Calculate Spherical Harmonics for perspective colors"
    )
    parser.add_argument(
        "in_folder",
        type=str,
        help="ply file with camera positions and their perceived colors",
    )
    parser.add_argument(
        "out_folder", type=str, help="ply file where results will be written to"
    )
    parser.add_argument(
        "--l_max",
        type=int,
        default=10,
        help="maximum order of spherical harmonics coefficients to calculate",
    )

    parser.add_argument(
        "--no-export",
        action="store_true",
        help="do not export ply files",
    )

    args = parser.parse_args()

    os.makedirs(args.out_folder, exist_ok=True)

    mp.set_start_method("spawn")

    l_max = args.l_max
    out_folder = args.out_folder

    startTime = time.time()
    with mp.Pool(8) as p:
        results = []
        args = [
            (f, out_folder, l_max,not args.no_export) for f in glob.glob(join(args.in_folder, "*.ply"))
        ]
        for result in tqdm(p.imap(export_sh, args), total=len(args)):
            results.append(result)

    executionTime = (time.time() - startTime)
    print(f"\nExecution time in seconds: {executionTime}s\n")
