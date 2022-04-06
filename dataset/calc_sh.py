import torch 
from plyfile import PlyData, PlyElement
import numpy as np

from sh import to_spherical,calc_coeficients,lm2flat_index



if __name__ == "__main__":
    import argparse

    from time import perf_counter
    from contextlib import contextmanager

    @contextmanager
    def measure_time() -> float:
        start = perf_counter()
        yield lambda: perf_counter() - start

    parser = argparse.ArgumentParser(description='Calculate Spherical Harmonics for perspective colors')
    parser.add_argument('in_file', type=str,
                        help='ply file with camera positions and their perceived colors')
    parser.add_argument('out_file', type=str,
                        help='ply file where results will be written to')
    parser.add_argument('--l_max', type=int, default=10,
                        help='maximum order of spherical harmonics coefficients to calculate')
    parser.add_argument('--cuda', default=False, action='store_true',
                        help='use CUDA (might be slower)')


    args = parser.parse_args()

    device = "cuda" if args.cuda else "cpu"

    with measure_time() as t:
        torch.zeros(1,device=device)
        print(f"pytorch startup took \t{t():.4f} secs")

    with measure_time() as t:
        plydata = PlyData.read(args.in_file)
        vertex_data = plydata["vertex"].data

        def unpack_data(data,field_names):
            return torch.from_numpy(np.stack([data[key]for key in field_names]).T)
            
        l_max = args.l_max

        cameras = unpack_data(vertex_data,["x","y","z"]).to(device)
        perceived_colors = unpack_data(vertex_data,["red","green","blue"]).float().to(device)/255.
        print(f"reading in took \t{t():.4f} secs")

    with measure_time() as t:
        cameras_spherical = to_spherical(cameras)
        coefs = calc_coeficients(l_max,cameras_spherical,perceived_colors)
        print(f"coef. calc. took \t{t():.4f} secs")

    with measure_time() as t:
        coef_data = []
        for l in range(l_max+1):
            for m,sh in enumerate(coefs.cpu()[lm2flat_index(l,l):lm2flat_index(l,-l)+1]):
                coef_data.append((l,-l+m,list(sh.numpy())))
                
        ply_sh_data =np.array(coef_data,dtype=[("l","u1"),("m","i1"),('coefficients', 'f4',(3,))])

        sh_elm = PlyElement.describe(ply_sh_data,"sh_coefficients")

        PlyData([sh_elm, plydata["vertex"]], text=True).write(args.out_file)
        print(f"exporting file took \t{t():.4f} secs")