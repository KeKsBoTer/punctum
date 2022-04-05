import torch 
from matplotlib import pyplot as plt
from math import pi as PI
from plyfile import PlyData, PlyElement
import numpy as np

from sh import to_spherical,get_spherical_harmonics


def calc_point_weights(coords_sperical,n=100):
    device = coords_sperical.device
    x = torch.arange(0,PI,PI/n,device=device)
    y = torch.arange(0,2*PI,2*PI/n,device=device)
    spherical_grid = torch.dstack(torch.meshgrid(x,y, indexing='ij')).flatten(0,1)

    distances = torch.cdist(coords_sperical,spherical_grid,p=2)

    nearest = distances.argmin(dim=0).reshape((n,n))

    _,counts = nearest.unique(sorted=True,return_counts=True)

    return counts/counts.sum()


def calc_coeficients(l_max,coords,target):
    device = coords.device

    coefs = torch.zeros((l_max+1,2*l_max+1,3),device=device)

    for l in range(l_max+1):
        y_lm = get_spherical_harmonics(l,coords[:,0],coords[:,1])
        a_lm = y_lm.T@target
        coefs[l,l_max-l:l_max+l+1]=a_lm

    return coefs



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
        print(perceived_colors)
        print(f"reading in took \t{t():.4f} secs")

    with measure_time() as t:
        cameras_spherical = to_spherical(cameras)
        # sample_area = 2*PI**2* calc_point_weights(cameras_spherical).unsqueeze(-1)
        coefs = calc_coeficients(10,cameras_spherical,perceived_colors)
        print(f"coef. calc. took \t{t():.4f} secs")

    with measure_time() as t:
        coef_data = []
        for l,c in enumerate(coefs.cpu()):
            for m,sh in enumerate(c[l_max-l:l_max+l+1]):
                coef_data.append((l,-l+m,list(sh.numpy())))
                
        ply_sh_data =np.array(coef_data,dtype=[("l","u1"),("m","i1"),('coefficients', 'f4',(3,))])

        sh_elm = PlyElement.describe(ply_sh_data,"sh_coefficients")

        PlyData([sh_elm, plydata["vertex"]], text=True).write(args.out_file)
        print(f"exporting file took \t{t():.4f} secs")
