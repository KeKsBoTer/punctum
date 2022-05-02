# adapted from https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/spherical_harmonics.py

from functools import lru_cache, reduce, wraps
from math import pi, sqrt
from operator import mul
from typing import Dict, Callable, Tuple

import torch


def cache(cache_store: Dict, key_fn: Callable):
    """Used to cache Lagrange polynomials

    Args:
        cache_store (_type_): cache dict
        key_fn (_type_): _description_
    """

    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache_store:
                return cache_store[key_name]
            res = fn(*args, **kwargs)
            cache_store[key_name] = res
            return res

        return inner

    return cache_inner


# constants

CACHE = {}


def clear_spherical_harmonics_cache():
    """clears global cache """
    CACHE.clear()


def lpmv_cache_key_fn(l: int, m: int, _x: torch.Tensor) -> Tuple[int, int]:
    """cache key for lagrange polynomials"""
    return (l, m)


# spherical harmonics


@lru_cache(maxsize=1000)
def semifactorial(x: int) -> int:
    return reduce(mul, range(x, 1, -2), 1.0)


@lru_cache(maxsize=1000)
def pochhammer(x: int, k: int) -> int:
    return reduce(mul, range(x + 1, x + k), float(x))


def negative_lpmv(l: int, m: int, y: torch.Tensor) -> torch.Tensor:
    if m < 0:
        y *= (-1) ** m / pochhammer(l + m + 1, -2 * m)
    return y


@cache(cache_store=CACHE, key_fn=lpmv_cache_key_fn)
def lpmv(l: int, m: int, x: torch.Tensor) -> torch.Tensor:
    """Associated Legendre function including Condon-Shortley phase.
    
    Args:
        l (int): order
        m (int): degree
        x (torch.Tensor): float argument tensor

    Returns:
        torch.Tensor: sh values
    """
    # Check memoized versions
    m_abs = abs(m)

    if m_abs > l:
        return None

    if l == 0:
        return torch.ones_like(x)

    # Check if on boundary else recurse solution down to boundary
    if m_abs == l:
        # Compute P_m^m
        y = (-1) ** m_abs * semifactorial(2 * m_abs - 1)
        y *= torch.pow(1 - x * x, m_abs / 2)
        return negative_lpmv(l, m, y)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2 * l - 1) / (l - m_abs)) * x * lpmv(l - 1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l + m_abs - 1) / (l - m_abs)) * CACHE[(l - 2, m_abs)]

    if m < 0:
        y = negative_lpmv(l, m, y)
    return y


def get_spherical_harmonics_element(
    l: int, m: int, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Tesseral spherical harmonic with Condon-Shortley phase.
    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.
    Args:
        l: int for degree
        m: int for order, where -l <= m < l
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape theta
    """
    m_abs = abs(m)
    assert m_abs <= l, "absolute value of order m must be <= degree l"

    N = sqrt((2 * l + 1) / (4 * pi))
    leg = lpmv(l, m_abs, torch.cos(theta))

    if m == 0:
        return N * leg

    if m > 0:
        Y = torch.cos(m * phi)
    else:
        Y = torch.sin(m_abs * phi)

    Y *= leg
    N *= sqrt(2.0 / pochhammer(l - m_abs + 1, 2 * m_abs))
    Y *= N
    return Y


def get_spherical_harmonics(
    l: int, theta: torch.Tensor, phi: torch.Tensor
) -> torch.Tensor:
    """Tesseral harmonic with Condon-Shortley phase.
    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.
    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, 2*l+1]
    """
    return torch.stack(
        [get_spherical_harmonics_element(l, m, theta, phi) for m in range(-l, l + 1)],
        dim=-1,
    )


def to_spherical(coords: torch.Tensor) -> torch.Tensor:
    """Cartesian to spherical coordinate conversion.
    Args:
        cords: [N,3] cartesian coordinates
    """
    assert (coords.norm(p=2, dim=1) - 1 < 1e-8).all(), "must be of length 1"

    spherical = torch.empty((coords.shape[0], 2), device=coords.device)

    spherical[:, 0] = coords[:, 2].acos()
    spherical[:, 1] = torch.atan2(coords[:, 1], coords[:, 0]) + torch.pi
    return spherical


def lm2flat_index(l: int, m: int) -> int:
    return l * (l + 1) - m


def evalute_sh(coefs: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    device = coefs.device
    l_max = coefs.shape[0] - 1
    Y = torch.zeros((*x.shape, 3), device=device)
    for l in range(l_max + 1):
        y_lm = get_spherical_harmonics(l, x, y)
        Y += y_lm @ coefs[lm2flat_index(l, l) : lm2flat_index(l, -l) + 1]

    return Y


def calc_sh(l_max: int, coords: torch.Tensor) -> torch.Tensor:
    assert (l_max + 1) ** 2 < coords.shape[0], "to few samples"
    values = torch.zeros((coords.shape[0], (l_max + 1) ** 2), device=coords.device)

    for l in range(l_max + 1):
        sh = get_spherical_harmonics(l, coords[:, 0], coords[:, 1])
        values[:, lm2flat_index(l, l) : lm2flat_index(l, -l) + 1] = sh
    return values


def calc_coeficients(
    l_max: int, coords: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """ Spherical Harmonics ceofficients calculation.
    Computes the ceofficients by formulating them as a least squares problem.
    See https://math.stackexchange.com/questions/54880/calculate-nearest-spherical-harmonic-for-a-3d-distribution-over-a-grid
    Args:
        l_max (int): maximum degree to compute
        coords ([N,2]): sperical coordinates 
        target ([N,D]): values for coords
    Returns:
        [(l_max+1)**2,D] ceofficients
    Throws:
        Assertion if (l_max+1)**2 >= N

    """
    sh = calc_sh(l_max, coords)
    A = sh.T @ sh
    B = sh.T @ target
    return torch.linalg.lstsq(A, B).solution

