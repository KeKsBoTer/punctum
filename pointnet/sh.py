# adapted from https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/spherical_harmonics.py

from functools import lru_cache, reduce, wraps
from math import pi, sqrt
import math
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
    assert (
        coords.norm(p=2, dim=1) - 1 < 1e-6
    ).all(), f"must be of length 1 but got ({coords.norm(p=2, dim=1) -1})"

    spherical = torch.empty((coords.shape[0], 2), device=coords.device)

    spherical[:, 0] = coords[:, 2].acos()
    spherical[:, 1] = torch.atan2(coords[:, 1], coords[:, 0]) + torch.pi
    return spherical


def to_cartesian(coords: torch.Tensor) -> torch.Tensor:
    """Spherical to cartesian coordinate conversion.
    Args:
        cords: [N,2] spherical coordinates
    """

    theta_sin = coords[:,0].sin()
    x=coords[:,1].cos() * theta_sin
    y=coords[:,1].sin() * theta_sin
    z=coords[:,0].cos()

    return torch.stack([x,y,z],dim=-1)


def lm2flat_index(l: int, m: int) -> int:
    return l * (l + 1) + m


def flat2lm_index(i: int) -> Tuple[int, int]:
    l = int(sqrt(i))
    m = l * (l + 1) - i
    return l, -m


def evalute_sh(coefs: torch.Tensor, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """evaluates spherical harmonics for given coefficients and position 

    Args:
        coefs (torch.Tensor[C]): sh coefficients
        x (torch.Tensor[N]): spherical positions theta
        y (torch.Tensor[N]): spherical positions phi

    Returns:
        torch.Tensor[N]: resulting function values
    """
    device = coefs.device
    l_max = int(math.sqrt(coefs.shape[0])) - 1
    Y = torch.zeros((*x.shape, coefs.shape[-1]), device=device)
    clear_spherical_harmonics_cache()
    for l in range(l_max + 1):
        y_lm = get_spherical_harmonics(l, x, y)
        Y += y_lm @ coefs[lm2flat_index(l, -l) : lm2flat_index(l, l) + 1]

    return Y


def calc_sh(l_max: int, coords: torch.Tensor) -> torch.Tensor:
    """calculates the coefficients up to l_max for the given spherical coordinates

    Args:
        l_max (int): maximum degree for l
        coords (torch.Tensor[N,2]): spherical coordinates

    Returns:
        torch.Tensor[N,(l_max+1)**2]: sh values
    """
    values = torch.zeros((coords.shape[0], (l_max + 1) ** 2), device=coords.device)
    clear_spherical_harmonics_cache()
    for l in range(l_max + 1):
        sh = get_spherical_harmonics(l, coords[:, 0], coords[:, 1])
        values[:, lm2flat_index(l, -l) : lm2flat_index(l, l) + 1] = sh
    return values


def calc_coeficients(
    l_max: int, coords: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    """ Spherical Harmonics coefficients calculation.
    Computes the coefficients by formulating them as a least squares problem.
    See https://math.stackexchange.com/questions/54880/calculate-nearest-spherical-harmonic-for-a-3d-distribution-over-a-grid
    Args:
        l_max (int): maximum degree to compute
        coords ([N,2]): spherical coordinates 
        target ([N,D]): values for coords
    Returns:
        [(l_max+1)**2,D] coefficients
    Throws:
        Assertion if (l_max+1)**2 >= N

    """
    assert (l_max + 1) ** 2 < coords.shape[0], "to few samples"
    sh = calc_sh(l_max, coords)
    A = sh
    B = target
    return torch.linalg.lstsq(A, B).solution

