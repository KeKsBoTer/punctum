# COPIED from https://github.com/lucidrains/se3-transformer-pytorch/blob/main/se3_transformer_pytorch/spherical_harmonics.py

from math import pi, sqrt
from functools import reduce
from operator import mul
import torch

from functools import lru_cache, wraps


def cache(cache, key_fn):
    def cache_inner(fn):
        @wraps(fn)
        def inner(*args, **kwargs):
            key_name = key_fn(*args, **kwargs)
            if key_name in cache:
                return cache[key_name]
            res = fn(*args, **kwargs)
            cache[key_name] = res
            return res

        return inner

    return cache_inner


# constants

CACHE = {}


def clear_spherical_harmonics_cache():
    CACHE.clear()


def lpmv_cache_key_fn(l, m, x):
    return (l, m,x)


# spherical harmonics


@lru_cache(maxsize=1000)
def semifactorial(x):
    return reduce(mul, range(x, 1, -2), 1.0)


@lru_cache(maxsize=1000)
def pochhammer(x, k):
    return reduce(mul, range(x + 1, x + k), float(x))


def negative_lpmv(l, m, y):
    if m < 0:
        y *= (-1) ** m / pochhammer(l + m + 1, -2 * m)
    return y


@cache(cache=CACHE, key_fn=lpmv_cache_key_fn)
def lpmv(l, m, x):
    """Associated Legendre function including Condon-Shortley phase.
    Args:
        m: int order
        l: int degree
        x: float argument tensor
    Returns:
        tensor of x-shape
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

    # Recursively precompute lower degree harmonics
    lpmv(l - 1, m, x)

    # Compute P_{l}^m from recursion in P_{l-1}^m and P_{l-2}^m
    # Inplace speedup
    y = ((2 * l - 1) / (l - m_abs)) * x * lpmv(l - 1, m_abs, x)

    if l - m_abs > 1:
        y -= ((l + m_abs - 1) / (l - m_abs)) * CACHE[(l - 2, m_abs,x)]

    if m < 0:
        y = negative_lpmv(l, m, y)
    return y


def get_spherical_harmonics_element(l, m, theta, phi):
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


def get_spherical_harmonics_2(l, theta, phi):
    """Tesseral harmonic with Condon-Shortley phase.
    The Tesseral spherical harmonics are also known as the real spherical
    harmonics.
    Args:
        l: int for degree
        theta: collatitude or polar angle
        phi: longitude or azimuth
    Returns:
        tensor of shape [*theta.shape, l+1]
    """
    return torch.stack(
        [get_spherical_harmonics_element(l, m, theta, phi) for m in range(0, l + 1)],
        dim=-1,
    )


def get_spherical_harmonics(l, theta, phi):
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


def to_spherical(coords):
    """Cartesian to spherical coordinate conversion.
    Args:
        cords: [N,3] cartesian coordinates
    """
    assert (coords.norm(p=2, dim=1) - 1 < 1e-8).all(), "must be of length 1"

    theta = coords[:, 2].acos()
    phi = torch.atan2(coords[:, 1], coords[:, 0]) + pi
    return torch.stack([theta, phi]).T



def lm2flat_index(l, m):
    return l * (l + 1) - m

def evalute_sh(coefs, x, y):
    device = coefs.device
    l_max = coefs.shape[0] - 1
    Y = torch.zeros((*x.shape, 3), device=device)
    for l in range(l_max + 1):
        y_lm = get_spherical_harmonics(l, x, y)
        Y += y_lm @ coefs[lm2flat_index(l,l):lm2flat_index(l,-l)+1]

    return Y


def calc_sh(l_max,coords):
    assert (l_max+1)**2 < coords.shape[0], "to few samples"
    values = torch.zeros((coords.shape[0],(l_max+1)**2),device=coords.device)

    for l in range(l_max+1):
        sh = get_spherical_harmonics(l,coords[:,0],coords[:,1])
        values[:,lm2flat_index(l,l):lm2flat_index(l,-l)+1] = sh
    return values
    
def calc_coeficients(l_max,coords,target):
    sh = calc_sh(l_max,coords)
    A = sh.T@sh 
    B = (sh.T@target)
    return torch.linalg.lstsq(A,B).solution