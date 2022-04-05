import torch
from math import sqrt
from .legendre import ScaledLegendrePolynomialFactory

class SphericalHarmonicsEvaluator:

  @staticmethod
  def lm2flat_index(l, m):
    return l * (l + 1) - m

  @staticmethod
  def flat2lm_index(i):
    l = int(sqrt(i))
    m = l * (l + 1) - i
    return l, m

  def __init__(self, max_degree, dtype=None, device=None):
    self.__dtype = dtype
    self.__device = device
    self.__max_degree = max_degree
    self.__factory = ScaledLegendrePolynomialFactory(dtype, device)
    self.__factory.populate_cache(max_degree)
    self.__one = self._as_tensor(1)
    self.__sqrt2 = self._as_tensor(1.4142135623730951)

  @property
  def coeficient_count(self):
      return (self.__max_degree + 1) ** 2

  def _as_tensor(self, x):
    return torch.as_tensor(x, dtype=self.__dtype, device=self.__device)

  def _evalute_sin_cos_theta(self, x, y):
    sin = [self.__one, y]
    cos = [self.__one, x]

    for m in range(2, self.__max_degree + 1):
      sinm = sin[m - 1] * cos[1] + cos[m - 1] * sin[1]
      cosm = cos[m - 1] * cos[1] - sin[m - 1] * sin[1]
      sin.append(sinm)
      cos.append(cosm)
    return sin, cos

  def __call__(self, x, y=None, z=None):
    results = []
    if y is None and z is None:
      x = self._as_tensor(x)
      y = x[..., 1]
      z = x[..., 2]
      x = x[..., 0]
    else:
      x = self._as_tensor(x)
      y = self._as_tensor(y)
      z = self._as_tensor(z)
    sin, cos = self._evalute_sin_cos_theta(x, y)

    legendre_results = {}
    for l in range(self.__max_degree + 1):
      sign = self.__one.clone()
      for m in range(l + 1):
        v = sign * self.__factory.get(l, m)(z)
        legendre_results[l, m] = v
        sign *= -self.__one

    for l in range(self.__max_degree + 1):
      for m in range(-l, l + 1):
        v = legendre_results[l, abs(m)]
        if m > 0:
          v = v * self.__sqrt2 * cos[m]
        if m < 0:
          v = v * self.__sqrt2 * sin[abs(m)]
        results.append(v)
    return torch.stack(results, dim=-1)