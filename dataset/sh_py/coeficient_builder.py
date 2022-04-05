from .statistics import RunningMean
from .evaluator import SphericalHarmonicsEvaluator

import torch

SPHERE_SURFACE_AREA = 12.566370614359172


class SphericalHarmonicsCoeficientBuilder:
  """
  Least squares fit of spherical harmonics according to
  https://math.stackexchange.com/questions/54880/calculate-nearest-spherical-harmonic-for-a-3d-distribution-over-a-grid
  """

  def __init__(self, max_degree, batch_shape=(), dtype=None, device=None, solve_ls=True):
    self.__dtype = dtype
    self.__device = device
    self.__max_degree = max_degree
    self.__batch_shape = batch_shape
    self.__evaluator = None
    self.__A_statistics = None
    self.__b_statistics = None
    self.__solve_ls = solve_ls

  def _check_dtype_and_device(self, *tensors):
    if self.__dtype is None or self.__device is None:
      self.__dtype = tensors[0].dtype
      self.__device = tensors[0].device
      self.__evaluator = SphericalHarmonicsEvaluator(self.__max_degree, self.__dtype, self.__device)
      n = self.__evaluator.coeficient_count
      if self.__solve_ls:
        self.__A_statistics = RunningMean((n, n), self.__dtype, self.__device)
      self.__b_statistics = RunningMean(self.__batch_shape + (n,), self.__dtype, self.__device)
    for t in tensors:
      if t.dtype != self.__dtype:
        raise ValueError(f"Got tensor of type {t.dtype}, but expected {self.__dtype}.")
      if t.device != self.__device:
        raise ValueError(f"Got tensor on device {t.device}, but expected {self.__device}.")

  def add_samples(self, direction, samples):
    if direction.shape != (3,):
      raise ValueError("Directions need to be 3D vector.")
    if samples.shape != self.__batch_shape:
      raise ValueError(f"Expected samples of shape {self.__batch_shape}, but got {samples.shape}.")
    self._check_dtype_and_device(direction, samples)

    sh = self.__evaluator(direction)
    if self.__solve_ls:
      add_A = SPHERE_SURFACE_AREA * sh[:, None] * sh[None, :]
      self.__A_statistics |= add_A

    values = SPHERE_SURFACE_AREA * sh * samples[..., None]
    self.__b_statistics |= values

  @property
  def coeficient_count(self):
    return (self.__max_degree + 1) ** 2

  @property
  def sample_count(self):
    if self.__b_statistics is None:
      return 0
    else:
      return self.__b_statistics.count

  def compute_coeficients(self):
    """
    Solves the least squares system and returns the spherical harmonics coeficients. If err is not None,
    the probability for each coeficient that its current value is within +/-err of the true value is also returned.
    """
    if self.__b_statistics is None:
      raise ValueError("No samples added yet.")

    b = self.__b_statistics.mean
    if self.__solve_ls:
      if self.sample_count < self.__evaluator.coeficient_count:
        raise ValueError(f"Not enough samples drawn so that the linear system is guaranteed to be underdetermined."
                         f"At least {self.__evaluator.coeficient_count} are required, but got {self.sample_count}.")
      A = self.__A_statistics.mean
      invA = torch.linalg.pinv(A)
      return torch.matmul(invA, b[..., None])[..., 0]
    else:
      return b
