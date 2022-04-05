import torch
import math
import numpy as np
from typing import Union


# Implement wrapper around torch.sum that catches the special case of supplying no dimensions to reduce along.
# In torch 1.10.1, torch.sum performs a full reduce for dims=(), but we actually want no reduce at all.
def _sum(x, dims):
  if len(dims) == 0:
    return x
  else:
    return torch.sum(x, dim=dims)


class RunningMean:

  @staticmethod
  def wrap(n, m):
    s = RunningMean()
    s.__n = n
    s.__m = m
    return s

  @staticmethod
  def from_samples(samples, reduce_dims: Union[bool, tuple, int] = True):
    if reduce_dims is True:
      statistics = RunningMean((), samples.dtype, samples.device)
      statistics.update(samples)
    elif reduce_dims is False:
      statistics = RunningMean.wrap(1, samples.clone())
    else:
      statistics = RunningMean.wrap(1, samples.clone())
      statistics.reduce_dims(reduce_dims)
    return statistics

  def __init__(self, shape=(), dtype=None, device=None):
    self.__n = 0
    self.__m = torch.zeros(shape, dtype=dtype, device=device)

  def copy(self):
    return RunningMean.wrap(self.__n, self.__m.clone())

  def __getitem__(self, item):
    return RunningMean.wrap(self.__n, self.__m[item].clone())

  @property
  def dtype(self):
    return self.__m.dtype

  @property
  def device(self):
    return self.__m.device

  @property
  def shape(self):
    return self.__m.shape

  @property
  def ndim(self):
    return self.__m.ndim

  @property
  def count(self):
    return self.__n

  @property
  def mean(self):
    return self.__m

  def reduce_dims(self, *dims):
    if len(dims) == 0:
      return
    if len(dims) == 1 and isinstance(dims[0], tuple):
      dims = dims[0]

    self.__n *= math.prod(self.shape[i] for i in dims)
    self.__m = torch.mean(self.__m, dim=dims)

  def _update_from_statistics(self, other, weight_per_elem):
    if self.shape != other.shape:
      raise ValueError(f"Inconsistent shape of statistics. Got {self.shape} and {other.shape}.")
    n2 = weight_per_elem * other.__n
    self.__n += n2
    nd = n2 * (other.__m - self.__m)
    self.__m += nd / self.__n

  def _update_from_tensor(self, samples, weight_per_elem):
    if samples.ndim < self.__m.ndim or samples.shape[samples.ndim - self.__m.ndim:] != self.__m.shape:
      shape_str = "".join(map(lambda x: f",{x}", self.__m.shape))
      raise ValueError(f"Samples tensor has to be of shape [...{shape_str}]")

    reduce_dims = tuple(range(samples.ndim - self.__m.ndim))
    reduce_size = math.prod(samples.shape[:samples.ndim - self.__m.ndim])

    self.__n += weight_per_elem * reduce_size
    d = samples - self.__m
    self.__m += _sum(d, reduce_dims) * weight_per_elem / self.__n

  def update(self, samples, weight_per_elem: Union[int, float] = 1):
    if weight_per_elem < 0:
      raise ValueError("Negative weights are not supported")

    if isinstance(samples, RunningStatistics):
      self._update_from_statistics(samples, weight_per_elem)
    else:
      samples = torch.as_tensor(samples, dtype=self.__m.dtype, device=self.__m.device)
      self._update_from_tensor(samples, weight_per_elem)

  def scale(self, value):
    self.__m *= value

  def __rmul__(self, other):
    return RunningMean.wrap(self.__n, other * self.__m)

  def __imul__(self, other):
    self.scale(other)
    return self

  def __ior__(self, other):
    self.update(other)
    return self

  def __or__(self, other):
    copy = self.copy()
    copy.update(other)
    return copy

  def __str__(self):
    shape = self.__m.shape
    m = self.mean.cpu().numpy()

    str_array = np.ndarray(shape, dtype=object)
    for i in np.ndindex(*shape):
      str_array[i] = f"{m[i]}"
    return str(str_array) + f" ({self.__n} elems)"



class RunningStatistics:
  """
  Implements a batched Welford's algorithm as discussed here:
  https://stackoverflow.com/questions/56402955/whats-the-formula-for-welfords-algorithm-for-variance-std-with-batch-updates
  """

  @staticmethod
  def wrap(n, m, m2):
    s = RunningStatistics()
    s.__n = n
    s.__m = m
    s.__m2 = m2
    return s

  @staticmethod
  def from_samples(samples, reduce_dims: Union[bool, tuple, int] = True):
    if reduce_dims is True:
      statistics = RunningStatistics((), samples.dtype, samples.device)
      statistics.update(samples)
    elif reduce_dims is False:
      statistics = RunningStatistics.wrap(1, samples.clone(), torch.zeros_like(samples))
    else:
      statistics = RunningStatistics.wrap(1, samples.clone(), torch.zeros_like(samples))
      statistics.reduce_dims(reduce_dims)
    return statistics

  def __init__(self, shape=(), dtype=None, device=None):
    self.__n = 0
    self.__m = torch.zeros(shape, dtype=dtype, device=device)
    self.__m2 = torch.zeros(shape, dtype=dtype, device=device)

  def copy(self):
    return RunningStatistics.wrap(self.__n, self.__m.clone(), self.__m2.clone())

  def __getitem__(self, item):
    return RunningStatistics.wrap(self.__n, self.__m[item].clone(), self.__m2[item].clone())

  @property
  def dtype(self):
    return self.__m.dtype

  @property
  def device(self):
    return self.__m.device

  @property
  def shape(self):
    return self.__m.shape

  @property
  def ndim(self):
    return self.__m.ndim

  @property
  def count(self):
    return self.__n

  @property
  def mean(self):
    return self.__m

  @property
  def var(self):
    return self.__m2 / self.__n

  @property
  def unbiased_var(self):
    return self.__m2 / (self.__n - 1)

  @property
  def std(self):
    return torch.sqrt(self.__m2 / self.__n)

  @property
  def unbiased_std(self):
    return torch.sqrt(self.__m2 / (self.__n - 1))

  def compute_confidence(self, err):
    """
    Returns the probability that the sample mean is within (E[X] - err, E[X] + err).
    """
    # We have no confidence at all when sample size is too small.
    if self.__n < 50:
      return torch.zeros_like(self.__m)
    z = err * math.sqrt(self.__n) / (self.unbiased_std + 1e-8)
    return torch.special.erf(0.7071067811865475 * z)

  def estimate_required_samples(self, err, confidence):
    std = self.unbiased_std + 1e-08
    confidence = torch.as_tensor(confidence, dtype=std.dtype, device=std.device)
    z = torch.special.erfinv(confidence) * 1.4142135623730951
    sqrtn = (z / err) * std
    return torch.clamp(sqrtn * sqrtn, min=50.0).to(torch.int64)

  def reduce_dims(self, *dims):
    if len(dims) == 0:
      return
    if len(dims) == 1 and isinstance(dims[0], tuple):
      dims = dims[0]

    reduce_slice = [slice(None, None)] * len(self.shape)
    for i in dims:
      reduce_slice[i] = 0
    reduce_slice = tuple(reduce_slice)

    n = self.__n
    self.__n *= math.prod(self.shape[i] for i in dims)
    m_prev = self.__m
    m_new = torch.mean(self.__m, dim=dims, keepdim=True)
    d = m_prev - m_new
    self.__m = m_new[reduce_slice]
    self.__m2 = torch.sum(self.__m2, dim=dims) + n * torch.sum(d * d, dim=dims)

  def _update_from_statistics(self, other, weight_per_elem):
    if self.shape != other.shape:
      raise ValueError(f"Inconsistent shape of statistics. Got {self.shape} and {other.shape}.")
    n2 = weight_per_elem * other.__n
    self.__n += n2
    nd = n2 * (other.__m - self.__m)
    self.__m += nd / self.__n
    d2 = other.__m - self.__m
    self.__m2 += weight_per_elem * other.__m2 + nd * d2

  def _update_from_tensor(self, samples, weight_per_elem):
    if samples.ndim < self.__m.ndim or samples.shape[samples.ndim - self.__m.ndim:] != self.__m.shape:
      shape_str = "".join(map(lambda x: f",{x}", self.__m.shape))
      raise ValueError(f"Samples tensor has to be of shape [...{shape_str}]")

    reduce_dims = tuple(range(samples.ndim - self.__m.ndim))
    reduce_size = math.prod(samples.shape[:samples.ndim - self.__m.ndim])

    self.__n += weight_per_elem * reduce_size
    d = samples - self.__m
    self.__m += _sum(d, reduce_dims) * weight_per_elem / self.__n
    d2 = samples - self.__m
    self.__m2 += weight_per_elem * _sum(d * d2, reduce_dims)

  def update(self, samples, weight_per_elem: Union[int, float] = 1):
    if weight_per_elem < 0:
      raise ValueError("Negative weights are not supported")

    if isinstance(samples, RunningStatistics):
      self._update_from_statistics(samples, weight_per_elem)
    else:
      samples = torch.as_tensor(samples, dtype=self.__m.dtype, device=self.__m.device)
      self._update_from_tensor(samples, weight_per_elem)

  def scale(self, value):
    self.__m *= value
    self.__m2 *= value * value

  def __rmul__(self, other):
    return RunningStatistics.wrap(self.__n, other * self.__m, other * other * self.__m2)

  def __imul__(self, other):
    self.scale(other)
    return self

  def __ior__(self, other):
    self.update(other)
    return self

  def __or__(self, other):
    copy = self.copy()
    copy.update(other)
    return copy

  def __str__(self):
    shape = self.__m.shape
    m = self.mean.cpu().numpy()
    std = self.std.cpu().numpy()

    str_array = np.ndarray(shape, dtype=object)
    for i in np.ndindex(*shape):
      str_array[i] = f"{m[i]} +/- {std[i]}"
    return str(str_array) + f" ({self.__n} elems)"


def main():
  def assert_close(l, r):
    all_close = torch.allclose(l.mean, r.mean) and torch.allclose(l.var, r.var)
    if not all_close:
      assert False, f"dmean = {l.mean - r.mean}, dvar = {l.var - r.var}"

  vall = torch.randn(5, 7, 11)
  v1 = vall[3:]
  v2 = vall[:3, 4:]
  v3 = vall[:3, :4]
  s_expected = RunningStatistics(11)
  s_expected |= vall

  # Test updating from other statistics
  s1 = RunningStatistics(11)
  s2 = RunningStatistics(11)
  s3 = RunningStatistics(11)
  s1.update(v1)
  s2.update(v2)
  s3.update(v3)
  s_actual = RunningStatistics(11)
  s_actual.update(s1)
  s_actual.update(s2)
  s_actual.update(s3)
  assert_close(s_expected, s_actual)

  # Test updating from tensors
  s_actual = RunningStatistics(11)
  s_actual.update(v1)
  s_actual.update(v2)
  s_actual.update(v3)
  assert_close(s_expected, s_actual)

  # Test weighting
  wt = 1.0 + float(np.random.rand())
  w1 = float(np.random.rand())
  w2 = wt - w1
  s_expected = RunningStatistics(11)
  s_expected.update(v1)
  s_expected.update(v2, weight_per_elem=w1)
  s_expected.update(v2, weight_per_elem=w2)
  s_actual = RunningStatistics(11)
  s_actual.update(v1)
  s_actual.update(v2, weight_per_elem=wt)
  assert_close(s_expected, s_actual)
  s_actual = RunningStatistics(11)
  s_actual.update(v1)
  s_actual.update(s2, weight_per_elem=wt)
  assert_close(s_expected, s_actual)

  # Test * operator
  scale = float(np.random.rand())
  s_expected = RunningStatistics(11)
  s_expected.update(scale * vall)
  s_actual = RunningStatistics(11)
  s_actual.update(vall)
  s_actual1 = s_actual.copy()
  s_actual1 *= scale
  s_actual2 = scale * s_actual
  assert_close(s_expected, s_actual1)
  assert_close(s_expected, s_actual2)

  # Test | operator
  s_expected = RunningStatistics(11)
  s_expected |= vall
  assert_close(s_expected, s1 | s3 | s2)
  s_actual = RunningStatistics(11)
  s_actual |= s3
  s_actual |= s1
  s_actual |= s2
  assert_close(s_expected, s_actual)

  # Test reduce
  s_expected = RunningStatistics.from_samples(vall, reduce_dims=(0, 1))
  s_actual = RunningStatistics.from_samples(vall, reduce_dims=False)
  s_actual.reduce_dims(0)
  s_actual.reduce_dims(0)
  assert_close(s_expected, s_actual)
  s_actual = RunningStatistics.from_samples(vall, reduce_dims=False)
  s_actual.reduce_dims(1)
  s_actual.reduce_dims(0)
  assert_close(s_expected, s_actual)
  s_actual = RunningStatistics.from_samples(vall, reduce_dims=False)
  s_actual.reduce_dims(0, 1)
  assert_close(s_expected, s_actual)
  s_actual = RunningStatistics.from_samples(vall, reduce_dims=False)
  s_actual.reduce_dims(1, 0)
  assert_close(s_expected, s_actual)

  print(s_expected)
  print()
  print("All checks successfull")


if __name__ == '__main__':
  main()
