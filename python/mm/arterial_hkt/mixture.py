'''
Created on Aug 27, 2012

@author: tjhunter

Representation of a mixture of gaussians (the main output of the 
model) and some functions to estimate the quality of the output.
'''
import numpy as np
from scipy.stats import norm

class GMixture(object):
  
  weights_tolerance = 1e-5
  
  """ Representation of a weighted mixture of Gaussian distributions.
  
  Fields:
  - weights 1D array that sums to 1
  - means 1D array of means
  - variances 1D array with the variances
  """
  
  def __init__(self, weights, means, variances):
    self.weights = weights
    self.means = means
    self.variances = variances
  
  def checkInvariants(self):
    assert isinstance(self.weights, np.ndarray)
    assert isinstance(self.means, np.ndarray)
    assert isinstance(self.variances, np.ndarray)
    assert len(self.weights.shape) == 1
    assert self.weights.shape == self.means.shape
    assert self.weights.shape == self.variances.shape
    assert self.weights.dtype == np.double
    assert self.variances.dtype == np.double
    assert self.means.dtype == np.double
    
    assert np.all(self.weights>=0)
    assert np.sum(self.weights) <= 1+GMixture.weights_tolerance
    assert np.sum(self.weights) >= 1-GMixture.weights_tolerance
    
  def __str__(self):
    s = 'Weights: {0} \n'.format(self.weights)
    s += 'Means: {0} \n'.format(self.means)
    s += 'Variances: {0} \n'.format(self.variances)
    return s
  
  
  def probability(self, x):
    """ The density of observing this observation.
    """
    return np.exp(self.logProbability(x))
  
  
  def probabilities(self, x):
    """ The density of observing these observations.
    Input
      x: array of points where the pdf is to be computed
    Output
      array with the values of the pdf at the points in x
    """
    return np.exp(self.logProbabilities(x))
  
  
  def logProbabilities(self, x):
    """ The log density of the observations.
    Input
      x: array of points where the log of the pdf is to be computed
    Output
      array with the values of the log-pdf at the points in x
    """
    x_ = np.outer(x, np.ones_like(self.means))
    ms = np.outer(np.ones_like(x), self.means)
    vs = np.outer(np.ones_like(x), self.variances)
    z = -0.5 * ((x_ - ms) ** 2) / vs - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(vs)
    m = np.max(z)
    return m + np.log(np.exp(z - m).dot(self.weights))
   
   
  def logProbability(self, x):
    return self.logProbabilities(x)[0]
  
  def maxVal(self):
    """ The maximum interesting value people should be looking at.
    """
    # % sigmas
    num_sigs = 5
    return np.max(self.means + np.sqrt(self.variances) * num_sigs)


  def minVal(self):
    """ The maximum interesting value people should be looking at.
    """
    # % sigmas
    num_sigs = 5
    return np.min(self.means - np.sqrt(self.variances) * num_sigs)
  
  
  def cumulative(self, x):
    """ Computes the value of the cdf at x
    """
    return self.cumulatives(np.array([x]))[0]
  
  
  def cumulatives(self, xs):
    """ Computes the value of the cdf at the points in x
    Input
      x: array of points where the cdf is to be computed
    Output
      array with the values of the lcdf at the points in x
    """
    n = len(self.means)
    zs = np.outer(xs, np.ones(n))
    ms = np.outer(np.ones_like(xs), self.means)
    vs = np.outer(np.ones_like(xs), self.variances)
    us = norm.cdf(zs, loc=ms, scale=np.sqrt(vs))
    return us.dot(self.weights)
  
  def cumulativeGrid(self, step=1.0, num_steps=0):
    """ returns a pair of (xs, ys) with xs some sampling values and the cumulative values
    """
    xs = np.arange(self.minVal(), self.maxVal(), step) if num_steps == 0 else self.minVal() + (self.maxVal() - self.minVal()) * np.arange(num_steps)
    return (xs, np.cumsum(self.probabilities(xs)))
  
  def bic(self, xs):
    n = len(xs)
    return -2 * np.sum(self.logProbabilities(xs)) + 3 * len(self.weights) * np.log(n)
  
  def assignments(self, xs):
    """ Returns the mixture index assignments of each value.s
    """
    x_ = np.outer(xs, np.ones_like(self.means))
    ms = np.outer(np.ones_like(xs), self.means)
    vs = np.outer(np.ones_like(xs), self.variances)
    z = -0.5 * ((x_ - ms) ** 2) / vs - 0.5 * np.log(2 * np.pi) - 0.5 * np.log(vs) + np.log(self.weights)
    return np.argmax(z, axis=1)

  def assignment(self, x):
    """ Returns the mixture index assignments of each value.s
    """
    return self.assignments(np.array([x]))[0]
  
  def inverseCumulative(self,
                        percentile,
                        precision=1e-2,
                        low_bound=None,
                        up_bound=None):
    def inner(a, b):
      assert a < b
      m = (a + b) / 2.0
      if (b - a) < precision:
        return m
      c = self.cumulative(m)
      if c > percentile:
        return inner(a , m)
      else:
        return inner(m, b)
    low_bound = self.minVal() if low_bound is None else low_bound
    up_bound = self.maxVal() if up_bound is None else up_bound
    return inner(low_bound, up_bound)
  
