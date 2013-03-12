'''
Created on Jan 31, 2013

@author: tjhunter
'''
import numpy as np
import numpy.linalg as la
from mm.arterial_hkt.gmrf_learning.quic_cpp.low_rank import random_projection_cholmod_csc

class GMRFEstimator(object):
  """ Interface to express covariance estimation algorithms.
  """
  
  def pathMean(self, var_ids):
    """ Returns the mean value for this list of var IDs.
    """
    pass
  
  def pathCovariance(self, var_ids):
    """ Returns a the covariance of the travel time corresponding to the 
    sum of variables encoded in the path.
    
    Arguments:
    - var_ids: list of indexes
    Returns:
    - variance (double)
    """

class ExactGMRFEstimator(GMRFEstimator):
  
  def __init__(self, translation, means, covariance):
    """ 
    Arguments:
    translation -- a dictionary of var_id -> index
    means -- 1D array
    precisions -- CSC sparse matrix
    """
    self.translation = translation
    self._means = means
#    self.precisions = precisions.todense()
    self._covariance = covariance #la.inv(self.precisions)
    
  def _idxs(self, var_ids):
    return np.array([self.translation[var_id] for var_id in var_ids],dtype=np.int)

  def pathMean(self, var_ids):
    idxs = self._idxs(var_ids)
    return self._means[idxs].sum()

  def pathCovariance(self, var_ids):
    idxs = self._idxs(var_ids)
    return self._covariance[np.ix_(idxs, idxs)].sum()

class JLGMRFEstimator(GMRFEstimator):
  
  def __init__(self, translation, means, Q):
    """ 
    Arguments:
    translation -- a dictionary of var_id -> index
    means -- 1D array
    """
    self.translation = translation
    self._means = means
    self.Q = Q
#    self.precisions = precisions
#    self.Q = random_projection_cholmod_csc(precisions,k)
    
  def _idxs(self, var_ids):
    return np.array([self.translation[var_id] for var_id in var_ids],dtype=np.int)

  def pathMean(self, var_ids):
    idxs = self._idxs(var_ids)
    return self._means[idxs].sum()

  def pathCovariance(self, var_ids):
    idxs = self._idxs(var_ids)
    z = self.Q.T[idxs].sum(axis=0)
    return z.dot(z)
  
class DiagonalGMRFEstimator(GMRFEstimator):
  
  def __init__(self, translation, means, diag_variance):
    """ 
    Arguments:
    translation -- a dictionary of var_id -> index
    means -- 1D array
    """
    self.translation = translation
    self._means = means
    self.diag_variance = diag_variance
#    self.precisions = precisions
#    self.Q = random_projection_cholmod_csc(precisions,k)
    
  def _idxs(self, var_ids):
    return np.array([self.translation[var_id] for var_id in var_ids],dtype=np.int)

  def pathMean(self, var_ids):
    idxs = self._idxs(var_ids)
    return self._means[idxs].sum()

  def pathCovariance(self, var_ids):
    idxs = self._idxs(var_ids)
    z = self.diag_variance[idxs]
    return z.sum()

