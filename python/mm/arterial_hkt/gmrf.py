'''
Created on Jul 17, 2012

@author: tjhunter

Super simple implementation of a Gaussian Markov Random Field.
'''
import numpy as np
import numpy.linalg as la
# from mm.arterial_hkt.tt_graph import GaussianParameters
from scipy.sparse.csc import csc_matrix

def CSCfromCompactRepresentation(diag_vals, upper_rows, upper_cols, upper_vals):
  n = len(diag_vals)
  rows = np.concatenate((np.arange(n), upper_rows, upper_cols))
  cols = np.concatenate((np.arange(n), upper_cols, upper_rows))
  ij = np.vstack((rows, cols))
  vals = np.concatenate((diag_vals, upper_vals, upper_vals))
  return csc_matrix((vals, ij), shape=(n, n))
  
class GMRF(object):
  
  def __init__(self, translations, rows, cols, means, diag_precision, upper_precision):
    """
    Translation: map var_id -> index
    """
    self.translations = translations
    self.rows = rows
    self.cols = cols
    self.means = means
    self.upper_precision = upper_precision
    self.diag_precision = diag_precision
    self.precision = CSCfromCompactRepresentation(self.diag_precision, self.rows, self.cols, self.upper_precision)
    self.check()
  
  def check(self):
    assert self.rows.dtype == np.int
    assert self.cols.dtype == np.int
    assert self.means.dtype == np.double
    assert self.upper_precision.dtype == np.double
    assert self.diag_precision.dtype == np.double
    m = len(self.upper_precision)
    assert self.rows.shape == (m,)
    assert self.cols.shape == (m,)
    n = len(self.translations)
    assert self.means.shape == (n,)
 