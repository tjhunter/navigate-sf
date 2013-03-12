'''
Created on Jan 29, 2013

@author: tjhunter
'''

import numpy as np
from scipy import linalg

from mm.arterial_hkt.gmrf_learning.utils import build_dense, build_sparse
from mm.arterial_hkt.gmrf_learning.psd import is_psd_dense, is_psd_cholmod
#from scikits.sparse.cholmod import cholesky


def logdet_dense(R,U,rows,cols):
  if not is_psd_dense(R, U, rows, cols):
    return -np.Inf
  X = build_dense(R, U, rows, cols)
  eis = linalg.eigvalsh(X)
  return np.sum(np.log(eis))

def logdet_dense_chol(R,U,rows,cols):
  if not is_psd_dense(R, U, rows, cols):
    return -np.Inf
  X = build_dense(R, U, rows, cols)
  C = linalg.cholesky(X).diagonal()
  return np.sum(np.log(C))*2

def logdet_cholmod(R,U,rows,cols, psd_tolerance=1e-6, factor=None):
  from scikits.sparse.cholmod import cholesky
  if not is_psd_cholmod(R, U, rows, cols, psd_tolerance, factor):
    return -np.Inf
  X = build_sparse(R, U, rows, cols)
#  print factor
  filled_factor = cholesky(X) if factor is None else factor.cholesky(X)
  D = filled_factor.D()
  return np.sum(np.log(D))
  
  
