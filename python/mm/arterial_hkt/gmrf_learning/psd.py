'''
Created on Jan 29, 2013

@author: tjhunter

Checks for positive semi-definiteness.
'''
import numpy as np
from mm.arterial_hkt.gmrf_learning.utils import build_dense, build_sparse
from scipy import linalg
#from scikits.sparse.cholmod import cholesky
from scipy.sparse.linalg.eigen.arpack.arpack import eigsh


def is_psd_dense(R,U,rows,cols,tolerance=1e-4):
  """ Simple check using dense matrices.
  """
  X = build_dense(R, U, rows, cols)
  return np.all(linalg.eigvalsh(X)>tolerance)

def is_psd_cholmod(R,U,rows,cols,tolerance=1e-6,factor=None):
  X = build_sparse(R, U, rows, cols)
  from scikits.sparse.cholmod import cholesky
  try:
    full_factor = cholesky(X) if factor is None else factor.cholesky(X)
  except:
    return False
  D = full_factor.D()
  return np.all(D>tolerance)

def is_psd_arpack(R,U,rows,cols,tolerance=1e-4):
  X = build_sparse(R, U, rows, cols)
  (eis,_)=eigsh(X, k=1, sigma=-1, which='LM')
  ei = eis[0]
  return ei>tolerance

def smallest_ev_arpack(R,U,rows,cols,tolerance=1e-4):
  X = build_sparse(R, U, rows, cols)
  (eis,_)=eigsh(X, k=1, sigma=-3, which='LM',tol=tolerance,maxiter=1000)
  ei = eis[0]
  return ei
