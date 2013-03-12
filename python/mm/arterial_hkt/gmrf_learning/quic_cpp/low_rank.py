'''
Created on Jan 30, 2013

@author: tjhunter

Low rank decomposition of a big matrix.
'''
import numpy as np

from mm.arterial_hkt.gmrf_learning.utils import build_sparse

def random_projection_cholmod(R,U,rows,cols,k,factor=None):
  X = build_sparse(R, U, rows, cols)
  return random_projection_cholmod_csc(X, k,factor)

def random_projection_cholmod_csc(X,k,factor=None):
  from scikits.sparse.cholmod import cholesky
  # TODO: add factor update
  full_factor = cholesky(X) if factor is None else factor.cholesky(X)
  (n,_) = X.shape
  D = full_factor.D()
  D12 = np.sqrt(D)
  Q = np.random.rand(k,n)
  Q[Q<0.5] = -1.0/np.sqrt(k)
  Q[Q>=0.5] = 1.0/np.sqrt(k)
  zA = (Q / D12).T
  zB = full_factor.solve_Lt(zA)
  A = full_factor.apply_Pt(zB).T
  return A
