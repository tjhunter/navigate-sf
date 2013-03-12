'''
Created on Jan 29, 2013

@author: tjhunter

Partial inversion of the matrix.
'''
from scipy import linalg
from mm.arterial_hkt.gmrf_learning.utils import build_dense

def inv_dense(R,U,rows,cols):
  X = build_dense(R, U, rows, cols)
  Y = linalg.inv(X)
  D = Y.diagonal()
  P = Y[rows,cols]
  return (D,P)
