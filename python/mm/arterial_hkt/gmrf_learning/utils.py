'''
Created on Jan 29, 2013

@author: tjhunter

Some utility functions to create data.
'''
import numpy as np
from scipy.sparse.csc import csc_matrix


def build_dense(R,U,rows,cols):
  n = len(R)
  m = len(U)
  S = np.zeros((n,n),dtype=np.double)
  S[np.arange(n),np.arange(n)] = R
  S[rows,cols]=U
  S[cols,rows]=U
  return S

def build_sparse(R,U,rows,cols):
  """ Returns an equivalent matrix in CSC format.
  """
  n = len(R)
  all_rows = np.concatenate((np.arange(n),rows,cols))
  all_cols = np.concatenate((np.arange(n),cols,rows))
  ij = np.vstack((all_rows,all_cols))
  vals = np.concatenate((R,U,U))
  return csc_matrix((vals,ij),shape=(n,n))
  
def test_data(R,U,rows,cols):
  """ Very fast checks on the input the data to ensure 
  compatibility.
  """
  n = len(R)
  m = len(U)
  assert R.shape == (n,)
  assert U.shape == (m,)
  assert rows.shape == (m,)
  assert cols.shape == (m,)
  assert np.all(rows<cols)
  assert U.dtype == np.double, U.dtype
  assert R.dtype == np.double, R.dtype
  assert rows.dtype == np.int64, rows.dtype
  assert cols.dtype == np.int64, cols.dtype

def normalized_problem(R,U,rows,cols):
  """ R = diag(M) * R2 * diag(M)
  """
  assert np.all(R>1e-8)
  M = np.sqrt(R)
  R2 = np.ones_like(R)
  U2 = U / (M[rows]*M[cols])
  return (M,R2,U2)
