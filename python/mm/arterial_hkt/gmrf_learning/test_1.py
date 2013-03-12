'''
Created on Jan 29, 2013

@author: tjhunter
'''

import numpy as np
from mm.arterial_hkt.gmrf_learning.utils import build_dense, build_sparse,\
  test_data, normalized_problem
from mm.arterial_hkt.gmrf_learning.psd import is_psd_dense, is_psd_cholmod
from mm.arterial_hkt.gmrf_learning.inv import inv_dense
from mm.arterial_hkt.gmrf_learning.quic import covsel_quick

from scipy.sparse.linalg import eigsh
from mm.arterial_hkt.gmrf_learning.log_det import logdet_dense,\
  logdet_dense_chol, logdet_cholmod

import numpy.linalg as la
from mm.arterial_hkt.gmrf_learning.cvx import run_cvx_dense, covsel_cvx_dense,\
  covsel_cvx_cholmod
from mm.arterial_hkt.gmrf_learning.quic_cpp.low_rank import random_projection_cholmod,\
  random_projection_cholmod_csc
from scikits.sparse.cholmod import analyze

def star(n,diag):
  m = n-1
  D = diag*np.ones(n,dtype=np.double)+np.arange(n)/float(n)
  P = np.arange(m)/float(m)+1
  rows = np.zeros((n-1,),dtype=np.int)
  cols = np.arange(1,n,dtype=np.int)
  return (D,P,rows,cols)

(D,P,rows,cols) = star(5,4)
X = build_dense(D, P, rows, cols)
Xs = build_sparse(D, P, rows, cols)

l1 = logdet_dense(D, P, rows, cols)
l2 = logdet_dense_chol(D, P, rows, cols)
l3 = logdet_cholmod(D, P, rows, cols)

(M,Dn,Pn) = normalized_problem(D, P, rows, cols)

test_data(D, P, rows, cols)
W = la.inv(X)
#Q = random_projection_cholmod(D, U, rows, cols, k, factor)
Q = random_projection_cholmod_csc(Xs, k=1000)
A = Q.T
print A.shape
R = np.sum(A*A,axis=1)
U = np.sum(A[rows]*A[cols],axis=1)
R_ = W.diagonal()
U_ = W[rows,cols]

#X = build_sparse(D, P, rows, cols)
#(eis,_)=eigsh(X, k=1, sigma=-1, which='LM')
#ei = eis[0]

#is_psd_dense(R, U, rows, cols)
#is_psd_cholmod(R, U, rows, cols)

(R,U) = inv_dense(D, P, rows, cols)

W = la.inv(X)

(D1,P1) = covsel_quick(R, U, rows, cols)

(D2,P2) = covsel_cvx_dense(R, U, rows, cols,num_iterations=150)
X2 = build_dense(D2, P2, rows, cols)

factor = analyze(Xs)
(D3,P3) = covsel_cvx_cholmod(R, U, rows, cols,
                             k=10000,num_iterations=100,
                             factor=factor,finish_early=False)

