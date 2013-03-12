'''
Created on Jan 29, 2013

@author: tjhunter
'''
import numpy as np
from scipy import linalg
from mm.arterial_hkt.gmrf_learning.quic_cpp import run_quic
from mm.arterial_hkt.gmrf_learning.utils import build_dense, build_sparse
from mm.arterial_hkt.gmrf_learning.psd import smallest_ev_arpack

def covsel_quick(R,U,rows,cols,lbda=1e6):
  min_ei = smallest_ev_arpack(R, U, rows, cols)
  if min_ei < 0:
    print "min_ei is %f"%min_ei
    R0 = R - min_ei + 1e-3
  else:
    R0 = R
  n = len(R)
  m = len(U)
  S = build_dense(R0, U, rows, cols)
  
  L=lbda * np.ones((n,n),dtype=np.double)
  L[np.arange(n),np.arange(n)] = 0
  L[rows,cols]=0
  L[cols,rows]=0
  
  (X_, W_, opt, time) = run_quic(S, L)
  D = X_.diagonal()
  P = X_[rows,cols]
  R2 = W_.diagonal()
  U2 = W_[rows,cols]
  print "Quic: Error is %f diag, %f outer"%(linalg.norm(R0-R2),linalg.norm(U-U2))
  return (D,P)


