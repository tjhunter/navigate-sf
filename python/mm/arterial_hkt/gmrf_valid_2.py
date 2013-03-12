'''
Created on Jan 24, 2013

@author: tjhunter
'''

from mm.arterial_hkt.gmrf import GMRF
from scikits.sparse.cholmod import cholesky
from scipy.linalg import inv
import numpy as np

#means = {'1':1,'3':3,'2':2}
#precisions={('1','2'):12,('2','3'):23,('1','1'):11,('3','3'):33,('2','2'):22}

means = {'1':1,'3':3,'2':2}
n = len(means)
precisions={('1','2'):1,('2','3'):1,('1','1'):2,('3','3'):2,('2','2'):2}

gmrf = GMRF(means, precisions)

sig_complete = inv(gmrf.precision.todense())

def fexact(idxs):
  return np.sum(sig_complete[np.ix_(idxs, idxs)])


factor = cholesky(gmrf.precision)
D = factor.D()
D12 = np.sqrt(D)

#X = gmrf.precision.todense()
#P=np.eye(n)
#P=P[factor.P()].T
#L=factor.L().todense()
#D=np.diag(factor.D())
#X2 = P.dot(L).dot(L.T).dot(P.T)
#I = P.dot(inv(L.T)).dot(inv(D)).dot(inv(L)).dot(P.T)

def fexact2(idxs):
  z = np.zeros(n)
  z[idxs] = 1.0
  z_perm = factor.solve_P(z)
  z_0 = factor.solve_L(z_perm)
  z_2 = z_0 / D12
  z_1 = factor.solve_D(z_0)
  return z_2.dot(z_2)


k = 10000
Q = np.random.rand(k,n)
Q[Q<0.5] = -1.0/np.sqrt(k)
Q[Q>=0.5] = 1.0/np.sqrt(k)

zA = (Q / D12).T
zB = factor.solve_Lt(zA)
R = factor.solve_Pt(zB).T

def fapprox(idxs):
  z0 = R[::,idxs].sum(axis=1)
  return z0.dot(z0)

#z = np.zeros(n)
#z[idxs] = 1.0
#z0 = R.dot(z)
#z0 = R[::,idxs].sum(axis=1)

idxs = np.array([2,0])

x_exact = fexact(idxs)
x_exact2 = fexact2(idxs)
x_approx1 = fapprox(idxs)

sig_approx = R.T.dot(R)

