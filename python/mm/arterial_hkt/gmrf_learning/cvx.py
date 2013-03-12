'''
Created on Jan 29, 2013

@author: tjhunter

Adapted from:

http://abel.ee.ucla.edu/cvxopt/userguide/spsolvers.html#example-covariance-selection
'''
from mm.arterial_hkt.gmrf_learning.log_det import logdet_dense_chol,\
  logdet_cholmod
import numpy.linalg as la
import numpy as np
from mm.arterial_hkt.gmrf_learning.utils import build_dense, build_sparse,\
  normalized_problem
from mm.arterial_hkt.gmrf_learning.psd import smallest_ev_arpack
from mm.arterial_hkt.gmrf_learning.quic_cpp.low_rank import random_projection_cholmod
from mm.arterial_hkt.utils import tic

def obj_dense(R_hat, U_hat, rows, cols, D, P):
  """ Dense objective of the minimzation problem.
  
  R_hat -- sufficient statistics for the learning problem 
  
  
  """
  return -logdet_dense_chol(D, P, rows, cols) \
    + (R_hat.dot(D) + 2 * U_hat.dot(P))

def obj_cholmod(R_hat, U_hat, rows, cols, D, P,
                psd_tolerance=1e-6, factor=None):
  """ Dense objective of the maximization problem.
  
  R_hat -- sufficient statistics for the learning problem 
  
  
  """
  return -logdet_cholmod(D, P, rows, cols,psd_tolerance,factor) \
    + (R_hat.dot(D) + 2 * U_hat.dot(P))

def grad_cholmod(R_hat, U_hat, rows, cols, D, P,
                 k,factor=None):
  Q = random_projection_cholmod(D, P, rows, cols, k, factor)
  A = Q.T
#  print A.shape
  R = np.sum(A*A,axis=1)
  U = np.sum(A[rows]*A[cols],axis=1)
  return (-R + R_hat, 2*(-U + U_hat))


def grad_dense(R_hat, U_hat, rows, cols, D, P):
  X = build_dense(D, P, rows, cols)
  W = la.inv(X)
#  print 'W eis:', (min(la.eigvalsh(W)),max(la.eigvalsh(W)))
  R = W.diagonal()
  U = W[rows, cols]
  return (-R + R_hat, 2*(-U + U_hat))

def hessian_dir_dense(R_hat, U_hat, rows, cols, D, P):
  """ Newton step seems to work now?
  """
  X = build_dense(D, P, rows, cols)
  W = la.inv(X)
  n = len(D)
  A = n * np.arange(n) + np.arange(n)
  B = n*rows + cols
  K = np.kron(W, W)
  # todo: finish here
  H = np.vstack((np.hstack((K[np.ix_(A, A)],2*K[np.ix_(A, B)])),
                 np.hstack((2*K[np.ix_(A, B)].T,4*K[np.ix_(B, B)]))))
  (g_D, g_P) = grad_dense(R_hat, U_hat, rows, cols, D, P)
  g = np.hstack((g_D, g_P))
  v = la.solve(H, g)
  v_D = v[:n]
  v_P = v[n:]
  return (-v_D, -v_P)

def iter_dense(R_hat, U_hat, rows, cols, D, P,num_lsearch_iter=100):
  (g_D, g_P) = grad_dense(R_hat, U_hat, rows, cols, D, P)
  v_D = -g_D
  v_P = -g_P
  
#  (v_D, v_P) = hessian_dir_dense(R_hat, U_hat, rows, cols, D, P)
#  print 'v',(v_D, v_P)
  # Stopping criterion:
  sqntdecr = -v_D.dot(g_D) - v_P.dot(g_P)
  print("Newton decrement squared:%- 7.5e" % sqntdecr)
  if sqntdecr < 1e-8:
    return None
  # line search
  dD = v_D
  dP = v_P
  s = 1.0
  f = obj_dense(R_hat, U_hat, rows, cols, D, P)
  print 'Current objective value: ', f
  for lsiter in range(num_lsearch_iter):
    curr_D = D + s * dD
    curr_P = P + s * dP
    fn = obj_dense(R_hat, U_hat, rows, cols, curr_D, curr_P)
#    print 'fn ',fn
    if fn == -np.Infinity:
      s *= 0.5
    else:
      if fn < f - 0.01 * s * sqntdecr:
        print "Update lsiter=", lsiter
        return (curr_D, curr_P,fn,lsiter)
      s *= 0.5
  print 'Too many iterations'
  return None

def iter_cholmod(R_hat, U_hat, rows, cols, D, P,
                 k,psd_tolerance=1e-6,factor=None,
                 num_lsearch_iter=10):
  tic("computing gradient", "iter_cholmod")
  (g_D, g_P) = grad_cholmod(R_hat, U_hat, rows, cols, D, P, k, factor)
  v_D = -g_D
  v_P = -g_P
  # Debug
#  (g_D_, g_P_) = grad_dense(R_hat, U_hat, rows, cols, D, P)
#  print 'g_D diff:',la.norm(g_D-g_D_)
#  print 'g_P diff:',la.norm(g_P-g_P_)
  # End debug
  # Stopping criterion:
  sqntdecr = -v_D.dot(g_D) - v_P.dot(g_P)
  tic("Newton decrement squared:%- 7.5e" % sqntdecr, "iter_cholmod")
  if sqntdecr < 1e-8:
    return None
  # line search
  dD = v_D
  dP = v_P
  s = 1.0
  f = obj_cholmod(R_hat, U_hat, rows, cols, D, P, 
                  psd_tolerance, factor)
  tic('Current objective value: {0}'.format(f), "iter_cholmod")
  for lsiter in range(num_lsearch_iter):
    curr_D = D + s * dD
    curr_P = P + s * dP
    fn = obj_cholmod(R_hat, U_hat, rows, cols, curr_D, curr_P, 
                  psd_tolerance, factor)
#    print 'fn ',fn
    tic("lsiter={0} fn={1}".format(lsiter,fn), "iter_cholmod")
    if fn == -np.Infinity:
      s *= 0.5
    else:
      if fn < f - 0.01 * s * sqntdecr:
        tic("Update lsiter={0}".format(lsiter), "iter_cholmod")
        return (curr_D, curr_P,fn,lsiter)
      s *= 0.5
  print 'Too many iterations'
  return None


def run_cvx_dense(R_hat, U_hat, rows, cols, num_iterations=500):
  D = np.ones_like(R_hat)
  P = np.zeros_like(U_hat)
  for iters in range(num_iterations):
    print "Iter ",iters
    z = iter_dense(R_hat, U_hat, rows, cols, D, P)
    if z is None:
      print 'Done early'
      return (D, P)
    (D2, P2,fn,lsiter) = z
    D = D2
    P = P2
  return (D, P)

def run_cvx_cholmod(R_hat, U_hat, rows, cols,
                    k,psd_tolerance=1e-6,factor=None,
                    num_iterations=500,finish_early=True,debug=True):
  D = np.ones_like(R_hat)
  P = np.zeros_like(U_hat)
  for iters in range(num_iterations):
    tic("Iter={0}".format(iters),"run_cvx_cholmod")
    # Debug
#    f1 = obj_dense(R_hat, U_hat, rows, cols, D, P)
#    f2 = obj_cholmod(R_hat, U_hat, rows, cols, D, P, psd_tolerance, factor)
#    delta = f1 - f2
#    print "True objective value:",f1
#    print "This objective value",f2
#    print "Difference",delta
    # End debug
    z = iter_cholmod(R_hat, U_hat, rows, cols, D, P, k,
                     psd_tolerance, factor)
    if finish_early and z is None:
      tic("done early","run_cvx_cholmod")
      return (D, P)
    if z is not None:
      (D2, P2,fn,lsiter) = z
      D = D2
      P = P2
  return (D, P)


def covsel_cvx_dense(R, U, rows, cols, num_iterations=500,min_ev=1e-2):
  min_ei = smallest_ev_arpack(R, U, rows, cols)
  print "min_ei is %f"%min_ei
  if min_ei < min_ev:
    R0 = R - min_ei + min_ev
  else:
    R0 = R
  return run_cvx_dense(R0, U, rows, cols, num_iterations)

def covsel_cvx_cholmod(R, U, rows, cols,
                    k,psd_tolerance=1e-6,factor=None,
                    num_iterations=500,finish_early=True,debug=True):
  if debug:
    tic('smallest ev', "covsel_cvx_cholmod")
  min_ei = smallest_ev_arpack(R, U, rows, cols)
  if debug:
    tic("min_ei is %f"%min_ei, "covsel_cvx_cholmod")
  if min_ei < 0:
    R0 = R - min_ei + 1e-3
  else:
    R0 = R
  return run_cvx_cholmod(R0, U, rows, cols,
                         k, psd_tolerance, factor, num_iterations,finish_early)

def independent_variables(n, rows, cols):
  """ Returns a mask that contains the independent variables.
  """
  m = len(rows)
  X= build_sparse(np.ones(n), np.ones(m), rows, cols)
  return np.array((X.sum(axis=1)==1).flatten())[0]

def gmrf_learn_cov_cvx(R, U, rows, cols, edge_count, 
                       min_variance=1e-2, min_edge_count=10,
                       num_iterations=50):
  n = len(R)
  m = len(U)
  mask = edge_count>=min_edge_count
  active_m = np.sum(mask)
  tic("m={0}, active m={1}".format(m,active_m), "gmrf_learn_cov_cvx")
  active_U = U[mask]
  active_rows = rows[mask]
  active_cols = cols[mask]
  # A number of variables hare independant (due to lack of observations)
  independent_mask = independent_variables(n, active_rows, active_cols)
  # Put them aside and use the independent strategy to solve them.
  indep_idxs = np.arange(n)[independent_mask]
  R_indep = R[indep_idxs]
  # Solve the regularized version for independent variables
  D_indep = 1.0 / np.maximum(min_variance*np.ones_like(R_indep), R_indep)
  # Putting together the dependent and independent parts
  D = np.zeros_like(R)
  D[independent_mask] = D_indep
  P = np.zeros_like(U)
  # No need to solve for the outer diagonal terms, they are all zeros.
  # Solve for the dependent terms
  dependent_mask = ~independent_mask
  n_dep = np.sum(dependent_mask)
  if n_dep > 0:
    idxs_dep = np.arange(n)[dependent_mask]
    reverse_idxs_dep = np.zeros(n,dtype=np.int64)
    reverse_idxs_dep[dependent_mask] = np.arange(n_dep)    
    rows_dep = reverse_idxs_dep[active_rows]
    cols_dep = reverse_idxs_dep[active_cols]
    R_dep = R[idxs_dep]
    U_dep = active_U
    (M, R_hat, U_hat) = normalized_problem(R_dep, U_dep, rows_dep, cols_dep)
    (D_norm_dep, P_norm_dep) = covsel_cvx_dense(R_hat, U_hat, rows_dep, cols_dep, num_iterations=num_iterations)  
    D[dependent_mask] = D_norm_dep / (M ** 2)
    P[mask] = P_norm_dep / (M[rows_dep] * M[cols_dep])
  return (D, P)

def gmrf_learn_cov_cholmod(R, U, rows, cols, edge_count,
                           k,
                           min_variance=1e-2, min_edge_count=10,
                           num_iterations=50,psd_tolerance=1e-3,finish_early=True):
  n = len(R)
  m = len(U)
  mask = edge_count>=min_edge_count
  active_m = np.sum(mask)
  tic("m={0}, active m={1}".format(m,active_m), "gmrf_learn_cov_cholmod")
  active_U = U[mask]
  active_rows = rows[mask]
  active_cols = cols[mask]
  # A number of variables hare independant (due to lack of observations)
  independent_mask = independent_variables(n, active_rows, active_cols)
  # Put them aside and use the independent strategy to solve them.
  indep_idxs = np.arange(n)[independent_mask]
  R_indep = R[indep_idxs]
  # Solve the regularized version for independent variables
  D_indep = 1.0 / np.maximum(min_variance*np.ones_like(R_indep), R_indep)
  # Putting together the dependent and independent parts
  D = np.zeros_like(R)
  D[independent_mask] = D_indep
  P = np.zeros_like(U)
  # No need to solve for the outer diagonal terms, they are all zeros.
  # Solve for the dependent terms
  dependent_mask = ~independent_mask
  n_dep = np.sum(dependent_mask)
  if n_dep > 0:
    idxs_dep = np.arange(n)[dependent_mask]
    reverse_idxs_dep = np.zeros(n,dtype=np.int64)
    reverse_idxs_dep[dependent_mask] = np.arange(n_dep)    
    rows_dep = reverse_idxs_dep[active_rows]
    cols_dep = reverse_idxs_dep[active_cols]
    R_dep = R[idxs_dep]
    U_dep = active_U
    (M, R_hat, U_hat) = normalized_problem(R_dep, U_dep, rows_dep, cols_dep)
    tic('Computing symbolic cholesky factorization of the graph...', "gmrf_learn_cov_cholmod")
    # Doing delayed import so that the rest of the code runs without sk-learn
    from scikits.sparse.cholmod import analyze
    Xs_dep = build_sparse(np.ones_like(R_hat), np.ones_like(U_hat), rows_dep, cols_dep)
    factor = analyze(Xs_dep)
    tic('Cholesky done', "gmrf_learn_cov_cholmod")
    # TODO add the other parameters
    (D_norm_dep, P_norm_dep) = covsel_cvx_cholmod(R_hat, U_hat, rows_dep, cols_dep,
                                          k, psd_tolerance, factor,
                                          num_iterations, finish_early)
    D[dependent_mask] = D_norm_dep / (M ** 2)
    P[mask] = P_norm_dep / (M[rows_dep] * M[cols_dep])
  return (D, P)

