from quic import Quic
import numpy as np
import numpy.linalg as la

def run_quic(S, L, tol=1e-4, max_iters=500):
  """ Runs the QUIC algorithm (using the reference c++ implementation)
  
  Arguments:
  S - pxp covariance matrix (2D array)
  L - pxp L1 weight matrix (2D array)
  
  Returns:
  (X, W, opts, times)
  
  X - sparse precision
  W - inverse of the sparse precision
  opts - list of optimal values computed
  times - cumulative running times
  """
  p, _ = S.shape
  assert L.shape == (p,p)
  q = Quic()
  # Corresponds to "default" mode
  pathLen = 1
  path = np.zeros(2, dtype=np.double)
  msg = 2
  iter0 = max_iters
  X = np.eye(p, dtype=np.double)
  W = np.eye(p, dtype=np.double)
  opt = np.zeros(iter0, dtype=np.double)
  time = np.zeros(iter0, dtype=np.double)
  iters = np.ones(iter0, dtype=np.uint32) * iter0
  optsize = 1
  q.compute2('d', p, S, L, pathLen, path, tol, msg, iters, X, W, opt, time)
  return (X, W, opt[opt!=0], time[time!=0])

p = 200
S = np.random.rand(p,p)
S = np.array(S.dot(S.T), dtype=np.double)

L = 55 * np.ones((p,p), dtype=np.double)

(X, W, opts, times) = run_quic(S, L, tol=1e-2, max_iters=20)

la.norm(S-np.eye(p))