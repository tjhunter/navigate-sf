'''
Created on Feb 18, 2013

@author: tjhunter
'''

import numpy as np
from mm.arterial_hkt.gmrf_learning.utils import test_data
from mm.arterial_hkt.gmrf_learning.cvx import *

R = np.ones(5)
U = np.ones(3)
cols = np.array([2,3,4])
rows = np.array([0,0,0])
test_data(R, U, rows, cols)
edge_count = np.array([20,20,20])
min_variance=1e-2
min_edge_count = 0
gmrf_learn_cov_cvx(R, U, rows, cols, edge_count,min_variance,min_edge_count)
