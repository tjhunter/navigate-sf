'''
Created on Feb 5, 2013

@author: tjhunter
'''
import numpy as np
from mm.arterial_hkt.gmrf import GMRF

n = 4
translations = dict(zip(['%d'%i for i in range(n)],range(n)))
rows = np.arange(1,n)
cols = np.zeros_like(rows)
m = len(rows)
diag_precision = np.ones(n)
upper_precision = np.zeros(m)
means = np.zeros_like(diag_precision)

gmrf_exact = GMRF(translations, rows, cols, means, diag_precision, upper_precision)

X = gmrf_exact.precision.todense()
W = np.linalg.inv(X)


