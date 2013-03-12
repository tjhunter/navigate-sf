'''
Created on Nov 15, 2012

@author: tjhunter

Tests for core travel time structures.
'''

#from mm.arterial_hkt.travel_times import completeDistribution, \
#  TravelTimeElements
#import numpy as np
#
#def test_1():
#  # Very simple test
#  gvids = [['1'], ['2']]
#  C = np.eye(2)
#  m = np.array([1.0, 2.0])
#  p0 = np.array([1.0])
#  ps = [np.array([[1.0]])]
#  tte = TravelTimeElements(gvids, p0, ps, m, C)
#  gm = completeDistribution(tte)
#  assert (gm.weights == np.array([1.0])).all()
#  assert (gm.variances == np.array([2.0])).all()
#  assert (gm.means == np.array([3.0])).all()
#
#def test_2():
#  # Very simple test
#  gvids = [['a', 'b'], ['c']]
#  C = np.eye(3)
#  m = np.array([1.0, 2.0, 3.0])
#  p0 = np.array([.5, .5])
#  ps = [np.array([[1.0, 1.0]])]
#  tte = TravelTimeElements(gvids, p0, ps, m, C)
#  gm = completeDistribution(tte)
#  assert (gm.weights == np.array([.5, .5])).all()
#  assert (gm.variances == np.array([2.0, 2.0])).all()
#  assert (gm.means == np.array([4.0, 5.0])).all()
#
#def test_3():
#  # Very simple test
#  gvids = [['a'], ['b', 'c']]
#  C = np.eye(3)
#  m = np.array([1.0, 2.0, 3.0])
#  p0 = np.array([1.0])
#  ps = [np.array([[.8], [.2]])]
#  tte = TravelTimeElements(gvids, p0, ps, m, C)
#  gm = completeDistribution(tte)
#  assert (gm.weights == np.array([.8, .2])).all()
#  assert (gm.variances == np.array([2.0, 2.0])).all()
#  assert (gm.means == np.array([3.0, 4.0])).all()
  