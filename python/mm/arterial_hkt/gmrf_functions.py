""" Functions for the GMRF.
"""
from mm.arterial_hkt.gmrf import GMRF
import numpy as np
from mm.arterial_hkt.gmrf_learning.quic import covsel_quick

def independantValues(tt_graph, min_variance=1e-1):
  """ Computes a GMRF from a learned TT graph, using only the diagonal elements.
  """
  # Very simple: only look at the diagonal elements
  means = tt_graph.means()
  diag_precision = 1.0 / np.maximum(min_variance*np.ones_like(tt_graph.variances()), tt_graph.variances())
  upper_precision = np.zeros_like(tt_graph.covariances())
  return GMRF(tt_graph.indexes, tt_graph.rows(), tt_graph.cols(),
              means, diag_precision, upper_precision)

def learnQuic(tt_graph,lbda=1e3):
  means = tt_graph.means()
  rows = tt_graph.rows()
  cols = tt_graph.cols()
  R = tt_graph.variances()
  U = tt_graph.covariances()
  (D,P) = covsel_quick(R, U, rows, cols, lbda)
  return GMRF(tt_graph.indexes, tt_graph.rows(), tt_graph.cols(),
              means, D, P)

def emptyValues(tt_graph):
  """ Initializes a GMRF from a TT graph, using only the structure.
  """
  means = np.zeros_like(tt_graph.means())
  diag_precision = np.zeros_like(tt_graph.variances())
  upper_precision = np.zeros_like(tt_graph.covariances())
  return GMRF(tt_graph.indexes, tt_graph.rows(), tt_graph.cols(),
              means, diag_precision, upper_precision)
