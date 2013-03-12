""" Structures for representing the mode graph.

@author: tjhunter
"""
import numpy as np

class GaussianParameters(object):
  """ Parameters of a gaussian distribution
  """
  
  def __init__(self, mean, variance):
    self.mean = mean
    self.variance = variance
  
  def __repr__(self):
    return "Gaussian[mean=%f,var=%f]" % (self.mean, self.variance)
    
    
class Variable(object):
  """ Variables of the travel time graph.
  
  Uniquely defined by a var ID (var_id)
  
  mode is string "go" or "stop" or "stopstop"
  
  Fields:
  - varID: variable ID
  - linkID: the link ID
  - incomingIDs: list of var IDs
  - outgoingIDs: list of var IDs
  - mode ("stop" or "go")
  - parameters: GaussianParameters
  - location: Coordinate (for plotting)
  """
  
  def __init__(self, var_id, incoming_ids, param=None, location=None):
    self.varID = var_id
    self.incomingIDs = incoming_ids
    self.outgoingIDs = []
    self.parameters = param
    self.location = location

  def __repr__(self):
    return "Var[id=%s, params=%s]" % (str(self.varID), str(self.parameters))

class TravelTimeGraph(object):
  """ Encodes the dependencies and structure of the variables.
  
  Fields:
  variables -- dic var_id -> Variable
  variable_counts -- dic of var_id -> number of observations
  n -- number of variables
  indexes -- map of var_ids -> indexes
  reverse_indexes -- list of var_ids
  edges -- map of (var_id,var_id) -> covariance
  edge_counts -- map of (var_id, var_id) -> number of observations
  m -- number of edges
  edge_indexes -- map of (var_id,var_id) -> index
  reverse_edge_indexes -- list of (var_id,var_id)
  """
  
  def __init__(self, variables, edges=None, weighted_edges=None,
               variable_counts={}, edge_counts={}):
    """ 
    Parameters:
    variables -- a list of Variable object
    edges -- a list of edge of ((from_var_id, to_var_id), value)
    """
    self.variables = dict([(var.varID, var) for var in variables])
    # Set up translation tables for variable indexing
    self.n = len(self.variables)
    self.reverse_indexes = [var.varID for var in variables]
    self.variable_keys = self.reverse_indexes
    self.indexes = dict(zip(self.reverse_indexes,
                            range(self.n)))
    # Make sure we do not have twice the same edge (undirected graph)
    # Reorder to make sure we encode an upper diagonal matrix
    filtered_edges = {}
    filtered_edges_list = []
    if edges is not None:
      for (from_var_id, to_var_id) in edges:
        if from_var_id != to_var_id:
          if (to_var_id, from_var_id) not in filtered_edges \
          and (from_var_id, to_var_id) not in filtered_edges:
            idx_from = self.indexes[from_var_id]
            idx_to = self.indexes[to_var_id]
            if idx_from < idx_to:
              filtered_edges[(from_var_id, to_var_id)] = 0.0
              filtered_edges_list.append((from_var_id, to_var_id))
            else:
              filtered_edges[(to_var_id, from_var_id)] = 0.0
              filtered_edges_list.append((to_var_id, from_var_id))

    if weighted_edges is not None:
      for ((from_var_id, to_var_id), val) in weighted_edges:
        if from_var_id != to_var_id:
          if (to_var_id, from_var_id) not in filtered_edges \
          and (from_var_id, to_var_id) not in filtered_edges:
            idx_from = self.indexes[from_var_id]
            idx_to = self.indexes[to_var_id]
            if idx_from < idx_to:
              filtered_edges[(from_var_id, to_var_id)] = val
              filtered_edges_list.append((from_var_id, to_var_id))
            else:
              filtered_edges[(to_var_id, from_var_id)] = val
              filtered_edges_list.append((to_var_id, from_var_id))
        else:
          self.variables[from_var_id].variance = val
    self.edges = filtered_edges
    # Set up translation tables
    self.m = len(self.edges)
    self.reverse_edge_indexes = filtered_edges_list
    self.edge_indexes = dict(zip(filtered_edges_list,
                            range(self.m)))
#    # Adding the reverse edge list as well for fast access
#    reversed_filtered_edges_list = [(to_var_id, from_var_id) for (from_var_id, to_var_id) in filtered_edges_list]
#    self.edge_indexes.update(zip(reversed_filtered_edges_list,
#                            range(self.m)))
    self._rows = np.array([self.indexes[from_var_id] 
                  for idx in range(self.m)
                  for (from_var_id, to_var_id) in [self.reverse_edge_indexes[idx]] ])
    self._cols = np.array([self.indexes[to_var_id] 
                  for idx in range(self.m)
                  for (from_var_id, to_var_id) in [self.reverse_edge_indexes[idx]] ])
    
    self.variable_counts = dict([(var.varID, 0) for var in variables])
    for (var_id, count) in variable_counts.items():
      assert var_id in self.variable_counts
      self.variable_counts[var_id] = count
    self.edge_counts = dict([(z, 0) for z in self.reverse_edge_indexes])
    for (var_id_p, count) in edge_counts.items():
      var_id_rev = (var_id_p[1], var_id_p[0])
      assert var_id_p in self.edge_counts or var_id_rev in self.edge_counts
      if var_id_p in self.edge_counts:
        self.edge_counts[var_id_p] = count
      else:
        self.edge_counts[var_id_rev] = count
#    self.checkInvariants()
  
  def checkInvariants(self):
    """ Checks ths invariants of the indexes
    """
    assert len(self.reverse_edge_indexes) == self.m
    assert len(self.edge_indexes) == self.m
    assert len(self._rows) == self.m
    assert len(self._cols) == self.m
    assert len(self.edges) == self.m
    for (from_var_id, to_var_id) in self.reverse_edge_indexes:
      idx_from = self.indexes[from_var_id]
      idx_to = self.indexes[to_var_id]
      assert idx_from < idx_to, (from_var_id, to_var_id, idx_from, idx_to)
      assert (from_var_id, to_var_id) in self.edge_indexes, (from_var_id, to_var_id)
    
  def setEdgeCovariance(self, from_var_id, to_var_id, cov):
    if from_var_id != to_var_id:
      if (from_var_id, to_var_id) in self.edges:
        self.edges[(from_var_id, to_var_id)] = cov
      else:
        assert (to_var_id, from_var_id) in self.edges
        self.edges[(to_var_id, from_var_id)] = cov
    else:
      self.variables[from_var_id].variance = cov
  
  def rows(self):
    return self._rows
  
  def cols(self):
    return self._cols
  
  def means(self):
    return np.array([self.variables[self.reverse_indexes[idx]].parameters.mean for idx in range(self.n)])
  
  def allVariables(self):
    """ Gives a sorted list of variables.
    The ordering is guaranteed to be always the same.
    """
    return [self.variables[self.reverse_indexes[idx]] for idx in range(self.n)]
  
  def allEdgeKeys(self):
    """ The ordering is guaranteed to be always the same.
    """
    return self.reverse_edge_indexes
  
  def variances(self):
    return np.array([self.variables[self.reverse_indexes[idx]].parameters.variance for idx in range(self.n)])

  def covariances(self):
    return np.array([self.edges[self.reverse_edge_indexes[idx]] for idx in range(self.m)])
  
  def variableCounts(self):
    return np.array([self.variable_counts[self.reverse_indexes[idx]] for idx in range(self.n)])
  
  def edgeCounts(self):
    return np.array([self.edge_counts[self.reverse_edge_indexes[idx]] for idx in range(self.m)])
