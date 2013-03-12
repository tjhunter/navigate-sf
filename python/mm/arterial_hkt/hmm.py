'''
Created on Jul 17, 2012

@author: tjhunter

The hidden Markov Model that describes the transition in the road network.

Each road link has a number of discrete states (typically 1, 2 or 3) and is represented
by a node in the HMM graph. Between each physical pair of the graph, there is
a transition (that is markov) represented by a matrix.

This is a first-order implementation. 2nd or 3rd order implementations are 
not available with these structures. To do later.
'''
import numpy as np
from mm.arterial_hkt.variable import VariableId
from mm.arterial_hkt.lru import lru_cache_function

class HMMNode(object):
  """ A vertex in the HMM graph. Corresponds to a road link.
  
  Fields:
  - nodeID: a unique identifier for the HMM node.
  - numModes:int, the number of states
  - modeProbs: 1D np array: the probability of being in any mode.
  - modeIDs: the ids of each mode in the ModeGraph
  """
  
  def __init__(self, node_id, prob_modes, mode_ids, location=None):
    """ 
    Arguments:
    node_id -- node id
    prob_modes -- 1D array
    mode_ids -- list of ids
    location -- Coordinate
    """
    self.nodeID = node_id
    self.modeProbs = prob_modes
    self.modes = mode_ids
    self.numModes = len(self.modeProbs)
    self.modeIndexes = dict(zip(mode_ids, range(self.numModes)))
    self.location = location
  
  def sampleMode(self):
    idx = np.argmax(np.random.multinomial(1, self.modeProbs))
    return (self.nodeID, self.modes[idx])


class HMMTransition(object):
  """ The transition between two links, described by transitions.
  probabilities between modes.
  
  The links are oriented with an outgoing direction. For the first 
  link, the outgoing direction is the second link. For the second
  link, the outgoing direction has to be specified.
  
  Fields:
  - fromNodeId: the node id of the originating node
  - toNodeId: the node id of the subsequent node
  - transitions: transition table where each row corresponds to a destination mode
    2D array, where the columns sum to one.
  """
  
  def __init__(self, from_node_id, to_node_id, transitions):
    self.transitions = np.array(transitions, dtype=np.double)
    self.fromNodeID = from_node_id
    self.toNodeID = to_node_id
    s = np.sum(self.transitions,axis=0)
    assert np.max(np.abs(s-1))<1e-6, (self.transitions, self)
  
  def __repr__(self):
    return "%s->%s: %s" % (str(self.fromNodeID), str(self.toNodeID), self.transitions)

class HMMGraph(object):
  """ Graph of HMM nodes and transitions corresponding to a road network.
  """
  
  def __init__(self, nodes, transitions):
    """ nodes: list of HMMNode objects
    transitions: list of transitions.
    """
    self._nodes = nodes
    self._transitions = transitions
    self.nodes_by_id = dict([(n.nodeID, n) for n in nodes])
    self.transitions_by_id = dict([((tr.fromNodeID, tr.toNodeID), tr) for tr in transitions])
    self.cache = {}
  
  def allNodes(self):
    return self._nodes
  
  def allTransitions(self):
    return self._transitions
  
  
#  @lru_cache_function(max_size=2**16, expiration=15*60)
  def conditionalDistribution(self, from_node_id, mode, to_node_id):
    """ Returns the conditional distribution starting from a node in a certain mode, and going to a subsequent node.
    
    Returns a list of (varID, weight)
    """
    key = (from_node_id, mode, to_node_id)
    if self.cache is not None and key in self.cache:
      return self.cache[key]
    start_node = self.node(from_node_id)
    start_idx = start_node.modeIndexes[mode]
    end_node = self.node(to_node_id)
    if (from_node_id, to_node_id) not in self.transitions_by_id:
      print 'debug'
    tr = self.transition(from_node_id, to_node_id)
    cond_probs = tr.transitions[:, start_idx]
    res = [(VariableId(to_node_id, end_mode), cond_probs[end_idx]) 
                 for (end_mode, end_idx) in end_node.modeIndexes.iteritems()]
    if self.cache  is not None:
      self.cache[key] = res
    return res

  def transition(self, from_node_id, to_node_id):
    """ Returns a transition object. Throws an error if no such transition exists.
    """
    return self.transitions_by_id[(from_node_id, to_node_id)]
  
  def node(self, node_id):
    return self.nodes_by_id[node_id]

  def _varDistribution(self, current_var_id, node_ids):
    """ Returns the distribution over list of variables for this sequences of modes.
    
    Returns a list of ([varID], weight)
    """
    if not node_ids:
      return [([current_var_id], 1.0)]
    last_node_id = current_var_id.nodeId
    last_mode = current_var_id.mode
    to_mode_id = node_ids[0]
    return [([current_var_id] + new_var_ids, w * next_w) 
            for (next_var_id, next_w) in self.conditionalDistribution(last_node_id, last_mode, to_mode_id) 
            for (new_var_ids, w) in self._varDistribution(next_var_id, node_ids[1:])]
  
  def _prob_trans(self, previous_var_id, var_ids):
    if not var_ids:
      return 1.0
    from_node_id = previous_var_id.nodeId
    to_node_id = var_ids[0].nodeId
    trans = self.transition(from_node_id, to_node_id)
    start_node = self.node(from_node_id)
    end_node = self.node(to_node_id)
    start_idx = start_node.modeIndexes[previous_var_id.mode]
    end_idx = end_node.modeIndexes[var_ids[0].mode]
    w = trans.transitions[end_idx, start_idx]
    return w * self._prob_trans(var_ids[0], var_ids[1:])
  
  def probability(self, var_ids):
    """ The probability of a sequence of var ids.
    
    Returns:
    a float
    """
    # The start probability:
    start_vid = var_ids[0]
    start_node = self.node(start_vid.nodeId)
    start_idx = start_node.modeIndexes[start_vid.mode]
    w = start_node.modeProbs[start_idx]
    return w * self._prob_trans(start_vid, var_ids[1:])
    
  
  def modeDistributionFromRoute(self, node_ids):
    """ The probability distribution of all the mode sequences that follow
    a sequence of links.
    
    Arguments:
    node_ids -- a list of node id
    
    Returns:
    A list of pairs of list of mode ids, probability weights.
    """
    return [([var_id.mode for var_id in var_ids], w) for (var_ids, w) in self.variableDistributionFromRoute(node_ids)]
  
  def variableDistributionFromRoute(self, node_ids):
    """ The complete probability distribution from all the sequences of variables
    that compose this sequences of nodes.
    """
    node_id = node_ids[0]
    start_varids = zip([VariableId(node_id, mode) for mode in self.node(node_id).modes],
                       self.node(node_id).modeProbs)
    return [(varids, w*start_w) for (start_varid, start_w) in start_varids 
            for (varids, w) in self._varDistribution(start_varid, node_ids[1:])]
  
  def _sampleVars(self, current_var_id, node_ids):
    if not node_ids:
      return []
    dis = self.conditionalDistribution(current_var_id.nodeId, current_var_id.mode, node_ids[0])
    probs = np.array([w for (_,w) in dis])
    #mt = np.argmax(np.random.multinomial(1, probs))
    # Probably more efficient way of doing it
    mt = probs.cumsum().searchsorted(np.random.random())
    (next_var_id,_) = dis[mt]
    return [next_var_id] + self._sampleVars(next_var_id, node_ids[1:])
  
  def sampleVariablesFromNodes(self, node_ids):
    start_node_id = node_ids[0]
    start_node = self.node(start_node_id)
    midx = np.argmax(np.random.multinomial(1, start_node.modeProbs))
    mode = start_node.modes[midx]
    start_var_id = VariableId(start_node_id, mode)
    return [start_var_id] + self._sampleVars(start_var_id, node_ids[1:])
  
  def sampleModes(self, node_ids):
    return [var_id.mode for var_id in self.sampleVariablesFromNodes(node_ids)]
