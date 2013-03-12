'''
Created on Jul 20, 2012

@author: tjhunter

Collection of functions related to the HMM.
'''
from collections import defaultdict
from mm.arterial_hkt.hmm import HMMNode, HMMTransition, HMMGraph
import numpy as np

def createHMMGraph(node_descriptions, node_relations):
  """ Instantiates a HMM graph with uniform probabilities, based on the description provided in the
  arguments.
  
  Arguments:
  node_description -- a list of (node_id, list of modes)
  node_relations -- a list of (from_node_id, to_node_id)
  
  Returns:
  a HMM graph
  """
  def createHMMNode(node_id, modes):
    prob_modes = np.ones(len(modes), dtype=np.double)
    prob_modes /= np.sum(prob_modes)
    return HMMNode(node_id, prob_modes, modes)
  nodes = [createHMMNode(node_id, modes) for (node_id, modes) in node_descriptions]
  nodes_by_id = dict([(node.nodeID, node) for node in nodes])
  def createHMMTransition(from_node_id, to_node_id):
    from_n_nodes = nodes_by_id[from_node_id].numModes
    to_n_nodes = nodes_by_id[to_node_id].numModes
    transitions = np.ones((to_n_nodes, from_n_nodes), dtype=np.double)
    transitions /= np.sum(transitions, axis=0)
    return HMMTransition(from_node_id, to_node_id, transitions)
  hmm_transitions = [createHMMTransition(from_node_id, to_node_id) for (from_node_id, to_node_id) in node_relations]
  return HMMGraph(nodes, hmm_transitions)

def fillProbabilitiesUniform(hmm_graph, skew=0.0):
  for node in hmm_graph.allNodes():
    prob_modes = np.arange(1, node.numModes + 1, dtype=np.double) ** skew
    prob_modes /= np.sum(prob_modes)
    node.modeProbs = prob_modes
    del node
  for trans in hmm_graph.allTransitions():
    (n_to_modes, n_from_modes) = trans.transitions.shape
    prob_modes = np.outer(np.arange(1, n_to_modes + 1) ** skew, np.ones(n_from_modes, dtype=np.double))
    prob_modes /= np.sum(prob_modes, axis=0)
    trans.transitions = prob_modes
    del trans


def fillProbabilitiesObservations(hmm_graph, var_seqs, smoothing_count=1e-4, smoothing_trans_count=1e-3):
  """ Computes the HMM graph using maximum likelihood on some observations.
  
  This function only compputes first order markov chains (should be updated to
  second order later).
  
  Arguments:
  hmm_graph -- a HMM graph
  var_seqs - an iterable of lists of variables
  smoothing_count - initial count for the uniform probability
  smoothing_trans_count -- smoothing values for the transitions (uniform probability)
  """
  # Build sufficient statistics from the observations
  # Sufficient statistics for first-order HMM is a dictionary of (mode_id1, mode_id2) -> count
  # TODO(tjh) explain the rationale for this data structure.
  sstats0 = defaultdict(lambda :defaultdict(int))
  sstats1 = defaultdict(lambda :defaultdict(int))
  for var_seq in var_seqs:
    # Check if the sequence of variables is bound to this network
    for var_id in var_seq:
      if var_id.nodeId not in hmm_graph.nodes_by_id:
        print("Warning: cannot fit %s to hmm graph" % (str(var_id)))
      del var_id
    # Only use the start values for the sufficient statistics.
    # Otherwise, it will be biased when the process is non-stationary.
    sstats0[var_seq[0].nodeId][var_seq[0].mode] += 1
    for (from_varid, to_varid) in zip(var_seq[:-1], var_seq[1:]):
      sstats1[(from_varid.nodeId, to_varid.nodeId)][(from_varid.mode, to_varid.mode)] += 1
    del var_seq
#  print sstats0
#  print sstats1
    # TODO(?) add 2nd order and maybe third order later
  # Update the HMM graph
  # The HMM nodes first
  fillProbabilitiesUniform(hmm_graph)
  for node in hmm_graph.allNodes():
    # Build new probabilities
    new_probs = float(smoothing_count) * np.ones_like(node.modeProbs)
    for (mode, count) in sstats0[node.nodeID].items():
      assert mode in node.modeIndexes, "mode %s does not exist for node %s" % (str(mode), str(node))
      idx = node.modeIndexes[mode]
      new_probs[idx] += count
      del mode, count, idx
    node.modeProbs = new_probs / np.sum(new_probs)
    del new_probs
    del node
  # The transitions second
  for trans in hmm_graph.allTransitions():
    new_trans = float(smoothing_trans_count) * np.ones_like(trans.transitions)
    from_node = hmm_graph.node(trans.fromNodeID)
    to_node = hmm_graph.node(trans.toNodeID)
    for ((from_mode, to_mode), count) in sstats1[(trans.fromNodeID, trans.toNodeID)].items():
      assert from_mode in from_node.modeIndexes, "from mode %s does not exist for node %s" % (str(from_mode), str(from_node))
      assert to_mode in to_node.modeIndexes, "from mode %s does not exist for node %s" % (str(to_mode), str(to_node))
      from_idx = from_node.modeIndexes[from_mode]
      to_idx = to_node.modeIndexes[to_mode]
      new_trans[to_idx][from_idx] += count
    trans.transitions = new_trans / np.sum(new_trans, axis=0)
    del new_trans, from_node, to_node, trans
  return hmm_graph

def drawHMMGraph(ax, hmm_graph, node_style=None, link_style=None):
  if link_style is not None:
    lats = []
    lons = []
    lats_ = []
    lons_ = []
    dlats = []
    dlons = []
    for (node1_id, node2_id) in hmm_graph.allTransitions():
      c1 = hmm_graph.node(node1_id).location
      c2 = hmm_graph.node(node2_id).location
      lats_.append(c1.lat)
      lons_.append(c1.lon)
      dlats.append(c2.lat - c1.lat)
      dlons.append(c2.lon - c1.lon)
      lats.append([c1.lat, c2.lat])
      lons.append([c1.lon, c2.lon])
    # pylint: disable=W0142
    ax.quiver(np.array(lons_), np.array(lats_), np.array(dlons), np.array(dlats), scale_units='xy', angles='xy', scale=1, **link_style)
#    ax.plot(np.array(lons).T, np.array(lats).T, zorder=10, **link_style)

  if node_style is not None:  
    lats = [node.location.lat for node in hmm_graph.allNodes()]
    lons = [node.location.lon for node in hmm_graph.allNodes()]
    # pylint: disable=W0142
    ax.scatter(lons, lats, zorder=11, **node_style)
