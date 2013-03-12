'''
Created on Dec 26, 2012

@author: tjhunter

Tools for the simple data model (a node == a link).
See bi for the direction-aware data model.
'''
from mm.arterial_hkt.hmm_functions import createHMMGraph
from mm.arterial_hkt.variable import VariableId
from mm.arterial_hkt.observation import Observation, TrajectoryObservation
from mm.arterial_hkt.tt_graph_functions import getCenter

all_modes = ["go", "stop", "stop-stop","stop3"]

def nodeDescriptionFromNetwork(net, num_modes=2, mode_counts=None):
  """ A description of the nodes that is suitable for instantiating a HMM graph and a TTGraph.
  
  Argument:
  net -- a dict of linkid -> link
  
  returns a list of (node_id, list of modes)
  """
  # Make sure to sort the keys to have a deterministic answer.
  all_link_ids = sorted(net.keys())
  if mode_counts is None:
    modes = all_modes[:num_modes]
    return [(link_id, modes) for link_id in all_link_ids]
  else:
    # A bit too short...
    return [(link_id, all_modes[:(mode_counts[link_id] if link_id in mode_counts else num_modes)])
            for link_id in all_link_ids]

def createHMMGraphFromNetwork(net, num_modes=2, mode_counts=None):
  """ Creates a HMM graph from a network.
  Returns a HMM graph
  """
  node_descriptions = nodeDescriptionFromNetwork(net, num_modes, mode_counts)
  node_relations = [(from_link_id, to_link_id) 
                    for (from_link_id, _) in node_descriptions
                    for to_link_id in net[from_link_id].outgoingLinks]
  hmm_graph = createHMMGraph(node_descriptions, node_relations)
  for node in hmm_graph.allNodes():
    link = net[node.nodeID]
    node.location = getCenter(link)
  return hmm_graph

def toTrajectoryObservation(seq):
  """ seq -- a list of tuples (linkID, mode_index, tt)
  """
  obs = [Observation(VariableId(link_id, all_modes[mode_index]), tt) for (link_id, mode_index, tt) in seq]
  return TrajectoryObservation(obs)

def toNodesSequence(link_ids):
  return link_ids