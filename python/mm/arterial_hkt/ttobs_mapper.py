'''
Created on Feb 4, 2013

@author: tjhunter

Travel time observation mapper.
'''
from mm.arterial_hkt.pipeline_functions import seqGroupBy, completeGroups, \
  getMixtures, completeTSpots
import mm.arterial_hkt.simple as model
from mm.arterial_hkt.stop_go_filter import detect_stops
from mm.arterial_hkt.mixture import GMixture
import math


class TTObsMapper(object):
  """ Defines an interface to transform an observation into a
  TrajectoryObservation; which is the only input for the algorithm.
  """
  
  def graphType(self):
    """ The type of the graph (simple or complex).
    """
    pass
  
  def modeCounts(self):
    """ A dictionary (node id -> int)
    Return the number of mode for each of the node.
    For simple networks, node is link id
    For complex networks, node is (link id, outgoing link id) 
    """
    pass
  
  def mapTrajectory(self, tspots, **param):
    """ Returns a list of TrajectoryObservation (can be empty)
    """
    pass



class SimpleAutoTTObsMapper(TTObsMapper):
  """
  """
  
  def __init__(self, learned_mixtures, network):
    """
    """
    self.network = network
    self.learned_mixtures = learned_mixtures
    self.mode_counts = dict([(link_id, len(mix.means)) 
                             for (link_id, mix) in learned_mixtures.items()])
  
  def graphType(self):
    return 'simple'
  
  def modeCounts(self):
    return self.mode_counts
  
  def mapTrajectory(self, tspots, **param):
    groups = seqGroupBy(tspots, keyf=lambda tsp:tsp.spot.linkId)
    ttob_seqs = completeGroups(groups, self.network)
    seqs = [[(ttob.linkId, self.learned_mixtures[ttob.linkId].assignment(ttob.tt),
              float(ttob.tt))
             for ttob in ttob_seq] for ttob_seq in ttob_seqs]
    return [model.toTrajectoryObservation(seq) for seq in seqs]

class LassoTTObsMapper(TTObsMapper):
  """
  """
  
  def __init__(self, network, default_params={}):
    """
    default_params: dictionary of parameters.
    """
    self.network = network
    self.mode_counts = dict([(link_id, 2) for link_id in network])
    self.default_params=default_params
  
  def graphType(self):
    return 'simple'
  
  def modeCounts(self):
    return self.mode_counts
  
  def mapTrajectory(self, tspots, **param):
    # Creating an update of all the parameters, if necessary.
    full_params = dict(self.default_params.items()+param.items())
    c_tspots = completeTSpots (tspots, self.network)
    seqs = [detect_stops.detect_mode(c_tspot, self.network, **full_params)
            for c_tspot in c_tspots]
    return [model.toTrajectoryObservation(seq) for seq in seqs if seq is not None]


def createTrajectoryConversion(graph_type, process, params, network, max_nb_mixture,n_jobs=1):
  if process == 'mixture_auto':
    assert graph_type == 'simple'
    dates = params['train_data']['dates']
    learned_mixtures = getMixtures(dates,
                                    network=network,
                                    max_n_links=None,
                                    return_tts=False,
                                    max_nb_mixture=max_nb_mixture,
                                    num_threads=n_jobs)
    learned_mixtures = createDefaultMixtureForLinkWithoutData(learned_mixtures, network, **params)
#    import ipdb; ipdb.set_trace();
    return SimpleAutoTTObsMapper(learned_mixtures, network)
  if process == 'mixture_lasso':
    assert graph_type == 'simple'
    dates = params['train_data']['dates']
    learned_mixtures = detect_stops.detect_stops(dates,
                                              network=network, **params)
    learned_mixtures = createDefaultMixtureForLinkWithoutData(learned_mixtures, network, **params)
    return SimpleAutoTTObsMapper(learned_mixtures, network)
  if process == 'lasso':
    assert graph_type == 'simple'
    return LassoTTObsMapper(network,params)
  assert False


def createDefaultMixtureForLinkWithoutData(learned_mixtures, network, **param):
  """ Fill in the learned mixtures for the link without data
  """
  default_speed = param['default_speed']
  avg_delay = param['avg_delay']
  non_stopping_default = param['non_stopping_default']
  default_var = param['default_variance']
  for lid in network:
    if lid not in learned_mixtures:
      llength = network[lid].length
      learned_mixtures[lid] = defaultMixture(llength,
                                             default_speed,
                                             avg_delay,
                                             non_stopping_default,
                                             default_var)
  return learned_mixtures
    
  
def defaultMixture(link_length,
                   default_speed,
                   avg_delay,
                   non_stopping_default,
                   default_var):
  """ Create a two component Gaussian mixtures for the link
  with specified length. 
  The first componenent assumes default_speed, default_var
  The second component assumes 
      default_speed for ff tt + avg_delay, std is half of avg delay
  The weight of first component is non_stopping_default
  """
  avg_tt = link_length / default_speed
  return GMixture([non_stopping_default, 1 - non_stopping_default],
           [avg_tt, avg_tt + avg_delay],
           [default_var, math.pow(avg_delay / 4.0, 2)])
  
  
  
