
'''
Created on Dec 28, 2012

@author: tjhunter

TODO: separate the part that relates to the mixture stop/go
'''
# pylint:disable=W0511
# --> todos
import datetime
import numpy as np
from joblib import Memory
from mm.data.viterbi import list_traj_filenames, read_trajectory
from collections import defaultdict
import mm.arterial_hkt.simple as simple_model
from mm.arterial_hkt.tt_graph_functions import createTravelTimeGraph, \
  fillTTGraph
from mm.arterial_hkt.mixture_functions import learnMixtureAuto
from mm.data.structures import Spot, TSpot
from mm.arterial_hkt.gmrf_estimator import DiagonalGMRFEstimator, \
  JLGMRFEstimator, ExactGMRFEstimator
from mm.arterial_hkt.pipeline_io import save_gmrf_estimator_DiagonalGMRFEstimator, \
  save_gmrf_estimator_JLGMRFEstimator, save_gmrf_estimator_ExactGMRFEstimator, \
  save_gmrf_values
from mm.arterial_hkt.gmrf_learning.quic_cpp.low_rank import random_projection_cholmod_csc
from mm.arterial_hkt.gmrf_functions import independantValues, learnQuic
from mm.arterial_hkt.gmrf_learning.utils import build_sparse
from mm.arterial_hkt.gmrf_learning.cvx import \
  gmrf_learn_cov_cvx, gmrf_learn_cov_cholmod
from mm.arterial_hkt.gmrf import GMRF
from mm.arterial_hkt.utils import tic
from joblib import Parallel, delayed

# TODO(tjh) make it independant from my path.
pipeline_memory = Memory("/tmp", verbose=1)

class TTObservation(object):
  """ A passage of a vehicle on a link, with some travel time information.
  """

  def __init__(self, start_time, end_time, link_id):
    self.start_time = start_time
    self.end_time = end_time
    self.linkId = link_id
    self.tt = (end_time - start_time).total_seconds()

  def __repr__(self):
    return "TTObs[%s, %s->%s]" % (str(self.linkId), str(self.start_time), str(self.end_time))


def seqGroupBy(l, keyf):
  def takeWhile(l, key):
    idx = 0
    while idx < len(l) and keyf(l[idx]) == key:
      idx += 1
    return (l[:idx], l[idx:])
  if not l:
    return []
  head = l[0]
  (group, reminder) = takeWhile(l, keyf(head))
  return [group] + seqGroupBy(reminder, keyf)

def consume_all_links(link_id, tspots):
  if not tspots:
    return ([], [])
  current_link_id = tspots[0].spot.linkId
  if current_link_id != link_id:
    return ([], tspots)
  else:
    (next_spots, next_remaining_spots) = consume_all_links(link_id, tspots[1:])
    return ([tspots[0]] + next_spots, next_remaining_spots)

def convertTrajToTTObservation(tspots):
  """ Takes a trajectory and convert it to simple travel time information.
  """
  # Just use the spots, and hope they are dense enough to cover all the links in between.
  def processor(tspots):
    if not tspots:
      return []
    else:
      link_id = tspots[0].spot.linkId
      (current_group, reminder) = consume_all_links(link_id, tspots)
      return [current_group] + processor(reminder)
  return processor(tspots)


def completeTSpots(tspots, net):
  """ Complete a list of TSpot by finding the time at which
  the veh entered and exited each link
  """
  groups = seqGroupBy(tspots, keyf=lambda tsp:tsp.spot.linkId)
  veh_id = tspots[0].id
  c_tspots = []
  res_tspots =[]
  def tspotFromInterpolation(interpolated_spot, veh_id):
    (lid, offset, time) = interpolated_spot
    return TSpot(Spot(lid, offset), veh_id, time)
  for i in range(len(groups) - 1):
    c_tspots += groups[i]
    interp_spot = interpolateTSpots(groups[i][-1], groups[i + 1][0], net)
    if len(interp_spot) == 0:
      res_tspots.append(c_tspots)
      c_tspots = []
    else:
      c_tspots += [tspotFromInterpolation(interp, veh_id) for interp in interp_spot]
  c_tspots += groups[-1]
  res_tspots.append(c_tspots)
  return res_tspots

  
def interpolateTSpots(previous_tsp, next_tsp, net):
  previous_link_id = previous_tsp.spot.linkId
  previous_link = net[previous_link_id]
  next_link_id = next_tsp.spot.linkId
  next_link = net[next_link_id]
  dt = (next_tsp.time - previous_tsp.time).total_seconds()
  assert dt > 0
  remaining_previous = previous_link.length - previous_tsp.spot.offset
  if remaining_previous < 1e-1:
    remaining_previous = 1e-1
  remaining_next = next_tsp.spot.offset
  if remaining_next < 1e-1:
    remaining_next = 1e-1
  if previous_link_id in next_link.incomingLinks:
    assert next_link_id in previous_link.outgoingLinks
    theta = remaining_previous / (remaining_next + remaining_previous)
    middle_time = previous_tsp.time + datetime.timedelta(seconds=theta * dt)
    assert middle_time <= next_tsp.time
    assert middle_time >= previous_tsp.time
    return [(previous_link_id, round(previous_link.length, 2), middle_time), (next_link_id, 0.0, middle_time)]
  # Check if there is a single obvious link that connects the two
  s = set(previous_link.outgoingLinks).intersection(next_link.incomingLinks)
  if len(s) == 1:
    middle_link_id = s.pop()
    middle_link = net[middle_link_id]
    theta1 = (remaining_previous) / (remaining_next + middle_link.length + remaining_previous)
    theta2 = (middle_link.length + remaining_previous) / (remaining_next + middle_link.length + remaining_previous)
    middle_time1 = previous_tsp.time + datetime.timedelta(seconds=theta1 * dt)
    middle_time2 = previous_tsp.time + datetime.timedelta(seconds=theta2 * dt)
    assert middle_time1 <= next_tsp.time
    assert middle_time2 >= previous_tsp.time
    return [(previous_link_id, round(previous_link.length, 2), middle_time1),
            (middle_link_id, 0.0, middle_time1),
            (middle_link_id, round(middle_link.length, 2), middle_time2),
             (next_link_id, 0.0, middle_time2)]
  # Completely disconnected.
  return []

def convertGroupToTTObservations(g):
  return [TTObservation(t1, t2, link_id1) for ((link_id1, _, t1), (_, _, t2)) in g]

# See if we need to add some data between the links to get a connected sequence of links.
def completeGroups(groups, net):
  """ Converts a group of tspots into a sequence of observations.
  """
  # Drop all the intermediate points
  lean_groups = [[g[0], g[-1]] for g in groups]
  g_pairs = zip(groups[:-1], lean_groups[1:])
  # TODO(tjh) explain what this code does,it is really dense.
  interp_groups = [x for (g1 , g2) in g_pairs for x in interpolateTSpots(previous_tsp=g1[-1], next_tsp=g2[0], net=net)]
  # Regroup/split by link
  regrouped = seqGroupBy(interp_groups, keyf=lambda z:z[0])
  # Split where we failed:
  # At least 2 subsequent links
  regrouped_no_breaks = [l for l in seqGroupBy(regrouped, keyf=len) if len(l[0]) > 1 and len(l) >= 2]
  obs = [convertGroupToTTObservations(g) for g in regrouped_no_breaks]
  return obs

def groupby(it, keyfunc):
  x = defaultdict(list)
  for z in it:
    x[keyfunc(z)].append(z)
  return x.items()

def filterOutsideNetwork(tspots, network):
  """ Returns list of list of tspots, all inside network.
  """
  groups = seqGroupBy(tspots, keyf=lambda tsp:tsp.spot.linkId in network)
  return [group for group in groups if group[0].spot.linkId in network]

# pylint:disable=W0613
@pipeline_memory.cache(ignore=['network'])
def getRawData(feed, nid, date, net_type, box, network):
  all_trajs = list(viterbi_list_traj_filenames(feed, nid, date, net_type))
  all_groups = []
  for traj_index in all_trajs:
    # pylint:disable=W0142
    (tspots, _) = read_trajectory(*traj_index)
    # Make sure we only have data for our sub network.
    for net_tspots in filterOutsideNetwork(tspots, network):
      all_groups.append(net_tspots)
  return all_groups

@pipeline_memory.cache(ignore=['traj_conv','network'])
def getDayTrajs(feed, nid, date, net_type, box, traj_conv_description,
                traj_conv, network):
  return [traj_ob for tspots_seq in getDayTSpots(feed, nid, date, net_type, box, network)
                  for traj_ob in traj_conv.mapTrajectory(tspots_seq)]
  

@pipeline_memory.cache(ignore=['network'])
def getDayTSpots(date, network):
  """ Returns a list of sequences of TSpot objects for this day.
  """
  all_traj_fns = list_traj_filenames(date)
  tspots_groups = []
  for fname in all_traj_fns:
    # pylint:disable=W0142
    try:
      tspots = read_trajectory(fname)
    except IOError:
      tic("Could not read trajectory: {0}".format(fname), "getDayTSpots")
      tspots = []
    # tspots is a list of TSpot
    # Make sure we only have data for our sub network.
    for net_tspots in filterOutsideNetwork(tspots, network):
      tspots_groups.append(net_tspots)
  return tspots_groups




@pipeline_memory.cache(ignore=['network'])
def getDayTTOBservations(date, network, print_stats=1000):
  """
  """
  all_trajs = list_traj_filenames(date)
  all_groups = []
  idx = 1
  for traj_index in all_trajs:
    idx += 1
    if print_stats > 0 and idx % print_stats == 0:
      tic("processed {0} observations".format(idx),"getDayTTOBservations")
    # pylint:disable=W0142
    try:
      tspots = read_trajectory(traj_index)
    except IOError:
      tic("ioerror when loading a trajectory {0}".format(traj_index), "getDayTTOBservations")
      tspots = []
    # Make sure we only have data for our sub network.
    for net_tspots in filterOutsideNetwork(tspots, network):
      groups = seqGroupBy(net_tspots, keyf=lambda tsp:tsp.spot.linkId)
      for g in completeGroups(groups, network):
        all_groups.append(g)
  return all_groups

def getMixtures_inner(args):
  (lid,tts,max_nb_mixture) = args
  return (lid, learnMixtureAuto(tts, max_nb_mixture))
  

@pipeline_memory.cache(ignore=['network',"num_threads"])
def getMixtures(dates, network,
                max_n_links=None, return_tts=False, max_nb_mixture=4,num_threads=1):
  tic("Running with {0} jobs.".format(num_threads),"getMixtures")
  ttob_seqs = (ttob_seq for date in dates
               for ttob_seq in getDayTTOBservations(date, network))
  # Get travel times for each link
  all_ttobs = (ttob for ttob_seq in ttob_seqs for ttob in ttob_seq)
  tic("starting groupby...","getMixtures")
  all_ttobs_by_lid = sorted([(lid, list(vals)) for (lid, vals) in groupby(all_ttobs, lambda ttob:ttob.linkId)], key=lambda z:-len(z[1]))
  tic("groupby done, {0} links".format(len(all_ttobs_by_lid)),"getMixtures")
  if max_n_links:
    all_ttobs_by_lid = all_ttobs_by_lid[:max_n_links]
  tts_by_link = [(lid, np.array([tto.tt for tto in vals])) for (lid, vals) in all_ttobs_by_lid]
  tic("vectorization done","getMixtures")
  if num_threads != 1:
    tic("Running with {0} jobs.".format(num_threads),"getMixtures")
    learned_mixtures = Parallel(n_jobs=num_threads,verbose=10)(delayed(getMixtures_inner)((lid, tts, max_nb_mixture)) for (lid, tts) in tts_by_link)
  else:
    learned_mixtures = [(lid, learnMixtureAuto(tts, max_nb_mixture)) for (lid, tts) in tts_by_link]
  if return_tts:
    return (dict(learned_mixtures), dict(tts_by_link))
  else:
    return dict(learned_mixtures)

class TrajectoryBuilder(object):
  """
  Attributes:
  network -- the network used to build this mapper.
  mode_counts -- dictionary node_id -> integer: the number of modes for each node.
  """

  def map(self, tspots):
    """ Transforms a sequence of TSpots into one or more Trajectory Observations.

    Returns a list of trajectory observations.
    """
    pass

class SimpleMixtureTrajectoryBuilder(TrajectoryBuilder):

  def __init__(self, network, learned_mixtures):
    TrajectoryBuilder.__init__(self)
    self.network = network
    self.learned_mixture = learned_mixtures
    self.mode_counts = dict([(link_id, len(mix.means))
                             for (link_id, mix) in learned_mixtures.items()])

  def ttObservationToTrajObservation(self, ttob_seq):
    # pylint:disable=E1101
    seq = [(ttob.linkId, self.learned_mixtures[ttob.linkId].assignment(ttob.tt), float(ttob.tt)) for ttob in ttob_seq]
    return simple_model.toTrajectoryObservation(seq)

  def map(self, tspots):
    groups = seqGroupBy(tspots, keyf=lambda tsp:tsp.spot.linkId)
    ttob_seqs = completeGroups(groups, self.network)
    traj_obs = [self.ttObservationToTrajObservation(ttob_seq) for ttob_seq in ttob_seqs]
    return traj_obs

def load_traj_obs_mapper_mixture_auto(network, graph_model,
                                      mixture_params, data_source, basic_geometry):
  assert graph_model == 'simple'
  args = data_source.copy()
  args.update(basic_geometry)
  args.update(mixture_params)
  args['network'] = network
  args['return_tts'] = False
  args['max_n_links'] = None
  # pylint:disable=W0142
  learned_mixtures = getMixtures(**args)
  return SimpleMixtureTrajectoryBuilder(network, learned_mixtures)


@pipeline_memory.cache(ignore=['network'])
def getTrajectoryObservations(builder, feed, nid, dates, net_type, box, network):
  return [traj_ob for date in dates
          for tspots in getRawData(feed, nid, date, net_type, box, network)
          for traj_ob in builder.map(tspots)]

@pipeline_memory.cache
def createFillTTGraph(hmm_graph, traj_observations):
  tt_graph = createTravelTimeGraph(hmm_graph, 1e-3)
  fillTTGraph(tt_graph, traj_observations)
  return tt_graph

  
def gmrf_est(gmrf, process, experiment_name=None, gmrf_estimation_parameters=None):
  ''' Saves and creates a gmfr estimator from a GMRF.
  
  Arguments:
  gmrf: a GMRF object
  process: the name of the process (diagonal, jl, exact)
  experiment_name -- (string), the name of the experiment
  gmrf_estimation_parameters --  (dict) the parameters, if any. depends on process
  
  Returns:
  a GMRFEstimator object
  
  Side effects:
  May save some information, if experiment_name is providide
  '''
  if process == 'diagonal':
    diag_variance = 1.0 / gmrf.diag_precision
    gmrf_estimator = DiagonalGMRFEstimator(gmrf.translations, gmrf.means, diag_variance)
    if experiment_name is not None:
      save_gmrf_estimator_DiagonalGMRFEstimator(gmrf_estimator, experiment_name=experiment_name)
  elif process == 'jl':
    precision = gmrf.precision
    k = gmrf_estimation_parameters['k']
    Q = random_projection_cholmod_csc(precision, k)
    print "Q shape",Q.shape
    gmrf_estimator = JLGMRFEstimator(gmrf.translations, gmrf.means, Q)
    if experiment_name is not None:
      save_gmrf_estimator_JLGMRFEstimator(gmrf_estimator, experiment_name=experiment_name)
  elif process == 'exact':
    X = gmrf.precision.todense()
    covariance = np.linalg.inv(X)
    gmrf_estimator = ExactGMRFEstimator(gmrf.translations, gmrf.means, covariance)
    if experiment_name is not None:
      save_gmrf_estimator_ExactGMRFEstimator(gmrf_estimator, experiment_name=experiment_name)
  else:
    assert False
  return gmrf_estimator

def independent_variables(n, rows, cols):
  """ Returns a mask that contains the independent variables.
  """
  m = len(rows)
  X= build_sparse(np.ones(n), np.ones(m), rows, cols)
  return X.sum(axis=1)==1

def gmrf_learn(tt_graph, process, experiment_name=None, gmrf_learning_params={}):
  """ Runs the gmrf learning procedure.
  
  If experiment_name is provided, will also save the values of the learned gmrf.
  
  Returns:
  a GMRF object
  """
  if process == 'independent':
    gmrf = independantValues(tt_graph)
  elif process == 'quic':
    lbda = gmrf_learning_params['lbda']
    gmrf = learnQuic(tt_graph, lbda=lbda)
  elif process == 'cvx':
    means = tt_graph.means()
    rows = tt_graph.rows()
    cols = tt_graph.cols()
    R = tt_graph.variances()
    U = tt_graph.covariances()
    # Only consider the edges that have sufficient observations.
    edge_count = tt_graph.edgeCounts()
    (D,P) = gmrf_learn_cov_cvx(R, U, rows, cols, edge_count, **gmrf_learning_params)
    gmrf = GMRF(tt_graph.indexes, tt_graph.rows(), tt_graph.cols(),
              means, D, P)
  elif process == 'jl':
    means = tt_graph.means()
    rows = tt_graph.rows()
    cols = tt_graph.cols()
    R = tt_graph.variances()
    U = tt_graph.covariances()
    # Only consider the edges that have sufficient observations.
    edge_count = tt_graph.edgeCounts()
    (D,P) = gmrf_learn_cov_cholmod(R, U, rows, cols, edge_count, **gmrf_learning_params)
    gmrf = GMRF(tt_graph.indexes, tt_graph.rows(), tt_graph.cols(),
              means, D, P)
  elif True:
    assert False
  if experiment_name is not None:
    save_gmrf_values(gmrf, experiment_name)
  return gmrf
