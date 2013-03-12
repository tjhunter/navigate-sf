""" TT graph functions
"""
import numpy as np
import datetime
from mm.data.structures import Coordinate, Route, Spot, RouteTT, CutTrajectory
from mm.arterial_hkt.tt_graph import Variable, TravelTimeGraph, \
  GaussianParameters
from mm.arterial_hkt.observation import Observation, TrajectoryObservation
from collections import defaultdict
from scipy.ndimage.interpolation import rotate
from mm.arterial_hkt.variable import VariableId
from mm.arterial_hkt.utils import tic

def createSplittingOffsets(net):
  """creates a dictionary of offsets with keys the link ids and value the 
  offset in meters
  
  net: a dictionary of link_id -> link object
  
  Returns a dictionary of link_id -> offset on the link.
  """
  return dict([ (link_id, min(0.25 * link.length, 6)) for (link_id, link) in net.iteritems()])

# Compute some centers for each link to be able to plot them:
def getCenter(link):
  """ Reasonable center based on link geometry.
  """
  if link.geom is None:
    return None
  mean_lat = np.mean([c.lat for c in link.geom])
  mean_lon = np.mean([c.lon for c in link.geom])
  return Coordinate(mean_lat, mean_lon)

def computeMultiCentersDirected(c1, c2, num_centers, center_relative_position=0.5, center_relative_radius=0.1):
  """ Computes a set of points on a circle predetermined.
  
  Arguments
  c1 -- start coordinate
  c2 -- end coordinate
  num_centers -- number of centers. 1 will will return the center
  """
  dlat = c2.lat - c1.lat
  dlon = c2.lon - c1.lon
  latA = c1.lat + center_relative_position * dlat
  lonA = c1.lon + center_relative_position * dlon
  if num_centers == 1:
    return [Coordinate(latA, lonA)]
  v = np.array([dlat, dlon])
#  print 'v',v
  x = np.array([latA, lonA])
#  print 'x',x
  def createCenter(i):
    theta = ((1.0 + i * 2) * np.pi) / num_centers
    R = np.array([(np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))])
    v2 = R.dot(v)
#    print "v2",v2
    x2 = x + center_relative_radius * v2
#    print "x2",x2
    lat = x2[0]
    lon = x2[1]
    return Coordinate(lat, lon)
  return [createCenter(i) for i in range(num_centers)]

def computeMultiCenters(c, num_centers, radius):
  """ Computes a set of points on a circle predetermined.
  
  Arguments
  c1 -- start coordinate
  c2 -- end coordinate
  num_centers -- number of centers. 1 will will return the center
  """
  if c is None:
    return [None for i in range(num_centers)]
  c2 = Coordinate(c.lat + 1, c.lon)
  return computeMultiCentersDirected(c, c2, num_centers, center_relative_position=0.0, center_relative_radius=radius)

def createTravelTimeGraph(hmm_graph, radius):
  """ Creates the graph of travel times, that will hold the sufficient statistics.
  
  Arguments:
  hmm_graph -- an HMM graph
  radius -- float, for creating coordinates to represent the graph
  
  Returns a TravelTimeGraph object
  """
  # Instantiate the variables
  def createVariable(node):
    cs = computeMultiCenters(node.location, max(3, node.numModes), radius)
    var_ids = [VariableId(node.nodeID, mode) for mode in node.modes]
    return [(var_id, Variable(var_id, [], location=c, param=GaussianParameters(0.0, 0.0))) for (var_id, c) in zip(var_ids, cs)]
    
  variables = dict([pair for node in hmm_graph.allNodes() 
                for pair in createVariable(node)])
  # Add connectivity
  edges = []
  for trans in hmm_graph.allTransitions():
    from_node = hmm_graph.node(trans.fromNodeID)
    to_node = hmm_graph.node(trans.toNodeID)
    for from_mode in from_node.modes:
      for to_mode in to_node.modes:
        from_var_id = VariableId(from_node.nodeID, from_mode)
        to_var_id = VariableId(to_node.nodeID, to_mode)
        variables[to_var_id].incomingIDs.append(from_var_id)
        variables[from_var_id].outgoingIDs.append(to_var_id)
        edges.append((from_var_id, to_var_id))
  return TravelTimeGraph(variables.values(), edges)


def preprocessCutTraj(cut_traj, net):
  """preprocess cut_traj by removing the first link of cut_traj
  
  TODO(jth) add a bit more documentation
  """
  rtts = []
  """
  index = 1
  if len(cut_traj.pieces[0].route.link_ids) > 1:
    piece = cut_traj.pieces[0]
    TTs = travelTimeByLink(piece, net)
    route = Route(piece.route.link_ids[1:], [Spot(piece.route.link_ids[1], 0), piece.route.lastSpot])
    rtts.append(RouteTT(route, piece.startTime + datetime.timedelta(0,TTs[0]), piece.endTime))
  
  if cut_traj.pieces[0].route.firstSpot == cut_traj.pieces[0].route.lastSpot:
    index = 2
    piece = cut_traj.pieces[1]
    TTs = travelTimeByLink(piece, net)
    route = Route(piece.route.link_ids[1:], [Spot(piece.route.link_ids[1], 0), piece.route.lastSpot])
    rtts.append(RouteTT(route, piece.startTime + datetime.timedelta(0,TTs[0]), piece.endTime))
  """  
  index1 = 0
  
  while len(cut_traj.pieces[index1].route.link_ids) == 1:
    index1 = index1 + 1
    
  piece = cut_traj.pieces[index1]
  TTs = travelTimeByLink(piece, net)
  route = Route(piece.route.link_ids[1:], [Spot(piece.route.link_ids[1], 0), piece.route.lastSpot])
  rtts.append(RouteTT(route, piece.startTime + datetime.timedelta(0, TTs[0]), piece.endTime))
    
  index1 = index1 + 1
    
  index2 = cut_traj.numPieces - 1
    
  while len(cut_traj.pieces[index2].route.link_ids) == 1:
    index2 = index2 - 1
    
  for piece in cut_traj.pieces[index1:(index2 + 1)]:
    rtts.append(piece)
  return CutTrajectory(rtts)
  
  
def getListLinks(cutTraj):
  """get list of linkIds of trajObs in reversed order
  listLinks(link_id[0],link_id[1],TTgo,TTstop)
  
  TODO(?) what does this function do???
  """
  listLinks = [((-1, -1), -1, -1)]
  for piece in reversed(cutTraj.pieces):
    for linkID in reversed(piece.route.link_ids):
      if linkID != listLinks[-1][0]:
        listLinks.append((linkID, 0, 0))
  return listLinks, len(listLinks)
  
    
def cutTrajToTrajObs(cut_traj, net, tt_graph, offsets):
  """convert a cut_trajectory into a trajectory_observation
  
  Arguments:
  cut_traj: a CutTrajectory object
  net: network (ductionary link_id -> Link)
  tt_graph: TTGraph
  offsets: dictionary of splitting offsets (see createSplittingOffsets)
  
  Return a TrajectoryObservation object.
  """
  cutTraj = preprocessCutTraj(cut_traj, net)
  
  # get list of linkIds of trajObs in reversed order
  # listLinks(link_id[0],link_id[1],TTgo,TTstop)
  (listLinks, lenListLinks) = getListLinks(cutTraj)

  # cursor on actual link observed in listLinks
  index1 = 1
  # cursor on link in listLinks
  index2 = 1
  index = 0
  # we look at each piece in reversed order
  for piece in reversed(cutTraj.pieces):
    
    # print index
    index = index + 1
    firstSpot = piece.route.firstSpot
    lastSpot = piece.route.lastSpot
    
    if firstSpot == lastSpot:
      # we have a stop
      if index1 == index2 and firstSpot.offset < offsets[firstSpot.linkId]:
        # we have the first stop before offset and increments cursor2
        index2 = index2 + 1
      # updates the element at index2 in listLinks with the stop
      
      # print 'we have a stop'
      # print 'index2',index2
      
      if index2 < lenListLinks:
        listLinks[index2] = (listLinks[index2][0], listLinks[index2][1], listLinks[index2][2] + piece.tt)
    else:
      # we have a go
      TTs = travelTimeByLink(piece, net)
      # updates the element at index2 in listlinks with the TT at first link
      if index2 < lenListLinks:
        listLinks[index2] = (listLinks[index2][0], listLinks[index2][1] + TTs[-1], listLinks[index2][2])
      
      for j in range(len(piece.route.link_ids) - 1):
        # if there are at least two link_ids for a go, we increment index1 and index1=index2
        index1 = index1 + 1
        index2 = index1
        
        # print 'we have a go'
        # print 'index2',index2
        
        if index2 < lenListLinks:
          listLinks[index2] = (listLinks[index2][0], listLinks[index2][1] + TTs[-j - 2], listLinks[index2][2])

  # construct trajObs
  observations = []
  for i in range(len(listLinks) - 2):
    link_id = listLinks[-i - 1][0]
    outgoing_id = listLinks[-i - 2][0]
    mode = "go"
    if listLinks[-i - 1][2] > 0:
      mode = "stop" 
    obs = Observation((link_id, outgoing_id, mode), listLinks[-i - 1][1] + listLinks[-i - 1][2])
    observations.append(obs)

  # note that the return contains observation on the first link and the last link, which can be discarded
  return TrajectoryObservation(observations)


def travelTimeByLink(routett, net):
  """gets the travel times for each link of routett
  
  Returns a list of floats (travel times), one for each link of the route.
  """
  tts = []
  firstSpot = routett.route.firstSpot
  lastSpot = routett.route.lastSpot
  links = routett.route.link_ids
  vel = averageRouteSpeed(routett, net)
  if len(links) == 1:
    tts.append(routett.tt)
  else:
    tts.append((net[links[0]].length - firstSpot.offset) / vel)
    for link in links[1:-1]:
      tts.append(net[link].length / vel)
    tts.append(lastSpot.offset / vel)
  return tts


def drawTTGraph(ax, tt_graph, var_style=None, edge_style=None):
  """ Plots a mode graph
  """
  # Plot the link_ids
  if edge_style is not None:
    lons = []
    lats = []
    for (varid1, varid2) in tt_graph.allEdgeKeys():
      loc1 = tt_graph.variables[varid1].location
      loc2 = tt_graph.variables[varid2].location
      this_lons = [loc1.lon, loc2.lon]
      this_lats = [loc1.lat, loc2.lat]
      lons.append(this_lons)
      lats.append(this_lats)
    # Do not forget to reorganize the shape of the data
    # pylint: disable=W0142
    ax.plot(np.array(lons).T, np.array(lats).T, **edge_style)
  # Plots last the modes
  if var_style is not None:
    node_lons = [var.location.lon for var in tt_graph.allVariables()]
    node_lats = [var.location.lat for var in tt_graph.allVariables()]
    # pylint: disable=W0142
    ax.scatter(node_lons, node_lats, **var_style)
 
  
def routeLength(route, net):
  """gets the length of a route
  """
  firstSpot = route.spots[0]
  lastSpot = route.spots[-1]
  return sum([net[link_id].length for link_id in route.link_ids[:-1]]) - firstSpot.offset + lastSpot.offset
  
  
def averageRouteSpeed(routeTT, net):
  """gets the velocity of a routeTT
  """
  return routeLength(routeTT.route, net) / routeTT.tt

def fillTTGraph_old(tt_graph, traj_obs, min_variance=1e-2,
                variance_prior=0.0, variance_prior_count=0.0):
  """ Computes the sufficient statistics of the travel time graph, and fills the
  corresponding TT graph.
  
  Arguments:
  tt_graph -- a travel time graph
  traj_obs -- an iterable of trajectory observations
  """
  assert False
  # DEBUG!!!!
  min_variance=1e-2
  variance_prior=0.0
  variance_prior_count=0.0
  # Compute the sufficient stats
  # First moments
  sstats0 = dict([(var_id, 0.0) for var_id in tt_graph.variable_keys])
  sstats0_counts = dict([(var_id, 0.0) for var_id in tt_graph.variable_keys])
  # Second moments
  all_edge_keys = [(from_var_id, to_var_id) 
                   for (from_var_id, to_var_id) in tt_graph.edges.keys()] \
                   + [(to_var_id, from_var_id) 
                      for (from_var_id, to_var_id) in tt_graph.edges.keys()] \
                   + [(var_id, var_id) 
                      for var_id in tt_graph.variable_keys]
  # Covariance term
  sstats1 = dict([((from_var_id, to_var_id),
                    (variance_prior * variance_prior_count if from_var_id == to_var_id else 0.0)) 
                   for (from_var_id, to_var_id) in all_edge_keys])
  # First order stat for the start
  sstats1_from = dict([((from_var_id, to_var_id),
                    variance_prior * variance_prior_count) 
                   for (from_var_id, to_var_id) in all_edge_keys])
  sstats1_to = dict([((from_var_id, to_var_id),
                    variance_prior * variance_prior_count) 
                   for (from_var_id, to_var_id) in all_edge_keys])
  sstats0_from = dict([((from_var_id, to_var_id), 0.0) 
                   for (from_var_id, to_var_id) in all_edge_keys])
  sstats0_to = dict([((from_var_id, to_var_id), 0.0) 
                   for (from_var_id, to_var_id) in all_edge_keys])
  sstats1_counts = dict([(z, variance_prior_count) for z in all_edge_keys])
  # Fill the sufficient stats for the variance and the mean:
  for traj_ob in traj_obs:
    obs = traj_ob.observations
    for ob in obs:
      sstats0[ob.varId] += ob.value
      sstats0_counts[ob.varId] += 1.0
      sstats1[(ob.varId, ob.varId)] += ob.value * ob.value
      sstats1_counts[(ob.varId, ob.varId)] += 1.0
    del obs,ob
  for traj_ob in traj_obs:
    obs = traj_ob.observations
    l = len(obs)
    for i in range(l):
      for j in range(i + 1, l):
        from_ob = obs[i]
        to_ob = obs[j]
        from_vid = from_ob.varId
        to_vid = to_ob.varId
        if (from_vid, to_vid) in sstats1:
          sstats0_from[(from_vid, to_vid)] += from_ob.value
          sstats0_to[(from_vid, to_vid)] += to_ob.value
          sstats1_from[(from_vid, to_vid)] += from_ob.value ** 2
          sstats1_to[(from_vid, to_vid)] += to_ob.value ** 2
          sstats1[(from_vid, to_vid)] += from_ob.value * to_ob.value
          sstats1_counts[(from_vid, to_vid)] += 1.0
        if (to_ob.varId, from_ob.varId) in sstats1:
          sstats0_from[(to_vid, from_vid)] += to_ob.value
          sstats0_to[(to_vid, from_vid)] += from_ob.value
          sstats1_from[(to_vid, from_vid)] += to_ob.value ** 2
          sstats1_to[(to_vid, from_vid)] += from_ob.value ** 2
          sstats1[(to_ob.varId, from_ob.varId)] += from_ob.value * to_ob.value
          sstats1_counts[(to_ob.varId, from_ob.varId)] += 1.0
    del obs,l,from_ob,to_ob,from_vid,to_vid,i,j
  
  # Put all the parameters to zero in the tt graph
  means = {}
  for var in tt_graph.allVariables():
    count0 = sstats0_counts[var.varID]
    mean = sstats0[var.varID] / float(count0) if count0 > 0 else 0.0
    means[var.varID] = mean
    del mean, count0,var
  variances = {}
  for var in tt_graph.allVariables():
    count1 = sstats1_counts[(var.varID, var.varID)]
    sstat1 = sstats1[(var.varID, var.varID)] / float(count1) if count1 > 0 else 0.0
    mean = means[var.varID]
    variance_ = sstat1 - mean * mean
    assert variance_ >= 0, (sstat1 - mean * mean, sstat1, mean)
    variance = max(variance_, min_variance)
    variances[var.varID] = variance
    var.parameters = GaussianParameters(mean, variance)
    del var,count1,sstat1,variance_,variance
  for (from_vid, to_vid) in tt_graph.edges.keys():
    count1 = sstats1_counts[(from_vid, to_vid)]
    sstat1 = sstats1[(from_vid, to_vid)] / float(count1) if count1 > 0 else 0.0
    local_mean_from = sstats0_from[(from_vid, to_vid)] / float(count1) if count1 > 0 else 0.0
    local_mean_to = sstats0_to[(from_vid, to_vid)] / float(count1) if count1 > 0 else 0.0
    sstat1_from = sstats1_from[(from_vid, to_vid)] / float(count1) if count1 > 0 else 0.0
    sstat1_to = sstats1_to[(from_vid, to_vid)] / float(count1) if count1 > 0 else 0.0
    local_var_from_ = sstat1_from - (local_mean_from ** 2)
    local_var_to_ = sstat1_to - (local_mean_to ** 2)
    # This epsilon should prevent the assertion from failing due to rounding
    # errors and a low number of samples.
    local_var_from = max(local_var_from_, min_variance)+1e-7
    local_var_to = max(local_var_to_, min_variance)+1e-7
    local_cov = sstat1 - local_mean_from * local_mean_to
    assert abs(local_cov) <= np.sqrt(local_var_from * local_var_to),()
    variance_from = tt_graph.variables[from_vid].parameters.variance
    variance_to = tt_graph.variables[to_vid].parameters.variance
    scale = np.sqrt((variance_from * variance_to) / (local_var_from * local_var_to))
    cov = scale * local_cov
    assert np.abs(cov) <= np.sqrt(variance_from * variance_to), (np.abs(cov), np.sqrt(variance_from * variance_to))
    tt_graph.edges[(from_vid, to_vid)] = cov
    del from_vid,to_vid,count1,sstat1,local_mean_from,local_mean_to,sstat1_from,sstat1_to

EPSI=1e-10

def fillTTGraph(tt_graph, traj_obs, min_variance=1e-2,
                variance_prior=0.0, variance_prior_count=0.0, traj_obs_copy=None):
  """ Computes the sufficient statistics of the travel time graph, and fills the
  corresponding TT graph.
  
  Arguments:
  tt_graph -- a travel time graph
  traj_obs -- an iterable of trajectory observations
  
  TODO: returns some stats about the number of elements seen
  """
  # This function is one of the most complicated
  # Compute the sufficient stats
  # First moments for means
  # The prior is on white noise
  sstats0 = dict([(var_id, 0.0) for var_id in tt_graph.variable_keys])
  sstats0_counts = dict([(var_id, variance_prior_count) for var_id in tt_graph.variable_keys])
  sstats0_true_counts = dict([(var_id, 0) for var_id in tt_graph.variable_keys])
  sstats1_var = dict([(var_id, variance_prior_count*variance_prior) for var_id in tt_graph.variable_keys])
  # Compute the sufficient statistics first for central elements
  tic_n_obs = 1000
  count_traj_obs = 0
  for traj_ob in traj_obs:
    count_traj_obs += 1
    if count_traj_obs % tic_n_obs == 0:
      tic("fillTTGraph: processed %d observations in pass #1"%count_traj_obs)
    obs = traj_ob.observations
    for ob in obs:
      sstats0[ob.varId] += ob.value
      sstats0_counts[ob.varId] += 1.0
      sstats0_true_counts[ob.varId] += 1
      sstats1_var[ob.varId] += ob.value * ob.value
    del obs,ob
  tic("fillTTGraph: processed %d observations in pass #1"%count_traj_obs)
  # Compute the means
  means = {}
  for var in tt_graph.allVariables():
    var_id = var.varID
    count0 = sstats0_counts[var_id]
    mean = sstats0[var_id] / float(count0) if count0 > 0 else 0.0
    means[var_id] = mean
    assert mean >= -EPSI # Specific to traffic problem
    mean = max(mean,0)
    del mean, count0,var,var_id
  # Compute the variances
  variances = {}
  for var in tt_graph.allVariables():
    var_id = var.varID
    count0 = sstats0_counts[var_id]
    sstat1 = sstats1_var[var_id] / float(count0) if count0 > 0 else 0.0
    mean = means[var_id]
    variance_ = sstat1 - mean * mean
    assert variance_ >= -EPSI, (sstat1 - mean * mean, sstat1, mean)
    variance = max(variance_, min_variance)
    variances[var_id] = variance
    del var,count0,sstat1,variance_,variance,var_id
  # Update the gaussian parameters
  for var in tt_graph.allVariables():
    var_id = var.varID
    mean = means[var_id]
    variance = variances[var_id]
    var.parameters = GaussianParameters(mean, variance)
    tt_graph.variable_counts[var_id] = sstats0_true_counts[var_id]
   
  # Second moments for outer terms
  all_edge_keys = tt_graph.edges.keys()
  # Covariance term
  # pvid = pair of var_ids
  sstats1 = dict([(pvid, 0.0) for pvid in all_edge_keys])
  # First order stat for the start
  sstats1_from = dict([(pvid, variance_prior * variance_prior_count) 
                   for pvid in all_edge_keys])
  sstats1_to = dict([(pvid, variance_prior * variance_prior_count) 
                   for pvid in all_edge_keys])
  sstats0_from = dict([(pvid, 0.0) for pvid in all_edge_keys])
  sstats0_to = dict([(pvid, 0.0) for pvid in all_edge_keys])
  sstats1_counts = dict([(pvid, variance_prior_count) for pvid in all_edge_keys])
  sstats1_true_counts = dict([(pvid, 0) for pvid in all_edge_keys])
  
  # Fill the sufficient stats for the variance and the mean:
  # Updates the sufficient stats
  # hack to make sure we can run twice over the data
  if traj_obs_copy is None:
    traj_obs_copy=traj_obs
  count_traj_obs = 0
  for traj_ob in traj_obs_copy:
    count_traj_obs += 1
    if count_traj_obs % tic_n_obs == 0:
      tic("fillTTGraph: processed %d observations in pass #2"%count_traj_obs)
    obs = traj_ob.observations
    l = len(obs)
    for i in range(l):
      for j in range(i + 1, l):
        from_ob = obs[i]
        to_ob = obs[j]
        from_vid = from_ob.varId
        to_vid = to_ob.varId
        assert not ((from_vid, to_vid) in sstats1 and (to_vid, from_vid) in sstats1)
        if (from_vid, to_vid) not in sstats1 and (to_vid, from_vid) not in sstats1:
          continue
        # We may need to flip around the vids and values because of the ordering in
        # the variables
        if (from_vid, to_vid) in sstats1:
          from_vid_c = from_vid
          to_vid_c = to_vid
          from_val_c = from_ob.value
          to_val_c = to_ob.value
        else:
          assert (to_vid, from_vid) in sstats1
          from_vid_c = to_vid
          to_vid_c = from_vid
          from_val_c = to_ob.value
          to_val_c = from_ob.value
        # Update the stats
        key = (from_vid_c, to_vid_c)
        sstats1[key] += from_val_c * to_val_c
        sstats1_from[key] += from_val_c * from_val_c
        sstats1_to[key] += to_val_c * to_val_c
        sstats0_from[key] += from_val_c
        sstats0_to[key] += to_val_c
        sstats1_counts[key] += 1.0
        sstats1_true_counts[key] += 1
  tic("fillTTGraph: processed %d observations in pass #2"%count_traj_obs)
  
  # Compute the new covariance terms
  for key in all_edge_keys:
    count = sstats1_counts[key]
    # The local means
    local_mean_from = sstats0_from[key] / float(count) if count > 0 else 0.0
    local_mean_to = sstats0_to[key] / float(count) if count > 0 else 0.0
    # The local variances
    sstat1_from = sstats1_from[key] / float(count) if count > 0 else 0.0
    sstat1_to = sstats1_to[key] / float(count) if count > 0 else 0.0
    local_var_from_ = sstat1_from - (local_mean_from ** 2)
    local_var_to_ = sstat1_to - (local_mean_to ** 2)
    # This epsilon should prevent the assertion from failing due to rounding
    # errors and a low number of samples.
    local_var_from = max(local_var_from_, min_variance)+1e-7
    local_var_to = max(local_var_to_, min_variance)+1e-7
    # The local covariance term
    sstat1 = sstats1[key] / float(count) if count > 0 else 0.0
    local_cov = sstat1 - local_mean_from * local_mean_to
    assert abs(local_cov) <= np.sqrt(local_var_from * local_var_to),()
    # The global variance terms
    (from_vid,to_vid) = key
    variance_from = variances[from_vid]
    variance_to = variances[to_vid]
    scale = np.sqrt((variance_from * variance_to) / (local_var_from * local_var_to))
    cov = scale * local_cov
    assert np.abs(cov) <= np.sqrt(variance_from * variance_to)+EPSI, \
      (np.abs(cov), np.sqrt(variance_from * variance_to))
    tt_graph.edges[key] = cov
    tt_graph.edge_counts[key] = sstats1_true_counts[key]
    del from_vid,to_vid,count,sstat1,local_mean_from,local_mean_to,sstat1_from,sstat1_to