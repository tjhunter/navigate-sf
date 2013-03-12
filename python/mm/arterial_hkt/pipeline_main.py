'''
Created on Feb 19, 2013

@author: tjhunter

The main procedures for learning and evaluation...
'''
from mm.arterial_hkt.gmrf import GMRF
from mm.data import get_network
from mm.arterial_hkt.utils import tic
from mm.arterial_hkt.ttobs_mapper import createTrajectoryConversion
import os
from mm.arterial_hkt.tt_graph_functions import createTravelTimeGraph, \
  fillTTGraph
from mm.arterial_hkt.ttobs_mapper import createTrajectoryConversion
from mm.data import get_network
from mm.data.structures import Coordinate
import mm.arterial_hkt.simple as model
import mm.arterial_hkt.validation.validation as validate
from mm.arterial_hkt.gmrf_functions import emptyValues
from mm.arterial_hkt.pipeline_io import save_ttg_structure, save_ttg_values, \
  save_hmm, save_gmrf_values, read_hmm, get_gmrf_estimator, experiment_directory,\
  read_hmm_pickle, test_traj_obs
from mm.arterial_hkt.hmm_functions import fillProbabilitiesObservations
import os
from mm.arterial_hkt.utils import tic, s_dump_elt, s_load
from mm.arterial_hkt.pipeline_functions import getDayTSpots, gmrf_learn,\
  gmrf_est, getDayTrajs
from mm.arterial_hkt.tt_graph import GaussianParameters
try:
  import cPickle as pickle
  print "Using fast version of pickle"
except ImportError:
  import pickle
from itertools import tee


def learn_procedure(experiment_design,num_jobs=1):
  experiment_name = experiment_design['name']
  # Get the network
  basic_geometry = experiment_design['basic_geometry']
  # Nearly everything will need a network.
  net = get_network(**basic_geometry)
  tic("Loaded network = {0} links".format(len(net)), experiment_name)
  graph_type = experiment_design['graph_type']
  traj_conv_param = experiment_design['trajectory_conversion']['params']
  
  # Trajectory conversion
  # Needed early because it gives the number of modes.
  global traj_conv_
  traj_conv_ = None
  def traj_conv():
    global traj_conv_
    if not traj_conv_:
      traj_conv_ = createTrajectoryConversion(graph_type=graph_type,
                                                process=experiment_design['trajectory_conversion']['process'],
                                                params=traj_conv_param,
                                                network=net,
                                                max_nb_mixture=traj_conv_param['max_n_modes'],
                                                n_jobs=num_jobs)
    return traj_conv_
  
  # Number of modes
  # Also stored on disk as pickle
  global mode_counts_
  mode_counts_ = None
  def mode_counts():
    global mode_counts_
    if not mode_counts_:
      tic("Loading trajectory conversion...")
      fname = "%s/mode_count.pkl"%experiment_directory(experiment_name)
      if not os.path.exists(fname):
        pickle.dump(traj_conv().modeCounts(), open(fname,'w'))
      mode_counts_ = pickle.load(open(fname,'r'))
      tic("Done loading trajectory conversion and mode counts")
    return mode_counts_
  
  # The HMM graph
  global hmm_graph_
  hmm_graph_ = None
  hmm_graph_fname = "%s/hmm_graph.pkl"%experiment_directory(experiment_name)
  def hmm_graph():
    global hmm_graph_
    if hmm_graph_ is None:
      if not os.path.exists(hmm_graph_fname):        
        if graph_type == 'simple':
          hmm_graph_ = model.createHMMGraphFromNetwork(net, mode_counts=mode_counts())
        else:
          # Complex model not implemented
          assert False
      else:
        tic("Reading completed hmm graph from %s"%hmm_graph_fname)
        hmm_graph_ = pickle.load(open(hmm_graph_fname,'r'))
    return hmm_graph_    
  
  # The TT gpaph
  # Also stored on disk as pickle by save_ttg_values (when it is filled).
  global tt_graph_
  tt_graph_ = None
  tt_graph_fname = "%s/tt_graph.pkl"%experiment_directory(experiment_name)
  def tt_graph():
    global tt_graph_
    if not tt_graph_:
      if not os.path.exists(tt_graph_fname):
        tic("creating empty tt graph", experiment_name)
        tt_graph_ = createTravelTimeGraph(hmm_graph(), radius=2e-4)
        tt_graph_.checkInvariants()
        save_ttg_structure(tt_graph_, experiment_name=experiment_name)
      else:
        tic("reading tt graph from %s"%tt_graph_fname, experiment_name)
        tt_graph_ = pickle.load(open(tt_graph_fname,'r'))
    return tt_graph_
  
  # The GMFR
  # Also stored on disk as pickle by save_gmrf_values (when it is filled).
  global gmrf_
  gmrf_ = None
  gmrf_fname = "%s/gmrf.pkl"%experiment_directory(experiment_name)
  def gmrf():
    global gmrf_
    if not gmrf_:
      if not os.path.exists(gmrf_fname):
        tic("creating empty gmrf", experiment_name)
        gmrf_ = emptyValues(tt_graph())
      else:
        tic("reading gmrf from %s"%gmrf_fname, experiment_name)
        gmrf_ = pickle.load(open(gmrf_fname,'r'))
    return gmrf_

  # The experiments data:
  data_source = experiment_design['data_source']
  dates = data_source['dates']
  basic_geometry = experiment_design['basic_geometry']
  
  # All this is lazy. Calling these functions does not create data.
  def tspots_seqs():
    return (ttob_seq for date in dates
               for ttob_seq in getDayTSpots(data_source['feed'],
                                                    basic_geometry['nid'],
                                                    date,
                                                    basic_geometry['net_type'],
                                                    basic_geometry['box'],
                                                    net))
  
  def traj_obs(print_num=1000):
    """ Returns the trajectory observations.
    
    If the obs have never been computed before, also stores them in a file.
    Otherwise reads the cached copy from the disk.
    """
    fname = "%s/traj_obs.pkl"%experiment_directory(experiment_name)
    fname_test = "%s/traj_obs_test.pkl"%experiment_directory(experiment_name)
    if not os.path.exists(fname):
      tic("traj_obs: Saving trajectory obs in %s"%fname, experiment_name)
      if num_jobs == 1:
        seq = (traj_ob for date in dates
                       for traj_ob in getDayTrajs(data_source['feed'],
                                                      basic_geometry['nid'],
                                                      date,
                                                      basic_geometry['net_type'],
                                                      basic_geometry['box'],
                                                      experiment_design['trajectory_conversion'],
                                                      traj_conv(), net))
      else:
        from joblib import Parallel, delayed
        tic("Using concurrent job code with {0} jobs".format(num_jobs),"learn_procedure")
        ls = Parallel(n_jobs=num_jobs)(delayed(wrapper)(data_source['feed'],
                      basic_geometry['nid'],
                      date,
                      basic_geometry['net_type'],
                      basic_geometry['box'],
                      experiment_design['trajectory_conversion'],
                      traj_conv(), net) for date in dates)
        seq = [traj_ob for l in ls
                       for traj_ob in l]

#      seq = (traj_ob for tspots_seq in tspots_seqs()
#                      for traj_ob in traj_conv().mapTrajectory(tspots_seq))
      kfold_cross_validation = data_source['kfold_cross_validation']
      test_k = data_source['test_k']
      assert kfold_cross_validation == 0 or test_k < kfold_cross_validation
      f = open(fname, 'w')
      if kfold_cross_validation > 0:
        tic("traj_obs: Saving test trajectory obs in %s"%fname_test, experiment_name)
        f_test = open(fname_test, 'w')
      idx = 0
      for traj_ob in seq:
        idx += 1
        if print_num > 0 and idx % print_num == 0:
          tic("traj_obs: Converted so far {0} observations".format(idx), experiment_name)
        if kfold_cross_validation > 0 and idx % kfold_cross_validation == test_k:
          s_dump_elt(traj_ob, f_test)
        else:
          s_dump_elt(traj_ob, f)
        yield traj_ob
    else:
      tic("traj_obs: opening trajectory obs in %s"%fname, experiment_name)
      f = open(fname, 'r')
      for traj_ob in s_load(f):
        yield traj_ob

  def var_seqs():
    return ([obs.varId for obs in traj_ob.observations] for traj_ob in traj_obs())

  # Starting learning here
  
  tic("HMM learning",experiment_name)
  tic("Loaded HMM = {0} nodes, {1} transitions".format(len(hmm_graph().allNodes()),
                                                       len(hmm_graph().allTransitions())), experiment_name)
  fillProbabilitiesObservations(hmm_graph(), var_seqs(), **experiment_design['hmm_learning']['parameters'])
  # Save to disk as well
  pickle.dump(hmm_graph(),open(hmm_graph_fname,'w'))
  save_hmm(hmm_graph(),experiment_name)
  
  tic("TT graph building", experiment_name)
  tic("Loaded TT graph = {0} edges, {1} variables".format(tt_graph().n,
                                                       tt_graph().m), experiment_name)
  gmrf_learning = experiment_design['gmrf_learning']
  fillTTGraph(tt_graph(), traj_obs(),traj_obs_copy=traj_obs(),**gmrf_learning['tt_graph_parameters'])
  pickle.dump(tt_graph(),open(tt_graph_fname,'w'))
  
  tic("GMRF learning", experiment_name)
  gmrf_learning = experiment_design['gmrf_learning']
  gmrf_learning_params = gmrf_learning['parameters']
  gmrf_ = gmrf_learn(tt_graph(), gmrf_learning['process'],
                    experiment_name, gmrf_learning_params)
  pickle.dump(gmrf_,open(gmrf_fname,'w'))
  save_gmrf_values(gmrf(), experiment_name=experiment_name)

  tic("GMRF estimation",experiment_name)
  gmrf_estimation = experiment_design['gmrf_estimation']
  gmrf_estimation_parameters = gmrf_estimation['parameters']
  # Saves all the GMRF estimators in the different formats
  # Will be reloaded when we do the estimation
  gmrf_est(gmrf(), gmrf_estimation['process'], experiment_name, gmrf_estimation_parameters)
  
  tic("End of learning", experiment_name)

def validation_procedure(experiment_design,
                         experiment_design_indep,
                         experiment_design_one_mode,
                         experiment_design_one_mode_indep,
                         validate_on_network=True):
  # Get the validation data
  experiment_name = experiment_design['name']
  # We will load the test data from this experiment
  experiment_name_one_mode = experiment_design_one_mode['name']
  # Get the network
  basic_geometry = experiment_design['basic_geometry']
  # Nearly everything will need a network.
  net = get_network(**basic_geometry)

  # reload the HMM and the GMRF estimator from the files
  # All we need for testing is a experiment_design
  test_hmm = read_hmm_pickle(experiment_name)
  # Read the estimator
  gmrf_estimation = experiment_design['gmrf_estimation']
  test_gmrf_estimator = get_gmrf_estimator(experiment_name, gmrf_estimation['process'])

  # Baseline Gaussian independent
  test_hmm_one_mode = read_hmm_pickle('{0}_one_mode'.format(experiment_name))
  test_gmrf_one_mode_indep = get_gmrf_estimator('{0}_one_mode_indep'.format(experiment_name), 'diagonal')

  # Baseline Gaussian
  test_gmrf_one_mode = get_gmrf_estimator('{0}_one_mode'.format(experiment_name), gmrf_estimation['process'])

  # Baseline MultiMode Gaussian independent
  test_gmrf_indep = get_gmrf_estimator('{0}_indep'.format(experiment_name), 'diagonal')

  tic('Validation')
  test_traj_obs_all = list(test_traj_obs(experiment_name))
  test_traj_obs_one_mode_all = list(test_traj_obs(experiment_name_one_mode))
  tic("Validation set: {0} trajectories".format(len(test_traj_obs_all)))
  model = [(test_traj_obs_all, test_gmrf_estimator, test_hmm, 'MM-GMRF')]
  baseline1 = [(test_traj_obs_one_mode_all, test_gmrf_one_mode_indep, test_hmm_one_mode, 'one mode indep')]
  baseline2 = [(test_traj_obs_one_mode_all, test_gmrf_one_mode, test_hmm_one_mode, 'one mode')]
  baseline3 = [(test_traj_obs_all, test_gmrf_indep, test_hmm, 'multi-modal indep')]
  val_model = model + baseline1 + baseline2 + baseline3
  tic('path validation')
  validate.validate_on_paths(val_model, net, 
                             estimation_sampling_process=experiment_design['estimation_sampling']['process'],
                             estimation_sampling_parameters=experiment_design['estimation_sampling']['parameters'], **experiment_design['evaluation'])
  if validate_on_network:
    tic('network validation')
    validate.validate_on_network(val_model, net, 
                               estimation_sampling_process=experiment_design['estimation_sampling']['process'],
                               estimation_sampling_parameters=experiment_design['estimation_sampling']['parameters'], **experiment_design['evaluation'])
  tic("Evaluation finished")


def wrapper(*args,**kwargs):
  tic("Wrapper called")
  res = getDayTrajs(*args, **kwargs)
  tic("Cached {0} trajs".format(len(res)))
  return res

def fillTrajectoryCache(graph_type,basic_geometry,data_source,traj_conv_description,n_jobs=1):
  net = get_network(**basic_geometry)
  tic("Loaded network = {0} links".format(len(net)), "fillTrajectoryCache")
  traj_conv = createTrajectoryConversion(graph_type=graph_type,
                                                process=traj_conv_description['process'],
                                                params=traj_conv_description['params'],
                                                network=net,
                                                max_nb_mixture=traj_conv_description['params']['max_n_modes'],
                                                n_jobs=n_jobs)
  dates = data_source['dates']
  from joblib import Parallel, delayed
  Parallel(n_jobs=n_jobs)(delayed(wrapper)(data_source['feed'],
                basic_geometry['nid'],
                date,
                basic_geometry['net_type'],
                basic_geometry['box'],
                traj_conv_description,
                traj_conv, net) for date in dates)


def learning_perf(experiment_design):
  """ Starting the main procedure.
  
  This script tries to reuse as much as it can from the disk, to avoid expensive recomputations.
  """
  experiment_name = experiment_design['name']
  # Get the network
  basic_geometry = experiment_design['basic_geometry']
  # Nearly everything will need a network.
  net = get_network(**basic_geometry)
  tic("Loaded network = {0} links".format(len(net)), experiment_name)
  graph_type = experiment_design['graph_type']
  traj_conv_param = experiment_design['trajectory_conversion']['params']
  
  # Trajectory conversion
  # Needed early because it gives the number of modes.
  global traj_conv_
  traj_conv_ = None
  def traj_conv():
    global traj_conv_
    if not traj_conv_:
      traj_conv_ = createTrajectoryConversion(graph_type=graph_type,
                                                process=experiment_design['trajectory_conversion']['process'],
                                                params=traj_conv_param,
                                                network=net,
                                                max_nb_mixture=traj_conv_param['max_n_modes'])
    return traj_conv_
  
  # Number of modes
  # Also stored on disk as pickle
  global mode_counts_
  mode_counts_ = None
  def mode_counts():
    global mode_counts_
    if not mode_counts_:
      tic("Loading trajectory conversion...")
      fname = "%s/mode_count.pkl"%experiment_directory(experiment_name)
      if not os.path.exists(fname):
        pickle.dump(traj_conv().modeCounts(), open(fname,'w'))
      mode_counts_ = pickle.load(open(fname,'r'))
      tic("Done loading trajectory conversion and mode counts")
    return mode_counts_
  
  # The HMM graph
  global hmm_graph_
  hmm_graph_ = None
  hmm_graph_fname = "%s/hmm_graph.pkl"%experiment_directory(experiment_name)
  def hmm_graph():
    global hmm_graph_
    if hmm_graph_ is None:
      if not os.path.exists(hmm_graph_fname):        
        if graph_type == 'simple':
          tic("creating empty hmm graph", experiment_name)
          hmm_graph_ = model.createHMMGraphFromNetwork(net, mode_counts=mode_counts())
          tic("done creating empty hmm graph", experiment_name)
          tic("saving hmm graph pickle", experiment_name)
          pickle.dump(hmm_graph(),open(hmm_graph_fname,'w'))
          tic("done saving hmm graph pickle", experiment_name)
        else:
          # Complex model not implemented
          assert False
      else:
        tic("Reading completed hmm graph from %s"%hmm_graph_fname)
        hmm_graph_ = pickle.load(open(hmm_graph_fname,'r'))
        tic("done reading completed hmm graph from %s"%hmm_graph_fname)
    return hmm_graph_    
  
  # The TT gpaph
  # Also stored on disk as pickle by save_ttg_values (when it is filled).
  global tt_graph_
  tt_graph_ = None
  tt_graph_fname = "%s/tt_graph.pkl"%experiment_directory(experiment_name)
  def tt_graph():
    global tt_graph_
    if not tt_graph_:
      if not os.path.exists(tt_graph_fname):
        tic("creating empty tt graph", experiment_name)
        tt_graph_ = createTravelTimeGraph(hmm_graph(), radius=2e-4)
        tt_graph_.checkInvariants()
        save_ttg_structure(tt_graph_, experiment_name=experiment_name)
      else:
        tic("reading tt graph from %s"%tt_graph_fname, experiment_name)
        tt_graph_ = pickle.load(open(tt_graph_fname,'r'))
    return tt_graph_
  
  # The GMFR
  # Also stored on disk as pickle by save_gmrf_values (when it is filled).
  global gmrf_
  gmrf_ = None
  gmrf_fname = "%s/gmrf.pkl"%experiment_directory(experiment_name)
  def gmrf():
    global gmrf_
    if not gmrf_:
      if not os.path.exists(gmrf_fname):
        tic("creating empty gmrf", experiment_name)
        gmrf_ = emptyValues(tt_graph())
        tic("created empty gmrf", experiment_name)
      else:
        tic("reading gmrf from %s"%gmrf_fname, experiment_name)
        gmrf_ = pickle.load(open(gmrf_fname,'r'))
        tic("done reading gmrf from %s"%gmrf_fname, experiment_name)
    return gmrf_
  
  
  tic("TT graph building", experiment_name)
  tic("Loaded TT graph = {0} edges, {1} variables".format(tt_graph().n,
                                                       tt_graph().m), experiment_name)
  tic("simulating sstat building", experiment_name)
  gmrf_learning = experiment_design['gmrf_learning']
  for var in tt_graph().allVariables():
    var_id = var.varID
    var.parameters = GaussianParameters(0.0, 1.0)
    tt_graph().variable_counts[var_id] = 1
  for key in tt_graph().edges.keys():
    tt_graph().edges[key] = -.1
    tt_graph().edge_counts[key] = 1
  tic("done simulating sstat building", experiment_name)
  
  if not os.path.exists(tt_graph_fname):
    tic("saving tt graph pickle", experiment_name)
    pickle.dump(tt_graph(),open(tt_graph_fname,'w'))
    tic("done saving tt graph pickle", experiment_name)
  
  tic("GMRF learning", experiment_name)
  gmrf_learning = experiment_design['gmrf_learning']
  gmrf_learning_params = gmrf_learning['parameters']
  gmrf_ = gmrf_learn(tt_graph(), gmrf_learning['process'],
                    experiment_name, gmrf_learning_params)
  tic("Done GMRF learning", experiment_name)
  
  tic("saving gmrf pickle", experiment_name)
  pickle.dump(gmrf_,open(gmrf_fname,'w'))
  save_gmrf_values(gmrf_, experiment_name=experiment_name)
  tic("done saving gmrf pickle", experiment_name)
  
  tic("GMRF estimation",experiment_name)
  gmrf_estimation = experiment_design['gmrf_estimation']
  gmrf_estimation_parameters = gmrf_estimation['parameters']
  # Saves all the GMRF estimators in the different formats
  # Will be reloaded when we do the estimation
  gmrf_est(gmrf(), gmrf_estimation['process'], experiment_name, gmrf_estimation_parameters)
  
  tic("End of learning", experiment_name)
 
