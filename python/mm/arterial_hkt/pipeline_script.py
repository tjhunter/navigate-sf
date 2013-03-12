'''
Created on Jan 21, 2013

@author: tjhunter

Main script to run the arterial model.

Should be good enough for now because it is the exact pipeline is still
unclear.

This script runs everything after the PIF.
'''
# pylint:disable=W0105


# All the imports

# pylint:disable=W0621
from mm.arterial_hkt.tt_graph_functions import createTravelTimeGraph, \
  fillTTGraph
from mm.arterial_hkt.ttobs_mapper import createTrajectoryConversion
from mm.data import get_network
from mm.data.structures import Coordinate
import mm.arterial_hkt.simple as model
import mm.arterial_hkt.validation.validation as validate
from mm.arterial_hkt.gmrf_functions import emptyValues
from mm.arterial_hkt.pipeline_io import save_ttg_structure, save_ttg_values, \
  save_hmm, save_gmrf_values, read_hmm, get_gmrf_estimator
from mm.arterial_hkt.pipeline_functions import getDayTSpots, gmrf_learn, \
  gmrf_est
from mm.arterial_hkt.hmm_functions import fillProbabilitiesObservations


''' All the parameters are fixed here.
No other constant should appear anywhere
'''

''' Description of the network geometry.
'''
basic_geometry = dict(
  full=True,
  box=(Coordinate(37.7816, -122.425), Coordinate(37.7874, -122.421)),
  fname='network.json'
)

''' Source of data
'''
data_source = dict(
  dates=['2012-03-%02d' % i for i in ([8] + range(13, 16) + range(20, 23))],
  test_dates=['2012-03-%02d' % i for i in (range(27, 30))] + ['2012-04-17'],
)

''' The node type (graph_type)  (see variable.VariableId)
'''
# TODO: maybe clean this up, remove graph_type
graph_type = 'simple'

''' Step 1: Conversion from trajectories to
trajectory observations (observation.TrajectoryObservation)
'''

trajectory_conversion = dict(
  process='mixture_auto',
  params=dict(
    min_variance=.1,
    default_variance=4.0,
    max_n_modes=3,
    train_data=data_source,
    basic_geometry=basic_geometry,
    default_speed=11.0,
    avg_delay=25.0,
    non_stopping_default=0.8
  ),
)
trajectory_conversion = dict(
  process='lasso',
  # process='both',
  params=dict(
    train_data=data_source,
    basic_geometry=basic_geometry,
    min_variance=0.1,
    default_variance=4.0,
    max_n_modes=3,
    speed_threshold=1.0,
    min_stop_duration=5.0,
    min_traj_speed=1.0,
    default_speed=10.0,
    avg_delay=20.0,
    non_stopping_default=0.7,
    perc_link_traveled=.98
  ),
)

''' The HMM learning procedure.
'''

""" Look at fillProbabilitiesObservations for an explanation of the parameters.
"""
hmm_learning = dict(
  process='max_ll',
  parameters=dict(
    smoothing_count=1e-4,
    smoothing_trans_count=1e-3
  )
)

''' GMRF procedure
'''

''' Parameters for the fillTTGraph function
Refer to this function for some (missing) documentation.
'''
tt_graph_parameters = dict(
  min_variance=1e-2,
  variance_prior=0.0,
  variance_prior_count=0.0
)

#gmrf_learning = dict(
#  process='independent',
#  parameters=dict(
#  ),
#  tt_graph_parameters=tt_graph_parameters
#)

gmrf_learning = dict(
 process='cvx',
 parameters=dict(
   num_iterations=100,
   min_edge_count=50,
   min_variance=1e-2,
 ),
 tt_graph_parameters=tt_graph_parameters
)

#gmrf_learning = dict(
# process='jl',
# parameters=dict(
#   k=100,
#   psd_tolerance=1e-6,
#   num_iterations=500,
#   finish_early=True
# ),
# tt_graph_parameters=tt_graph_parameters
#)

''' GMRF Estimation algorithm (exact/approximate)
'''
gmrf_estimation = dict(
 process='diagonal',
 parameters=dict(
 )
)

#gmrf_estimation = dict(
# num_samples=1000,
# process='exact',
# parameters=dict(
# )
#)

#gmrf_estimation = dict(
# num_samples=1000,
# process='jl',
# parameters=dict(
#   k=200
# )
#)

''' Parameters for the HMM estimation (sampling).

See  getTTDistributionSamplingFast Found at: mm.arterial_hkt.mixture_functions
for an explanation of the relevant parameters.
'''
estimation_sampling_normal=dict(
  process='sampling',
  parameters=dict(
    num_samples=1000,
    max_modes=10000,
  )
)

estimation_sampling_fast=dict(
  process='sampling_fast',
  parameters=dict(
    num_samples=2000,
    max_modes=10000,
    precision_target=.01,
  )
)

''' evaluation procedure
min_validation_path_length:
  minimum length of a path to be considered for validate_on_paths
  set to 0 is no minimum length is required
min_nb_validation_points:
  min number of points on a given path for this path to be
  considered for validate_on_paths. It needs to be enough for the
  statistical validate_on_paths of the results to be meaningful
max_nb_validation_paths:
  max number of path to consider for the validate_on_paths of the
  results. This should mostly be limited for plotting validate_on_paths
  For other metrics, all valid validate_on_paths data should be considered
  Set to None if no max should be considered.
length_bin_size:
  Size of the bins for the likelihood validation so that path
  of the same length are compared together
'''
evaluation = dict(
  min_validation_path_length=200.0,
  min_nb_validation_points=10,
  max_nb_validation_paths = 7,
  length_bin_size=100.0,
  max_nb_paths = 2000
)

''' Complete experiment description.
'''
experiment_design = dict(
  name='minimal',
  graph_type=graph_type,
  basic_geometry=basic_geometry,
  data_source=data_source,
  trajectory_conversion=trajectory_conversion,
  hmm_learning=hmm_learning,
  gmrf_learning=gmrf_learning,
  gmrf_estimation=gmrf_estimation,
  estimation_sampling=estimation_sampling_fast,
  evaluation=evaluation,
)


if __name__ == '__main__':
  """ Starting the main procedure.

  TODO: put all learning in a function
  """
  # pylint:disable=W0142
  experiment_design = experiment_design

  experiment_name = experiment_design['name']
  # Get the network
  basic_geometry = experiment_design['basic_geometry']
  net = get_network(**basic_geometry)
  graph_type = experiment_design['graph_type']
  traj_conv_param = experiment_design['trajectory_conversion']['params']
  traj_conv = createTrajectoryConversion(graph_type=graph_type,
                                            process=experiment_design['trajectory_conversion']['process'],
                                            params=traj_conv_param,
                                            network=net,
                                            max_nb_mixture=traj_conv_param['max_n_modes'])

  traj_conv_one_mode = createTrajectoryConversion(graph_type=graph_type,
                                            process='mixture_auto',
                                            params=traj_conv_param,
                                            network=net,
                                            max_nb_mixture=1)

  #  mode_counts = dict([(link_id,1) for link_id in net.keys()])
  if graph_type == 'simple':
    hmm_graph = model.createHMMGraphFromNetwork(net, mode_counts=traj_conv.modeCounts())
    hmm_graph_one_mode = model.createHMMGraphFromNetwork(net, mode_counts=traj_conv_one_mode.modeCounts())
    # hmm_graph = model.createHMMGraphFromNetwork(net, mode_counts=mode_counts)
  else:
    # Complex model not implemented
    assert False

  tt_graph = createTravelTimeGraph(hmm_graph, radius=2e-4)
  tt_graph.checkInvariants()

  tt_graph_one_mode = createTravelTimeGraph(hmm_graph_one_mode, radius=2e-4)
  tt_graph_one_mode.checkInvariants()


  gmrf = emptyValues(tt_graph)
  gmrf_one_mode = emptyValues(tt_graph)

  # Checkpoint: save the structures

  save_ttg_structure(tt_graph, experiment_name=experiment_name)
  # The TTG structure is required when loading the GMRF (and the GMRF estimators)
  # Make sure they are saved in all the directories
  save_ttg_structure(tt_graph_one_mode, experiment_name='{0}_one_mode'.format(experiment_name))
  save_ttg_structure(tt_graph_one_mode, experiment_name='{0}_one_mode_indep'.format(experiment_name))
  save_ttg_structure(tt_graph, experiment_name='{0}_indep'.format(experiment_name))


  # Loading the learning data

  data_source = experiment_design['data_source']
  dates = data_source['dates']
  basic_geometry = experiment_design['basic_geometry']
  tspots_seqs = [ttob_seq for date in dates
               for ttob_seq in getDayTSpots(data_source['feed'],
                                                    basic_geometry['nid'],
                                                    date,
                                                    basic_geometry['net_type'],
                                                    basic_geometry['box'],
                                                    net)]
  traj_obs = [traj_ob for tspots_seq in tspots_seqs
                      for traj_ob in traj_conv.mapTrajectory(tspots_seq, **traj_conv_param)]
  traj_obs_one_mode = [traj_ob for tspots_seq in tspots_seqs
                      for traj_ob in traj_conv_one_mode.mapTrajectory(tspots_seq, **traj_conv_param)]


  gmrf_learning = experiment_design['gmrf_learning']
  fillTTGraph(tt_graph, traj_obs, **gmrf_learning['tt_graph_parameters'])
  fillTTGraph(tt_graph_one_mode, traj_obs_one_mode, **gmrf_learning['tt_graph_parameters'])

  # CHECKPOINT HERE: SAVE TT GRAPH Values
  save_ttg_values(tt_graph, experiment_name=experiment_name)
  save_ttg_values(tt_graph_one_mode, experiment_name='{0}_one_mode'.format(experiment_name))

  var_seqs = [[obs.varId for obs in traj_ob.observations] for traj_ob in traj_obs]
  var_seqs_one_mode = [[obs.varId for obs in traj_ob.observations] for traj_ob in traj_obs_one_mode]
  fillProbabilitiesObservations(hmm_graph, var_seqs, **experiment_design['hmm_learning']['parameters'])
  fillProbabilitiesObservations(hmm_graph_one_mode, var_seqs_one_mode, **experiment_design['hmm_learning']['parameters'])

  # CHECKPOINT HERE: SAVE HMM GRAPH values
  save_hmm(hmm_graph, experiment_name=experiment_name)
  save_hmm(hmm_graph_one_mode, experiment_name='{0}_one_mode'.format(experiment_name))

  gmrf_learning_params = gmrf_learning['parameters']
  gmrf = gmrf_learn(tt_graph, gmrf_learning['process'],
                    experiment_name, gmrf_learning_params)
  gmrf_one_mode_indep = gmrf_learn(tt_graph_one_mode, 'independent',
  			'{0}_one_mode_indep'.format(experiment_name),
  			gmrf_learning_params)
  gmrf_one_mode = gmrf_learn(tt_graph_one_mode, gmrf_learning['process'],
  			'{0}_one_mode'.format(experiment_name),
  			gmrf_learning_params)
  gmrf_indep = gmrf_learn(tt_graph, 'independent',
  			'{0}_indep'.format(experiment_name),
  			gmrf_learning_params)

  # CHECKPOINT HERE: SAVE GMRF values
  save_gmrf_values(gmrf, experiment_name=experiment_name)
  save_gmrf_values(gmrf_one_mode_indep, experiment_name='{0}_one_mode_indep'.format(experiment_name))
  save_gmrf_values(gmrf_one_mode, experiment_name='{0}_one_mode'.format(experiment_name))
  save_gmrf_values(gmrf_indep, experiment_name='{0}_indep'.format(experiment_name))


  gmrf_estimation = experiment_design['gmrf_estimation']
  gmrf_estimation_parameters = gmrf_estimation['parameters']
  # Saves all the GMRF estimators in the different formats
  # Will be reloaded when we do the estimation
  gmrf_est(gmrf, gmrf_estimation['process'], experiment_name, gmrf_estimation_parameters)
  gmrf_est(gmrf_one_mode_indep, 'diagonal', '{0}_one_mode_indep'.format(experiment_name), gmrf_estimation_parameters)
  gmrf_est(gmrf_one_mode, gmrf_estimation['process'], '{0}_one_mode'.format(experiment_name), gmrf_estimation_parameters)
  gmrf_est(gmrf_indep, 'diagonal', '{0}_indep'.format(experiment_name), gmrf_estimation_parameters)

  # Testing

  # Get the validation data
  test_dates = data_source['test_dates']
  test_tspots_seqs = [ttob_seq for date in test_dates
               for ttob_seq in getDayTSpots(date, net)]
  test_traj_obs = [traj_ob for tspots_seq in test_tspots_seqs
                      for traj_ob in traj_conv.mapTrajectory(tspots_seq, **traj_conv_param)]
  test_traj_obs_one_mode = [traj_ob for tspots_seq in test_tspots_seqs
                      for traj_ob in traj_conv_one_mode.mapTrajectory(tspots_seq, **traj_conv_param)]
  test_traj_obs = traj_obs
  test_traj_obs_one_mode = traj_obs_one_mode

  # reload the HMM and the GMRF estimator from the files
  # All we need for testing is a experiment_design
  test_hmm = read_hmm(experiment_name)
  # Read the estimator
  test_gmrf_estimator = get_gmrf_estimator(experiment_name, gmrf_estimation['process'])

  # Baseline Gaussian independent
  test_hmm_one_mode = read_hmm('{0}_one_mode'.format(experiment_name))
  test_gmrf_one_mode_indep = get_gmrf_estimator('{0}_one_mode_indep'.format(experiment_name), 'diagonal')

  # Baseline Gaussian
  test_gmrf_one_mode = get_gmrf_estimator('{0}_one_mode'.format(experiment_name), gmrf_estimation['process'])

  # Baseline MultiMode Gaussian independent
  test_gmrf_indep = get_gmrf_estimator('{0}_indep'.format(experiment_name), 'diagonal')
  # test_gmrf_estimator_exact = get_gmrf_estimator(experiment_name,{'process':'exact'})
  # test_gmrf_estimator_diagonal = get_gmrf_estimator(experiment_name,{'process':'diagonal'})
  # TODO: perform cross validate_on_paths
  # This is after stop/go, so it is easier than pure new paths
#  test_traj_obs = test_traj_obs[1:10]
#  test_traj_obs_one_mode = test_traj_obs_one_mode[1:10]
#
  print 'Validation'
  model = [(test_traj_obs, test_gmrf_estimator, test_hmm, 'model')]
  baseline1 = [(test_traj_obs_one_mode, test_gmrf_one_mode_indep, test_hmm_one_mode, 'one mode indep')]
  baseline2 = [(test_traj_obs_one_mode, test_gmrf_one_mode, test_hmm_one_mode, 'one mode')]
  baseline3 = [(test_traj_obs, test_gmrf_indep, test_hmm, 'multi-modal indep')]
  val_model = model + baseline1 + baseline2 + baseline3
  print 'path validation'
  validate.validate_on_paths(val_model, net, 
                             estimation_sampling_process=experiment_design['estimation_sampling']['process'],
                             estimation_sampling_parameters=experiment_design['estimation_sampling']['parameters'], **experiment_design['evaluation'])
  print 'network validation'
  validate.validate_on_network(val_model, net, 
                             estimation_sampling_process=experiment_design['estimation_sampling']['process'],
                             estimation_sampling_parameters=experiment_design['estimation_sampling']['parameters'], **experiment_design['evaluation'])


