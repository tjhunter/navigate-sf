'''
Created on Feb 4, 2013

@author: audehofleitner
Module to test the functions in validation_path
'''

import validation_paths as vp
from mm.arterial_hkt.pipeline_script_1 import experiment_design, createTrajectoryConversion
from mm.data import get_network
from mm.arterial_hkt.pipeline_functions import getDayTSpots

design = experiment_design
basic_geometry = design['basic_geometry']
net = get_network(**basic_geometry)
graph_type=design['graph_type']
traj_conv = createTrajectoryConversion(graph_type=design['graph_type'],
                                            process=design['trajectory_conversion']['process'],
                                            params=design['trajectory_conversion']['params'],
                                            network=net)
data_source = design['data_source']
dates = design['data_source']['dates']
basic_geometry = design['basic_geometry']
tspots_seqs = [ttob_seq for date in dates 
               for ttob_seq in getDayTSpots(data_source['feed'], 
                                                    basic_geometry['nid'],
                                                    date, 
                                                    basic_geometry['net_type'],
                                                    basic_geometry['box'], 
                                                    net)]
traj_obs = [traj_ob for tspots_seq in tspots_seqs 
                      for traj_ob in traj_conv.mapTrajectory(tspots_seq)]


param = design['evaluation']
nb_obs_per_path = vp.number_measurements_per_path(traj_obs, net, debug=True, **param)
validation_paths = vp.select_validation_paths(traj_obs, nb_obs_per_path, debug=True, **param)
val_data = vp.select_validation_data_given_paths(traj_obs, validation_paths, debug=True)