'''
Created on Feb 20, 2013

@author: tjhunter
'''
from mm.data import get_network
from mm.arterial_hkt.utils import tic
from mm.arterial_hkt.ttobs_mapper import createTrajectoryConversion
from mm.arterial_hkt.pipeline_functions import getDayTrajs
from mm.arterial_hkt.pipeline_main import fillTrajectoryCache

if __name__ == '__main__':
  from mm.arterial_hkt.pipeline_script_gmm import *
  traj_conv_description = experiment_design['trajectory_conversion']
  fillTrajectoryCache(graph_type,basic_geometry, 
                      data_source, traj_conv_description,n_jobs=3)
