'''
Created on Feb 7, 2012

@author: tjhunter

Filter for elements of trajectories generated using the Viterbi algorithm.
'''

from string import Template
import json as json
from data_dir import data_dir
import os
from codec_json import decode_TSpot


def list_traj_filenames(date):
  dir_tpl = '${data_dir}/trajectories/${date}/'
  dir_name = Template(dir_tpl).substitute(date=date, data_dir=data_dir())
  if os.path.exists(dir_name):
    files = list(os.listdir(dir_name))
  else:
    print 'no data for {0}'.format(date)
    files = []
  return files

def traj(fname):
  fin_obj = open(fname)
  for line in fin_obj:
    dct = json.loads(line[:-1])
    dct_paths = dct['routes']
    dct_pc = dct['point']
    yield ([], dct_paths, [], dct_pc)
  fin_obj.close()


def read_trajectory(fname):
  """ Read a trajectory
  
  Arguments:
  fname: name of trajectory file (format in README)
  
  Returns tspots with tspots a list of TSpot
  """
  traj_gen = traj(fname)
  tspots = []
  for traj_gen in traj_gen:
    (_, dct_paths, _, dct_sc) = traj_gen
    tsp = decode_TSpot(dct_sc)
    tspots.append(tsp)
  return tspots

