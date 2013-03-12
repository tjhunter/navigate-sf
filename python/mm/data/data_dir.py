'''
Created on Feb 7, 2012

@author: tjhunter

Finding the data directories.
'''

import os


def data_dir():
  """ Directory that contains the local data.
  
  The default value is overriden if the environment variable 
  DATA_DIR is defined. In this case, the value of this variable
  is returned instead.
  """
  if 'DATA_DIR' in os.environ:
    return os.environ['DATA_DIR']
  raise Exception("Set your DATA_DIR environment variable")


