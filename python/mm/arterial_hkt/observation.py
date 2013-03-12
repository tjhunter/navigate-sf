'''
Created on Nov 15, 2012

@author: tjhunter

The main input to the learning process.
'''

class TrajectoryObservation(object):
  """ A CutTrajectory transformed into a list of observations
  
  Fields:
  - observations: a list of Observation object
  - numObs: the number of observations
  """
  
  def __init__(self, observations):
    self.observations = observations
    self.numObs = len(self.observations)

class Observation(object):
  """ A single observation of a variable on a Mode Graph.
  
  Fields:
  - varId the identifier of the variable mode: VariableID object (node ID, mode)
  - value: travel time value on this mode (float)
  """
  
  def __init__(self, var_id, val):
    self.varId = var_id
    self.value = val
