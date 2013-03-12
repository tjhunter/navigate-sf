'''
Created on Dec 26, 2012

@author: tjhunter
'''

class VariableId(object):
  """ A unique identifier for a travel time variable.
  
  It is a pair of a node id (usually a link id, or a (link_id, outgoing link id))
  and a mode associated to this mode (go, stop, stopstop).
  
  Can be used as a dictionary key.
  """
  
  def __init__(self, node_id, mode):
    self.nodeId = node_id
    self.mode = mode
  
  def __hash__(self):
    return hash((self.nodeId, self.mode))
    
  def __eq__(self, other):
      if type(other) is type(self):
          return self.__dict__ == other.__dict__
      return False
    
  def __ne__(self, other):
      return not self.__eq__(other)
  
  def __repr__(self):
    return "VarID[node=%s,mode=%s]" % (str(self.nodeId), str(self.mode))
