'''
Created on Feb 7, 2012

@author: tjhunter

Code to represent a network.
'''
from data_dir import data_dir
import json as json
from codec_json import decode_Coordinate, decode_link_id
from collections import defaultdict
from mm.data.codec_json import decode_node_id
from mm.data.structures import Coordinate
import numpy as np

class DefaultLink(object):
  """ A representation of a link
  
  Useful fields
  -------------
  linkId : the ID of the link
  geom: the geometry (sequence of Coordinate)
  length: length of the link (float)
  outgoingLinks, incomingLinks: list of link IDs
  """
  
  def __init__(self, link_id, geom, length, outgoing_links, incoming_links):
    self.linkId = link_id
    self.geom = geom
    self.length = length
    self.outgoingLinks = outgoing_links
    self.incomingLinks = incoming_links


def get_default_network(fname=None,full=True): 
  if fname is None:
    # TODO: create line network
    'todo'
  fin = open('{0}/{1}'.format(data_dir(),fname), "r")
  res = {}
  incomingLinksByNode = defaultdict(list)
  outgoingLinksByNode = defaultdict(list)
  # In a first step, create the links without information about 
  # incoming or outgoing links
  for line in fin:
    print 'end of line is ', line[-1]
    dct = json.loads(line.strip())
    link_id = decode_link_id(dct['id'])
    geom = None
    if full:
      geom = [decode_Coordinate(dct_c) for dct_c in dct['geom']['points']]
    start_node = decode_node_id(dct['startNodeId'])
    end_node = decode_node_id(dct['endNodeId'])
    # Python magic: the mutable lists in the default dictionary will get
    # filled while we build the structure
    incomingLinksByNode[end_node].append(link_id)
    outgoingLinksByNode[start_node].append(link_id)
    res[link_id] = DefaultLink(link_id, geom, dct['length'], 
                              outgoingLinksByNode[end_node], 
                              incomingLinksByNode[start_node])
    print 'finished line', line
  fin.close()
  return res

def lineNetwork(num_links, link_length=10.0):
  net = {}
  link_ids = ["%d"%idx for idx in range(num_links)]
  coords = [Coordinate(link_length * idx, 0.0) for idx in range(num_links+1)]
  for idx in range(num_links):
    geom = [coords[idx], coords[idx+1]]
    incoming = [] if idx == 0 else [link_ids[idx-1]]
    outgoing = [] if idx == num_links - 1 else [link_ids[idx+1]]
    link = DefaultLink(link_id=link_ids[idx],
                      geom=geom,
                      length=link_length,
                      outgoing_links=outgoing,
                      incoming_links=incoming)
    net[link_ids[idx]] = link
  return net

def circleNetwork(num_links, radius=30.0):
  net = {}
  link_ids = ["%04d"%idx for idx in range(num_links)]
  coords = [Coordinate(radius * np.cos(2*np.pi*idx/float(num_links)), radius * np.sin(2*np.pi*idx/float(num_links))) for idx in range(num_links+1)]
  for idx in range(num_links):
    geom = [coords[idx], coords[idx+1]]
    incoming = [link_ids[(idx-1)%num_links]]
    outgoing = [link_ids[(idx+1)%num_links]]
    link = DefaultLink(link_id=link_ids[idx],
                      geom=geom,
                      length=np.pi * radius / num_links,
                      outgoing_links=outgoing,
                      incoming_links=incoming)
    net[link_ids[idx]] = link
  return net

