'''
Created on Jan 10, 2012

@author: tjhunter

All the data structures as used in the python files.
'''

class Coordinate(object):
  """ A geolocation representation.
  """
  def __init__(self, lat, lng):
    self.lat = lat
    self.lon = lng

  def __eq__(self, other):
    return self.lat == other.lat and self.lon == other.lon

  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __repr__(self):
    return "Coord(%s,%s)"%(self.lat,self.lon)


class Spot(object):
  """ Spot representation.
  """
  
  def __init__(self, link_id, offset, coordinate=None):
    self.linkId = link_id
    self.offset = offset
    self.coordinate = coordinate
    
  def __eq__(self, other):
    return self.linkId == other.linkId and int(10 * self.offset) == int(10 * other.offset)
    
  def __ne__(self, other):
    return not self.__eq__(other)
  
  def __repr__(self):
    return "Spot[%s, %.1f]" % (str(self.linkId), self.offset)


class Route(object):
  """ Representation of a netconfig.Route
  """
  
  def __init__(self, links, spots, geometry=None):
    self.link_ids = links
    self.spots = spots
    self.geometry = geometry
    #self.length = sum([link.length for link in self.link_ids[:-1]]) - self.firstSpot.offset + self.lastSpot.offset
    
  @property
  def firstSpot(self):
    return self.spots[0]
  
  @property
  def lastSpot(self):
    return self.spots[-1]
  
  @staticmethod
  def fromPair(links, start_spot, end_spot):
    return Spot(links, [start_spot, end_spot])

  def __repr__(self):
    return "Route[%s,link_ids=%s,%s]" % (str(self.firstSpot), str(self.link_ids), str(self.lastSpot))

class TSpot(object):
  """ A timed spot.
  """
  def __init__(self, spot, vehicle_id, time, 
               hired=None, speed=None, obsCoordinate=None):
    self.spot = spot
    self.id = vehicle_id
    self.time = time
    self.hired = hired
    self.speed = speed
    self.obsCoordinate = obsCoordinate

  def __repr__(self):
    return "TSpot[%s,%s, %s]" % (str(self.spot), str(self.time), str(self.id)) 

class RouteTT(object):
  """ Python representation of the RouteTT object
  """
  
  def __init__(self, route, start_time, end_time, vehicle_id=None):
    self.route = route
    self.startTime = start_time
    self.endTime = end_time
    self.id = vehicle_id
    self.tt = (self.endTime - self.startTime).total_seconds()
    #self.vel = self.route.length / self.tt

  def __repr__(self):
    return "RouteTT[%s, %s, %s, %s]" % (str(self.id), str(self.startTime), str(self.endTime), str(self.route)) 

# TODO(?) rename to something else
class Point_pts(object):
  """ A point in the time-space diagram associated with time, space and speed
  """
  
  def __init__(self, space, time, speed):
    self.space = space
    self.time = time
    self.speed = speed

class CutTrajectory(object):
  """ A simplified trajectory for the stop/go model.
  """
  
  def __init__(self, rtts):
    """
    Arguments:
     - rtts: list of RouteTT objects
    """
    self.pieces = rtts
    self.numPieces = len(self.pieces)
