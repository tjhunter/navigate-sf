'''
Created on Jan 10, 2012

@author: tjhunter

Encoding/decoding conversions to the common JSON format.
'''
from structures import Coordinate
import datetime
from mm.data.structures import Spot, Route, TSpot

def encode_Coordinate(gps):
  return {'lat' : gps.lat, 'lon' : gps.lng}

def decode_Coordinate(dct):
  return Coordinate(dct['lat'], dct['lon'])

def encode_link_id(link_id):
  (primary, secondary) = link_id
  return {'primary': primary, 'secondary': secondary}

def decode_link_id(dct):
  return (dct['primary'], dct['secondary'])

def decode_node_id(dct):
  return (dct['primary'], dct['secondary'])

def encode_Spot(spot):
  res = {}
  res['linkId'] = encode_link_id(spot.linkId)
  res['offset'] = spot.offset
  if spot.coordinate:
    res['coordinate'] = encode_Coordinate(spot.coordinate)
  return res

def decode_Spot(dct):
  coordinate = decode_Coordinate(dct['coordinate']) if 'coordinate' in dct else None
  return Spot(decode_link_id(dct['linkId']), \
               dct['offset'], coordinate)

def decode_Route(dct):
  links = [decode_link_id(dct2) for dct2 in dct['links']]
  spots = [decode_Spot(dct2) for dct2 in dct['spots']]
  geometry = None
  if 'geometry' in dct:
    geometry = [decode_Coordinate(c_dct) for c_dct in dct['geometry']['points']]
  return Route(links, spots, geometry)

def encode_time(time):
  return {'year':time.year, \
          'month':time.month, \
          'day':time.day, \
          'hour':time.hour, \
          'minute':time.minute, \
          'second':time.second}

def decode_time(dct):
  return datetime.datetime(dct['year'], dct['month'], dct['day'], \
                           dct['hour'], dct['minute'], dct['second'])

def decode_TSpot(dct):
  speed = dct['speed'] if 'speed' in dct else None
  hired = dct['hired'] if 'hired' in dct else None
  coord = decode_Coordinate(dct['coordinate'])
  spot = decode_Spot(dct['spots'][0])
  return TSpot(spot, dct['id'],
               decode_time(dct['time']),
               hired, speed, obsCoordinate=coord)

