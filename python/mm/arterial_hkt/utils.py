'''
Created on Feb 18, 2013

@author: tjhunter
'''
import time
try:
  from cPickle import dumps, loads
except ImportError:
  from pickle import dumps, loads



def tic(msg, experiment_name=None):
  print "[%12.4f] %s: %s" % (time.time(), (experiment_name if experiment_name else ""), msg)

"""
The following code comes from:
http://code.google.com/p/streaming-pickle/
"""



def s_dump(iterable_to_pickle, file_obj):
  '''dump contents of an iterable iterable_to_pickle to file_obj, a file
  opened in write mode'''
  for elt in iterable_to_pickle:
    s_dump_elt(elt, file_obj)


def s_dump_elt(elt_to_pickle, file_obj):
  '''dumps one element to file_obj, a file opened in write mode'''
  pickled_elt_str = dumps(elt_to_pickle)
  file_obj.write(pickled_elt_str)
  # record separator is a blank line
  # (since pickled_elt_str might contain its own newlines)
  file_obj.write('\n\n')


def s_load(file_obj):
  '''load contents from file_obj, returning a generator that yields one
  element at a time'''
  cur_elt = []
  for line in file_obj:
    cur_elt.append(line)

    if line == '\n':
      pickled_elt_str = ''.join(cur_elt)
      elt = loads(pickled_elt_str)
      cur_elt = []
      yield elt