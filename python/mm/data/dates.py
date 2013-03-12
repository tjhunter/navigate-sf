'''
Created on Feb 7, 2012

@author: tjhunter
'''

def parse_date(date_str):
  (y,m,d) = date_str.split('-')
  return (int(y), int(m), int(d))

def date_range(date_from, date_to):
  res = []
  (y,m,d) = (date_from)
  (y_to, m_to, d_to) = date_to
  while y < y_to or (y==y_to and m < m_to) or (y==y_to and m == m_to and d <= d_to):
    res.append((y,m,d))
    d += 1
    if d > 31:
      m += 1
      d = 1
    if m > 12:
      y += 1
      m = 1
  return res

def date_str(date):
  (y, m, d) = date
  return "%04i-%02i-%02i" % (y, m, d) 

def date_range_str(range_str):
  (start, end) = range_str.split(":")
  return [date_str(d) for d in date_range(parse_date(start), parse_date(end))]