'''
Created on Feb 1, 2013

@author: audehofleitner
'''

import matplotlib
#matplotlib.use("Agg")
from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Times New Roman']})
#matplotlib.rcParams['ps.useafm'] = True
#matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
rc('font', **{'family':'serif'})

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as stats
import math

font = {'size'   : 12}
lines = {'linewidth': 2}
legend = {'fontsize':10}
matplotlib.rc('legend',**legend)
matplotlib.rc('font', **font)
matplotlib.rc('lines', **lines)
COLORS = ['b', 'g', 'r', 'c', 'm', 'k', 'y']
LINE_STYLE = ['-', '--', '-.', ':']
MARKER = ['o', 's', 'v', '*', '^', 'x']
HATCH = [ '/' , '|' , '-' , '+' , 'x' , 'o' , 'O' , '.' , '*' ]

# Jack added this, looks better?
# from matplotlib import rc
#rc('font',**{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

def cumulative(points, models):
  """
  points = array of experimental travel times
  """
  points.sort()
  cum = np.linspace(0, 1, len(points))
  plt.plot(points, cum, 'k*', linewidth=5, label='Validation data')

  min_x = 0
  max_x = max(points)
  xx = np.linspace(min_x, max_x, 100)
  for i, (name, dist) in enumerate(models.items()):
    learned_cum = dist.cumulatives(xx)
    plt.plot(xx, learned_cum,
             '{0}{1}'.format(COLORS[(i + 1) % len(COLORS)],
                                LINE_STYLE[(i + 1) % len(LINE_STYLE)]),
             label=name)
  min_y = 0
  max_y = 1
  plt.xlim([min_x, max_x])
  plt.ylim([min_y, max_y])
  
  
def pp_plot(points, models):
  """
  points = array of experimental travel times
  """
  points.sort()
  cum = np.linspace(0, 1, len(points))
  plt.plot([0, 1], [0, 1], '-b', linewidth=0.5, label='Perfect estimation')

  for i, (name, dist) in enumerate(models.items()):
    learned_cum = dist.cumulatives(points)
    plt.plot(learned_cum, cum,
             '{0}{1}'.format(COLORS[(i + 1) % len(COLORS)],
                                LINE_STYLE[(i + 1) % len(LINE_STYLE)]),
             label=name)
  plt.xlim([0, 1])
  plt.ylim([0, 1])
  
  
def qq_plot(points, models):
  """
  points = array of experimental travel times
  """
  points.sort()
  cum = np.linspace(1.0 / (1 + len(points)),
                    len(points) / (1.0 + len(points)),
                    len(points))
  plt.plot(points, points, '-b', linewidth=0.5, label='Perfect estimation')
  xmin = 400
  xmax = 0
  for i, (name, dist) in enumerate(models.items()):
    learned_quantiles = np.zeros_like(cum)
    b = None
    for j, c in enumerate(cum):
      learned_quantiles[j] = dist.inverseCumulative(c, low_bound=b)
      b = learned_quantiles[j]
    plt.plot(learned_quantiles, points,
             '{0}{1}'.format(COLORS[(i + 1) % len(COLORS)],
                                LINE_STYLE[(i + 1) % len(LINE_STYLE)]),
             label=name)
    xmin = min(xmin, learned_quantiles[0])
    xmax = max(xmax, learned_quantiles[-1])
  plt.xlim([xmin, xmax])
  plt.ylim([points[0], points[-1]])


def ksdensity_(obs, models):
  # hist = plt.hist(obs, normed=True, facecolor='c', label='Validation data')
  kde = stats.kde.gaussian_kde(obs)
  xmin = 0
  xmax = max(obs)
  xx = np.linspace(xmin, xmax, 100)
  yy = kde(xx)
  plt.plot(xx, yy, '{0}{1}'.format(COLORS[0], LINE_STYLE[0]), label='empirical')
  for i, (name, dist) in enumerate(models.items()):
    y_learned = dist.probabilities(xx)
    plt.plot(xx, y_learned,
             '{0}{1}'.format(COLORS[(i + 1) % len(COLORS)],
                                LINE_STYLE[(i + 1) % len(LINE_STYLE)]),
             label=name)

  # lims
  plt.xlim([xmin, xmax])
  # plt.grid(True)


def scatter_box(bounds,
                observations):
  ax = plt.gca()
  
  interval = 1.0 / (len(bounds) + 1)
  # plt.scatter(observations, [0.05] * len(observations), marker='o', s=20, facecolor='none', c='k')
  max_b = max([b[-1] for b in bounds.values()])
  xmin = math.floor(100 * min(observations)) / 100.0
  xmax = math.ceil(max(observations) * 100) / 100.0
  for i, (model, bound) in enumerate(bounds.items()):
    xmin = min(xmin, bound[0])
    xmax = max(xmax, bound[-1])
    plt.axvspan(bound[0], bound[-1],
                i * interval,
                (i + 0.8) * interval,
                hold=True,
                facecolor=COLORS[i + 1], alpha=0.5)
    plt.axvspan(bound[1], bound[-2],
                i * interval,
                (i + 0.8) * interval,
                hold=True,
                facecolor=COLORS[i + 1], alpha=0.7,
                label=model)
    plt.plot([bound[2]] * 2, [i * interval, (i + 0.8) * interval], '-k')
    # Only put label for one drawing of the validation data
    if i == 0:
      plt.scatter(observations, [(i + 0.4) * interval] * len(observations),
                marker='x', s=20, c='k',
                linewidth=1, zorder=1000 + i)
    else:
      plt.scatter(observations, [(i + 0.4) * interval] * len(observations),
                marker='x', s=20, c='k',
                linewidth=1, zorder=1000 + i)
  observations.sort()
  n = len(observations)
  plt.axvspan(observations[int(0.05 * n)], observations[int(0.95 * n)],
              (i + 1) * interval,
              (i + 1.8) * interval,
              hold=True,
              facecolor='w', alpha=0.5)
  plt.axvspan(observations[int(0.25 * n)], observations[int(0.75 * n)],
              (i + 1) * interval,
              (i + 1.8) * interval,
              hold=True,
              hatch=HATCH[0],
              facecolor='w', alpha=0.7,
              label='Validation points')
  plt.plot([observations[int(0.5 * n)]] * 2, [(i + 1) * interval, (i + 1.8) * interval], '-k')
#  plt.scatter(observations, [(i + 1.4) * interval] * len(observations),
#                marker='x', s=20, linewidth=1, 
#                c='k', zorder=1000 + i + 1)
  # plt.boxplot(observations, vert=False, positions=(i + 1.4) * interval)
  n_ticks = 5
  plt.ylim(0, 1)
  plt.xlim(xmin - 5, xmax + 5)
#  tiks = (xmax - xmin + 10) * np.arange(0,1.0,1.0/n_ticks) + (xmin - 5)
#  ax.set_xticks(tiks)
  ax.locator_params(nbins=n_ticks)
  ax.yaxis.set_visible(False)
  ax.yaxis.set_ticks_position("none")


def validate_intervals(conf_levels, conf):
  plt.plot(conf_levels,
           conf_levels,
           '{0}{1}{2}'.format(LINE_STYLE[0],
                              MARKER[0],
                              COLORS[0]),
           label='perfect estimation')
  for i, (lab, c) in enumerate(conf.items()):
    plt.plot(conf_levels,
             c[0],
             '{0}{1}{2}'.format(LINE_STYLE[(i + 1) % len(LINE_STYLE)],
                              MARKER[(i + 1) % len(MARKER)],
                              COLORS[(i + 1) % len(COLORS)]),
             label='{0}: a={1:.3f}, b={2:.3f}'.format(lab, c[1], c[2]))
  # plt.xlabel('Level of confidence of the interval')
  # plt.ylabel('\\% of obs. in the interval')
  plt.xlim(0, 1)
  plt.ylim(0, 1)
  plt.legend(loc=2)


def likelihood_dist_per_bins(ll, bin_width):
  all_stds = []
  all_means = []
  for model_ll in ll.values():
    xmin = np.min(model_ll.keys()) * bin_width
    xmax = (np.max(model_ll.keys()) + 1) * bin_width
    break
  for model_ll in ll.values():
    all_means += [m[0] for m in model_ll.values()]
    all_stds += [m[1] for m in model_ll.values()]
#  min_ll, max_ll = min(all_means), max(all_means)
  abs_mean = np.abs(all_means)
  abs_mean.sort()
  # max_std = abs_mean[-4]
  for i, (lab, model_ll) in enumerate(ll.items()):
    means = [m[0] for m in model_ll.values()]
#    stds = [m[1] for m in model_ll.values()]
    bins = [(k + 0.5) * bin_width for k in model_ll.keys()]
    plt.plot(bins, means,
             '{0}{1}{2}'.format(LINE_STYLE[(i + 1) % len(LINE_STYLE)],
                              MARKER[(i + 1) % len(MARKER)],
                              COLORS[(i + 1) % len(COLORS)]),
             label=lab)
#    for m, s, b in zip(means, stds, bins):
#      plt.plot([b, b],
#               [m - s, m + s],
#               '-{0}{1}'.format(MARKER[(i + 1) % len(MARKER)],
#                                COLORS[(i + 1) % len(COLORS)]))
  plt.xlabel('Length of the path')
  plt.ylabel('Average loglikelihood')
  #plt.ylim(min_ll - max_std, max_ll + max_std)
  plt.xlim(xmin, xmax)
  plt.legend()
