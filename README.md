navigate-sf
======================

Travel time estimation model for arterial

Description
------------

Name: Tim Hunter, Tasos Kouvelas, Jerome Thai, Aude Hofleitner, Walid Krichène, Jack Reilly

Group: Arterial estimation

Goal: The goal is to predict and estimate travel times and link traffic state. The model uses streaming probe data
through the path inference, and returns global travel time estimators (with link correlation), and network state.
The model is statistical only and models travel times at link level using mixtures of Gaussian distribution.
Correlations are inputed using HMM and GMRF.

This project also provides data compression strategies for high- to medium-frequency sampling of probe data
(stop/go model) and tools for performance evaluation of travel time-based models.


Data format
============

Network file
------
A network is a list of json dicts, each on one line (no line breaks inside of a dict), where each dict specifies a link.

* id: a dict with fields 'primary' and 'secondary', where 'primary' is the link id and 'secondary' is the direction on the link (0 or 1).
* length: length of link
* startNodeID/endNodeID: intersection id of the beginning/end of the link. Node id's are used to define the topology.
* geom: dictionary with one field: 'points', where 'points' is a list of dictionaries with fields of 'lat' and 'lon'.

Example link dict:

```
{
"id": {
    "primary": 123,
    "secondary": 0
    },
"length": 120.2,
"startNodeId": {
    "primary": 847489,
    "secondary": 0
    },
"endNodeId": {
    "primary": 84740,
    "secondary": 0
    },
},
"geom": {
    "points": [
      {"lat": 30.3, "lon": -122.2},
      {"lat": 30.33, "lon": -122.1},
      ...
    ]
    }
}
```

Trajectory file
------
A trajectory file is a list of json dicts, each on one line (no line breaks inside of a dict), where each dict specifies a point and time along a trajectory. Each dict has only one field: 'point' (for historical reasons). A 'point' is a dict with the following fields:

* time: a dict with the following fields: 'year', 'month', 'day', 'hour', 'minute', 'second'.
* spots: a list with only one entry: a dict with the following fields:
    * linkId: a dict representing a link id as specified in the "Network" section above.
    * offset: an offset distance from the start of the link.

Example spot dict:

```
{
"point": {
    "time": {
        "year": 2012, "month": 3, "day": 13, "hour": 12, "minute": 13, "second": 14
    },
    "spots": [{
        "linkId": {
            "primary": 1,
            "secondary": 0
            },
        "offset": 0.5
    }]
}
```

$DATA_DIR directory format
------
Look at "Setting up the environment" to set up $DATA_DIR environment variable.

The folder contains both the data for constructing the network and the trajectory files to use as input for learning.

The network filename is specified in the parameters of the pipeline script, where the $DATA_DIR directory is assumed to be the root folder.

Inside the $DATA_DIR directory is another folder called 'trajectories', which contains a folder for each date. Each folder date has a list of json trajectory files.

Example $DATA_DIR folder:

```
$DATA_DIR/
    small_network.json
    big_network.json
    trajectories/
        2013-03-12/
            any_name.json
            any_other_name.json
            any_name_6.json
            …
        2013-03-13/
            …
```


Setting up the environment
==========================

\* The QUIC module is not required for running the pipeline

Dependencies
------------


### Python ###

You have to install the following python packages:

+ numpy
+ scipy
+ [matplotlib](http://matplotlib.org/)
+ [joblib](http://pypi.python.org/pypi/joblib)
+ [scikit-learn](http://scikit-learn.org/stable/)
+ nose (optional, for automated testing)
+ pylint (optional, for code quality assessment)

The GMRF learning algorithm requires the following packages:

+ [scikit-sparse]() to use the sparse Cholesky decomposition
+ The QUIC algorithm, which is embedded in the code (optional soon, see below for compilation instructions)


### Compiling the QUIC module ###
There is a Makefile to run in the directory ```python/mm/arterial_hkt/gmrf_learning/quic_cpp/```. It has to be adapted for Mac computers


### Environment variables ###

Edit your ```.bashrc``` to include the following:

```
export DATA_DIR='location of the folder that contains the data'
export PYTHONPATH=<location of the folder navigate-sf>/navigate-sf/python:$PYTHONPATH
```

Development
============

The files follow this naming convention:

 - my_data_structure.py for files containing a class called MyDataStructure and a few functions
 - my_data_structure_functions.py contains long, complicated functions related to MyDataStructure
 - my_data_structure_test[_{1,2,3,4,...}].py is a set of tests related to MyDataStructure
 - my_data_structure_valid[_{1,2,3,4}].py is some potentially useful random code to plot, test, look into MyDataStructure (the scratchpad). Not covered by tests or by code coverage.
 - launch_rocket_script.py executable script that one can run from the command line.

*Tabulation* Default tabulation is **2 spaces** (and not 4 as in regular python code). **No tab allowed**

Global functions, fields and methods of classes should follow the camelCase convention. Variables should
follow the underscore_every_where convention.