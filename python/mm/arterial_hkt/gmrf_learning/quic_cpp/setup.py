'''
Created on May 23, 2012

@author: tjhunter

To build the extension:
go into this directory
python setup.py build_ext --inplace
'''
import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext

setup(
    cmdclass = {'build_ext': build_ext},
    ext_modules = [Extension("sic", ["optim_barrier.pyx"],
                             include_dirs=[np.get_include()])]
)