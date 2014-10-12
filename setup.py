#!/bin/env python
from distutils.core import setup, Extension
from os import environ

pysais = Extension('pysais',
    sources = ['sais.c', 'pysais.c'],
    #libraries = ['gsl', 'gslcblas'],
    library_dirs = ['/home/nfs/alexeygritsenk/env/sys_enhance/lib64/'])

setup(name = 'Py-SAIS',
      version = '0.1',
      description = 'A Python module wrapper for the SA-IS algorithm.',
      author = 'Alexey A. Gritsenko',
      author_email = 'a.gritsenko@tudelft.nl',
      long_description = 'A Python module wrapper for the SA-IS algorithm implementation by Yuta Mori.',
      include_dirs = ['/home/nfs/alexeygritsenk/env/sys_enhance/include/python2.7/', '/home/nfs/alexeygritsenk/env/sys_enhance/lib64/python2.7/site-packages/numpy/core/include/numpy/', '/home/nfs/alexeygritsenk/env/sys_enhance/include/'],
      ext_modules = [pysais])
