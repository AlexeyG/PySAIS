#!/bin/env python
from distutils.core import setup, Extension
import numpy as np

pysais = Extension('pysais',
    sources = ['sais.c', 'pysais.c'],
    )

setup(name = 'Py-SAIS',
      version = '0.1',
      description = 'A Python module wrapper for the SA-IS algorithm.',
      author = 'Alexey A. Gritsenko',
      author_email = 'a.gritsenko@tudelft.nl',
      long_description = 'A Python module wrapper for the SA-IS algorithm implementation by Yuta Mori.',
      include_dirs = [np.get_include() + '/numpy'],
      ext_modules = [pysais])
