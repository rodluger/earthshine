#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division, print_function, absolute_import
from setuptools import setup


# Setup!
setup(name='earthshine',
      version="0.0.1",
      description="Earthshine in TESS",
      long_description="",
      url='http://github.com/rodluger/everest',
      author='Rodrigo Luger',
      author_email='rodluger@gmail.com',
      license='MIT',
      packages=['earthshine'],
      install_requires=[
          'numpy>=1.8',
          'scipy',
          'matplotlib',
          'starry==1.0.0.dev0',
          'healpy'
      ],
      dependency_links=[
        'https://github.com/rodluger/starry/tarball/linear#egg=starry-1.0.0.dev0'
      ],
      include_package_data=False,
      zip_safe=False
      )