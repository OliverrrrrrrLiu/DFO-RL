#!/usr/bin/env python
# (C)2011 Arpad Buermen
# (C)2018 Jaroslav Fowkes, Lindon Roberts
# Licensed under GNU GPL V3

#
# Do not edit. This is a computer-generated file.
#

# Ensure compatibility with Python 2
from __future__ import absolute_import, division, print_function, unicode_literals

from distutils.core import setup, Extension
import os
import numpy as np
from subprocess import call
from glob import glob

#
# OS specific
#


define_macros=[('LINUX', None)]
include_dirs=[os.path.join(np.get_include(), 'numpy')]
objFileList=glob('*.o')
objFileList.append('/usr/local/opt/cutest/libexec/objects/mac64.osx.gfo/double/libcutest.a')
libraries=['gfortran']
library_dirs=[max(glob('/usr/local/Cellar/gcc/*/lib/gcc/*/'),key=os.path.getmtime),'/usr/local/gfortran/lib/']
extra_link_args=['-Wl,-no_compact_unwind']


#
# End of OS specific
#

# Module
module1 = Extension(
      str('_pycutestitf'),
      [str('cutestitf.c')],
      include_dirs=include_dirs,
      define_macros=define_macros,
      extra_objects=objFileList,
      libraries=libraries,
      library_dirs=library_dirs,
      extra_link_args=extra_link_args
    )

# Settings
setup(name='PyCUTEst automatic test function interface builder',
    version='1.0',
    description='Builds a CUTEst test function interface for Python.',
    long_description='Builds a CUTEst test function interface for Python.',
    author='Arpad Buermen, Jaroslav Fowkes, Lindon Roberts',
    author_email='arpadb@fides.fe.uni-lj.si, fowkes@maths.ox.ac.uk, robertsl@maths.ox.ac.uk',
    url='',
    platforms='Linux',
    license='GNU GPL',
    packages=[],
    ext_modules=[module1]
)
