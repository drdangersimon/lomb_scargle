"""used to make wrapper from boost c++ code on cython code"""
 
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from glob import glob
from shutil import move
import os

setup(name="PackageName",
    ext_modules=[
        Extension("cpp_imp", ["cpp_imp.cpp"],
        libraries = ["boost_python"])
    ])
setup(
    ext_modules = cythonize("*.pyx")
)

# move .so files from build
to_copy = glob('build/lib*/*.so')
if len(to_copy) == 0:
    print 'type $python setup.py build to test'
[move(copy_path, os.path.abspath('.')+'/') for copy_path in to_copy]

