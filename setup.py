"""used to make wrapper from boost c++ code"""
 
from distutils.core import setup
from distutils.extension import Extension
 
setup(name="PackageName",
    ext_modules=[
        Extension("cpp_imp", ["cpp_imp.cpp"],
        libraries = ["boost_python"])
    ])
