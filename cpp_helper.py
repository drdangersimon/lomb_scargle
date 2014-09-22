import numpy as np
from cpp_imp import lomb_cpp


def cpp_imp(x, y, f):
    '''C++ implimation of lomb-scargle. This is a helper function that calls
    the C++ code'''
    # make sure C order and numpy arrays
    x = np.asarray(x, order='C')
    y = np.asarray(y, order='C')
    f = np.asarray(f, order='C')
    # create out function
    out = np.empty_like(f, order='C')
    lomb_cpp(x, y, f, out)
    return out

if __name__ == '__main__':
    import benchmarks
    import pylab as lab
    print benchmarks.short_example.scipy_example(cpp_imp)[0]
    lab.show()
