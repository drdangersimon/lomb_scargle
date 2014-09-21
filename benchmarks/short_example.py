# from scipy
from time import time
import numpy as np
import pylab as lab


def scipy_example(periodogram_fn, show=False):
    '''Uses periodgoram_fn to calculate periodogram and times it'''
    # First define some input parameters for the signal:
    A = 2.
    w = 1.
    phi = 0.5 * np.pi
    nin = 1000
    nout = 100000
    frac_points = 0.9 # Fraction of points to select

    # Randomly select a fraction of an array with timesteps:

    r = np.random.rand(nin)
    x = np.linspace(0.01, 10*np.pi, nin)
    x = x[r >= frac_points]
    normval = x.shape[0] # For normalization of the periodogram
     
    # Plot a sine wave for the selected times:

    y = A * np.sin(w*x+phi)

    #Define the array of frequencies for which to compute the periodogram:
    
    f = np.linspace(0.01, 10, nout)
     
    # Calculate Lomb-Scargle periodogram:

    t = []
    for i in xrange(5):
        t.append(time())
        pgram = periodogram_fn(x, y, f)
        t[-1] = time() - t[-1]

    # Now make a plot of the input data:
    fig = lab.figure()
    plt1 = fig.add_subplot(2, 1, 1)
    plt1.plot(x, y, 'b+')

    # Then plot the normalized periodogram:

    plt2 = fig.add_subplot(2, 1, 2)

    plt2.plot(f, np.sqrt(4*(pgram/normval)))
    if show:
        lab.show()
    return t, fig
