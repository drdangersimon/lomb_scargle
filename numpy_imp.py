import numpy as np
import numexpr as ne
'''Numpy implimantation of lomb-scargle periodgram'''


def lombscargle_num(x, y, freqs):
    
    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    # Create empty array for output periodogram
    pgram = np.empty(freqs.shape[0], dtype=np.float64)

    for i in xrange(freqs.shape[0]):
        c = np.cos(freqs[i] * x)
        s = np.sin(freqs[i] * x)
        xc = np.sum(y * c)
        xs = np.sum(y * s)
        cc = np.sum(c**2)
        ss = np.sum(s**2)
        cs = np.sum(c * s)
        
        tau = np.math.atan2(2 * cs, cc - ss) / (2 * freqs[i])
        c_tau = np.cos(freqs[i] * tau)
        s_tau = np.sin(freqs[i] * tau)
        c_tau2 = c_tau * c_tau
        s_tau2 = s_tau * s_tau
        cs_tau = 2 * c_tau * s_tau

        pgram[i] = 0.5 * (((c_tau * xc + s_tau * xs)**2 / \
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + \
            ((c_tau * xs - s_tau * xc)**2 / \
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))

    return pgram


def lombscargle_ne(x, y, freqs):
    '''uses numexp to do numpy stuff'''
    # Check input sizes
    if x.shape[0] != y.shape[0]:
        raise ValueError("Input arrays do not have the same size.")

    # Create empty array for output periodogram
    pgram = np.empty(freqs.shape[0], dtype=np.float64)

    for i in xrange(freqs.shape[0]):
        f = freqs[i]
        c = ne.evaluate('cos(f * x)')
        s = ne.evaluate('sin(f * x)')
        xc = ne.evaluate('sum(y * c)')
        xs = ne.evaluate('sum(y * s)')
        cc = ne.evaluate('sum(c**2)')
        ss = ne.evaluate('sum(s**2)')
        cs = ne.evaluate('sum(c * s)')
        
        tau = ne.evaluate('arctan2(2 * cs, cc - ss) / (2. * f)')
        c_tau = ne.evaluate('cos(f * tau)')
        s_tau = ne.evaluate('sin(f * tau)')
        c_tau2 = ne.evaluate('c_tau * c_tau')
        s_tau2 = ne.evaluate('s_tau * s_tau')
        cs_tau = ne.evaluate('2 * c_tau * s_tau')

        pgram[i] = ne.evaluate('''0.5 * (((c_tau * xc + s_tau * xs)**2 / 
            (c_tau2 * cc + cs_tau * cs + s_tau2 * ss)) + 
            ((c_tau * xs - s_tau * xc)**2 / 
            (c_tau2 * ss - cs_tau * cs + s_tau2 * cc)))''')

    return pgram
