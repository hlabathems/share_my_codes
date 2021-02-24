import numpy as np
from scipy.special import wofz
from scipy.integrate import simps
from lmfit import minimize, Parameters, report_fit
from matplotlib import pyplot as plt
import argparse, os

def straight_line(x, m, c):
    '''
        Equation of a straight line
    '''
    return m * x + c

def voigt_plus_line(params, x):
    '''
        This function combines the Voigt line profile for the OIII line and a straight line for the continuum underneath.
    '''
    amp = params['amp'].value # Amplitude of the profile
    sigma = params['sigma'].value # The standard deviation of the Normal distribution part
    center = params['center'].value
    gamma = params['gamma'].value # The half-width at half-maximum of the Cauchy distribution part
    c = params['c'].value # Intercept
    m = params['m'].value # Slope

    z = (x - center + 1j * gamma) / (sigma * np.sqrt(2))

    return (amp * np.real(wofz(z))) / (sigma * np.sqrt(2 * np.pi)) + m * x + c

# Function that compute the difference between the fit iteration and data
def voigt_plus_line_err(p, x, y):
    return voigt_plus_line(p, x) - y

def extract_columns(spec):
    w, f, e = np.loadtxt(spec, usecols = (0, 1, 2), unpack = True)
    return w, f, e

def mask_wave_array(w, region):
    return (w >= region[0]) & (w <= region[1])

# Integrate using Simpson's rule
def integrate(w, f):
    return simps(f, w)

def output_results(result):
    fitout = f + result.residual
    m = result.params['m'].value
    c = result.params['c'].value
    continuum = straight_line(w, m, c)
    
    y_subtract = fitout - continuum
    f_tot = integrate(w, y_subtract)
    
    spec = os.path.basename(args.fname)
    spec_save = os.path.splitext(args.fname)[0]
    center = result.params['center'].value
    
    report_fit(result)
    
    print(spec, f_tot, center)
    
    plt.figure()
    plt.plot(w, f)
    plt.plot(w, fitout)
    plt.plot(w, continuum, color = 'r')
    plt.savefig('%s_voigt.pdf' % (spec_save), format = 'pdf')
    #plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'This program fits O[III]5007 line with the Voigt profile')
    parser.add_argument('--fname', help = 'The spectrum whose O[III]5007 line is to be fitted', type = str)
    parser.add_argument('--reg', nargs='+', help = 'Wavelength region of interest around the O[III]5007 line', type=int)
    
    args = parser.parse_args()

    w, f, e = extract_columns(args.fname)
    mask = mask_wave_array(w, [args.reg[0], args.reg[1]])
    w, f, e = w[mask], f[mask], e[mask]

    params = Parameters()
    params.add('amp', value = f.max())
    params.add('center', value = w[np.argmax(f)])
    params.add('sigma', value = 2)
    params.add('gamma', value = 0)
    params.add('m', value = 0)
    params.add('c', value = 0)

    result = minimize(voigt_plus_line_err, params, args = (w, f))

    output_results(result)
