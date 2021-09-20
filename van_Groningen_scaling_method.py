# Let's start by importing
from lmfit import minimize, Parameters, report_fit
from scipy.integrate import simps
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Models the continuum as a straight line or first order polynomial.
def cont_fit(params, x):
    a = params['a'].value # Slope
    b = params['b'].value # Intercept
    
    return a*x + b

# Function that compute the difference between the fit iteration and data
cont_fit_err = lambda p, x, y: cont_fit(p, x) - y

# This tells where the continuum region to be fitted is, and fits it.
def region_around_line(x, y, region):
    indcont = ((x >= region[0][0]) & (x <= region[0][1])) | ((x >= region[1][0]) & (x <= region[1][1]))
    indrange = (x >= region[0][0]) * (x <= region[1][1])
    
    params = Parameters()
    params.add('a', value = 0)
    params.add('b', value = 0)
    
    # Perform minimization
    fitout = minimize(cont_fit_err, params, args = (x[indcont], y[indcont]))
    continuum = cont_fit(fitout.params, x[indrange])
    
    return x[indrange], y[indrange] - continuum

# This function will derive the three parameters required to scale the input spectrum
def fit_profile(params, wavelength, data):
    dx = params['dx'].value # Wavelength shift
    sigma = params['sigma'].value # Resolution
    amplitude = params['amplitude'].value # Flux scaling factor
    
    center = wavelength[np.argmax(data)]
    g = (wavelength - (center + dx)) / sigma
    gaussian = amplitude * np.exp(-.5 * g ** 2)
    convolved = np.convolve(data, gaussian, mode = 'same') # Convolve
    
    return convolved

# Function to minimize the difference between the model spectrum and input spectrum
def fit_profile_err(p, x, y, z):
    residuals = fit_profile(p, x, y) - z

    return residuals

def main():
    # Read in reference/model spectrum
    ref = np.loadtxt('ref.txt', unpack = True)

    # Read in input spectrum to be scaled
    spec = np.loadtxt(sys.argv[1], unpack = True)

    region = [[6507, 6512], [6560, 6565]] # Background or continuum region

    ref_w, ref_f = region_around_line(ref[0], ref[1], region)
    spec_w, spec_f = region_around_line(spec[0], spec[1], region)
    
    divisor = np.max(ref_f) # To normalize spectrum
    
    ref_f, spec_f = ref_f / divisor, spec_f / divisor

    # Initial parameters
    params = Parameters()
    params.add('dx', value = 0)
    params.add('sigma', value = 1)
    params.add('amplitude', value = 1)
    
    # Minimize
    fitout = minimize(fit_profile_err, params, args = (spec_w, spec_f, ref_f))
    fitted_p = fitout.params
    scaled = fit_profile(fitted_p, spec_w, spec_f)

    # Best-fit parameters
    report_fit(fitout)
    
    # Save new scaled spectrum
    save_scaled_spectrum = True
    if save_scaled_spectrum:
        ref_center = ref_w[np.argmax(ref_f)] # Model spectrum center
        new_center = spec_w[np.argmax(scaled)] # This center is from the fit
        
        shift = ref_center - new_center # Wavelength shift
        spec_w_new = shift + spec[0] # Apply shift
    
        scaling_factor = simps(ref_f, ref_w) / simps(scaled, spec_w)  # To scale the spectrum
        spec_f_new = spec[1] * scaling_factor # Scaled
        spec_e_new = spec[2] * scaling_factor
    
        # Re-bin
        spec_f_new = np.interp(ref[0], spec_w_new, spec_f_new)
        spec_e_new = np.interp(ref[0], spec_w_new, spec_e_new)
    
        out_name = os.path.splitext(sys.argv[1])[0].split('.fits')[0] + '_scaled.txt'
    
        # Save
        np.savetxt(out_name, np.transpose([ref[0], spec_f_new, spec_e_new]))
    
    plt.plot(ref_w, ref_f)
    plt.plot(spec_w, scaled)

    plt.show()

main()
