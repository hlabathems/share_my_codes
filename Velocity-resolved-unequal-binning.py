import glob
import os
import sys
import warnings
import numpy as np
import PYCCF as myccf
from astropy import constants as const
import scipy
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings('ignore')

plt.rc('font', family='serif')
plt.rc('xtick', labelsize='medium')
plt.rc('ytick', labelsize='medium')

def get_specs(infiles):
    '''
        Unpacks input 1-D spectra, useful for constructing the mean
        
    Parameters:
    infiles (list or array): 1-D spectra
    
    Returns:
        tuple
    '''
    idx_sort = np.argsort(infiles)
    infiles =  [infiles[i] for i in idx_sort] # For some reason(s), files returned by glob are not sorted or ordered, this sorts them
    
    ws = []
    fs = []
    fes = []
    
    for infile in infiles:
        w, f, e = np.loadtxt(infile, usecols = (0, 1, 2), unpack = True)
        ws.append(w)
        fs.append(f)
        fes.append(e)
    
    return np.array(ws), np.array(fs), np.array(fes)

def plot_spec(x, y, bin_edges = None, coeffs = None):
    '''
        Plots spectrum indicating where the bin edges are if bin_edges != None.
    '''
    plt.figure()
    if bin_edges is not None:
        for idx in range(len(bin_edges)):
            plt.axvline(x = bin_edges[idx], color = 'sienna', ls = 'dotted')

    plt.plot(x, y, 'k')
    plt.xlabel('Velocity (km/s)', fontsize = 'x-large')
    plt.ylabel('Flux (arbitrary units)', fontsize = 'x-large')
    plt.show()

def plot_lcvs(time_series1, time_series2, label = ''):
    '''
        This plots continuum plus line (constructed from bins) light curves
    '''
    fig, ax = plt.subplots(2, 1, sharex = True)
    fig.subplots_adjust(hspace = 0.2)
    
    ax[0].errorbar(time_series1[0], time_series1[1], yerr = time_series1[2], fmt = '.', color = 'salmon', markeredgecolor = 'k', markeredgewidth = 0.5, elinewidth = 0.5, capsize = 0, label = 'LC (Continuum)')
    ax[1].errorbar(time_series2[0], time_series2[1], yerr = time_series2[2], fmt = '.', color = 'dodgerblue', markeredgecolor = 'k', markeredgewidth = 0.5, elinewidth = 0.5, capsize = 0, label = label)
    
    ax[0].set_ylabel('Flux (arbitrary units)', fontsize = 'x-large')
    ax[1].set_ylabel('Flux (arbitrary units)', fontsize = 'x-large')
    ax[1].set_xlabel('HJD - 2450000 (days)', fontsize = 'x-large')
    
    ax[0].legend(loc = 'best', fontsize = 'large')
    ax[1].legend(loc = 'best', fontsize = 'large')
    fig.suptitle(label, fontsize = 'large')
    plt.show()

def get_bin_centers(bin_edges):
    '''
        Calculates bin centers given bin edges
    '''
    bin_width = bin_edges[1:] - bin_edges[:-1]
    bin_centers = bin_edges[1:] - bin_width / 2.
    
    return bin_centers

def calc_ccf(time_series1, time_series2):
    '''
        Measures cross-correlation between time_series1 and time_series2. This uses PYCCF code that is publicly available.
        
    Parameters:
        time_series1 (tuple): Continuum light curve
        time_series2 (tuple): Line light curve
    
    Returns:
        List: Cross-correlation results
    
    References:
        See https://www.ascl.net/1805.032
    '''
    date1, flux1, err1 = time_series1
    date2, flux2, err2 = time_series2
   
    lag_range = [-10, 50]  # Time lag range to consider in the CCF (days). Must be small enough that there is some overlap between light curves at that shift (i.e., if the light curves span 80 days, these values must be less than 80 days)
    interp = 0.25  # Interpolation time step (days). Must be less than the average cadence of the observations, but too small will introduce noise.
    nsim = 2000  # Number of Monte Carlo iterations for calculation of uncertainties
    mcmode = 0  # 0 means use both FR/RSS method, 1 means RSS only, 2 means FR only in evaluating errors
    sigmode = 0.2 # Choose the threshold for considering a measurement "significant". sigmode = 0.2 will consider all CCFs with r_max <= 0.2 as "failed"
    imode = 0 # Cross-correlation mode: 0, twice (default); 1, interpolate light curve 1; 2, interpolate light curve 2.
    
    tlag_peak, status_peak, tlag_centroid, status_centroid, ccf_pack, max_rval, status_rval, pval = myccf.peakcent(date1, flux1, date2, flux2, lag_range[0], lag_range[1], interp)
    tlags_peak, tlags_centroid, nsuccess_peak, nfail_peak, nsuccess_centroid, nfail_centroid, max_rvals, nfail_rvals, pvals = myccf.xcor_mc(date1, flux1, abs(err1), date2, flux2, abs(err2), lag_range[0], lag_range[1], interp, nsim = nsim, mcmode = mcmode, imode = imode, sigmode = sigmode)
    
    lag = ccf_pack[1]
    r = ccf_pack[0]
    
    perclim = 84.1344746
    
    # Calculate 68% confidence interval on the median lag
    centau = scipy.stats.scoreatpercentile(tlags_centroid, 50)
    centau_uperr = (scipy.stats.scoreatpercentile(tlags_centroid, perclim)) - centau
    centau_loerr = centau - (scipy.stats.scoreatpercentile(tlags_centroid, (100. - perclim)))
    
    peaktau = scipy.stats.scoreatpercentile(tlags_peak, 50)
    peaktau_uperr = (scipy.stats.scoreatpercentile(tlags_peak, perclim)) - peaktau
    peaktau_loerr = peaktau - (scipy.stats.scoreatpercentile(tlags_peak, (100. - perclim)))
    
    return (tlags_centroid, tlags_peak, lag, r, centau, centau_uperr, centau_loerr, peaktau, peaktau_uperr, peaktau_loerr, max_rval)

def plot_ccf(lag, r, centtab, title, savefile = None):
    '''
        Plots the correlation function together with the centroid probability distribution.
    '''
    
    plt.figure()
    
    plt.plot(lag, r, color = 'k')
    plt.hist(centtab, lag, density = True, histtype = 'step', color = 'r', label = 'CCCD')
    
    plt.xlabel('Time Lag (days)', fontsize = 'x-large')
    plt.ylabel('r', fontsize = 'x-large')
    plt.title(title, fontsize = 'large')
    
    plt.ylim([-1, 1])
    plt.legend(loc = 'best', fontsize = 'large')
    plt.grid()
    plt.show()
    
def wav_to_vel(w, wc):
    '''
        Converts wavelength to velocity
        
    Parameters:
        w (float): Wavelength to be converted
        wc (float): Central wavelength or rest-frame wavelength of the broad emission-line
        
    Returns:
    float: Converted wavelength in km/s
    '''
    c = const.c.to('km/s').value
    
    return ((w - wc) / wc) * c

def vel_to_wav(v, wc):
    c = const.c.to('km/s').value

    return wc * ((v/c) + 1)

# Validate user input
def is_valid_file(parser, arg):
    
    if not os.path.isfile(arg):
        parser.error('The file {} does not exist!'.format(arg))
    else:
        return arg

# Validate user input
def is_valid_directory(parser, arg):
    
    if not os.path.isdir(arg):
        parser.error('The directory {} does not exist!'.format(arg))
    else:
        return glob.glob(arg + '/3C120*cont.txt') # Return continuum-subtracted spectra

def main():

    import argparse

    parser = argparse.ArgumentParser(description='This program measures the velocity-resolved time lags of the broad emission-line of the target AGN')
    parser.add_argument('-f1', help='LC (Continuum)', metavar='FILE', type=lambda x: is_valid_file(parser, x))
    parser.add_argument('-f2', help='Time file containing dates for spectra', metavar='FILE', type=lambda x: is_valid_file(parser, x))
    parser.add_argument('-p', help='Folder containing continuum-subtracted spectra', type=lambda x: is_valid_directory(parser, x))
    parser.add_argument('-wc', help='Centroid wavelength of the line', type=float, default=4862.68)

    args = parser.parse_args()

    time_series1 = np.loadtxt(args.f1, usecols = (0, 1, 2), unpack = True) # LC (Continuum) from photometry/spectroscopy
    timefile = np.loadtxt(args.f2, usecols = (0, 1), dtype = str, unpack = True) # File with .hjd

    tsub = 0
    tline = np.array([float(t) for t in timefile[1]]) - tsub # Times of the line light curves to be constructed from bins
    
    ws, fs, fes = get_specs(args.p)

    mean_spec = np.mean(fs, axis = 0)

    bins = np.array([-5000, -2000, -1500, -1000, -500, 0, 500, 1000, 1500, 2000, 5000]) # Bins as a sequence
    nbins = len(bins) - 1 # Number of bins
    
    Sums = [[] for _ in range(nbins)]
    Sums_errs = [[] for _ in range(nbins)]

    for j in range(len(ws)):
        vspace = np.array([wav_to_vel(w, args.wc) for w in ws[j]])
        # Binning
        
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.binned_statistic.html
        bin_sums, bin_edges, binnumber = scipy.stats.binned_statistic(vspace, fs[j], statistic = 'sum', bins = bins)
        bin_errors_sq, bin_edges, binnumber = scipy.stats.binned_statistic(vspace, fes[j] ** 2, statistic = 'sum', bins = bins)
        
        for k in range(nbins):
            Sums[k].append(bin_sums[k])
            Sums_errs[k].append(bin_errors_sq[k])
        plot_all = False
        if plot_all:
            plot_spec(vspace, fs[j] / np.mean(fs[j]), bin_edges = bin_edges)

    bin_centers = get_bin_centers(bin_edges)
    Sums = np.array(Sums)
    Sums_errs = np.array(Sums_errs)
    Sums_errs = np.sqrt(Sums_errs)
    _cal_ccf = True
    if _cal_ccf:
        # Perform cross-correlation
        centaus = np.zeros(nbins)
        centau_uperrs = np.zeros(nbins)
        centau_loerrs = np.zeros(nbins)
        max_rvals = np.zeros(nbins)
        
        _display = False # Set true to display light curve for each of the bins
        for i in range(nbins):
            time_series_from_bins = (tline, Sums[i], Sums_errs[i])
            # Display light curves
            if _display:
                plot_lcvs(time_series1, time_series_from_bins, label = 'Bin {}'.format(i + 1))
            result = calc_ccf(time_series1, time_series_from_bins)
            centau, centau_uperr, centau_loerr = result[4], result[5], result[6]
            max_rval = result[10]
            lag = result[2]
            r = result[3]
            tlags_centroid = result[0]
            
            centaus[i] = centau
            centau_uperrs[i] = centau_uperr
            centau_loerrs[i] = centau_loerr
            max_rvals[i] = max_rval
            
            show_ccf = False
            if show_ccf:
                plot_ccf(lag, r, tlags_centroid, 'Bin {}'.format(i + 1))
            
            print('#####Bin {} #####'.format(i + 1))
            print('Centroid Lag: {} (+{} -{})'.format(centau, centau_uperr, centau_loerr))
    
        # H-beta lag from Table 5 of the paper
        tau = 21.6 # Average lag
        usig = tau + 1.6 # Upper uncertainty
        lsig = tau - 1.0 # lower uncertainty
        
        bin_width = bin_edges[1:] - bin_edges[:-1]
        _show_plot = True
        if _show_plot:
            fig, ax = plt.subplots(2, 1, sharex = True, figsize = (8, 6), facecolor = 'w', edgecolor = 'k', dpi = 100)
            fig.subplots_adjust(hspace = 0.1)
            ax[0].axvline(x = 0, color = 'k', ls = 'solid')
            ax[0].errorbar(bin_centers, centaus, yerr = (centau_loerrs, centau_uperrs), xerr = bin_width / 2., fmt = '.', color = 'k', capsize = 0, elinewidth = 0.5, markeredgewidth = 0.5, markeredgecolor = 'k', label = r'H$\beta$')
            ax[0].axhline(y = tau, color = 'k', linestyle = 'dotted')
            ax[0].axhspan(ymin = lsig, ymax = usig, color = 'dimgray', alpha = 0.5)
            ax[1].plot(vspace, mean_spec / np.max(mean_spec), '-k', label = 'Mean')
            for idx in range(len(bin_edges)):
                ax[1].axvline(x = bin_edges[idx], color = 'dimgray', ls = '--')
            '''
            for idx in range(nbins):
                ax[1].text(bin_centers[idx], 0.1, '{:.2f}'.format(max_rvals[idx]), rotation = 90, fontsize = 'medium')
            '''
            ax[0].set_yticks([10, 20, 30])
            #ax[0].set_ylim([0, 30])
            #ax[0].axhline(y = 0, ls = '--', color = 'k')
            ax[1].set_xlim([bins[0], bins[-1]])
            ax[1].set_xlabel('Velocity (km/s)', fontsize = 'x-large')
            ax[0].set_ylabel(r'$\tau_{\mathrm{cent}}$ (days)', fontsize = 'x-large')
            ax[1].set_ylabel(r'$\overline{F}_{\lambda}$', fontsize = 'x-large')
            ax[0].legend(loc = 'upper right', fontsize = 'large')
            ax[1].legend(loc = 'upper right', fontsize = 'large')
            plt.savefig('3C120_velocity-resolved.pdf', format = 'pdf', bbox_inches = 'tight', dpi = 1000)
            plt.show()
    
main()
