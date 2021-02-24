'''
    Implementation of the interpolation cross-correlation (ICCF) method based on Gaskell & Peterson 1987; White & Peterson 1994
    
    Author: Michael Hlabathe - mh@saao.ac.za
    
'''
print(__doc__)

import numpy as np
import matplotlib.pyplot as plt
import sys
import warnings

warnings.filterwarnings('ignore')

plt.style.use('./MNRAS_Style.mplstyle')

def sort_timeseries(ts):
    
    argsort = np.argsort(ts[:, 0])
    
    return np.array([ts[:, 0][argsort], ts[:, 1][argsort], ts[:, 2][argsort]])

def get_timeseries(infile1, infile2):

    ts1 = np.genfromtxt(infile1)
    ts2 = np.genfromtxt(infile2)

    return sort_timeseries(ts1), sort_timeseries(ts2)

def xcor(ts1, ts2, t_start, t_end, tstep, imode = 1):

    lags = np.arange(t_start, t_end + tstep, tstep)
    ccf = np.zeros(np.size(lags), dtype = float)

    for k, lag in enumerate(lags):

        if imode == 1: # Interpolate continuum
            mask = (ts1[0] - lag > ts1[0][0]) & (ts1[0] - lag < ts1[0][-1])
            ynew = np.interp(ts1[0][mask] - lag, ts2[0], ts2[1])
            xnew = ts1[1][mask]

        elif imode == 2: # Interpolate line
            mask = (ts2[0] + lag > ts2[0][0]) & (ts2[0] + lag < ts2[0][-1])
            xnew = np.interp(ts2[0][mask] + lag, ts1[0], ts1[1])
            ynew = ts2[1][mask]

        x_diff = xnew - np.mean(xnew)
        y_diff = ynew - np.mean(ynew)

        ccf[k] = np.sum(x_diff * y_diff) / (np.size(x_diff) * np.std(x_diff) * np.std(y_diff))

    return lags, ccf

def get_delay(lags, ccf, threshold = 80):

    if threshold:
        
        pk = lags[np.argmax(ccf)]
        rmax = np.max(ccf)
        
        if (ccf >= (rmax * threshold / 100.0)).any():
            mask = (ccf >= rmax * threshold / 100.0)
            ccf = ccf[mask]
            lags = lags[mask]

            cen = np.sum(ccf * lags) / np.sum(ccf)
        else:
            cen = np.NAN

        return cen, pk, rmax

def plot_ccf(lags, xccf, yccf, save_output = True):

    avg = np.mean(np.array([xccf, yccf]), axis = 0)

    cen, pk, rmax = get_delay(lags, avg)
    
    print(cen, pk, rmax)

    plt.figure()

    plt.plot(lags, avg, color = 'k', label = 'CCF')
    plt.axvline(x = 0, color = 'k')

    plt.xlabel('Time Lag (days)', fontsize = 'x-large')
    plt.ylabel('CCF', fontsize = 'x-large')
    
    plt.ylim([0, 1])

    plt.grid()
    plt.legend(loc = 'best')

    if save_output:
        np.savetxt('PG2304+043/g_CCF.dat', np.transpose([avg, lags]))

def plot_hist(lags, cen_dist, pk_dist, save_output = True):
    
    mask = ~np.isnan(cen_dist)
    
    cen_lw, cen, cen_up = calc_uncertainties(cen_dist[mask])
    pk_lw, pk, pk_up = calc_uncertainties(pk_dist)
    
    print('Centroid: %s (%s, %s) days' % (cen, cen_lw, cen_up))
    print('Peak: %s (%s, %s) days' % (pk, pk_lw, pk_up))
    
    plt.figure()

    plt.hist(cen_dist[mask], bins = 50, density = True, color = 'r', label = 'CCCD')
    
    plt.xlabel(r'$\tau_{cent}\,$ (days)', fontsize = 'x-large')
    plt.ylabel('Frequency', fontsize = 'x-large')
    
    plt.grid()
    plt.legend(loc = 'best')
    
    if save_output:
        np.savetxt('PG2304+043/g_CCCD.dat', np.transpose([cen_dist[mask]]))

    plt.figure()

    plt.hist(pk_dist, bins = 50, density = True, color = 'r', linestyle = '--', label = 'CCPD')
    
    plt.xlabel(r'$\tau_{peak}\,$ (days)', fontsize = 'x-large')
    plt.ylabel('Frequency', fontsize = 'x-large')

    plt.grid()
    plt.legend(loc = 'best')

def subset(ts):

    t = np.random.choice(ts[0], ts.shape[1], replace = True)
    
    t = np.unique(t)
    
    t = np.sort(t)
    
    idx_1 = np.searchsorted(ts[0], t, 'left')
    idx_2 = np.searchsorted(ts[0], t, 'right')
    
    idx = idx_1[idx_1 != idx_2]
    
    y = ts[1][idx]
    ye = ts[2][idx]
    
    ynew = np.random.normal(y, ye)

    return np.array([t, ynew])

def FR_RSS(ts1, ts2, t_start, t_end, tstep, iterations = None):

    if iterations:

        cen_dist = np.zeros(iterations)
        pk_dist = np.zeros(iterations)

        for k in range(iterations):
            ts1_ = subset(ts1)
            ts2_ = subset(ts2)

            lags, yccf = xcor(ts1_, ts2_, t_start, t_end, tstep, imode = 1)
            lags, xccf = xcor(ts1_, ts2_, t_start, t_end, tstep, imode = 2)

            avg = np.mean(np.array([xccf, yccf]), axis = 0)

            cen, pk, rmax = get_delay(lags, avg, threshold = 80)

            cen_dist[k] = cen
            pk_dist[k] = pk
    
        return cen_dist, pk_dist

def calc_uncertainties(dist):

    dist = np.sort(dist)

    lag = np.percentile(dist, 50)

    plus_sigma = np.percentile(dist, 84)
    minus_sigma = np.percentile(dist, 16)

    up = plus_sigma - lag
    lw = minus_sigma - lag

    return lw, lag, up

if __name__ == '__main__':
    
    if len(sys.argv[1:]) == 5:

        ts1, ts2 = get_timeseries(sys.argv[1], sys.argv[2])

        t_start, t_end, tstep = float(sys.argv[3]), float(sys.argv[4]), float(sys.argv[5])
        
        lags, yccf = xcor(ts1, ts2, t_start, t_end, tstep, imode = 1)
        lags, xccf = xcor(ts1, ts2, t_start, t_end, tstep, imode = 2)
        lags, acf = xcor(ts2, ts2, t_start, t_end, tstep, imode = 1)
        
        uncertainties = True
        
        plot_ccf(lags, xccf, yccf)
        
        if uncertainties:

            cen_dist, pk_dist = FR_RSS(ts1, ts2, t_start, t_end, tstep, iterations = 1000)

            plot_hist(lags, cen_dist, pk_dist)
    
        plt.show()

    else:
        print('python iccf.py line continuum minlag maxlag step')
        sys.exit()
