import argparse, glob, os, sys, warnings
import numpy as np
import pandas as pd
from astropy.io import fits, ascii
from astropy.coordinates import SkyCoord
from astropy import units as u
import matplotlib.pyplot as plt

warnings.filterwarnings('ignore')

def readText(cat):
    
    data = ascii.read(cat)
    
    return data

def readFits(fits_file):
    
    return fits.getheader(fits_file)

def find_match(args, tolerance = 4e-4):

    radec = readText(args.file)
    indices = np.arange(len(radec['RA']))
    comp_m = indices != 0
    tar_m = indices == 0
    cs_ra = radec['RA'][comp_m]
    cs_dec = radec['DEC'][comp_m]

    cs_ra = [SkyCoord('%s %s' % (RA, DEC), unit = (u.hourangle, u.deg)).ra.value for RA, DEC in zip(cs_ra, cs_dec)]
    cs_dec = [SkyCoord('%s %s' % (RA, DEC), unit = (u.hourangle, u.deg)).dec.value for RA, DEC in zip(cs_ra, cs_dec)]

    cats = np.array([cat for cat in glob.iglob('%s/*_flux*.cat' % (args.dcats))])
    
    Nframes = len(cats)
    Nref_stars = len(cs_ra)

    tar_c = SkyCoord('%s %s' % (radec['RA'][tar_m][0], radec['DEC'][tar_m][0]), unit = (u.hourangle, u.deg))
    
    tar_ra = tar_c.ra.value
    tar_dec = tar_c.dec.value

    cs_rawmag = np.zeros((Nref_stars, Nframes))
    cs_uncer = np.zeros((Nref_stars, Nframes))
    tar_rawmag = np.zeros(Nframes)
    tar_uncer = np.zeros(Nframes)
    HJD_ims = np.zeros(Nframes)
    observ = []
    tel = []

    for k, cat in enumerate(cats):
        data = readText(cat)
        cat_ra = data['ALPHA_J2000']
        cat_dec = data['DELTA_J2000']
        flags = data['FLAGS']
        fwhm = data['FWHM_IMAGE']
        cat_mags = data['MAG_APER']
        cat_magerr = data['MAGERR_APER']

        splitext = cat.split('_flux')[0] + '.fits'
        fits_file = '%s/%s' % (args.dfits, os.path.basename(splitext))

        hdr = readFits(fits_file)
        hjd = hdr['HJD']
        site_id = hdr['SITEID']

        if site_id == 'cpt':
            site_id = 'SAAO'
        if site_id == 'elp':
            site_id = 'McD'
        if site_id == 'lsc':
            site_id = 'CTIO'
        if site_id == 'coj':
            site_id = 'SSO'
        if site_id == 'ogg':
            site_id = 'CFHT'

        HJD_ims[k] = hjd
        observ.append(site_id)
        tel.append(os.path.basename(cat).split('-')[0][3:])

        for j, ra, dec in zip(range(Nref_stars), cs_ra, cs_dec):
            dist_ra = cat_ra - ra
            dist_dec = cat_dec - dec

            dist = np.sqrt(np.square(dist_ra) + np.square(dist_dec))
            
            if len(dist) == 0:
                dist = np.array([99])
            
            idx = np.argmin(dist)

            if dist[idx] <= tolerance and cat_mags[idx] != 99 and flags[idx] == 0:
                cs_rawmag[j, k] = cat_mags[idx]
                cs_uncer[j, k] = cat_magerr[idx]
            else:
                cs_rawmag[j, k] = np.NAN
                cs_uncer[j, k] = np.NAN

        dist_ra = cat_ra - tar_ra
        dist_dec = cat_dec - tar_dec
        
        dist = np.sqrt(np.square(dist_ra) + np.square(dist_dec))
        
        if len(dist) == 0:
            dist = np.array([99])
                
        idx = np.argmin(dist)

        if dist[idx] <= tolerance and cat_mags[idx] != 99 and flags[idx] == 0:
            tar_rawmag[k] = cat_mags[idx]
            tar_uncer[k] = cat_magerr[idx]
        else:
            tar_rawmag[k] = np.NAN
            tar_uncer[k] = np.NAN

    observ = np.array(observ)
    tel = np.array(tel)
    
    return cs_rawmag, cs_uncer, tar_rawmag, tar_uncer, HJD_ims, observ, cats, tel

def phot_table(t, ms, mes):
    light_curves = {}
    light_curves['HJD'] = t - 2450000

    for j in range(len(ms)):
        light_curves['comp %s' % (j + 1)] = ms[j]
        light_curves['error %s' % (j + 1)] = mes[j]

    return pd.DataFrame(light_curves)

# Check goodness of fit for constant-flux model of each star
def goodness_of_fit(f, ferr):
    m = ~np.isnan(f)
    x, w = f[m], 1. / np.square(ferr[m])

    xbar = np.sum(x * w) / np.sum(w)
    res = x - xbar
    dof = np.size(x) - 1
    chisqr = np.sum(np.square(res))
    
    return chisqr / dof

def reference_frame(y, idx_frame = None):
    if idx_frame is not None:
        ref_frame = np.array([row[idx_frame] for row in y])
    
    return ref_frame

def dphot(y, ye, frame_num = 0):
    ref_frame_mags = reference_frame(y, idx_frame = frame_num)
    ref_frame_mag_errs = reference_frame(ye, idx_frame = frame_num)
    
    dm = y - np.vstack(ref_frame_mags)
    dmerr = np.sqrt(np.square(ye) + np.square(np.vstack(ref_frame_mag_errs)))
    
    return dm, dmerr

def check_variable_stars(y, ye, ref_star_idx = None):
    if ref_star_idx is not None:
        ref_star_dm = y[ref_star_idx]
        ref_star_dmerr = ye[ref_star_idx]
        
        subtract_ref_star_dm = y - ref_star_dm

        subtract_ref_star_dmerr = np.sqrt(np.square(ye) + np.square(ref_star_dmerr))
        
        return subtract_ref_star_dm, subtract_ref_star_dmerr

def plot_lcurves(y, ye):
    star_num = 1

    print('#####################################################')
    
    plt.figure()
    
    for k in range(len(y)):
        
        seq_num = np.arange(1, len(y[k]) + 1)
        
        print('RMS variability: %s' % (np.nanstd(y[k] - np.nanmean(y[k]))))
        
        plt.errorbar(seq_num, y[k], yerr = ye[k], fmt = '.', label = 'comp %s' % (star_num))
        
        star_num += 1
    
    plt.gca().invert_yaxis()
    plt.xlabel('Sequence Number', fontsize = 'x-large')
    plt.ylabel(r'$\Delta m = m - m_{frame}$', fontsize = 'x-large')
    plt.legend(loc = 'best')
    plt.grid()

def plot_comps(y, ye):
    star_num = 1
    
    print('#####################################################')
    
    plt.figure()
    
    for k in range(len(y)):
        
        seq_num = np.arange(1, len(y[k]) + 1)
        
        print('The LC scatter: %s' % (np.nanstd(y[k])))
        
        plt.errorbar(seq_num, y[k], yerr = ye[k], fmt = '.', label = 'comp %s' % (star_num))
        
        star_num += 1
    
    plt.axhline(y = 0.01, color = 'k', linestyle = '--')
    plt.axhline(y = -0.01, color = 'k', linestyle = '--')
    plt.gca().invert_yaxis()
    plt.xlabel('Sequence Number', fontsize = 'x-large')
    plt.ylabel(r'$\Delta m - \Delta m_{star}$', fontsize = 'x-large')
    plt.legend(loc = 'best')
    plt.grid()

def final_lcurve(x, tar_y, tar_ye, y, ye, observ, cats, tel, catalog_mag = 0, ref_star_idx = 0):
    tar_dm = (tar_y - y[ref_star_idx]) + catalog_mag
    tar_dmerr = np.sqrt(np.square(tar_ye) + np.square(ye[ref_star_idx]))

    nan_mask = ~np.isnan(tar_dm)
    
    x, y, ye, observ, cats, tel = x[nan_mask], tar_dm[nan_mask], tar_dmerr[nan_mask], observ[nan_mask], cats[nan_mask], tel[nan_mask]

    print('The LC scatter = %5.3f mag' % np.std(y))
    
    plt.figure()

    if 'SAAO' in observ:
        saao_m = observ == 'SAAO'
        plt.errorbar(x[saao_m] - 2450000, y[saao_m], ye[saao_m], fmt = '.', color = 'red', label = 'SAAO')
    if 'SSO' in observ:
        sso_m = observ == 'SSO'
        plt.errorbar(x[sso_m] - 2450000, y[sso_m], ye[sso_m], fmt = '.', color = 'brown', label = 'SSO')
    if 'McD' in observ:
        mcd_m = observ == 'McD'
        plt.errorbar(x[mcd_m] - 2450000, y[mcd_m], ye[mcd_m], fmt = '.', color = 'blue', label = 'McD')
    if 'CTIO' in observ:
        ctio_m = observ == 'CTIO'
        plt.errorbar(x[ctio_m] - 2450000, y[ctio_m], ye[ctio_m], fmt = '.', color = 'green', label = 'CTIO')
    if 'CFHT' in observ:
        cfht_m = observ == 'CFHT'
        plt.errorbar(x[cfht_m] - 2450000, y[cfht_m], ye[cfht_m], fmt = '.', color = 'orange', label = 'CFHT')

    ascii.write([x - 2450000, y, ye, tel, observ], '%s/temp.dat' % (args.dcats), overwrite = True)

    plt.gca().invert_yaxis()
    plt.xlabel('HJD - 2450000 (days)', fontsize = 'x-large')
    plt.ylabel(r'$m$', fontsize = 'x-large')
    plt.legend(loc = 'best')
    plt.grid()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description = 'Performs Differential Photometry')

    parser.add_argument('--dfits', type = str, help = 'Directory containing .fits files', required = True)
    parser.add_argument('--dcats', type = str, help = 'Directory containing .cat files from Sextractor', required = True)
    parser.add_argument('--file', type = str, help = 'The file containing RA and DEC of target and comparison star(s)', required = True)

    args = parser.parse_args()

    cs_rawmag, cs_uncer, tar_rawmag, tar_uncer, HJD_ims, observ, cats, tel = find_match(args)
    phot_table = phot_table(HJD_ims, cs_rawmag, cs_uncer)
    
    dm, dmerr = dphot(cs_rawmag, cs_uncer, frame_num = 0)
    
    plot_lcurves(dm, dmerr)

    subtract_ref_star_dm, subtract_ref_star_dmerr = check_variable_stars(dm, dmerr, ref_star_idx = 0)

    plot_comps(subtract_ref_star_dm, subtract_ref_star_dmerr)
    
    final_lcurve(HJD_ims, tar_rawmag, tar_uncer, cs_rawmag, cs_uncer, observ, cats, tel, catalog_mag = 15.411, ref_star_idx = 0)
    
    plt.show()
