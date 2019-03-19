import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import glob
import sys
import h5py

data_dir = '/mnt/ceph/users/mbedell/tess/'

def save_all_sc(filenames, out_file):
    N = len(filenames)
    with fits.open(filenames[0]) as hdul:
        time = np.copy(hdul[1].data['TIME']) # assumes all TPFs have the same number of cadences
    M = len(time)
    
    # read all the files:
    cols = np.empty(N)
    rows = np.empty(N)
    ccds = np.empty(N)
    cameras = np.empty(N)
    bkgs = np.empty((N,M))
    bkg_errs = np.empty((N,M))
    raw_bkgs = np.empty((N,M))
    raw_bkg_errs = np.empty((N,M))
    raw_times = np.empty((N,M))
    raw_qualities = np.empty((N,M))
    bad = [] # store indices that may need deleting
    tenpercent = 1000 # guess
    for i,fn in enumerate(filenames):
        if i//tenpercent == 0:
            print('{0:3.1f}% done'.format(i/N * 100))
        try:
            with fits.open(fn) as hdul:
                cols[i] = np.copy(hdul[1].header['1CRV5P'])
                rows[i] = np.copy(hdul[1].header['2CRV5P'])
                ccds[i] = np.copy(hdul[0].header['CCD'])
                cameras[i] = np.copy(hdul[0].header['CAMERA'])
                # get raw time series:
                bkg = np.copy(hdul[1].data['FLUX_BKG'][:,0,0]) # assumes all pixels get the same bkg
                bkg_err = np.copy(hdul[1].data['FLUX_BKG_ERR'][:,0,0])                
                t = np.copy(hdul[1].data['TIME'])
                raw_bkgs[i,:] = np.copy(bkg)
                raw_bkg_errs[i,:] = np.copy(bkg_err)
                raw_times[i,:] = np.copy(t)
                raw_qualities[i,:] = np.copy(hdul[1].data['QUALITY'])
                # interpolate good quality measurements to standard time grid and save:
                q = hdul[1].data['QUALITY'] == 0
                bkgs[i,:] = np.interp(time, t[q], bkg[q])
                bkg_errs[i,:] = np.interp(time, t[q], bkg_err[q])
        except:
            print('WARNING: disregarding file {0}'.format(fn))
            bad.append(i)
    if len(bad) > 0: # remove any files that failed to be read
        cols = np.delete(cols, bad)
        rows = np.delete(rows, bad)
        ccds = np.delete(ccds, bad)
        cameras = np.delete(cameras, bad)
        bkgs = np.delete(bkgs, bad, axis=0)
        bkg_errs = np.delete(bkg_errs, bad, axis=0)
        raw_times = np.delete(raw_times, bad, axis=0)
        raw_bkgs = np.delete(raw_bkgs, bad, axis=0)
        raw_bkg_errs = np.delete(raw_bkg_errs, bad, axis=0)
        raw_qualities = np.delete(raw_qualities, bad, axis=0)
        
    # save the data:
    bkg = np.nanmedian(bkgs, axis=0)
    with h5py.File(out_file,'w') as f:
        f.create_dataset('cols', data=cols)
        f.create_dataset('rows', data=rows)
        f.create_dataset('ccds', data=ccds)
        f.create_dataset('cameras', data=cameras)
        f.create_dataset('median_bkg', data=bkg)
        f.create_dataset('bkgs', data=bkgs)
        f.create_dataset('bkg_errs', data=bkg_errs)
        f.create_dataset('time', data=time)
        f.create_dataset('raw_times', data=raw_times)
        f.create_dataset('raw_bkgs', data=raw_bkgs)
        f.create_dataset('raw_bkg_errs', data=raw_bkg_errs)
        f.create_dataset('raw_qualities', data=raw_qualities)
    return
   

if __name__ == "__main__":
    script, sector = sys.argv
    search = data_dir+'s{0:02.0f}/tess*_tp.fits'.format(int(sector))
    print('searching for SC files as {0}'.format(search))
    filenames = glob.glob(search)
    print('reading {0} SC files...'.format(len(filenames)))
    out_file = data_dir+'sector{0:02.0f}.hdf5'.format(int(sector))
    save_all_sc(filenames, out_file)
    print('saved to {0}'.format(out_file))
