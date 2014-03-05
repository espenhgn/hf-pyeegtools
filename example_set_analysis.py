#!/usr/bin/env python
'''
Procedures to read and analyze EEG channels Axona dacqUSB file formats

These files require only a python environment with common scientific
modules, i.e., numpy, scipy and matplotlib. Full distributions like
"enthought canopy" or "anaconda python" comes with everything.
To run the script, either run it from your python editor, or in ipython
like:
"run example_set_analysis.py"

The file "pyeegtools.py" must be in the same folder,
or added to "PYTHONPATH"


Project home: https://code.google.com/p/hf-pyeegtools/

No warranties, released under GPLv3

(c) espenhgn@gmail.com, Hafting-Fyhn lab, UIO, 2013

TODO: use animal speed info

'''
from __future__ import division

import numpy as np
import matplotlib.pyplot as plt
import glob
import os

import pyeegtools


# ##############################################################################
# Main
# ##############################################################################
if __name__ == '__main__':
    #interactive plotting
    plt.ion()
    plt.close('all')


    # ##############
    # parameters
    # ##############
    time_window = (0, 20)     #window in time series plots
    
    #frequency domain analysis
    NFFT = 512       #number of datapoints in each block for Fourier analysis
    waveletfreqs = np.arange(2, 125, 2) #morlet wavelet center frequencies
    thetafreqs = (0, 20)    #lower/upper bounds of theta band
    gammafreqs = (20, waveletfreqs[-1]) #100)   #ditto, gamma band
    
    
    #phase-amplitude plots for upper/lower gamma (ala Tort et al. 2010)
    #however, here the continous wavelets will be used.
    f_theta = 8
    f_gamma_low = 30
    f_gamma_high = 80
    phasebins = np.linspace(-np.pi, np.pi, 20) #bins for theta phases
    
    
    waveletfreqs = np.array([f_theta, f_gamma_low, f_gamma_high])
    
    #position file, arbitrary units, probably corresponding to 1 x 1 m
    #boxbounds = [30, 430, 60, 455] #[left, right, bottom, top], guessed box size
    boxbounds = [0, 500, 0, 500] #[left, right, bottom, top], guessed box size
    properboxsize = (1., 1.)  #normalize pos in (m), assume animal went edge-to-edge  
    
    maxspeed = 200    #max possible speed in boxbounds units (acquisition units)
    #speedlimit = 0.01 #1 cm/s, less than this, animal sit still
    f_cut=2. #spatial low-pass filter cutoff frequency, Hz

   
   
    # #####################
    # File list
    # #####################

    ##list of datasets to go through

    #load from z-area, setfiles can be added to list at will
    setfiles = []
    #Windows
    if os.sys.platform in ['win32', 'win64', 'windows']:
        #setfiles += glob.glob("Z:\\Espen\\rats\\1079\\*.set")[:4]   
        #setfiles += glob.glob("Z:\\Espen\\rats\\1199\\*.set")[:4]    
        #setfiles += glob.glob("Z:\\Espen\\rats\\1227\\*.set")[:4]
        setfiles += glob.glob("Z:\\Espen\\rats\\1399\\*.set")[:4]
        setfiles += glob.glob("Z:\\Espen\\rats\\1400\\*.set")[:3]
        setfiles += glob.glob("Z:\\Espen\\rats\\1416\\*.set")
        
    #OS X, Linux, Unix etc
    else:
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1079/*.set")[:4]   
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1199/*.set")[:4]    
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1227/*.set")[:4]    
        setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1399/*.set")[:4]    
        setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1400/*.set")[:3]    
        setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1416/*.set")    

    if len(setfiles) == 0:
        raise Exception, 'no files matched file pattern!'

    #container with all loaded datasets
    datas = []

    # #######################
    # iterate over .set-files
    # #######################
    for setfile in setfiles:
        
        # ##########################
        # load and process datasets
        # ##########################
    
        #get the datasets from preconverted recording output
        datasets = pyeegtools.get_datasets(setfile=setfile)


        #load pos file if present using Pos class, interpolate between points
        posfile = setfile.split('.set')[0] + '.pos'
        if os.path.isfile(posfile):
            datasets['posfile'] = posfile
            datasets['position'] = pyeegtools.Posfile(posfile, postprocess=False)
            datasets['position'].process_dataset_positions(boxbounds=boxbounds,
                                       properboxsize=properboxsize,
                                       maxspeed=maxspeed,
                                       UniVarSpl_k=1, UniVarSpl_s=0.2,
                                       N=1, f_cut=f_cut)
        
        #get the wavelet coefficients
        wavelets = pyeegtools.get_morlet_wavelets(waveletfreqs=waveletfreqs, w=7, s=1.)
        #w=7 from Colgin et al. 2009
    
        #compute the continuous wavelet transforms
        pyeegtools.apply_wavelets(datasets, wavelets, method='fftconvolve')

        #extract wavelet events and times at given frequencies:
        cwt_freqs = (f_theta, f_gamma_low, f_gamma_high)
        pyeegtools.compute_cwt_events(datasets, wavelets, cwt_freqs, window=0.5, num_stds=2)
        
        
        #store datasets object
        datas.append(datasets)


    #compare several datasets
    fig = plt.figure(figsize=(10,10))
    fig.suptitle('')
    fig.subplots_adjust(bottom=0.2, wspace=0.3, hspace=0.8)
    titles = []
    for i, freq in enumerate(cwt_freqs):
        data = {}
        xticklabels = {}
        num_channels = len(datas[0].keys()) - 4
        for j in xrange(num_channels):
            data['eeg%i' % (j+1)] = []
            xticklabels['eeg%i' % (j+1)] = []
        ind = np.where(wavelets['freqs'] == freq)[0]
        titles.append('$\omega$=%i Hz' % freq)

        for datasets in datas:
            for setname in datasets.keys():
                eegX = setname.split('_')[-1]
                if setname in ['set', 'setfile', 'position', 'posfile']:
                    continue
                dataset = datasets[setname]

                events = dataset.cwt_events[freq]
                times = dataset.cwt_event_times[freq]
    
                #get the local maxima
                maxima = np.abs(events[:, ind, :]).max(axis=2).flatten()
    
                data[eegX].append(maxima)
                xticklabels[eegX].append(os.path.split(setname)[-1])

        for k in xrange(num_channels):
            ax = fig.add_subplot(num_channels, len(cwt_freqs), i+1 + k*(num_channels+1))
            pyeegtools.remove_axis_junk(ax)
            ax.boxplot(data['eeg%i' % (k+1)])
            ax.set_xticklabels(xticklabels['eeg%i' % (k+1)], rotation='vertical')
            if i ==0:
                ax.set_ylabel(r'$|X_\omega|$ amplitudes (-)')
                
            if k == 0:
                ax.set_title(titles[i])
    
    #save figure
    fig.savefig('example_set_analysis.pdf', dpi=100)

    