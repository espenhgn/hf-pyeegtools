#!/usr/bin/env python
'''
Procedures to read and analyze EEG channels Axona dacqUSB file formats

These files require only a python environment with common scientific
modules, i.e., numpy, scipy and matplotlib. Full distributions like
"enthought canopy" or "anaconda python" comes with everything.
To run the script, either run it from your python editor, or in ipython
like:
"run example_full_analysis.py"

The file "pyeegtools.py" must be in the same folder,
or added to "PYTHONPATH"


Project home: https://code.google.com/p/hf-pyeegtools/

No warranties, released under GPLv3

(c) espenhgn@gmail.com, Hafting-Fyhn lab, UIO, 2013
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
    
    
    #position file, arbitrary units, probably corresponding to 1 x 1 m
    #boxbounds = [30, 430, 60, 455] #[left, right, bottom, top], guessed box size, acquisition units
    boxbounds = [0, 500, 0, 500] #[left, right, bottom, top], guessed box size, acquisition units
    properboxsize = (1., 1.)  #normalize pos in (m), assume animal went edge-to-edge  
    
    maxspeed = 1    #max possible speed in (ms-1)
    #speedlimit = 0.01 #1 cm/s, less than this, animal sit still
    speedlimit = [[0, 0.01], [0.01, 0.10], [0.01, np.inf]] 
    f_cut=2. #spatial low-pass filter cutoff frequency, Hz

   
   
    # #####################
    # File list
    # #####################

    ##list of datasets to go through
    #load from z-area, setfiles can be added to list at will
    #the last indexing operation select just a few .set-files
    setfiles = []
    #Windows
    if os.sys.platform in ['win32', 'win64', 'windows']:
        #setfiles += glob.glob("Z:\\Espen\\rats\\1079\\*.set")[:4]   
        #setfiles += glob.glob("Z:\\Espen\\rats\\1199\\*.set")[:4]    
        #setfiles += glob.glob("Z:\\Espen\\rats\\1227\\*.set")[5:6]
        #setfiles += glob.glob("Z:\\Espen\\rats\\1371\\*.set")
        setfiles += glob.glob("Z:\\Espen\\rats\\1399\\*.set")[:4]
        setfiles += glob.glob("Z:\\Espen\\rats\\1400\\*.set")[:3]
    #OS X, Linux, Unix etc
    else:
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1079/*.set")[:4]   
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1199/*.set")[:4]    
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1227/*.set")[5:6]
        setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1371/*.set")
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1399/*.set")[:4]    
        #setfiles += glob.glob("/Volumes/imbv-hafting/Espen/rats/1400/*.set")[:3]    

        setfiles = setfiles[:1]
        
    if len(setfiles) == 0:
        raise Exception, 'no files matched file pattern!'

    # #######################
    # iterate over .set-files
    # #######################
    for setfile in setfiles:

        #set up figure destination, creating a subfolder of files
        figdest = os.path.join(os.path.split(setfile)[0], 'figures')
        if not os.path.isdir(figdest):
            os.mkdir(figdest)
        print 'writing figure output to %s' % figdest    
        
        # ##########################
        # load and process datasets
        # ##########################
    
        #get the datasets from preconverted recording output
        datasets = pyeegtools.get_datasets(setfile=setfile)


        ##load pos file if present using Pos class, interpolate between points
        #posfile = setfile.split('.set')[0] + '.pos'
        #if os.path.isfile(posfile):
        #    datasets['posfile'] = posfile
        #    datasets['position'] = pyeegtools.Posfile(posfile, postprocess=False)
        #    datasets['position'].process_dataset_positions(boxbounds=boxbounds,
        #                               properboxsize=properboxsize,
        #                               maxspeed=maxspeed,
        #                               UniVarSpl_k=1, UniVarSpl_s=0.2,
        #                               N=1, f_cut=f_cut)
        
        #get the wavelet coefficients
        wavelets = pyeegtools.get_morlet_wavelets(waveletfreqs=waveletfreqs, w=7, s=1.)
        #w=7 from Colgin et al. 2009
        
        #compute the continuous wavelet transforms
        pyeegtools.apply_wavelets(datasets, wavelets, method='fftconvolve')

        
        ##extract wavelet events and times at given frequencies:
        cwt_freqs = [f_theta, f_gamma_low, f_gamma_high]
        pyeegtools.compute_cwt_events(datasets, wavelets, cwt_freqs, window=0.5, num_stds=2)
        


        # ######################
        # do some plotting
        # ######################

        #some plots will use this filenamepostfix
        figname_postfix = os.path.split(datasets['setfile'])[-1].split('.set')[0]
        
        
        for setname, dataset in datasets.items():        
            if setname in ['set', 'setfile', 'position', 'posfile']:
                continue
            fig = pyeegtools.figure1(setname=setname,
                          dataset=dataset,
                          wavelets=wavelets,
                          time_window=time_window,
                          thetafreqs=thetafreqs,
                          gammafreqs=gammafreqs,
                          NFFT=NFFT,
                          )
            fig.savefig(os.path.join(figdest, 'overview_') + setname + '.pdf', dpi=100)
            plt.close(fig)
            
            
            fig = pyeegtools.figure2(setname=setname,
                          dataset=dataset,
                          wavelets=wavelets,
                          time_window=time_window,
                          f_theta=f_theta,
                          f_gamma_low=f_gamma_low,
                          f_gamma_high=f_gamma_high,
                          phasebins=phasebins)
            fig.savefig(os.path.join(figdest, 'phase_amplitude_') + setname + '.pdf', dpi=100)
            plt.close(fig)
            
            
            #plot event triggered frequency events
            for i, freq in enumerate(cwt_freqs):
                if i==0:
                    fig1 = None
                    fig2 = None
                #get the data for this freq
                cwt_events = dataset.cwt_events[freq]
                event_times = dataset.cwt_event_times[freq]
            
                fig1 = pyeegtools.figure4(setname, dataset, wavelets, np.abs(cwt_events),
                               freq=freq, fig=fig1,
                               column=i, numcolumns=len(cwt_freqs), columnwidth=0.18)
                fig2, data = pyeegtools.figure6(datasets, setname, wavelets,
                               np.abs(cwt_events), event_times, freq=freq,
                               speedlimit=speedlimit,
                               fig=fig2, column=i, numcolumns=len(cwt_freqs),
                               columnwidth=0.18)
            
            
            fig1.savefig(os.path.join(figdest, 'mean_cwt_envelope_events_') + setname + '.pdf', dpi=100)
            fig2.savefig(os.path.join(figdest, 'cwt_envelope_amplitudes_') + setname + '.pdf', dpi=100)
            plt.close(fig1)
            plt.close(fig2)
            
            
            #compute cycle-averaged response across all frequencies
            cycles, events, event_times = pyeegtools.compute_cycle_average(dataset,
                                                                wavelets,
                                                                freq=f_theta,
                                                                numcycles=200)
            
            fig = pyeegtools.figure5(setname, dataset, wavelets, cycles, events,
                          thetafreqs=thetafreqs, gammafreqs=gammafreqs, f_theta=f_theta,
                          whitening=True)
            fig.savefig(os.path.join(figdest, 'cycle_average_') + setname + '.pdf', dpi=100)
            plt.close(fig)
        
        
        
        
            fig, Cxy, f = pyeegtools.draw_crossfreq_coherence(setname, dataset, wavelets,
                                                          NFFT=NFFT, fcutoff=2, clim=None)
            fig.savefig(os.path.join(figdest, 'xcoherence_') + setname + '.pdf', dpi=100)
            plt.close(fig)
            
        
        
            #plot a comparison of mean oscillatory events
            fig = pyeegtools.figure7(setname, dataset, wavelets, freqs = cwt_freqs)
            fig.savefig(os.path.join(figdest, 'mean_triggered_oscillations') + setname + '.pdf', dpi=100)
            plt.close(fig)
        
        
        if datasets['position'].valid:
            fig = pyeegtools.figure3(datasets)
            fig.savefig(os.path.join(figdest, 'position_') + figname_postfix + '.pdf', dpi=100)
            plt.close('all')
        
        
        #plot some comparisons if there are more than one dataset per setfile
        if len(datasets.keys()) > 4:
            #just the EEG traces
            fig = pyeegtools.figure8(datasets, time_window)
            fig.savefig(os.path.join(figdest, 'EEGs_') + figname_postfix + '.pdf', dpi=100)
            plt.close(fig)
        
            #cross-cohereneces
            fig = pyeegtools.figure9(datasets, NFFT=NFFT)
            fig.savefig(os.path.join(figdest, 'EEG_coherences_') + figname_postfix + '.pdf', dpi=100)
            plt.close(fig)
        
            #plot event amplitude distributions
            fig = pyeegtools.plot_datasets_event_amplitudes(datasets, wavelets, speedlimit,
                                                 freqs=cwt_freqs)
            fig.savefig(os.path.join(figdest, 'EEG_event_amplitudes_') + figname_postfix + '.pdf', dpi=100)
            plt.close(fig)
