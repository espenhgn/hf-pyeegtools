#!/usr/bin/env python
'''
Procedures to read analyze EEG channels Axona dacqUSB file formats

Project home: https://code.google.com/p/hf-pyeegtools/

No warranties, released under GPLv3

(c) espenhgn@gmail.com, Hafting-Fyhn lab, UIO, 2013
'''
from __future__ import division

import numpy as np
import scipy as sc
import scipy.signal as ss
from scipy.interpolate import UnivariateSpline
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import glob
import os


# ##############################################################################
# class definitions
# ##############################################################################

class EEGfile(object):
    '''class EEG - axona .eeg file reader'''

    def __init__(self, fname, headerlines=11, verbose=False):
        '''
        Axona .eeg file reader.

        File header information is set as class attributes,
        the eeg data is set as class attribute 'eeg', of type
        np.ndarray dtype='int8'.

        kwargs:
        ::
            fname: str, path to .eeg file
            headerlines: int, number of text lines in file header

        usage:
        ::
            eegfile = EEGfile("/path/to/filename.eeg")
            print eegfile.eeg
        '''
        #open file

        #set attributes
        self.fname = fname

        #length of file header
        headerchars = 0

        #open file
        f = open(self.fname, 'r')
        #loop over header lines
        for i, line in enumerate(f):
            #first lines in file is info 
            if i >= headerlines:
                continue
            #split attribute name and value
            attr, value = line.split('\r\n')[0].split(' ', 1)

            if verbose:
                print attr, value

            #attempt cast to int, float, str in that order:
            try:
                setattr(self, attr, int(value))
            except:
                try:
                    setattr(self, attr, float(value))
                except:
                    setattr(self, attr, str(value))

            headerchars += len(line)

        headerchars += 10

        #numeric value for sample rate in Hz
        self.Fs = float(self.sample_rate.split(' ')[0])

        #close file
        f.close()

        #read binary part of file containing num_EEG_samples EEG datapoints
        f = open(fname, 'rb')
        eeg = np.fromfile(f, dtype='int8', count=-1)
        f.close()

        self.eeg = eeg[headerchars:headerchars+self.num_EEG_samples]
        

class Setfile(object):
    '''class Setfile - axona .set file reader'''

    def __init__(self, fname):
        '''
        Axona .set file reader

        class init of class setfile, used to import all content of an
        axona .set file.

        Unless value is numlike, output will be treated as a string,
        and set as class attributes.

        kwargs:
        ::
            fname: str, path to setfile

        usage:
        ::
            setfile = Setfile("/path/to/filename.set")
        '''

        #set attributes
        self.fname = fname
        f = open(self.fname)

        #read set-file line by line
        for line in f:
            #split attribute name and value
            attr, value = line.split('\r\n')[0].split(' ', 1)

            #a little warning
            if hasattr(self, attr):
                print 'attr %s=%s exist!' % (attr, str(getattr(self, attr)))

            #attempt cast to int, float, str in that order:
            try:
                setattr(self, attr, int(value))
            except:
                try:
                    setattr(self, attr, float(value))
                except:
                    setattr(self, attr, str(value))


class Posfile(object):
    def __init__(self, fname, postprocess=True, headerlines=27, verbose=False):
        '''
        Axona .pos-file reader implemented as a class.
        
        Header information from file will be set as class attributes as is,
        while the raw x- and y-coordinates with corresponding timestamps will be
        set as attributes Posfile.xpos, Posfile.ypos, Posfile.tpos. For these raw datapoints
        invalid entries will be removed (i.e., nan-values), however not if they
        are all nans.
        
        If valid points are found, the class-method
        Posfile.process_dataset_positions() will by default interpolate between
        points using univariate spline interpolation to construct a continuouos
        trajectory (see docstring in function)
        
        
        kwargs:
        ::
            fname: str, path to axona .pos file
            postprocess: bool, switch on interpolation between datapoints
            headerlines: int, number of text lines in file header

        Usage:
        ::
            import pylab as plt
            pos = Posfile('/path/to/filename.pos')
            plt.figure()
            plt.plot(pos.xpos, pos.ypos, 'r.')
            plt.plot(pos.x, pos.y, 'k', lw=1)
            plt.show()
            
        '''
        #open file
        f = open(fname, 'r')
        
        #set attributes
        self.fname = fname
        
        print fname
        numchars = 0 #count the header length until "data_start"
        #loop over lines
        for i, line in enumerate(f):
            #first few lines is info 
            if i >= headerlines:
                continue
            
            #count
            numchars += len(line)
            
            #split attribute name and value
            attr, value = line.split('\r\n')[0].split(' ', 1)
            
            if verbose: 
                print attr, value
            
            #attempt cast to int, float, str in that order:
            try:
                setattr(self, attr, int(value))
            except:
                try:
                    setattr(self, attr, float(value))
                except:
                    setattr(self, attr, str(value))
        
        #EEG sample rate
        Fs_pos = float(self.sample_rate.split(' ')[0])
        self.Fs = self.EEG_samples_per_position * Fs_pos
        
        #close file
        f.close()
        
        
        #read binary data until end of file
        f = open(fname, 'rb')
        bin_data = np.fromfile(f, dtype='uint8', count=-1)
        f.close()
        
        #some needed vars
        offset = numchars + 10
        poslen = self.bytes_per_timestamp + 4*self.bytes_per_coord + 8
        
        big_endian_mat = np.array([[256, 256, 256, 256], [1,1,1,1]])
        big_endian_vec = 256**np.arange(self.bytes_per_timestamp)[::-1].reshape(
                                                -1, self.bytes_per_timestamp)
        
        self.tpos = np.zeros(self.num_pos_samples)
        self.xpos = np.zeros((self.num_pos_samples, 2))
        self.ypos = np.zeros((self.num_pos_samples, 2))

        #loop over each timestamp
        for i in xrange(self.num_pos_samples):
            #read and convert binary data:
            pos_offset = offset + i*poslen
            t_bytes = bin_data[pos_offset:pos_offset + self.bytes_per_timestamp]
            pos_offset += self.bytes_per_timestamp
            c_bytes = bin_data[pos_offset:pos_offset+4*self.bytes_per_coord].reshape(
                                                    -1, self.bytes_per_coord).T
            coords = (c_bytes * big_endian_mat).sum(axis=0)
            
            #fill in values
            self.tpos[i] = (t_bytes * big_endian_vec).sum()
            self.xpos[i, ] = coords[::2]
            self.ypos[i, ] = coords[1::2]
        
        
        #fix timestamps, sometimes self.tpos is not monotonically increasing
        badtimeinds = np.where(np.diff(self.tpos) != 1)[0] + 1
        if badtimeinds.size > 0:
            #rewrite self.tpos, we know how large it should be
            self.tpos = np.arange(self.tpos.size).astype(float)

        #convert timestamp vector to unit of s
        self.tpos /= float(self.timebase.split(' ')[0])
        
        
        #value 1023 correspond to missing value, mask with nans:
        inds = self.xpos == 1023
        self.xpos[inds] = np.nan
        self.ypos[inds] = np.nan

        #mask indices with bad timestamps as nans:
        self.xpos[badtimeinds] = np.nan
        self.ypos[badtimeinds] = np.nan

        #appears as though last axis is just nonsense
        self.xpos = self.xpos[:, 0]
        self.ypos = self.ypos[:, 0]
        
        
        #flag for valid processing of position data, i.e., for only nan-values
        self.valid = False        
        
        #interpolate between datapoints, removing nans:
        self.process_dataset_positions()


    def process_dataset_positions(self, boxbounds=[0, 500, 0, 500],
                                   properboxsize=[1., 1.],
                                   maxspeed=200,
                                   UniVarSpl_k=1, UniVarSpl_s=0.2,
                                   N=1, f_cut=2.):
        '''
        if valid position data attributes are found, continous time series is
        reconstructed at time-resolution of EEG, accessible as
        Posfile.time, Posfile.x, Posfile.y
        
        The function allows overriding boundaries of box, set box size etc.
        
        kwargs:
        ::
            datasets: dict, data stricture
            boxbounds: list/tuple, [left, right, bottom, top], bounds of box ? units
            properboxsize: list/tuple, width and height of box in (m)
            maxspeed: float, maximum speed per s in raw coordinates
            UniVarSpl_k: float, param for univariate spline interpolation
            UniVarSpl_s: float, param for univariate spline interpolation
            N: int, butterworth filter order applied to contin. position data
            f_cut: float, low pass filter cutoff freq. on position data
        
        '''
        #non-nan value indices (not really needed, box check will detect)
        nanfilter = np.logical_and(np.isnan(self.xpos),
                                   np.isnan(self.ypos)) != True
        
        #remove coordinates outside of box
        posfilter = ((self.xpos >= boxbounds[0]) &
                        (self.xpos <= boxbounds[1]) &
                        (self.ypos >= boxbounds[2]) &
                        (self.ypos <= boxbounds[3]))
        
        #combine
        inds = np.logical_and(nanfilter, posfilter)
        
        #initial removal of invalid points
        tpos = self.tpos[inds]
        xpos = self.xpos[inds]
        ypos = self.ypos[inds]
                    
        #some datasets have only NAN values for positions, if so: return
        if np.all(np.isnan(tpos)) or np.all(np.isnan(xpos)) or np.all(np.isnan(ypos)):
            print 'no valid position data found'
            self.valid = False
            return
    
        #find large jumps in speed, then mask out succeeding time steps
        badinds = np.where(abs(np.diff(np.sqrt(xpos**2 + ypos**2)) / np.diff(tpos)) > maxspeed)[0]
        inds = np.ones(tpos.size).astype(bool)
        for i in badinds:
            inds[i-1:i+2] = False


        #mask out data
        tpos = tpos[inds]
        xpos = xpos[inds]
        ypos = ypos[inds]


        #additional step, remove datapoints existing by themselves
        inds = np.r_[True, np.diff(tpos) < 1./float(self.sample_rate.split(' ')[0])]
        
        tpos = tpos[inds]
        xpos = xpos[inds]
        ypos = ypos[inds]
        

        ##normalize to units of m using info in properboxsize
        xpos -= xpos.min()
        xpos /= xpos.max()*properboxsize[0]
        ypos -= ypos.min()
        ypos /= ypos.max()*properboxsize[1]
        
    
        #use Univariate Spline interpolation of order 1 between datapoints to
        #time res of EEG.
        #Interpolators. Higher orders (k > 1) will give large overshoots,
        #better with linearlike spline, s-values (0<s<0.1) needed to follow points
        stx = UnivariateSpline(tpos, xpos, k=UniVarSpl_k, s=UniVarSpl_s)
        sty = UnivariateSpline(tpos, ypos, k=UniVarSpl_k, s=UniVarSpl_s)
        
        #interpolate position at EEG resolution, only for times in "tpos"
        tpos_interp = np.arange(tpos.min(), tpos.max()+1./self.Fs, 1./self.Fs)
        xpos_interp = stx(tpos_interp)
        ypos_interp = sty(tpos_interp)


        #low-pass filter on position data
        b, a = ss.butter(N=1, Wn=f_cut*2/self.Fs)
        #zero phase shift filter
        xpos_interp = ss.filtfilt(b, a, xpos_interp)
        ypos_interp = ss.filtfilt(b, a, ypos_interp)
        
        ##apply nanmask before and after valid positions
        #inds = (tpos_interp < tpos[0]) | (tpos_interp > tpos[-1])
        
        #tpos_interp[inds] = np.nan
        #xpos_interp[inds] = np.nan
        #ypos_interp[inds] = np.nan
        
        
        #update the values
        self.tpos = tpos
        self.xpos = xpos
        self.ypos = ypos
        
        
        self.time = tpos_interp
        self.x = xpos_interp
        self.y = ypos_interp
        
        #change flag if interpolations were successful
        self.valid = True


# ##############################################################################
# function definitions
# ##############################################################################

def get_datasets(setfile):
    '''identify, and load data in defaultfolder, identified by
    corresponding .mat-files containing EEG from one or several channels
    
    Axona .set-, .pos- and .eeg*-files will be loaded, if present

    kwargs:
    ::
        setfile: str, path to .set-file
        
    output:
        dict, containing data structure similar to this:
            {'DHV_2012060119_eeg1': <pyeegtools.EEGfile at 0x11b513c50>,
             'DHV_2012060119_eeg2': <pyeegtools.EEGfile at 0x11b513510>,
             'posfile': 'rat1079-2012060119/DHV_2012060119.pos',
             'position': <pyeegtools.Posfile at 0x11b53c0d0>,
             'set': <pyeegtools.Setfile at 0x13c40b550>,
             'setfile': 'rat1079-2012060119/DHV_2012060119.set'}
    
    i.e., we're using different classes included here to read in .eeg*-files,
    .pos-files and corresponding .set-files
    
    '''
    
    #container
    datasets = {}
    
    #load setfile if present, use it as a global identifier
    datasets['setfile'] = setfile
    datasets['set'] = Setfile(setfile)
    print "loaded setfile %s" % setfile


    #read available .eeg datafiles
    eegfiles = glob.glob(setfile.split('.set')[0] + '*.eeg*')
    
    print eegfiles
    
    for fil in eegfiles:
        print 'loading eeg file %s' % fil
        filname, ending = os.path.split(fil)[-1].split('.eeg')
        if ending == '24':
            print 'skipping file %s' % fil
        else:
            if ending == '':
                ending = '1'
            datasets[filname + '_eeg%s' % ending] = EEGfile(fil)
        
    
    #proper units of mV for EEG traces:
    for setname, dataset in datasets.items():
        if setname in ['set', 'setfile', 'position', 'posfile']:
            continue
        ending = setname.split('_eeg')[-1]
        if ending == '':
            ending = '1'
        linkedch = getattr(datasets['set'], 'EEG_ch_%s' % ending)
        gain = getattr(datasets['set'], 'gain_ch_%i' % (linkedch-1))

        #convert int8 to floats
        if dataset.eeg.dtype != float:
            dataset.eeg = dataset.eeg.astype(float)
            
        #normalize from int8 values to units of mV
        dataset.eeg /= 2**8
        #Sturla confirms that ADC_fullscale_mv 1500 sets range +- 750 muV.
        dataset.eeg *= datasets['set'].ADC_fullscale_mv / 2.
        
        #convert to proper mV
        dataset.eeg /= gain
            
    return datasets

    
def get_morlet_wavelets(waveletfreqs=np.arange(2, 81, 2), Fs=250,
                        w=6., s=1.):
    '''
    Get a set of wavelet coefficients, only tested with scipy.signal.morlet.
    
    Compute the wavelet coefficients for each frequency
    The fundamental frequency of morlet wavelet in Hz is given
    by ``f = 2*s*w*r / M`` where r is the sampling rate.
    Defaulting to w=6 cycles (Roach2006):
    -> M = 2*s*w*2*r/f
    
    kwargs:
    ::
        waveletfun: scipy.signal.wavelet-function, kind of wavelet
        waveletfreqs: np.ndarray, center frequencies of wavelets
        w: float, number of cycles in morlet wavelet
        s: float, scaling factor
        
    return:
        wavelets: dict, values:
            'coeffs' : list of dtype complex np.ndarray, set of complex morlets
            'freqs' : np.ndarray, center frequencies of wavelets
    '''
    waveletfun = ss.morlet #or ss.ricker or ...
    #set up a dictionary storing the wavelet coefficients
    wavelets = {
        'freqs' : waveletfreqs,
        'coeffs' : []
    }

    #compute the wavelets for each freq
    for i, f in enumerate(waveletfreqs):
        waveletkwargs = {
            'M' : 2.*s*Fs*w/f,
            'w' : w,
            's' : s,
            'complete' : True,
        }
        #compute the wavelet
        wavelets['coeffs'].append(waveletfun(**waveletkwargs))
        
    return wavelets


def apply_wavelets(datasets, wavelets):
    '''
    Recursively apply wavelet coefficients for each wavelet center frequency to
    signal in datasets, discarding imaginary component
    
    kwargs:
    ::
        datasets: dict, as output by pyeegtools.get_datasets(path/to/file.set')
        wavelets : dict, as output by pyeegtools.get_morlet_wavelets(**kwargs)
    
    '''
    for setname, value in datasets.items():
        if setname in ['set', 'setfile', 'position', 'posfile']:
            continue
        #continous wavelet transforms, signal envelope, signal phase 
        value.cwt = np.empty((wavelets['freqs'].size, value.eeg.size),
                                        dtype=complex)
        for i, wavelet in enumerate(wavelets['coeffs']):
            #compute the wavelet transform:
            value.cwt[i, ] = ss.convolve(value.eeg, wavelet, 'same')


def compute_cwt_events(datasets, wavelets,
                       freqs=(8, 30, 60), window=0.5, num_stds=2):
    '''
    for each EEG dataset in datasets, compute for each frequency in freqs
    
    kwargs:
    ::
        datasets: dict, as output by pyeegtools.get_datasets(path/to/file.set')
        wavelets : dict, as output by pyeegtools.get_morlet_wavelets(**kwargs)
        freqs: list of integers
        window: float
        num_stds: float
    
    '''
    for setname, dataset in datasets.items():        
        if setname in ['set', 'setfile', 'position', 'posfile']:
            continue

        #containers set as dataset attributes
        dataset.cwt_events = {}
        dataset.cwt_event_times = {}
        dataset.cwt_freqs = freqs
        
        #compute event triggered frequency events
        for i, freq in enumerate(freqs):
            #extract some events
            events, times = extract_events(dataset, wavelets,
                                           window=window,
                                           num_stds=num_stds,
                                           freq=freq)
            #fill in values
            dataset.cwt_events[freq] = events
            dataset.cwt_event_times[freq] = times


def extract_events(dataset, wavelets, window=0.5, num_stds=2, freq=40):
    '''
    extract and return events in dataset.cwt by threshold crossing
    at a given frequency.
    threshold is abs(X).mean() + num_stds*abs(X).std(), similar to Colgin2009,
    but X is now the contionuous wavelet transform at given frequency f
    
    kwargs:
    ::
        dataset: <pyeegtools.EEGfile> object
        wavelets : dict, as output by pyeegtools.get_morlet_wavelets(**kwargs)
        window : float, extracted time window in (s)
        num_stds : float, cutting threshold is a multiple of signal std above mean
        freq : int/float, frequency of contionous wavelet used for detection
    
    return:
        cwt_events : detected events
        event_times : np.array, timestamp of events  
    
    '''
    # detection of events:
    window = dataset.Fs * window
    ind = wavelets['freqs'] == freq

    
    #get the corresponding traces and magnitude envelopes
    trace = dataset.cwt[ind, ].real.flatten()
    envelope = np.abs(dataset.cwt)[ind, ].flatten()
    
    #find threshold crossings, threshold is signal.mean() + num_stds*signal.std()
    #edges are detected from below
    [crossings] = np.where((envelope[:-1] < envelope.mean() + num_stds*envelope.std()) & \
                            (envelope[1:] > envelope.mean() + num_stds*envelope.std()))
    
    #skip crossings with windows falling outside of bounds of each trace
    crossings = crossings[(crossings>=window) & (crossings <= trace.size-window)]

    #get event times
    event_times = crossings.astype(float) / dataset.Fs
    
    #fill up 3d matrix
    cwt_events = np.zeros((crossings.size, wavelets['freqs'].size, 2*window), dtype=complex)
    
    #iterate over each crossing
    for i, j in enumerate(crossings):
        cwt_events[i, ] = dataset.cwt[:, j-window:j+window]

    return cwt_events, event_times   


def compute_cycle_average(dataset, wavelets, freq=8, numcycles=100):
    '''
    compute the frequency resolved average associated with a given frequency
    over a full period.
    
    cycle average is triggered on the zero-crossings of hilbert-space phase
    at given frequency
    
    kwargs:
    ::
        dataset: <pyeegtools.EEGfile> object
        wavelets : dict, as output by pyeegtools.get_morlet_wavelets(**kwargs)
        freq : int/float, frequency at which mean is computed
        numcycles : int or 'all', number of cycles used for averaging
        
    return:
    ::
        cycles : np.array, real component of X(f) during each cycle
        events : np.array, abs(X) during each cycle
        times : np.array, timestamps of cycles and events
    '''
    #corresponding index, period
    ind = wavelets['freqs'] == freq
    T = 1./freq #period
    window = np.ceil(T*dataset.Fs/2) #period in units of samples rounded even
    
    #find zero crossings of phase angle from hilbert transform
    phase = np.arctan2(dataset.cwt[ind,].imag,
                       dataset.cwt[ind,].real).flatten()
    
    #detect crossings
    [crossings] = np.where((phase[:-1] > 0) & (phase[1:] < 0))
    #skip crossings less than one half period away from each end point
    crossings = crossings[(crossings>=window) & \
                          (crossings <= phase.size-window)]
    
    #constrain number of cycles in average
    if numcycles=='all':
        pass
    else:
        if crossings.size > numcycles:
            crossings = crossings[:numcycles]

    #get event times
    times = crossings.astype(float) / dataset.Fs

    #cycles from phase crossing times
    cycles = np.empty((crossings.size, window*2))
    #continuous wavelets envelopes from phase crossing times
    events = np.empty((crossings.size, wavelets['freqs'].size, window*2))
    for i, j in enumerate(crossings):
        cycles[i, ] = dataset.cwt[ind, j-window:j+window].real
        events[i, ] = np.abs(dataset.cwt)[:, j-window:j+window]
        
    return cycles, events, times


def get_high_speed_inds(position, event_times, speedlimit):
    '''
    return indices corresponding to event_times recorded when animal was
    running faster than speed limit
    
    kwargs:
    ::
        position : <pyeegtools.Posfile> object
        event_times : np.array, time stamps of EEG events
        speedlimit : float, speed in m/s
        
    return:
    ::
        inds : np.array, indices of event_times where speed >= speedlimit
    
    '''
    
    time = position.time
    x = position.x
    y = position.y
    
    speed = np.r_[0, np.abs(np.diff(np.sqrt(x**2 + y**2)) / np.diff(time))]

    #remove event times without proper position
    event_times = event_times[(event_times >= time[0]) & (event_times <= time[-1])]
    
    #use times in units of indices, avoid roundoff errors
    utime = np.arange(time.size) + int(time[0]*position.Fs)
    uevent_times = (event_times*position.Fs).astype(int)
    
    event_speeds = np.zeros(event_times.size)
    for j, t in enumerate(uevent_times):
        event_speeds[j] = speed[utime==t]


    #setup of bar plots
    #distribute events to bins
    inds = event_speeds >= speedlimit

    return inds


def remove_axis_junk(ax, which=['right', 'top']):
    '''remove upper and right axis
    
    kwargs:
    ::
        ax : <matplotlib.axes.AxesSubplot>
        which : list of strings, which axes lines to remove
    '''
    for loc, spine in ax.spines.iteritems():
        if loc in which:
            spine.set_color('none')            
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')


def colorbar(fig, ax, im, label):
    '''better formatted colorbar.
    
    kwargs:
    ::
        fig : <matplotlib.figure.Figure>
        ax : <matplotlib.axes.AxesSubplot>
        im : <matplotlib.image.AxesImage>
        label: str, colorbar label
    '''
    rect = np.array(ax.get_position().bounds)
    rect[0] += rect[2] + 0.01
    rect[2] = 0.01
    cax = fig.add_axes(rect)
    cbar = plt.colorbar(im, cax=cax)
    cbar.set_label(label, ha='center')


def draw_crossfreq_coherence(setname, dataset, wavelets, NFFT=256):
    '''do the cross-frequency coherence image plot as function of
    frequency resolved signal phase from hilbert_transforms at each
    wavelet center frequency
    
    kwargs:
    ::
        setname: str, figure suptitle
        dataset: <pyeegtools.EEGfile> object
        wavelets : dict, as output by pyeegtools.get_morlet_wavelets(**kwargs)
        NFFT : int in base 2, frequency resolution of 
    
    return:
    ::
        fig : <matplotlib.figure.Figure>
        Cxy : np.array, crossfrequency coherence matrix
        f : frequency axis of coherence
    
    '''
    #container
    Cxy = []
    x = dataset.eeg
    for i, freq in enumerate(wavelets['freqs']):
        y = dataset.cwt[i, ].flatten()
        #power defined as absolute magnitude squared
        y = np.abs(y)**2
        
        #compute coherence
        cxy, f = plt.mlab.cohere(y, x, NFFT=NFFT, Fs=dataset.Fs)
        Cxy.append(cxy)
    Cxy = np.array(Cxy)
    
    
    extent = [
        f.min(), f.max(),
        wavelets['freqs'].min(), wavelets['freqs'].max(),
        ]
    
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(setname, va='bottom')
    
    ax = fig.add_subplot(111)
    im = ax.imshow(Cxy,
                extent=extent,
                origin='bottom',
                interpolation='nearest',
                cmap=plt.get_cmap('jet', 51),
                rasterized=True)
    plt.axis('tight')
    ax.set_title('cross-frequency coherence')
    ax.set_xlabel(r'f$_\mathrm{phase}$ (Hz)')
    ax.set_ylabel(r'f$_\mathrm{power}$ (Hz)')
    colorbar(fig, ax, im, label=r'coherence (-)') 

    return fig, Cxy, f


def figure1(setname, dataset, wavelets,
            time_window=(0, 5),
            thetafreqs = (0, 20),
            gammafreqs = (20, 80),
            NFFT=512,
            ):
    '''plot spectral contents and continuous wavelet transforms'''
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(setname, va='bottom')

    tvec = np.arange(dataset.eeg.size).astype(float) / dataset.Fs
    time_inds = (tvec >= time_window[0]) & (tvec <= time_window[1])


    #draw FFT PSD
    ax = fig.add_axes([0.1, 0.5, 0.15, 0.45])
    Pxx, freqs = plt.mlab.psd(dataset.eeg, NFFT=NFFT, Fs=dataset.Fs)
    plt.semilogx(Pxx, freqs, 'k', rasterized=True)
    plt.xlabel(r'PSD (mV$^2$/Hz)')
    plt.ylabel('f (Hz)')
    plt.title('spectral density')
    plt.axis('tight')
    
    
    #draw FFT spectogram
    ax = fig.add_axes([0.3, 0.5, 0.6, 0.45])
    Pxx, freqs, bins, im = plt.specgram(dataset.eeg, Fs=dataset.Fs, NFFT=NFFT,
                                        interpolation='nearest',
                                        cmap=plt.get_cmap('jet', 51),
                                        rasterized=True)
    plt.axis('tight')
    plt.title('FFT spectogram')
    colorbar(fig, ax, im, r'power (mV$^2$/Hz)')
    
        

    
    #draw timeseries
    ax = fig.add_axes([0.3, 0.35, 0.6, 0.1])
    plt.plot(tvec[time_inds],
             dataset.eeg[time_inds],
             'k', rasterized=True)
    plt.title('EEG timeseries')
    plt.ylabel('(mV)')
    plt.axis('tight')
    
    
    
    #draw image plot of envolope of continous wavelet transform,
    #first higher freqs
    ax = fig.add_axes([0.3, 0.2, 0.6, 0.10])
    gammainds = (wavelets['freqs'] >= gammafreqs[0]) & (wavelets['freqs'] <= gammafreqs[1])
    wfreqs = wavelets['freqs'][gammainds]
    xlims = [tvec[time_inds][0],
             tvec[time_inds][-1]]
    ylims = [wfreqs[0] - 1,
             wfreqs[-1] + 1]
    
    im = plt.imshow(np.abs(dataset.cwt[gammainds, :][:, time_inds]),
                    extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                    interpolation='nearest',
                    origin='bottom',
                    cmap=plt.get_cmap('jet', 51),
                    rasterized=True)
    plt.ylabel('f (Hz)')
    plt.title('continous wavelet transform (%i <= f <= %i Hz)' % gammafreqs)
    plt.axis('tight')
    colorbar(fig, ax, im, r'$|X_w|$ (-)')
    

    #low frequency wavelets
    ax = fig.add_axes([0.3, 0.05, 0.6, 0.10])
    thetainds = (wavelets['freqs'] >= thetafreqs[0]) & (wavelets['freqs'] <= thetafreqs[1])
    wfreqs = wavelets['freqs'][thetainds]
    xlims = [tvec[time_inds][0],
             tvec[time_inds][-1]]
    ylims = [wfreqs[0] - 1,
             wfreqs[-1] + 1]
    
    im = plt.imshow(np.abs(dataset.cwt[thetainds, :][:, time_inds]),
                    extent=[xlims[0], xlims[1], ylims[0], ylims[1]],
                    origin='bottom',
                    interpolation='nearest',
                    cmap=plt.get_cmap('jet', 51),
                    rasterized=True)
    plt.ylabel('f (Hz)')
    plt.xlabel('time (s)')
    plt.title('continous wavelet transform, (%i <= f <= %i Hz)' % thetafreqs)
    plt.axis('tight')
    colorbar(fig, ax, im, label=r'$|X_w|$ (-)')  
    
    return fig



def figure2(setname, dataset, wavelets,
            time_window=(0, 5),
            f_theta=8, f_gamma_low=40, f_gamma_high=60,
            phasebins=np.linspace(-np.pi, np.pi, 20)):
    '''
    phase amplitude plots from continuous wavelet transforms
    '''
    
    #phase-amplitude plots
    #find corresponding indices:
    f_theta_ind = wavelets['freqs'] == f_theta
    f_gamma_ind_low = wavelets['freqs'] == f_gamma_low
    f_gamma_ind_high = wavelets['freqs'] == f_gamma_high
    
    #compute phase-amplitude histogram (see Tort et al. 2010)
    binsize = np.diff(phasebins)[0]
    thetaphases = np.arctan2(dataset.cwt[f_theta_ind, ].imag,
                             dataset.cwt[f_theta_ind, ].real).flatten()
    
    
    #appropriate plotting window
    tvec = np.arange(dataset.eeg.size).astype(float) / dataset.Fs
    time_inds = (tvec >= time_window[0]) & (tvec <= time_window[1])
    
    
    fig = plt.figure(figsize=(10, 10))
    fig.suptitle(setname, va='bottom')
    fig.subplots_adjust(left=0.1, bottom=0.075,
                        right=0.95, top=0.925,
                        wspace=0.4, hspace=0.4)
    
    
    ax = fig.add_subplot(421)
    plt.plot(tvec[time_inds], dataset.eeg[time_inds], 'k')
    plt.ylabel(r'$\Phi_\mathrm{EEG}(t)$ (mV)')
    plt.title(r'EEG signal')
    
    
    ax = fig.add_subplot(423)
    plt.plot(tvec[time_inds], dataset.cwt[f_theta_ind, time_inds].real, 'k')
    plt.plot(tvec[time_inds], np.abs(dataset.cwt[f_theta_ind, time_inds]), 'r')
    plt.ylabel(r'$\mathrm{Re}(X_\omega)$ (-)')
    plt.title(r'theta signal ($\mathrm{Re}(X_\omega)$, f=%i Hz)' % f_theta)
    
    
    ax = fig.add_subplot(425)
    plt.plot(tvec[time_inds], thetaphases[time_inds], 'k')
    plt.ylim(-np.pi, np.pi)
    ax.set_yticks([-np.pi,-np.pi/2, 0, np.pi/2, np.pi])
    ax.set_yticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    plt.ylabel(r'$\theta_\omega (t)$ (rad)')
    plt.xlabel(r't (s)')
    plt.title(r'phase ($\arctan(\mathrm{Re}(X_\omega)/\mathrm{Im}(X_\omega))$, f=%i Hz)' % f_theta)
    
    
    ax = fig.add_subplot(422)
    plt.plot(tvec[time_inds], dataset.cwt[f_gamma_ind_low, time_inds].real, 'k')
    plt.plot(tvec[time_inds], np.abs(dataset.cwt[f_gamma_ind_low, time_inds]), 'r')
    plt.ylabel(r'$\mathrm{Re}(X_\omega)$ (-)')
    plt.title(r'low gamma ($\mathrm{Re}(X_\omega)$, f=%i Hz)' % f_gamma_low)
    
    
    ax = fig.add_subplot(424)
    plt.plot(tvec[time_inds], dataset.cwt[f_gamma_ind_high, time_inds].real, 'k')
    plt.plot(tvec[time_inds], np.abs(dataset.cwt[f_gamma_ind_high, time_inds]), 'r')
    plt.ylabel(r'$\mathrm{Re}(X_\omega)$ (-)')
    plt.title(r'high gamma ($\mathrm{Re}(X_\omega)$, f=%i Hz)' % f_gamma_high)
    
    
    #compute phase-amplitude histogram (see Tort et al. 2010)
    gamma_ampl = np.abs(dataset.cwt[f_gamma_ind_low, ]).flatten()
    gamma_hist = []
    gamma_hist_err = []
    #loop over each phase angle bin:
    for phasebin in phasebins:
        inds = (thetaphases >= phasebin) & (thetaphases < phasebin + binsize)
        gamma_hist = np.r_[gamma_hist, gamma_ampl[inds].mean()]
        gamma_hist_err = np.r_[gamma_hist_err, gamma_ampl[inds].std()]
    
    
    ax = fig.add_subplot(426)
    plt.bar(phasebins, gamma_hist, width=binsize*0.8)
    #plt.bar(phasebins, gamma_hist, width=binsize*0.8, yerr=gamma_hist_err)
    plt.axis('tight')
    ax.set_xticks([-np.pi,-np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    plt.title(r'phase-amplitude, f=%i Hz vs. f=%i Hz' % (f_theta, f_gamma_low))
    plt.ylabel(r'$|(X_\omega)|$ (-)')

    
    #compute phase-amplitude histogram (see Tort et al. 2010)
    gamma_ampl = np.abs(dataset.cwt[f_gamma_ind_high, ]).flatten()
    gamma_hist = []
    gamma_hist_err = []
    #loop over each phase angle bin:
    for phasebin in phasebins:
        inds = (thetaphases >= phasebin) & (thetaphases < phasebin + binsize)
        gamma_hist = np.r_[gamma_hist, gamma_ampl[inds].mean()]
        gamma_hist_err = np.r_[gamma_hist_err, gamma_ampl[inds].std()]
    
    
    ax = fig.add_subplot(428)
    plt.bar(phasebins, gamma_hist, width=binsize*0.8)
    #plt.bar(phasebins, gamma_hist, width=binsize*0.8, yerr=gamma_hist_err)
    plt.axis('tight')
    ax.set_xticks([-np.pi,-np.pi/2, 0, np.pi/2, np.pi])
    ax.set_xticklabels([r'$-\pi$', r'$-\pi/2$', r'0', r'$\pi/2$', r'$\pi$'])
    plt.title(r'phase-amplitude, f=%i Hz vs. f=%i Hz' % (f_theta, f_gamma_high))
    plt.xlabel(r'$\theta_\omega$ (rad)')
    plt.ylabel(r'$|(X_\omega)|$ (-)')


    return fig


def figure3(datasets, setname):
    '''
    plot interpolated position data,
    if no valid data is found, return empty figure
    '''
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(setname, va='bottom')

    if datasets['position'].valid:
        post_interp = datasets['position'].time
        posx_interp = datasets['position'].x
        posy_interp = datasets['position'].y
    
        
        ax = fig.add_axes([0.4, 0.4, 0.5, 0.5])
        #plt.plot(posx, posy, 'k.')
        plt.plot(posx_interp, posy_interp, lw=2)
        plt.axis('tight')
        plt.title('position')
    
        ax = fig.add_axes([0.4, 0.1, 0.5, 0.25])
        #plt.plot(posx, post, 'k.')
        plt.plot(posx_interp, post_interp, lw=2)
        plt.ylabel('t (s)')
        plt.xlabel('x (m)')
        plt.axis('tight')
    
        ax = fig.add_axes([0.1, 0.1, 0.25, 0.25])
        plt.plot(post_interp[1:],
                 abs(np.diff(np.sqrt(posx_interp**2+posy_interp**2)) / np.diff(post_interp)),
                 lw=2)
        plt.xlabel('t (s)')
        plt.ylabel('speed (m/s)')
        plt.axis('tight')
        
        ax = fig.add_axes([0.1, 0.4, 0.25, 0.5])
        #plt.plot(post, posy, 'k.')
        plt.plot(post_interp, posy_interp, lw=2)
        plt.xlabel('t (s)')
        plt.ylabel('y (m)')
        plt.axis('tight')
        
    return fig


def figure4(setname, dataset, wavelets, events, freq=8, thetafreqs=(0, 20), gammafreqs=(20, 80),
            fig=None, figsize=(10,10), column=0, numcolumns=3, columnwidth=0.25):
    '''
    plot mean events, frequency resolved
    
    kwargs:
    ::
        setname: 
    '''
    
    #get indices for upper and lower band
    thetainds = (wavelets['freqs'] >= thetafreqs[0]) & \
                (wavelets['freqs'] <= thetafreqs[1])
    gammainds = (wavelets['freqs'] >= gammafreqs[0]) & \
                (wavelets['freqs'] <= gammafreqs[1])

    #get duration of events
    window = events.shape[2] / dataset.Fs
    #get the imshow extents
    thetaextent = [-window/2.,
                   window/2.,
              wavelets['freqs'][thetainds][0] - 1,
              wavelets['freqs'][thetainds][-1] + 1]
    
    gammaextent = [-window/2., window/2.,
              wavelets['freqs'][gammainds][0] - 1,
              wavelets['freqs'][gammainds][-1] + 1]


    #plotting window
    if fig == None:
        fig = plt.figure(figsize=figsize)
        fig.suptitle(setname, va='bottom')
    
    #find appropriate position of plot
    xpos = np.linspace(0.1, 0.9-columnwidth, numcolumns)[column]
    
    #draw in figure
    ax = fig.add_axes([xpos, 0.35, columnwidth, 0.55])
    im = ax.imshow(events.mean(axis=0)[gammainds],
               extent=gammaextent,
               interpolation='nearest',
               cmap=plt.get_cmap('jet', 51),
               origin='bottom')
    if column == 0:
        ax.set_ylabel(r'f (Hz)')
    ax.set_title(r'$|(X_\omega)|$, f=%i Hz' % freq)
    ax.axis(ax.axis('tight'))
    colorbar(fig, ax, im, 'amplitude  (-)')

    ax = fig.add_axes([xpos, 0.1, columnwidth, 0.2])
    im = ax.imshow(events.mean(axis=0)[thetainds],
               extent=thetaextent,
               interpolation='nearest',
               cmap=plt.get_cmap('jet', 51),
               origin='bottom')
    ax.set_xlabel('t (s)')
    ax.set_yticks(wavelets['freqs'][thetainds])
    if column == 0:
        ax.set_ylabel(r'f (Hz)')
    ax.axis(ax.axis('tight'))
    colorbar(fig, ax, im, 'amplitude (-)')

    return fig


def figure5(setname, dataset, wavelets, cycles, events,
            thetafreqs=(0, 20), gammafreqs=(20, 80), f_theta=8):
    '''
    draw image plots of cycle averaged responses at a given frequency
    
    
    '''
    #get indices for upper and lower band
    thetainds = (wavelets['freqs'] >= thetafreqs[0]) & \
                (wavelets['freqs'] <= thetafreqs[1])
    gammainds = (wavelets['freqs'] >= gammafreqs[0]) & \
                (wavelets['freqs'] <= gammafreqs[1])

    #get duration of events
    window = float(events.shape[2]) / dataset.Fs
    
    #get the imshow extents
    thetaextent = [-window/2.,
                   window/2.,
              wavelets['freqs'][thetainds][0] - 1,
              wavelets['freqs'][thetainds][-1] + 1]
    
    gammaextent = [-window/2., window/2.,
              wavelets['freqs'][gammainds][0] - 1,
              wavelets['freqs'][gammainds][-1] + 1]

    
    #set up figure object
    fig = plt.figure(figsize=(10,10))
    fig.suptitle(setname, va='bottom')
    

    #draw in figure
    ax = fig.add_axes([0.1, 0.5, 0.8, 0.4])
    im = ax.imshow(events.mean(axis=0)[gammainds],
               extent=gammaextent,
               interpolation='nearest',
               cmap=plt.get_cmap('jet', 51),
               origin='bottom')
    ax.set_ylabel(r'f (Hz)')
    ax.set_title(r'$|(X_\omega)|$, f=%i Hz average' % f_theta)
    ax.axis(ax.axis('tight'))
    colorbar(fig, ax, im, r'$|(X_\omega)|$ (-)')

    ax = fig.add_axes([0.1, 0.25, 0.8, 0.2])
    im = ax.imshow(events.mean(axis=0)[thetainds],
               extent=thetaextent,
               interpolation='nearest',
               cmap=plt.get_cmap('jet', 51),
               origin='bottom')
    ax.set_ylabel(r'f (Hz)')
    ax.axis(ax.axis('tight'))
    colorbar(fig, ax, im, r'$|(X_\omega)|$ (-)')
    
    ax = fig.add_axes([0.1, 0.1, 0.8, 0.1])
    ax.plot((np.arange(cycles.shape[1])-cycles.shape[1]/2)*1./dataset.Fs,
                        cycles.mean(axis=0))
    ax.axis(ax.axis('tight'))
    ax.set_ylabel(r'$|(X_\omega)|$, f=%i Hz' % f_theta)
    ax.set_xlabel('t (s)')
    
    #return figure object
    return fig


def figure6(datasets, setname, wavelets,
           events, event_times, freq=40, speedlimit=0.01,
           fig=None, figsize=(10,10), column=0,
           numcolumns=3, columnwidth=0.18):
    '''
    plot the event amplitude distributions with/without motion
    '''

    #get the peak amplitude for given frequency
    ind = wavelets['freqs'] == freq
    maxima = events[:, ind, :].max(axis=2).flatten()

    #get the continous rat speed
    if datasets.has_key('position') & datasets['position'].valid:        
        inds = get_high_speed_inds(datasets['position'], event_times, speedlimit)
        
        #data for boxes
        data = [
            maxima[inds==False],
            maxima[inds]
        ]
        
        #ticklabels
        xticklabels=[r'$\less %.2f$ ms$^{-1}$' % speedlimit,
                     r'$\geq %.2f$ ms$^{-1}$' % speedlimit]
    else:
        data = [maxima]
        xticklabels=['data']

    #plotting window
    if fig == None:
        fig = plt.figure(figsize=figsize)
        fig.suptitle(setname, va='bottom')

    #find appropriate position of plot
    xpos = np.linspace(0.1, 0.9-columnwidth, numcolumns)[column]

    ax = fig.add_axes([xpos, 0.1, columnwidth, 0.8])
    ax.boxplot(data)
    ax.set_title('f=%i Hz' % freq)
    if column == 0:
        ax.set_ylabel(r'$\mathrm{Re}(X_\omega)$ (-)')
    ax.set_xticklabels(xticklabels, rotation=45)
    
    return fig, data


def figure7(setname, dataset, wavelets,
            freqs=[8, 30, 60], window=0.5, num_stds=2):
    '''
    cut and extract continous wavelet transform at given frequencies, and
    plot the averaged responses.
    '''
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(wspace=0.4, hspace=0.4)
    fig.suptitle(setname, va='bottom')

    #counter
    subplot = 1 

    #compute event triggered frequency events
    for i, freq in enumerate(freqs):
        #get the data for this freq
        cwt_events = dataset.cwt_events[freq]
        event_times = dataset.cwt_event_times[freq]

        
        #freq above is frequency for triggering,
        #now, we plot the means at each
        for j, f in enumerate(freqs):
            tvec = np.arange(dataset.Fs*2*window).astype(float) / dataset.Fs - 0.5
            ind = wavelets['freqs'] == f
            ax = fig.add_subplot(len(freqs), len(freqs), subplot)
            ax.plot(tvec, cwt_events.mean(axis=0).real[ind].flatten(), 'k', label='f=%i Hz' % f)
            ax.plot(tvec, np.abs(cwt_events.mean(axis=0))[ind].flatten(), 'r')
            ax.set_title(r'$\mathrm{Re}(\bar{X_\omega})$, f=%iHz, %iHz' % (freq, f))
            ax.axis(ax.axis('tight'))
            
            subplot += 1
            

        if i == len(freqs) - 1:
            ax.set_xlabel('t (s)')
        
    return fig


def figure8(datasets, time_window=(0, 5)):
    '''plot each end every eeg trace of datasets'''
    fig  = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.3)

    fig.suptitle(datasets['setfile'])

    subplot = 1
    keys = []
    for key in np.sort(datasets.keys()):
        if key in ['set', 'setfile', 'position', 'posfile']:
            continue
        else:
            keys.append(key)
    
    #global ylims:
    ylims = []
    for setname in keys:
        ylims = np.r_[ylims, np.abs(datasets[setname].eeg).max()]
    
    
    
    for i, setname in enumerate(keys):
        dataset = datasets[setname]
        tvec = np.arange(dataset.eeg.size).astype(float) / dataset.Fs
        time_inds = (tvec >= time_window[0]) & (tvec <= time_window[1])
    
        ax = fig.add_subplot(len(keys), 1, subplot)
        remove_axis_junk(ax)        
        ax.plot(tvec[time_inds], dataset.eeg[time_inds])            
        if i == 0:
            ax.set_title('EEG time series')
        ax.set_ylabel('ch. %i (mV)' % (i+1))
        ax.set_ylim(-ylims.max(), ylims.max())
        
        subplot += 1
    ax.set_xlabel('t (ms)')

    return fig


def figure9(datasets, NFFT=256):
    '''
    plot the coherences across all available datasets
    '''
    fig  = plt.figure(figsize=(10,10))
    fig.subplots_adjust(hspace=0.5, wspace=0.5, right=0.98, left=0.07, bottom=0.05, )
    fig.suptitle(os.path.split(datasets['setfile'])[-1])

    #plot a series of coherence plots
    subplot = 1
    keys = []
    for key in np.sort(datasets.keys()):
        if key in ['set', 'setfile', 'position', 'posfile']:
            continue
        else:
            keys.append(key)
    for i in xrange(len(keys)-1):
        for j in xrange(1, len(keys)):
            if i < j:
                ax = fig.add_subplot(len(keys)-1, len(keys)-1, subplot)
                remove_axis_junk(ax)
                Cxy, f = plt.mlab.cohere(datasets[keys[i]].eeg, datasets[keys[j]].eeg,
                           NFFT=NFFT, Fs=datasets[keys[i]].Fs)
                ax.plot(f, Cxy, 'k', clip_on=False)
                ax.set_xticks([0, 50, 100])
                ax.axis('tight')
                ax.set_title('ch%i:ch%i' % (i+1, j+1))
                ax.set_ylim((0, 1))
                if i == j-1:
                    ax.set_xlabel('f (Hz)')
                    ax.set_ylabel(r'coherence $c_\Phi$ (-)')
            else:
                pass

            subplot += 1
    
    return fig


#draw boxplots for amplitudes of events, comparing detected amplitudes
def plot_datasets_event_amplitudes(datasets, wavelets, speedlimit,
                                   freqs=(8, 30, 60)):
    '''
    detect events, plot boxplots comparing datasets given in each
    frequency band
    '''
    #set up axes titles
    titles = []
    for freq in freqs:
        titles.append('f=%i Hz' % freq)
    
    #use the sorted datakeys
    keys = []
    for key in np.sort(datasets.keys()):
        if key in ['set', 'setfile', 'position', 'posfile']:
            continue
        else:
            keys.append(key)

    #set up figure
    fig = plt.figure(figsize=(10,10))
    fig.subplots_adjust(bottom=0.15)
    fig.suptitle(datasets['setfile'])

    #iterate over frequencies
    for i, freq in enumerate(freqs):
        #container:
        data = []
        xticklabels = []
        #get the peak amplitude for given frequency
        ind = np.where(wavelets['freqs'] == freq)[0]

        for j, setname in enumerate(keys):
            dataset = datasets[setname]

            #get the data for this freq
            events = dataset.cwt_events[freq]
            times = dataset.cwt_event_times[freq]

            #get the local maxima
            maxima = np.abs(events[:, ind, :]).max(axis=2).flatten()
            
            #wanna sort on animal movement:
            if datasets.has_key('position') & datasets['position'].valid:
                inds = get_high_speed_inds(datasets['position'], times, speedlimit)
                #extract the corresponding amplitudes
                data.append(maxima[inds==False])
                data.append(maxima[inds])
                
                xticklabels.append(r'ch.%i, $\leq%.2f$ ms$^{-1}$' % (j+1, speedlimit))
                xticklabels.append(r'ch.%i, $>%.2f$ ms$^{-1}$' % (j+1, speedlimit))
            else:
                data.append(maxima)
                xticklabels.append('ch.%i' % (j+1))
            
        ax = fig.add_subplot(1, len(freqs), i+1)
        ax.boxplot(data)
        ax.set_xticklabels(xticklabels, rotation='vertical')
        ax.set_title(titles[i])
        if i ==0:
            ax.set_ylabel(r'$|X_w|$ amplitudes (-)')
            
    return fig

