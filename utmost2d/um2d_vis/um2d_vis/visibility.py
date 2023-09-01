"""
Class for handling SNAP-12x visibility files.

Also gives a command line plotting utility
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
from astropy.time import Time
from astropy.units import Quantity
import aoflagger

def db(x):
    return 10*np.log10(x)


def decode_list(a):
    return [x.decode('ascii') for x in a]


def decode_bytes(d):
    for key, value in d.items():
        if isinstance(value, bytes):
            d[key] = value.decode('ascii')
    return d


class Visibility(object):
    def __init__(self, fn, flaginputs=[]):
       
        a = h5py.File(fn, 'r')
        md = dict(a.attrs)
        md = decode_bytes(md)
        md['labels'] = decode_list(md['labels'])
        md['units'] = decode_list(md['units'])
        f_idx = list(md['labels']).index('freq')
        t_idx = list(md['labels']).index('time')
        ai_idx = list(md['labels']).index('station_i')
        aj_idx = list(md['labels']).index('station_j')
        f0, fdelt = md['scales'][f_idx]
        t0, tdelt = md['scales'][t_idx]
        t_off = tdelt * md['data_idx']
        t = Time(np.arange(a['data'].shape[t_idx]) * tdelt + t0 + t_off, format='unix')
        f = Quantity(np.arange(a['data'].shape[f_idx]) * fdelt + f0, unit=md['units'][f_idx])
        if len(flaginputs) > 0:
            self.data = np.delete(a['data'][:], flaginputs, ai_idx) # Get rid of any antennas we don't actually want
            self.data = np.delete(self.data, flaginputs, aj_idx) # Get rid of any antennas we don't actually want
        else:
            self.data = a['data'][:]
        self.freqs = f
        self.time  = t
        self.header = md
        self.ant_ids = range(self.data.shape[-1])
        self.f_idx = f_idx
        self.t_idx = t_idx
        self.ai_idx = ai_idx
        self.aj_idx = aj_idx
    
    def __repr__(self):
        s = "<Visibility: " + self.header['source_name'] + " "
        s += " UTC_START: " + self.header['utc_start'] + ">\n"
        s += "Shape: " + str(list(zip(self.header['labels'], self.data.shape)))
        return s
    
    def extract_antennas(self, ant_list):
        self.ant_ids = ant_list
        Nant = len(self.ant_ids)
        ds = self.data.shape
        data = np.zeros((ds[0], ds[1], Nant, Nant), dtype='complex128')
        for ii, ant in enumerate(self.ant_ids):
            for jj, ant2 in enumerate(self.ant_ids):
                data[..., ii, jj] = self.data[..., ant, ant2]
        self.data = data

    def normalise(self):
        # First set any zeros in the autocorrelations to 1.0
        for ii, ant in enumerate(self.ant_ids):
            self.data[...,ii,ii] = np.where(self.data[...,ii,ii] == 0 + 0j, 1 + 0j, self.data[...,ii,ii])
        # Then normalise
        for ii, ant in enumerate(self.ant_ids):
            for jj, ant2 in enumerate(self.ant_ids):
                if ii != jj:
                    self.data[..., ii, jj] = self.data[..., ii, jj] / np.abs(np.sqrt(self.data[..., ii, ii] * self.data[..., jj, jj]))
        for ii, ant in enumerate(self.ant_ids):
            self.data[..., ii, ii] = 1 + 0j

    def apply_gains(self, gains):
        if isinstance(gains, str):
            import hickle as hkl
            gains = hkl.load(gains)

        nant = self.data.shape[-1]
        for ii in range(nant):
            for jj in range(nant):
                g = np.conj(gains[:, ii]) * gains[:, jj]
                self.data[..., ii, jj] /= g

    def determine_validity(self):
        self.valid = np.mean(np.abs(self.data), (self.f_idx, self.ai_idx, self.aj_idx)) > 0
        print(self.valid)

    def append_data(self, fn):
        a = h5py.File(fn)
        self.data = np.concatenate((self.data, a['data']))
        self.time = Time(np.arange(self.data.shape[self.t_idx]) * self.tdelt + self.t0 + self.t_off, format='unix')

    def append_data_freq(self, fn, flaginputs=[], verbose=False):
        #a = h5py.File(fn)
        a = Visibility(fn)
        tdelt = self.time.unix[1] - self.time.unix[0]
        newtdelt = a.time.unix[1] - a.time.unix[0]
        fdelt = self.freqs.value[1] - self.freqs.value[0]
        newfdelt = a.freqs.value[1] - a.freqs.value[0]
        f_idx = list(self.header['labels']).index('freq') 
        t_idx = list(self.header['labels']).index('time')
        ai_idx = list(self.header['labels']).index('station_i')
        aj_idx = list(self.header['labels']).index('station_j')
        if verbose:
            print("About to append a file with frequencies starting at",a.freqs.value[0],"to me, which has frequencies starting at",self.freqs.value[0])
            print("indices (t,f,ai,aj) are", t_idx, f_idx, ai_idx, aj_idx)
        #t0, tdelt = self.header['scales'][t_idx]
        #f0, fdelt = self.header['scales'][f_idx]
        #newt0, newtdelt = a.header['scales'][t_idx]
        #newf0, newfdelt = a.header['scales'][f_idx]
        # Check that the integration time is the same
        if np.abs(tdelt - newtdelt) > 1e-6: # Check that integration times are "close enough"
            print("Incompatible files to append: tdelt({0},{1})".format(tdelt, newtdelt))
            raise Exception("BYE!")
        if np.abs(tdelt - newtdelt) != 0: # They are close enough, but still not exactly the same - raise a warning
            print("Warning: tdelt is not identical ({0},{1})".format(tdelt, newtdelt))
        # Check that there is some overlap in the time
        if not (self.time.unix[-1] > a.time.unix[0] and a.time.unix[-1] > self.time.unix[0]):
            print("No overlap in files: start times({0},{1}), end times ({2},{3})".format(self.time.unix[0], a.time.unix[0], self.time.unix[-1], a.time.unix[1]))
            raise Exception("BYE!")
        # Check that the new file is adjacent in frequency
        if not (self.freqs.value[0] == a.freqs.value[0] + len(a.freqs)*newfdelt or a.freqs.value[0] == self.freqs.value[0] + len(self.freqs)*newfdelt):
            print("Files not adjacent in frequency: start frequencies({0},{1}), last end frequencies ({2},{3})".format(self.freqs.value[0], a.freqs.value[0],  self.freqs.value[0] + len(self.freqs)*fdelt,  a.freqs.value[0] + len(a.freqs)*newfdelt))
            raise Exception("BYE!")
        # Work out the time offset
        if verbose:
            print(a.time.isot[0], a.time.isot[0], tdelt)
        starttimediff = np.rint((a.time.unix[0] - self.time.unix[0]) / tdelt).astype(int)
        if verbose:
            print("Time offset in samples:",starttimediff)

        # Add appropriate zero padding to the data to be added
        if starttimediff != 0:
            padding = np.zeros(2*len(self.data.shape), dtype=int).reshape(len(self.data.shape), 2)
            if starttimediff > 0:
                padding[t_idx][0] = starttimediff # Pad the start of the t axis with zeros
                start = 0
                end = len(self.time)
            else:
                padding[t_idx][1] = -starttimediff # Pad the end of the t axis with zeros
                start = -starttimediff
                end = len(self.time) - starttimediff
            appenddata = np.pad(a.data,padding,mode='constant') # Defaults to constant value of zero for the padding
            appenddata = appenddata.take(indices=range(start, end), axis=t_idx)
        else:
            appenddata = a.data
        if len(flaginputs) > 0:
            appenddata = np.delete(appenddata, flaginputs, ai_idx)
            appenddata = np.delete(appenddata, flaginputs, aj_idx)

        # And then add the data in the right order
        if verbose:
            print("Before concatenation:",self.data.shape)
        if a.freqs.value[0] < self.freqs.value[0]:
            self.header['scales'][f_idx] = (a.freqs.value[0], fdelt)
            self.data = np.concatenate((appenddata, self.data), axis=f_idx)
            self.freqs = Quantity(np.arange(self.data.shape[f_idx]) * fdelt + a.freqs.value[0], unit=self.header['units'][f_idx])
        else:
            self.data = np.concatenate((self.data, appenddata), axis=f_idx)
            self.freqs = Quantity(np.arange(self.data.shape[f_idx]) * fdelt + self.freqs.value[0], unit=self.header['units'][f_idx])
        if verbose:
            print("After concatenation:",self.data.shape)

    def aoflag(self,firsttimeindex,lasttimeindex, plot_reference_input=-1):
        # Create the AOFlagger object
        flagger = aoflagger.AOFlagger()
        # Set up the strategy and the data
        path = flagger.find_strategy_file(aoflagger.TelescopeId.LOFAR)
        strategy = flagger.load_strategy_file(path)
        aodata = flagger.make_image_set(len(self.freqs), lasttimeindex + 1 - firsttimeindex, 1)
        nant = self.data.shape[-1]
        #if plot_reference_input >= 0:
        #    fig, axes = plt.subplots(1, 2)
        for i in range(nant):
            for j in range(nant):
                if i == j or not i%2 == j%2:
                    continue # Skip intra-module baselines or cross-polarisation baselines
                # Get the visibility amplitude from this baseline and give it to the AOFlagger
                baselineamplitudes = np.log(np.abs(self.data[firsttimeindex:lasttimeindex+1,:,i,j]))
                aodata.set_image_buffer(0, baselineamplitudes)
    
                # Run the flagging
                flags = strategy.run(aodata)
                flagvalues = flags.get_buffer()
                if i == plot_reference_input:
                    plt.imshow(baselineamplitudes)
                    plt.savefig("aoflagger.raw.amps.{0}-{1}.png".format(i,j))
                    plt.clf()
                    plt.imshow(baselineamplitudes*(1 - flagvalues))
                    plt.savefig("aoflagger.flagged.amps.{0}-{1}.png".format(i,j))
                    plt.clf()
                    #axes[0].imshow(baselineamplitudes)
                    #axes[1].imshow(baselineamplitudes*(1 - flagvalues))
                    #plt.savefig("aoflagger.amps.{0}-{1}.png".format(i,j))
                    #fig.clf()
                self.data[firsttimeindex:lasttimeindex+1:,:,i,j] = self.data[firsttimeindex:lasttimeindex+1,:,i,j]*(1 - flagvalues)

    def flagtimes(self, ftimes):
        for ft in ftimes:
            self.data[ft,:,:,:] = 0j + 0
            for n in range(self.data.shape[3]):
                self.data[ft,:,n,n] = 0j + 1

    def flagchans(self, fchans):
        for fc in fchans:
            self.data[:,fc,:,:] = 0j + 0
            for n in range(self.data.shape[3]):
                self.data[:,fc,n,n] = 0j + 1

    def flagbaselines(self, fbaselines):
        for fb in fbaselines:
            self.data[:,:,fb[0],fb[1]] = 0j + 0


    def plot_bandpass(self, ant_id, ant_id2=None, t_idx='all', logged=False, phase=False, *args, **kwargs):
        if ant_id2 is None:
            ant_id2 = ant_id
        if phase:
            pfunc = np.angle
        else:
            pfunc = np.abs
        if t_idx == 'all':
             d = pfunc(self.data[:, :, ant_id, ant_id2]).nanmean(axis=0)
        else:
             d = pfunc(self.data[t_idx, :, ant_id, ant_id2]).nanmean(axis=0)
            
        if logged:
            d = db(d)
            plt.ylabel("Power [dB counts]")
        elif phase:
            plt.ylim(-3.5, 3.5)
            plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], ["$-\pi$", "$-\pi/2$", 0, "$\pi/2$", "$\pi$"])
        else:
            plt.ylabel("Power [counts]")
        plt.plot(self.freqs, d, *args, **kwargs)
        plt.xlabel("Frequency [%s]" % str(self.freqs.unit))
        plt.title("Bandpass ANT %i" % ant_id)
    
    def plot_waterfall(self, ant_id, ant_id2=None, phase=False, time_fmt='elapsed', logged=False, *args, **kwargs):
        if ant_id2 is None:
            ant_id2 = ant_id
        if phase:
            d = np.angle(self.data[:, :, ant_id, ant_id2])
        else:
            d = np.abs(self.data[:, :, ant_id, ant_id2])
        if logged:
            d = db(d)
        kwargs['aspect'] = kwargs.get('aspect', 'auto')
        
        if time_fmt == 'elapsed':
            t0 = 0
            t1 = (self.time.unix[-1] - self.time.unix[0])
            plt.ylabel("Elapsed time [s]")
        else:
            plt.ylabel("%s [%s]" %(self.time.scale, self.time.format))
        
        kwargs['extent'] = kwargs.get('extent', [self.freqs[0].value, self.freqs[-1].value, t1, t0])
        plt.imshow(d, *args, **kwargs)
        plt.xlabel("Frequency [%s]" % str(self.freqs.unit))
        plt.title("%i-%i" % (ant_id, ant_id2))

    def plot_matrix(self, xlim=None, ylim=None, logged=False, phase=False, waterfall=False, 
                    triangle='lower', figscale=1.0, *args, **kwargs):
        plt.figure(figsize=(figscale*len(self.ant_ids),figscale*len(self.ant_ids)))
        Nmat = self.data.shape[-1]
        for ii in range(Nmat):
            for jj in range(Nmat):
                if triangle == 'lower':
                    do_plot = ii >= jj
                elif triangle == 'upper':
                    do_plot = ii <= jj
                elif triangle == 'all':
                    do_plot = True
                else:
                    raise RuntimeError("Need to select triangle=lower, upper or all")
                if do_plot:
                    plt.subplot(Nmat, Nmat, Nmat*ii + jj + 1)
                    if waterfall:
                        self.plot_waterfall(ii, jj, logged=logged, phase=phase, *args, **kwargs)
                    else:
                        self.plot_bandpass(ii, jj, logged=logged, phase=phase, *args, **kwargs)
                    plt.title("")
                    plt.xticks([])
                    plt.yticks([])
                    if jj == 0:
                        plt.ylabel("A%i" % ii)
                    else:
                        plt.ylabel("")
                    if ii == 11:
                        plt.xlabel("A%i" % jj)
                    else:
                        plt.xlabel("")
                    if xlim is not None:
                        plt.xlim(*xlim)
                    if ylim is not None:
                        plt.ylim(*ylim)
                    
        plt.subplots_adjust(hspace=0, wspace=0)            


def load_vis(fn):
    return Visibility(fn)
