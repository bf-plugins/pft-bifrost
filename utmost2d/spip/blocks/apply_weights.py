# Copyright (c) 2016, The Bifrost Authors. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
# * Redistributions of source code must retain the above copyright
#   notice, this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright
#   notice, this list of conditions and the following disclaimer in the
#   documentation and/or other materials provided with the distribution.
# * Neither the name of The Bifrost Authors nor the names of its
#   contributors may be used to endorse or promote products derived
#   from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from __future__ import absolute_import

import bifrost as bf
from bifrost.pipeline import TransformBlock
from bifrost.DataType import DataType

from astropy.time import Time, TimeDelta
from astropy import units as u
from astropy.coordinates import Angle
import astropy.constants as c

from um2d_vis.source_db import source_db
from um2d_vis.sky_model import make_sky_model
from um2d_vis.ant_array import make_antenna_array

from copy import deepcopy
import numpy as np

LIGHT_SPEED = c.c.value

INT_CMULT_KERNEL = """
// Compute b = w * a

Complex<float> a_cf;
a_cf.real = a.real;
a_cf.imag = a.imag;

Complex<float> w_cf;
w_cf.real = w.real;
w_cf.imag = w.imag;

Complex<float> b_cf;
b_cf = a_cf * w_cf;

b.real = (int) b_cf.real;
b.imag = (int) b_cf.imag;
"""

FLOAT_CMULT_KERNEL = """
// Compute b = w * a

Complex<float> a_cf;
a_cf.real = a.real;
a_cf.imag = a.imag;

Complex<float> w_cf;
w_cf.real = w.real;
w_cf.imag = w.imag;

b = a_cf * w_cf;
"""

class ApplyWeightsBlock(TransformBlock):
    def __init__(self, iring, weights_callback, output_dtype, update_frequency, 
                 *args, **kwargs):
        super(ApplyWeightsBlock, self).__init__(iring, *args, **kwargs)
        self.weights_callback = weights_callback
        self.update_frequency = update_frequency
         
        # Slightly different kernels for float / int output
        self.output_dtype = output_dtype
        assert output_dtype in ('cf32', 'ci8', 'ci16', 'ci32')
        if output_dtype == 'cf32':
            self.kernel = FLOAT_CMULT_KERNEL
        else:
            self.kernel = INT_CMULT_KERNEL
        self.on_data_cnt = 0

    def define_valid_input_spaces(self):
        """Return set of valid spaces (or 'any') for each input"""
        return ('cuda',)
    
    def on_sequence(self, iseq):
        ihdr = iseq.header
        itensor = ihdr['_tensor']
        ohdr = deepcopy(ihdr)
        
        # Send ihdr to weights generator class
        self.weights_callback.init(ihdr)
        ohdr['_tensor']['dtype'] = self.output_dtype

        # reset the on_data_count
        print("on_data_cnt was",self.on_data_cnt,", resetting to zero")
        self.on_data_cnt = 0
        print("on_data_cnt is ",self.on_data_cnt)
        
        return ohdr

    def on_data(self, ispan, ospan):
        idata = ispan.data
        odata = ospan.data
        
        # Update weights
        if self.on_data_cnt % self.update_frequency == 0:
            self.weights_callback.generate_weights(datacount=self.on_data_cnt)
        w = self.weights_callback.weights_gpu
        
        bf.map(self.kernel, {'a': idata, 'b': odata, 'w': w})
        
        self.on_data_cnt += 1

class WeightsGenerator(object):
    """ Weights Generator object

    Contains the knowledge about antenna and source positions, and fixed delays, necessary to calculate weights at a given time

    Args:
        cassetteposition (dict): xyz positions, indexed by antenna ID (e.g., M100C1)
        snapmap (array):         array of (SNAP_ID, ADC_INPUT, CASSETTE_NAME)
        cabledelays (array):     array of cable delays [MAX_SNAP_ID, 12] in seconds 
        applieddelays (array):   array of delays already applied at the SNAP [MAX_SNAP_ID, 12] in seconds
        recorded_snaps (list):   List of the recorded SNAP IDs (NSNAPs)
        flaginputs (list):       List of the inputs to flag
        bandpass (array):        array of the gain scales to apply [channel, antenna]
        verbose (bool):          Perform verbose printing
        declination (float):     Declination to phase to in degrees. If set (not NaN) then overrides the source coordinates and fixes to this declination.
        nfanbeams (bool);        Number of fanbeams (negative to skip calculation of fanbeamweights
        fanbeamspacing (float):  Fan beam spacing in arcmin
    """    
    def __init__(self, cassettepositions, snapmap, cabledelays, applieddelays, recorded_snaps, flaginputs, bandpass, verbose, declination=np.nan, nfanbeam=-1, fanbeamspacing=-1):
        self.cassettepositions = cassettepositions
        self.snapmap = snapmap
        self.cabledelays = cabledelays
        self.applieddelays = applieddelays
        self.recorded_snaps = recorded_snaps
        self.lat = '-35:22:15'
        self.lon = '149:25:26'
        self.elev = 750
        self.latangle = Angle(self.lat, unit=u.deg)
        self.flaginputs = flaginputs
        self.bandpass = bandpass
        self.verbose = verbose
        self.declination = declination
        self.tracking = True
        if not np.isnan(declination):
            self.tracking = False
        self.nfanbeam = nfanbeam
        self.fanbeamspacing = fanbeamspacing

    def init(self, ihdr):
        """ Initialize by reading sequence header """
        self.itensor = deepcopy(ihdr["_tensor"])
        self.shape   = self.itensor["shape"]
        self.shape[0] = abs(self.shape[0])
        if self.verbose:
            print("Shape:",self.shape)
            print("Itensor keys:",self.itensor.keys())
        unixt0, unixtd = self.itensor['scales'][self.itensor['labels'].index('time')]
        self.t0 = Time(unixt0, format='unix')
        self.td = TimeDelta(unixtd, format='sec')
        if self.verbose:
            print("Itensor labels:",self.itensor['labels'])

        ''' This is all from when I thought the data was arranged differently...
            Keeping it for posterity now but it should be deleted eventually.

        sb_index = self.itensor['labels'].index('subband')
        f_index  = self.itensor['labels'].index('freq')
        if not self.itensor['units'][f_index] == self.itensor['units'][sb_index]:
            raise RuntimeError("Mis-matched frequency units- {0}, {1}".format(self.itensor['units'][f_index], self.itensor['units'][sb_index]))
        self.f0, self.fd = self.itensor['scales'][f_index]
        self.nf = self.itensor['labels'].index('freq').shape
        self.sb0, self.sbd = self.itensor['scales'][sb_index]
        self.nsb = self.itensor['labels'].index('subband').shape
        self.source_name = ihdr['SOURCE']

        # Create the frequency array
        # 'shape': [-1, 512, 8, 2, 8, 32, 6, 2] # third (subband) and 6th (channel) are the important ones
        finefreq = self.f0 + np.arange(self.nf) * self.fd
        sbfreq   = self.sb0 + np.arange(self.nsb) * self.sbd
        self.f = u.Quantity(np.outer(sbfreq, finefreq), unit=self.itensor['units'][f_index]) 

        # Expand this frequency array into the dimensions that are easiest to expand later when multiplying by delays
        ffillshape = deepcopy(self.shape)
        ffillshape[self.itensor['labels'].index('heap')] = 1
        ffillshape[self.itensor['labels'].index('frame')] = 1
        ffillshape[self.itensor['labels'].index('pol')] = 1
        ffillshape[self.itensor['labels'].index('snap')] = 1
        ffillshape[self.itensor['labels'].index('station')] = 1
        self.f.reshape(ffillshape)
        self.w_ang = 2 * np.pi * self.f.to('Hz').value
        #self.f = np.repeat(self.f, self.shape[self.itensor['labels'].index('snap')], self.itensor['labels'].index('snap'))
        #fillshape[self.itensor['labels'].index('snap')] = self.shape[self.itensor['labels'].index('snap')]
        #self.f = np.repeat(self.f, self.shape[self.itensor['labels'].index('station')], self.itensor['labels'].index('station'))
        #fillshape[self.itensor['labels'].index('station')] = self.shape[self.itensor['labels'].index('station')]

        dfillshape = deepcopy(self.shape)
        dfillshape[self.itensor['labels'].index('heap')] = 1
        dfillshape[self.itensor['labels'].index('frame')] = 1
        dfillshape[self.itensor['labels'].index('pol')] = 1
        dfillshape[self.itensor['labels'].index('subband')] = 1
        dfillshape[self.itensor['labels'].index('freq')] = 1
        self.d = np.zeros(shape=dfillshape)
        '''

        # Set up the source name
        self.source_name = ihdr['SOURCE']

        # Create the frequency array
        # 'shape':  [1, 256, 4096, 24]
        # 'labels': ['time', 'freq', 'fine_time', 'station']
        # the second (freq) is the important one
        f_index  = self.itensor['labels'].index('freq')
        self.f0, self.fd = self.itensor['scales'][f_index]
        self.nf = self.shape[self.itensor['labels'].index('freq')]
        self.f = u.Quantity(self.f0 + np.arange(self.nf) * self.fd, unit=self.itensor['units'][f_index]) 
        #self.f = u.Quantity(self.f0 + 2*self.fd + np.arange(self.nf) * self.fd, unit=self.itensor['units'][f_index]) #WAR: Monkey patch frequency axis
        if self.verbose:
            print("Frequencies:",self.f)
        #self.w_ang = 2 * np.pi * self.f.to('Hz').value

        # Expand this frequency array into the dimensions that are easiest to expand later when multiplying by delays
        ffillshape = deepcopy(self.shape)
        ffillshape[self.itensor['labels'].index('station')] = 1
        ffillshape[self.itensor['labels'].index('fine_time')] = 1
        if self.verbose:
            print("ffillshape:",ffillshape)
        self.w_ang = 2 * np.pi * self.f.reshape(ffillshape).to('Hz').value

        # Create an array to hold the delays, but with unity length on the other dimensions
        dfillshape = deepcopy(self.shape)
        dfillshape[self.itensor['labels'].index('freq')] = 1
        dfillshape[self.itensor['labels'].index('fine_time')] = 1
        self.d = np.zeros(shape=dfillshape)
        if self.verbose:
            print("dfillshape:",dfillshape)

        # Generate a flag array to apply to the weights
        self.flag = np.ones(shape=dfillshape)
        for i in range(len(self.recorded_snaps)):
            for j in range(12):
               ant_idx = i*12 + j
               if ant_idx in self.flaginputs:
                   self.flag.flat[ant_idx] = 0.0

        # Generate a bandpass-correcting array
        bpfillshape = deepcopy(self.shape)
        bpfillshape[self.itensor['labels'].index('fine_time')] = 1
        self.bp = np.power(np.reshape(self.bandpass, bpfillshape), -1)*self.flag
        if self.verbose:
            print("bpfillshape:",bpfillshape)

        # Create a array which can be used to broadcast along the fine_time axis
        ftfillshape = deepcopy(self.shape)
        ftfillshape[self.itensor['labels'].index('freq')] = 1
        ftfillshape[self.itensor['labels'].index('station')] = 1
        self.ftfill = np.ones(shape=ftfillshape)

        # Create the antenna list, based on the SNAP id and the SNAP map
        snapinputs = [""] * 12 * len(self.recorded_snaps)
        for row in self.snapmap:
            if row[0] in self.recorded_snaps:
                snapinputs[12*self.recorded_snaps.index(row[0]) + row[1]] = row[2]
        self.antennas = []
        for a in snapinputs:
            self.antennas.append(self.cassettepositions[a[:-1]])
        self.um = make_antenna_array(self.lat, self.lon, self.elev, self.t0.to_datetime(), self.antennas)

        # If we are tracking, then set up the source
        if self.tracking:
            if self.verbose:
                print("Source DB information:",source_db[self.source_name])
            self.sky_model = make_sky_model(source_db[self.source_name])
            self.sky_model.compute_ephemeris(self.um)

            # Check if there is more than one source associated with this pointing
            if len(self.sky_model.sources) > 1:
                print("WARNING - Multiple sources at this position, will phase to the first one!")

#    def generate_fanbeam_weights(self, nchan, nbeam, npol, nant, beamspacing):
#        """ Generate the offset weights to point a bunch of fan beams
#        Args:
#            nchan (int): Number of channels
#            nbeam (int): Number of fan beam
#            npol  (int): Number of polarisations
#            nant  (int): Number of antennas
#            beamspacing (float): Spacing of the beams in arcmin
#
#        Returns:
#            fanbeamweights (array): Array of complex, 8 bit weights
#        """
    def generate_fanbeam_weights(self):
        """ Generate the offset weights to point a bunch of fan beams
        Returns:
            fanbeamweights (array): Array of complex, 8 bit weights
        """

        # Check the nchan, npol, nant against existing setup, and that a declination has been specified
        try:
#            assert nchan == self.shape[self.itensor['labels'].index('freq')]
#            assert nant*npol == self.shape[self.itensor['labels'].index('station')]
            assert not self.tracking
        except AssertionError:
#            print("Asked to create a fanbeam weight array with nchan",nchan,",npol",npol,",nant",nant)
#            print("But self.shape says nchan is", self.shape[self.itensor['labels'].index('freq')])
#            print("and self.shape says npol*nant is",self.shape[self.itensor['labels'].index('station')])
            print("Tracking must be false, but it is",self.tracking)

        nchan = self.shape[self.itensor['labels'].index('freq')]
        npol = 2
        nant = self.shape[self.itensor['labels'].index('station')] // npol
        nbeam = self.nfanbeam
        beamspacing = self.fanbeamspacing

        # Create the weights array, a delay array to fill it with, and an array containing 2*pi*f
        #fanbeamweights = np.zeros((nchan, nbeam, npol, nant), dtype=[('re', 'int8'), ('im', 'int8')])
        d = np.zeros((1, nbeam, npol, nant), dtype=np.float64)
        w = 2 * np.pi * self.f.reshape((nchan, 1, 1, 1)).to('Hz').value

        # Derive the offset weights
        el = np.pi * (self.declination - self.latangle.degree) / 180.0
        for k in range(nbeam):
            for i in range(len(self.recorded_snaps)):
                for j in range(12):
                    p = j % npol
                    a = i*6 + (j // npol)
                    ant_idx = i*12 + j
                    centre_tau = -self.um.xyz[ant_idx, 0] * np.sin(el) / LIGHT_SPEED # delay for this antenna for centre beam
                    #antbeam_idx = ant_idx*nbeam + k
                    antbeam_idx = k*len(self.recorded_snaps)*12 + ant_idx
                    deltael = (k-nbeam//2) * beamspacing * np.pi / (180.0*60.0)
                    tau = -self.um.xyz[ant_idx, 0] * np.sin(el + deltael) / LIGHT_SPEED # Delay for this antenna for this beam
                    #d.flat[antbeam_idx] = tau - centre_tau # Differential delay for this beam of this antenna, relative to the centre beam
                    d[0][k][p][a] =  tau - centre_tau # Differential delay for this beam of this antenna, relative to the centre beam
                    #d[0][k][p][a] = 0 # Test with no beamformer offsets

        self.fanbeamweights_cpu = bf.ndarray(128*(np.exp(-1j * w * d)), dtype=[('re', 'int8'), ('im', 'int8')])
        print("Fan beam weights shape:", self.fanbeamweights_cpu.shape)
        self.fanbeamweights_gpu = self.fanbeamweights_cpu.copy('cuda')

        # Return the fanbeamweights (on the GPU) array
        return self.fanbeamweights_gpu

    def generate_weights(self, datacount):
        """ Main function call that is required for weight generation """
        if self.verbose:
            print("Generating weights, with datacount",datacount)
            print("Time is", (self.t0 + self.td*datacount).mjd)

        # Update the model for the current time and update the source ephemeris
        self.um = make_antenna_array(self.lat, self.lon, self.elev, (self.t0 + self.td*datacount).to_datetime(), self.antennas) 
        if self.tracking:
            self.sky_model.compute_ephemeris(self.um)
            if self.verbose:
                self.sky_model.report()
                print("Elevation is", (np.pi/2 - self.sky_model.sources[0].alt)*180./np.pi, "azimuth is", self.sky_model.sources[0].az*180/np.pi)
            az = self.sky_model.sources[0].az
            el = np.pi/2 - self.sky_model.sources[0].alt
        else:
            if self.verbose:
                print("Fixed declination of", self.declination, ", elevation is", self.declination - self.latangle.degree)
            el = np.pi * (self.declination - self.latangle.degree) / 180.0
            az = 0

        # Loop through, calculating delays, and multiplying/broadcasting to fill the weights array
        for i in range(len(self.recorded_snaps)):
            for j in range(12):
               ant_idx = i*12 + j
               t_g = -self.um.xyz[ant_idx, 0] * np.cos(az) * np.sin(el) / LIGHT_SPEED
               self.d.flat[ant_idx] =  t_g - self.applieddelays[self.recorded_snaps[i], j] + self.cabledelays[self.recorded_snaps[i], j]
               #print(i, j, self.um.xyz[ant_idx, 0], self.sky_model.sources[0].alt, t_g, self.d.flat[ant_idx], self.cabledelays[self.recorded_snaps[i], j])

        self.weights_cpu = bf.ndarray((self.bp * np.exp(-1j * self.w_ang * self.d)) * self.ftfill, dtype='cf32')
        self.weights_gpu = self.weights_cpu.copy('cuda')

        ## If required, also generate fanbeam weights
        #if self.nfanbeam > 0:
        #    fanbeamweights = self.generate_fanbeam_weights()
        #    fanbeamweightsfile = "fanbeamweights.hkl"
        #    print("Dumping fanbeamweights to fanbeamweights.hkl")
        #    hkl.dump(fanbeamweights, fanbeamweightsfile)

        # Return the GPU weights
        return self.weights_gpu


def apply_weights(iring, weights_callback, output_dtype='cf32', update_frequency=100, 
                  *args, **kwargs):
    """ Apply complex gain calibrations to antennas
    Args:
        iring (Ring or Block): Input data source.
        weights_callback: WeightsGenerator class that generates weights.  
                          Should return a bf.ndarray in CUDA space with the same shape as ispan.data
        *args: Arguments to ``bifrost.pipeline.TransformBlock``.
        **kwargs: Keyword Arguments to ``bifrost.pipeline.TransformBlock``.
   Returns:
        ApplyWeightsBlock: A new block instance.
    """
    return ApplyWeightsBlock(iring, weights_callback, output_dtype, update_frequency, *args, **kwargs)


