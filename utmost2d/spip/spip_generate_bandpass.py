#!/home/npsr/miniconda2/envs/bifrost/bin/python
"""
# psrdada_2x.py

Start bifrost pipeline for SNAP spectrometer (2x UDP capture)
"""

import glob
import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import blocks as byip
from blocks.dada_header_spip2 import dada_dict_to_bf_dict
from astropy import units as u
import numpy as np
import os

class PrintStuffBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, n_gulp_per_print=128, print_on_data=True, *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
        self.n_gulp_per_print = n_gulp_per_print
        self.print_on_data = print_on_data

    def on_sequence(self, iseq):
        print("[%s]" % datetime.now())
        print(iseq.name)
        pprint(iseq.header)
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % self.n_gulp_per_print == 0 and self.print_on_data:
            d = ispan.data
            #d = np.array(d).astype('float32')
            print("[%s] %s %s" % (now, str(ispan.data.shape), str(ispan.data.dtype)))
        self.n_iter += 1

class GetFreqBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, *args, **kwargs):
        super(GetFreqBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0

    def on_sequence(self, iseq):
        self.itensor = iseq.header['_tensor']
        f_index  = self.itensor['labels'].index('freq')
        self.f0, self.fd = self.itensor['scales'][f_index]
        self.nf = self.itensor['shape'][self.itensor['labels'].index('freq')]
        self.f = u.Quantity(self.f0 + np.arange(self.nf) * self.fd, unit=self.itensor['units'][f_index])

    def on_data(self, ispan):
        self.n_iter += 1

class SliceBlock(bf.pipeline.TransformBlock):
    """ Extract a slice along a given axis, selecting a given entry """
    def __init__(self, iring, axis, axis_idx, *args, **kwargs):
        """ Slice along axis, selecting by index 
        
        This is essentially equivalent to numpy.take() 

        Args:
            axis (str or int): Axis identifier, e.g. 'freq'
            axis_idx (ing): Integer ID to slice over
        
        Example: Slice along 'antenna' axis, selecting antenna 8.
             [-1 time, 320 channels, 12 antennas, 2 pol] --->
             [-1 time, 320 channels, 2 pol]  (antenna axis now gone)
        """
        super(SliceBlock, self).__init__(iring, *args, **kwargs)
        self.axis = axis
        self.axis_idx = axis_idx

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        itensor = iseq.header['_tensor']
        if isinstance(self.axis, str):
            self.axis = ohdr["_tensor"]["labels"].index(self.axis)
        
        # Remove axis from tensor
        for tkey in ['shape', 'labels', 'units', 'scales']:
            ohdr['_tensor'][tkey].pop(self.axis)

        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        idata = ispan.data
        odata = ospan.data

        odata[...] = idata.take(self.axis_idx, self.axis)
        return out_nframe

class ExtractAntennaBlock2(bf.pipeline.TransformBlock):
    def __init__(self, iring, ant_id, nchan, *args, **kwargs):
        super(ExtractAntennaBlock2, self).__init__(iring, *args, **kwargs)
        self.ant_id = ant_id
        self.nchan = nchan

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        itensor = iseq.header['_tensor']
        ohdr["_tensor"]["shape"] = [-1, 1, self.nchan]
        ohdr["_tensor"]['labels'] = ['time', 'pol', 'freq']
        u = itensor['scales']
        ohdr["_tensor"]['scales'] = [u[0], [0.0, 0.0], u[-1]]
        ohdr["_tensor"]['units']  = ['s', None, 'MHz']
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        ospan.data[...] = ispan.data[..., self.ant_id, :, :]
        return out_nframe

class RenameBlock(bf.pipeline.TransformBlock):
    def __init__(self, iring, basename, *args, **kwargs):
        super(RenameBlock, self).__init__(iring, *args, **kwargs)
        self.basename = basename

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        name = ohdr['name']
        name = self.basename + '_' + name.split('_')[1]
        ohdr['name'] = name
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        ospan.data[...] = ispan.data[...]
        return out_nframe



class StitchBlock(bf.pipeline.MultiTransformBlock):
    """ Concatenate two rings along axis """
    def __init__(self, irings, axis=-1, *args, **kwargs):
        super(StitchBlock, self).__init__(irings, *args, **kwargs)
        self.axis = axis

    def on_sequence(self, iseqs):
        ohdr = deepcopy(iseqs[0].header)
        if isinstance(self.axis, str):
            self.axis = ohdr["_tensor"]["labels"].index(self.axis)
        ohdr["_tensor"]["shape"][self.axis] *= 2
        return [ohdr,]

    def on_data(self, ispans, ospans):
        out_nframe = ispans[0].nframe
        d0 = ispans[0].data
        d1 = ispans[1].data
        d  = np.concatenate((d0, d1), axis=self.axis)

        odata = ospans[0].data
        odata[...] = d
        return [out_nframe,]

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='SNAP filterbanking beamformer (beanfarmer).')
    p.add_argument('-b', '--buffer', default=0xBABA, type=lambda x: int(x,0), help="PSRDADA buffer to connect to")
    p.add_argument('-f', '--filename', default=None, type=str, help="If set, will read from file instead of PSRDADA.")
    p.add_argument('-d', '--outdir', default='/data/npsr', type=str, help='Output directory')
    p.add_argument('-n', '--ntavg', default=256, type=int, help='Number of frames to average over (time resolution)')
    p.add_argument('-a', '--numant', default=72, type=int, help='Number of antennas in the dataset')
    p.add_argument('-c', '--numchan', default=64, type=int, help='Number of channels in the dataset')
    args = p.parse_args()


    hdr_callback    = dada_dict_to_bf_dict
    n_tavg  = args.ntavg
    ant_ids = range(args.numant)
    outdir  = args.outdir
    os.chdir(outdir)

    # First DADA buffer
    if args.filename is None:
        b_dada = bf.blocks.psrdada.read_psrdada_buffer(args.buffer, hdr_callback, 1, single=True, core=0)
    else:
        b_dada = bf.blocks.read_dada_file([args.filename], hdr_callback, gulp_nframe=1, core=0)
    PrintStuffBlock(b_dada)
    
    # GPU processing
    b_gpu = bf.blocks.copy(b_dada, space='cuda', core=1, gpu=0)
    with bf.block_scope(fuse=False, gpu=0):
        b_gpu = bf.views.merge_axes(b_gpu, 'station', 'pol')
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'subband', 'freq', 'heap', 'frame', 'snap', 'station'] )
        b_gpu = bf.views.merge_axes(b_gpu, 'subband', 'freq', label='freq')
        b_gpu = bf.views.merge_axes(b_gpu, 'snap', 'station', label='antenna')
        b_gpu = bf.views.merge_axes(b_gpu, 'heap', 'frame', label='fine_time')
        b_gpu = byip.detect(b_gpu, 'stokes_I')
        # Beanfarmer requires (time, freq, fine_time, pol, station) 
        b_gpu = bf.blocks.reduce(b_gpu, 'fine_time', n_tavg, op='mean')
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'fine_time', 'antenna', 'freq'])
        b_gpu = bf.views.merge_axes(b_gpu, 'time', 'fine_time')
        b_gpu = bf.blocks.reduce(b_gpu, 'time', 16, op='mean')
        PrintStuffBlock(b_gpu)
        fb = GetFreqBlock(b_gpu)

    b_cpu = bf.blocks.copy(b_gpu, space='cuda_host')
    bf.blocks.binary_write(b_cpu, file_ext='amplitudes.out')

    print "Running pipeline"
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()
    ampfile = sorted(glob.glob("*amplitudes.out"))[-1]
    amps = np.fromfile(ampfile, dtype=np.float32)
    ntimes = amps.shape[0] / (args.numchan * args.numant)
    bandpass = np.sqrt( np.mean(amps[:ntimes*args.numant*args.numchan].reshape(ntimes, args.numant, args.numchan), axis=0)).transpose() 
    bandpassplusfreqs = np.hstack((fb.f.value.reshape(args.numchan, 1), bandpass))
    np.savetxt("bandpass.txt",bandpassplusfreqs, delimiter=',')
    np.savetxt("amps.txt",amps)
    print ("Finished", fb.f.value)
