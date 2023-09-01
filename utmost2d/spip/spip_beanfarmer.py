#!/home/npsr/miniconda2/envs/bifrost/bin/python
"""
# spip_beanfarmer.py

fan beam filterbank generator
"""
import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import blocks as byip
import numpy as np
#from blocks import correlate_dp4a, h5write
from blocks import h5write, apply_weights, WeightsGenerator
#from blocks import adam_beanfarmer
from blocks.dada_header_spip2 import dada_dict_to_bf_dict
import hickle as hkl
import os
import h5py
from um2d_vis.um2dconfig import get_cassette_dictionary, read_snap_mapping, get_delays, get_bandpass
from bifrost.libbifrost import _bf

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

class ExtractBeamBlock(bf.pipeline.TransformBlock):
    def __init__(self, iring, beam_id, nchan, *args, **kwargs):
        super(ExtractBeamBlock, self).__init__(iring, *args, **kwargs)
        self.beam_id = beam_id
        self.nchan = nchan

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        itensor = iseq.header['_tensor']
        print("EXTRACTBLOCK: input shape:", itensor['shape'])
        ohdr["_tensor"]["shape"] = [-1, 1, self.nchan]
        ohdr["_tensor"]['labels'] = ['time', 'pol', 'freq']
        u = itensor['scales']
        ohdr["_tensor"]['scales'] = [u[0], [0.0, 0.0], u[-1]]
        ohdr["_tensor"]['units']  = ['s', None, 'MHz']
        #print("EXTRACTBLOCK: output shape:", ohdr["_tensor"])
        return ohdr

    def on_data(self, ispan, ospan):
        in_nframe  = ispan.nframe
        out_nframe = in_nframe

        ospan.data[...] = ispan.data[..., self.beam_id, :, : ]
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

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='SNAP fan-beam filterbank beamformer (using beanfarmer).')
    p.add_argument('-m', '--mode', default='lo', type=str, help="mode: 'lo' (1 sec) or 'hi' (1 ms) time resolution.")
    p.add_argument('-b', '--buffer', default=0xBABA, type=lambda x: int(x,0), help="PSRDADA buffer to connect to")
    p.add_argument('-O', '--outbuffer', default=None, type=lambda x: int(x,0), help="PSRDADA buffer to write to.")
    p.add_argument('-B', '--benchmark', action='store_true', default=False, help="Enable benchmark mode. If set, will not write to disk.")
    p.add_argument('-f', '--filename', default=None, type=str, help="If set, will read from file instead of PSRDADA. Can be single file or comma-separated list")
    p.add_argument('-o', '--outdir',  default='/data/npsr/', type=str, help="Path to output data to.")
    p.add_argument('--declination', default=-45, type=float, help="Declination to phase centre beam to, in degrees.")
    p.add_argument('--nchan',       default=256, type=int, help='Number of channels in the file')
    p.add_argument('--nbeam',       default=128, type=int, help='Number of fan beams to produce')
    p.add_argument('--beamspacing', default=6, type=float, help='Beam spacing in arcmin')
    p.add_argument('--configdir',   default='/home/adeller/packages/src/utmost2d_snap/config/', help='Directory in which config files (antenna, snap etc) are kept')
    p.add_argument('--antposfile',  default='2020.08.27.txt', help='antenna position file name')
    p.add_argument('--snapmapfile', default='2020.08.25.txt', help='snap mapping file name')
    p.add_argument('--cabledelayfile', default='2020.11.29.txt', help='cable delay file name')
    p.add_argument('--applieddelayfile', default='appliedsnapdelays.txt', help='Already-applied SNAP delays file name')
    #p.add_argument('--fanbeamweightsfile', default='128beam4module.fanbeamweights.hkl', help='pre-cooked fan beam weights in a hickle file')
    p.add_argument('--bandpassfile',    default='', help='per-antenna frequency dependent scaling')
    p.add_argument('--flaginputs',      default='', help='comma-separated list of input indices to flag')
    p.add_argument('--recordedsnaps',   default='0,2', help='Comma-separated list of SNAP ids')
    p.add_argument('--filterbank',      default=False, action='store_true', help='Write out filterbank rather than voltages in h5 format')
    p.add_argument('--verbose',      default=False, action='store_true', help='Print out lots of stuff')
    args = p.parse_args()

    htr_mode = True if args.mode == 'hi' else False

    outdir  = args.outdir
    file_prefix = 'spip_beamformer'
    hdr_callback    = dada_dict_to_bf_dict
    n_int_per_file  = 128
    n_tavg  = 32 # Number of times to average

    # Check if there is a cable_delays.txt file in this directory, warn if so
    if os.path.exists("cable_delays.txt") and not args.cabledelayfile == "cable_delays.txt":
        print("WARNING! There is a cable_delays.txt file in this directory, but you are not using it! Do you intend to use {0}/{1} as the calibration file?".format(args.configdir, args.cabledelayfile))

    # Load up antenna details
    antposfile = args.configdir + '/antenna_positions/' + args.antposfile
    cassettepositions = get_cassette_dictionary(antposfile)

    # Load up the SNAP mapping
    snapmapfile = args.configdir + '/snap_mapping/' + args.snapmapfile
    snapmap = read_snap_mapping(snapmapfile)

    # Load up the cable delays
    cabledelayfile = args.configdir + '/cable_delays/' + args.cabledelayfile
    cabledelays = get_delays(cabledelayfile) / 1e9 # convert to seconds

    # Load up the already-applied SNAP delays (this should really be loaded through a safer / less error-prone mechanism, via REDIS?)
    applieddelays = get_delays(args.applieddelayfile) / 1e9 # convert to seconds

    # Load up the frequency-dependent per-antenna scales
    if args.bandpassfile == '' or not os.path.exists(args.bandpassfile):
        p.error('Please supply a valid bandpassfile: I got ' + args.bandpassfile)
    bandpass = get_bandpass(args.bandpassfile)
    if args.verbose:
        print ("Shape of the bandpass file is (should be nchan,nant):",bandpass.shape)

    # Load up which SNAPs are being recorded (this should really be done via something which is saved to REDIS when they are configured)
    recorded_snaps = [int(r) for r in args.recordedsnaps.split(',')]
    nant = 6*len(recorded_snaps)

    # Get a list of inputs to flag
    if args.flaginputs.strip() == "":
        flaginputs = []
    else:
        flaginputs =  map(int, args.flaginputs.rstrip(',').split(','))

    # apply weights function
    wg = WeightsGenerator(cassettepositions, snapmap, cabledelays, applieddelays, recorded_snaps, flaginputs, bandpass, args.verbose, args.declination, args.nbeam, args.beamspacing)
    weight_update_frequency = 100000 # No need to update when pointing at a fixed declination

    # First DADA buffer
    if args.filename is None:
        b_dada = bf.blocks.psrdada.read_psrdada_buffer(args.buffer, hdr_callback, 1, single=True, core=0)
    else:
        b_dada = bf.blocks.read_dada_file(args.filename.split(','), hdr_callback, gulp_nframe=1, core=0)
    PrintStuffBlock(b_dada)

    ## Generate some fanbeamweights
    #npol = 2
    #fanbeamweightsfile = "fanbeamweights.hkl"
    #fanbeamweights = wg.generate_fanbeam_weights(args.nchan, args.nbeam, npol, nant, args.beamspacing)
    #hkl.dump(fanbeamweights, fanbeamweightsfile)
    # Load up the fan beam weights, to figure out for many beams there are, and check the dimensions
    #fanbeamweights = hkl.load(args.fanbeamweightsfile)
    #nfanbeam = fanbeamweights.shape[1]
    #try:
    #    assert fanbeamweights.shape == (args.nchan, nfanbeam, 2, nant)
    #    assert fanbeamweights.dtype.names[0] == 're'
    #    assert fanbeamweights.dtype.names[1] == 'im'
    #    assert str(fanbeamweights.dtype[0]) == 'int8'
    #except AssertionError:
    #    print('ERR: beam weight shape/dtype is incorrect')
    #    print('ERR: beam weights shape is: %s' % str(fanbeamweights.shape))
    #    print('ERR: shape should be %s' % str((args.nchan, nfanbeam, 2, nant, 2)))
    #    print('ERR: dtype should be int8, dtype: %s' % fanbeamweights.dtype.str)
    
    # GPU processing
    b_gpu = bf.blocks.copy(b_dada, space='cuda', core=1, gpu=0)
    with bf.block_scope(fuse=False, gpu=0):
        b_gpu = bf.views.merge_axes(b_gpu, 'station', 'pol')
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'subband', 'freq', 'heap', 'frame', 'snap', 'station'] )
        b_gpu = bf.views.merge_axes(b_gpu, 'subband', 'freq', label='freq')
        b_gpu = bf.views.merge_axes(b_gpu, 'snap', 'station', label='station')
        b_gpu = bf.views.merge_axes(b_gpu, 'heap', 'frame', label='fine_time')
        #b_gpu = bf.views.add_axis(b_gpu, axis=3, label='pol')
        # Beanfarmer requires (time, freq, fine_time, pol, station)
        b_gpu = apply_weights(b_gpu, weights_callback=wg, output_dtype='ci8', 
                              update_frequency=weight_update_frequency)
        # The following only works if the input data are arranged X, Y, X, Y (as they are)
        # Remove to go back to the old approach that didn't handle polarisation
        b_gpu = bf.views.split_axis(b_gpu, 'station', 2, label='pol')
        PrintStuffBlock(b_gpu)
        b_gpu = bf.blocks.beanfarmer(b_gpu, weights_callback=wg, n_avg=n_tavg, n_beam=args.nbeam, n_pol=2, 
                                     n_chan=args.nchan, n_ant=nant)
        #b_gpu = adam_beanfarmer(b_gpu, n_avg=n_avg, n_beam=args.nbeam, n_pol=2,
        #                        n_chan=args.nchan, n_ant=6*len(recorded_snaps), 
        #                        weights=wg.getFanbeamWeights(args.nbeam, args.beamspacing))
        if args.filterbank:
            # Transpose to time, beam, freq for writing out later
            b_gpu = bf.blocks.transpose(b_gpu, ['time', 'fine_time', 'beam', 'freq'])
            b_gpu = bf.views.merge_axes(b_gpu, 'time', 'fine_time')
            b_gpu = bf.views.add_axis(b_gpu, axis=2, label='pol')
        else:
            # Transpose to time, beam, freq, fine_time order for subsequent searching
            b_gpu = bf.blocks.transpose(b_gpu, ['time', 'beam', 'freq', 'fine_time'])
        PrintStuffBlock(b_gpu)

    # Back to CPU and to disk
    b_cpu = bf.blocks.copy(b_gpu, space='cuda_host', core=2)
    if args.outbuffer is not None:
        b_dada = bf.blocks.psrdada.write_psrdada_buffer(b_cpu, args.outbuffer, gulp_nframe=1)
    elif not args.benchmark:
        if args.filterbank:
            b_cpus = {}
            for beam_id in range(args.nbeam):
                b_cpus[beam_id] = ExtractBeamBlock(b_cpu, beam_id=beam_id, nchan=args.nchan)
                b_cpus[beam_id] = RenameBlock(b_cpus[beam_id], 'bf0_fb%i' % beam_id)
            for beam_id, blk in b_cpus.items():
                bf.blocks.write_sigproc(blk, path=outdir)
        else:
            # Write out in h5 format
            h5write(b_cpu, outdir=outdir, prefix=file_prefix, n_int_per_file=n_int_per_file, core=3)
    

    print "Running pipeline"
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()
