#!/home/npsr/miniconda2/envs/bifrost/bin/python
"""
# psrdada_2x.py

Start bifrost pipeline for SNAP spectrometer (2x UDP capture)
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
from blocks.dada_header_spip2 import dada_dict_to_bf_dict
import hickle as hkl
import os
import h5py
from um2d_vis.um2dconfig import get_cassette_dictionary, read_snap_mapping, get_delays, get_bandpass, get_flagged_inputs, get_recorded_snaps
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
        #pprint(iseq.header)
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % self.n_gulp_per_print == 0 and self.print_on_data:
            d = ispan.data
            #d = np.array(d).astype('float32')
            print("[%s] %s %s" % (now, str(ispan.data.shape), str(ispan.data.dtype)))
        self.n_iter += 1

class ExtractAntennaBlock2(bf.pipeline.TransformBlock):
    def __init__(self, iring, ant_id, nchan, *args, **kwargs):
        super(ExtractAntennaBlock2, self).__init__(iring, *args, **kwargs)
        self.ant_id = ant_id
        self.nchan = nchan

    def on_sequence(self, iseq):
        ohdr = deepcopy(iseq.header)
        itensor = iseq.header['_tensor']
        print("EXTRACTBLOCK: input shape:", itensor['shape'])
        ohdr["_tensor"]["shape"] = [-1, 4, self.nchan]
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


if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='SNAP tied array beamformer. Default output is voltages in h5 format.')
    p.add_argument('-m', '--mode', default='lo', type=str, help="mode: 'lo' (1 sec) or 'hi' (1 ms) time resolution.")
    p.add_argument('-b', '--buffer', default=0xBABA, type=lambda x: int(x,0), help="PSRDADA buffer to connect to")
    p.add_argument('-O', '--outbuffer', default=None, type=lambda x: int(x,0), help="PSRDADA buffer to write to.")
    p.add_argument('-B', '--benchmark', action='store_true', default=False, help="Enable benchmark mode. If set, will not write to disk.")
    p.add_argument('-f', '--filename', default=None, type=str, help="If set, will read from file instead of PSRDADA. Can be single file or comma-separated list")
    p.add_argument('-o', '--outdir', default='/data/dprice', type=str, help="Path to output data to.")
    p.add_argument('--nchan',       default=256, type=int, help='Number of channels in the file')
    p.add_argument('--configdir',   default='/home/adeller/packages/src/utmost2d_snap/config/', help='Directory in which config files (antenna, snap etc) are kept')
    p.add_argument('--antposfile',         default='active.txt', help='antenna position file name')
    p.add_argument('--snapmapfile',        default='active.txt', help='snap mapping file name')
    p.add_argument('--cabledelayfile',     default='active.txt', help='cable delay file name')
    p.add_argument('--flaggedinputsfile',  default='active.txt', help='flagged inputs file name')
    p.add_argument('--flaginputs',         default='', help='comma-separated list of input indices to flag')
    p.add_argument('--recordedsnapsfile',  default='active.txt', help='SNAP ids inputs file')
    p.add_argument('--recordedsnaps',      default='0,2', help='Comma-separated list of SNAP ids')
    p.add_argument('--applieddelayfile', default='appliedsnapdelays.txt', help='Already-applied SNAP delays file name')
    p.add_argument('--bandpassfile',     default='', help='per-antenna frequency dependent scaling')
    p.add_argument('--filterbank',       default=False, action='store_true', help='Write out filterbank rather than voltages in h5 format')
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
    if args.recordedsnaps != "":
        recorded_snaps = [int(r) for r in args.recordedsnaps.split(',')]
    else:
        recordedsnapsfile = args.configdir + '/recorded_snaps/' + args.recordedsnapsfile
        recorded_snaps = get_recorded_snaps(recordedsnapsfile)

    # Get a list of inputs to flag
    flagged_inputs = []
    if not args.flaginputs.strip() == "":
        flagged_inputs = map(int, args.flaginputs.rstrip(',').split(','))
    else:
        if args.flaggedinputsfile != "":
            flaggedinputsfile = args.configdir + '/flagged_inputs/' + args.flaggedinputsfile
            flagged_inputs = get_flagged_inputs(flaggedinputsfile)

    # apply weights function
    wg = WeightsGenerator(cassettepositions, snapmap, cabledelays, applieddelays, recorded_snaps, flagged_inputs, bandpass, args.verbose)
    if args.verbose:
        print("WeightsGenerator has been created")
    weight_update_frequency = 100

    # First DADA buffer
    if args.filename is None:
        b_dada = bf.blocks.psrdada.read_psrdada_buffer(args.buffer, hdr_callback, 1, single=True, core=0)
    else:
        b_dada = bf.blocks.read_dada_file(args.filename.split(','), hdr_callback, gulp_nframe=1, core=0)
    PrintStuffBlock(b_dada)
    
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
        b_gpu = apply_weights(b_gpu, weights_callback=wg, output_dtype='cf32', 
                              update_frequency=weight_update_frequency)
        # The following only works if the input data are arranged X, Y, X, Y (as they are)
        # Remove to go back to the old approach that didn't handle polarisation
        b_gpu = bf.views.split_axis(b_gpu, 'station', 2, label='pol')
        b_gpu = bf.blocks.reduce(b_gpu, axis='station')

        if args.filterbank:
            #b_gpu = byip.detect(b_gpu, mode='stokes', axis='pol')
            b_gpu = byip.detect(b_gpu, mode='coherence', axis='pol')
            ### THIS IS THE OLD APPROACH that doesn't handle polarisation
            ##b_gpu = byip.detect(b_gpu, 'stokes_I')
            ##b_gpu = bf.views.add_axis(b_gpu, axis=3, label='pol')
            # Beanfarmer requires (time, freq, fine_time, pol, station) 
            b_gpu = bf.blocks.reduce(b_gpu, 'fine_time', n_tavg)
        else:
            # transpose to time, station, freq, pol for DSPSR to process
            b_gpu = bf.blocks.transpose(b_gpu, ['time', 'fine_time', 'station', 'freq', 'pol'])
            b_gpu = bf.views.merge_axes(b_gpu, 'time', 'fine_time')

    # Back to CPU and to disk
    b_cpu = bf.blocks.copy(b_gpu, space='cuda_host', core=2)
    PrintStuffBlock(b_cpu)
    if args.outbuffer is not None:
        b_dada = bf.blocks.psrdada.write_psrdada_buffer(b_cpu, args.outbuffer, gulp_nframe=1)
        b_dada.add_header_keywords( {"INSTRUMENT": "dspsr", "TELESCOPE": "MOST"} )
    elif not args.benchmark:
        if args.filterbank: # Convert to filterbank
            b_cpu = bf.blocks.transpose(b_cpu, ['time', 'fine_time', 'station', 'pol', 'freq'])
            b_cpu0 = bf.views.merge_axes(b_cpu, 'time', 'fine_time')
            b_cpu0 = ExtractAntennaBlock2(b_cpu0, ant_id=0, nchan=args.nchan)
            bf.blocks.write_sigproc(b_cpu0, path=outdir)
        else: # Just write out voltages in h5 format
            h5write(b_cpu, outdir=outdir, prefix=file_prefix, n_int_per_file=n_int_per_file, core=3)
    else:
        print("Enabling benchmark mode...")

    print("Running pipeline")
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()
