#!/home/npsr/miniconda2/envs/bifrost/bin/python
"""
# spip_beamformer.py

Combined beamformer script that can produce multiple tied beams plus fan beams
"""
import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import blocks as byip
import numpy as np
from blocks import h5write
from blocks.extraction import extract_antenna, extract_beam
from blocks.apply_weights2_temperature import apply_weights2, WeightsGenerator2
from blocks.dada_header_spip2 import dada_dict_to_bf_dict
import hickle as hkl
import os
import h5py
from um2d_vis.um2dconfig import get_cassette_dictionary, read_snap_mapping, get_delays, get_bandpass, get_flagged_inputs, get_recorded_snaps
from bifrost.libbifrost import _bf
from NormaliseBlock_FST import normalise

class PrintStuffBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, n_gulp_per_print=128, print_on_data=True, printlabel='', *args, **kwargs):
        super(PrintStuffBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
        self.n_gulp_per_print = n_gulp_per_print
        self.printlabel = printlabel
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
            print("[%s:%s] %s %s" % (now, self.printlabel, str(ispan.data.shape), str(ispan.data.dtype)))
        self.n_iter += 1

class DumpBlock(bf.pipeline.SinkBlock):
    def __init__(self, iring, dumpprefix='/data/npsr/logged', dobandpass=True, *args, **kwargs):
        super(DumpBlock, self).__init__(iring, *args, **kwargs)
        self.n_iter = 0
        self.dobandpass = dobandpass
        if dobandpass:
            self.dumpfilename = dumpprefix + ".bandpass.npy"
        else:
            self.dumpfilename = dumpprefix + ".timeseries.npy"

    def on_sequence(self, iseq):
        self.n_iter = 0
        print("Dump block header")
        print(iseq.header)
        datashape = iseq.header['_tensor']['shape']
        if self.dobandpass:
            self.loggeddata = np.empty((0,datashape[1],datashape[2]))
            self.currentdump = np.zeros((1,datashape[1],datashape[2]))
        else:
            self.loggeddata = np.empty((datashape[1],0))

    def on_data(self, ispan):
        d = ispan.data
        self.n_iter += 1
        if self.dobandpass:
            self.currentdump += d
            if self.n_iter % 100 == 0:
                self.loggeddata = np.append(self.loggeddata, self.currentdump, axis=0)
                self.currentdump.fill(0.0)
        else:
            self.loggeddata = np.append(self.loggeddata, d[0], axis=1)

    def on_sequence_end(self, iseq):
        with open(self.dumpfilename, 'wb') as f:
            np.save(f, self.loggeddata)
            f.flush()
            os.fsync(f.fileno())
            f.close()

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
    p.add_argument('-b', '--buffer', default=0xBABA, type=lambda x: int(x,0), help="PSRDADA buffer to connect to")
    p.add_argument('-O', '--tboutbuffer', default=None, action='append', type=lambda x: int(x,0), help="PSRDADA buffer to write tied beam voltages out to. Call multiple times for multiple tied beams.")
    p.add_argument('-F', '--fboutbuffer', default=None, type=lambda x: int(x,0), help="PSRDADA buffer to write fan beam filterbank data to.")
    p.add_argument('-B', '--benchmark', action='store_true', default=False, help="Enable benchmark mode. If set, will not write to disk.")
    p.add_argument('-f', '--filename', default=None, type=str, help="If set, will read from file instead of PSRDADA. Can be single file or comma-separated list")
    p.add_argument('-o', '--outdir', default='/data/npsr', type=str, help="Path to output data to.")
    p.add_argument('-c', '--core', default=0, type=int, help="CPU core to bind input thread to, subsequent processes bind to subsequent cores")
    p.add_argument('-n', '--n_count', default=None, help="Mean and variance are calculated every very n_count seconds for normalisation.")
    p.add_argument('--declination', default=-45, type=float, help="Declination to phase centre beam to, in degrees.")
    p.add_argument('--fixed_boresight', default=False, action="store_true", help="Ignore the source name in the header and make boresight correction to args.declination")
    p.add_argument('--nchan',       default=256, type=int, help='Number of channels in the file')
    p.add_argument('--nfanbeam',    default=0, type=int, help='Number of fan beams to produce')
    p.add_argument('--fanbeamspacing', default=5, type=float, help='Fan beam spacing in arcmin')
    p.add_argument('--individualcassettes', default=False, action="store_true", help="Produce individual cassette filterbanks")
    p.add_argument('-C', '--cboutbuffer',  default=None, type=lambda x: int(x,0), help="PSRDADA buffer to write cassette beam filterbank data to.")
    p.add_argument('--configdir',   default='/home/npsr/software/utmost2d_snap/config/', help='Directory in which config files (antenna, snap etc) are kept')
    p.add_argument('--antposfile',  default='2020.08.27.txt', help='antenna position file name')
    p.add_argument('--snapmapfile', default='2020.08.25.txt', help='snap mapping file name')
    p.add_argument('--cabledelayfile', default='2020.11.29.txt', help='cable delay file name')
    p.add_argument('--applieddelayfile', default='appliedsnapdelays.txt', help='Already-applied SNAP delays file name')
    p.add_argument('--bandpassfile',     default='', help='per-antenna frequency dependent scaling')
    p.add_argument('--flaginputs',       default='', help='comma-separated list of input indices to flag')
    p.add_argument('--flaggedinputsfile',  default='active.txt', help='flagged inputs file name')
    p.add_argument('--recordedsnaps',   default='', help='Comma-separated list of SNAP ids')
    p.add_argument('--recordedsnapsfile',  default='active.txt', help='SNAP ids inputs file')
    p.add_argument('--tbfilterbank',      default=False, action='store_true', help='Write out filterbank rather than voltages in h5 format')
    p.add_argument('--tempcoefffile',   default='temperature_correction.txt', help='Temperature coefficient file name')
    p.add_argument('--currenttempfile', default='/home/npsr/temperature_logs/current_temp.txt', help='File continuously updated with current temperature')
    p.add_argument('--overridetemp', default=-999, type=float, help='If set, will override the current logged temperature - use for re-processing saved voltages')
    p.add_argument('--verbose',      default=False, action='store_true', help='Print out lots of stuff')
    args = p.parse_args()
    outdir  = args.outdir

    file_prefix = 'spip_beamformer'
    hdr_callback    = dada_dict_to_bf_dict
    n_int_per_file  = 128
    n_tavg  = 32 # Number of times to average
    #log_bandpasses = True
    log_bandpasses = False

    if args.tboutbuffer is None:
        args.tboutbuffer = []

    # Check for valid setup
    if args.nfanbeam == 0 and len(args.tboutbuffer) == 0 and not args.individualcassettes:
        parser.error("One or more of nfanbeam>0, len(tboutbuffer)>0, or individualcassettes must be true!")

    if len(args.tboutbuffer) > 3:
        parser.error("A maximum of 3 tied-beams is supported")

    # Check if there is a cable_delays.txt file in this directory, warn if so
    if os.path.exists("cable_delays.txt") and not args.cabledelayfile == "cable_delays.txt":
        print("WARNING! There is a cable_delays.txt file in this directory, but you are not using it! Do you intend to use {0}/{1} as the calibration file?".format(args.configdir, args.cabledelayfile))

    # Load up antenna details
    antposfile = args.configdir + '/antenna_positions/' + args.antposfile
    if args.verbose:
        print("antposfile is " + antposfile)
    cassettepositions = get_cassette_dictionary(antposfile)

    # Load up the SNAP mapping
    snapmapfile = args.configdir + '/snap_mapping/' + args.snapmapfile
    snapmap = read_snap_mapping(snapmapfile)

    # Load up the cable delays
    cabledelayfile = args.configdir + '/cable_delays/' + args.cabledelayfile
    cabledelays = get_delays(cabledelayfile) / 1e9 # convert to seconds

    # Load up the already-applied SNAP delays (this should really be loaded through a safer / less error-prone mechanism, via REDIS?)
    applieddelays = get_delays(args.applieddelayfile) / 1e9 # convert to seconds

    # Load up the temperature coefficients (they are in units of ns per degree C initially)
    temp_coeffs = get_delays(args.configdir + '/temperature_coefficients/' + args.tempcoefffile) / 1e9 # convert to seconds per degree C

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

    nant = 6*len(recorded_snaps)

    # Get a list of inputs to flag
    flagged_inputs = []
    if not args.flaginputs.strip() == "":
        flagged_inputs = map(int, args.flaginputs.rstrip(',').split(','))
    else:
        if args.flaggedinputsfile != "":
            flaggedinputsfile = args.configdir + '/flagged_inputs/' + args.flaggedinputsfile
            flagged_inputs = get_flagged_inputs(flaggedinputsfile)

    # Tied beam correctors
    wgtiedbeams = []
    for i in range(len(args.tboutbuffer)):
        wgtiedbeams.append(WeightsGenerator2(cassettepositions, snapmap, cabledelays, applieddelays, temp_coeffs, args.currenttempfile, args.overridetemp, recorded_snaps, flagged_inputs, bandpass, args.verbose, False, args.declination, 0, 0, i, args.configdir)) # No fan beams here ever

    ## Boresight corrector defaults to the first tied beam if there is one
    #if len(args.tboutbuffer) > 0:
    #    wgboresight = wgtiedbeams[0]

    # Fan beam corrector, if utilised
    if args.nfanbeam > 0:
        wgfanbeam = WeightsGenerator2(cassettepositions, snapmap, cabledelays, applieddelays, temp_coeffs, args.currenttempfile, args.overridetemp, recorded_snaps, flagged_inputs, bandpass, args.verbose, args.fixed_boresight, args.declination, args.nfanbeam, args.fanbeamspacing, -1, args.configdir)
        ## If a fixed boresight correction is used, rather than around a tied beam, then we need a new boresight corrector too
        #if args.fixed_boresight:
        #    wgboresight = WeightsGenerator2(cassettepositions, snapmap, cabledelays, applieddelays, recorded_snaps, flagged_inputs, bandpass, args.verbose, args.fixed_boresight, args.declination, 0, 0, -1) # No fan beams
        #elif len(args.tboutbuffer) == 0:
        #    wgboresight = WeightsGenerator2(cassettepositions, snapmap, cabledelays, applieddelays, recorded_snaps, flagged_inputs, bandpass, args.verbose, args.fixed_boresight, args.declination, 0, 0, -1) # No fan beams
    if args.individualcassettes:
        cassette_flags = [] # Don't flag any individual outputs
        wgcassettebeam = WeightsGenerator2(cassettepositions, snapmap, cabledelays, applieddelays, temp_coeffs, args.currenttempfile, args.overridetemp, recorded_snaps, cassette_flags, bandpass, args.verbose, args.fixed_boresight, args.declination, 0, 0, -1, args.configdir)

    # Set the weight update frequency
    if args.fixed_boresight:
        boresight_update_frequency = 4000 # Update every 4 minutes or so, to take temperature changes into effect
    else:
        boresight_update_frequency = 200
    tiedbeam_update_frequency = 200 # Always update the tied beam(s) frequently enough

    if args.verbose:
        print("WeightsGenerator has been created")

    # Set up data read
    if args.filename is None:
        b_gpu = bf.blocks.psrdada.read_psrdada_buffer(args.buffer, hdr_callback, 1, single=True, core=args.core, space='cuda')
    else:
        #b_gpu = bf.blocks.read_dada_file(args.filename.split(','), hdr_callback, gulp_nframe=1, core=args.core, space='cuda')
        b_file = bf.blocks.read_dada_file(args.filename.split(','), hdr_callback, gulp_nframe=1, core=args.core)
        b_gpu  = bf.blocks.copy(b_file, space='cuda', core=args.core+1, gpu=0)
    if args.verbose:
        PrintStuffBlock(b_gpu)
    
    # GPU processing
    tb_gpus = {}
    with bf.block_scope(fuse=True, gpu=0):
        b_gpu = bf.views.merge_axes(b_gpu, 'station', 'pol')
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'subband', 'freq', 'heap', 'frame', 'snap', 'station'] )
        b_gpu = bf.views.merge_axes(b_gpu, 'subband', 'freq', label='freq')
        b_gpu = bf.views.merge_axes(b_gpu, 'snap', 'station', label='station')
        b_gpu = bf.views.merge_axes(b_gpu, 'heap', 'frame', label='fine_time')
        #b_gpu = apply_weights2(b_gpu, weights_callback=wgboresight, output_dtype='cf32', 
        #                       update_frequency=boresight_update_frequency, tiedbeam=-1)
        # Produce the fan beams, if needed
        if args.nfanbeam > 0:
            #fb_gpu = apply_weights2(b_gpu, weights_callback=wgboresight, output_dtype='cf32',
            #                        update_frequency=boresight_update_frequency, tiedbeam=-1)
            #fb_gpu = bf.views.split_axis(b_gpu, 'station', 2, label='pol')
            #fb_gpu = bf.blocks.beanfarmer(fb_gpu, weights_callback=wgfanbeam, n_avg=n_tavg, n_beam=args.nbeam, n_pol=2, n_chan=args.nchan, n_ant=nant)
            fb_gpu = apply_weights2(b_gpu, weights_callback=wgfanbeam, output_dtype='ci8',
                                    update_frequency=boresight_update_frequency)
            fb_gpu = bf.views.split_axis(fb_gpu, 'station', 2, label='pol')
            fb_gpu = bf.blocks.transpose(fb_gpu, ['time','freq','fine_time','pol','station'])
            fb_gpu = bf.blocks.beanfarmer(fb_gpu, weights_callback=wgfanbeam, n_avg=n_tavg, n_beam=args.nfanbeam, n_pol=2, n_chan=args.nchan, n_ant=nant)
            #print('Normalising data...')
            #fb_gpu = normalise(fb_gpu, args.n_count)
            if args.fboutbuffer is not None:
                # Transpose to time, beam, freq, fine_time order for subsequent searching
                fb_gpu = bf.blocks.transpose(fb_gpu, ['time', 'beam', 'freq', 'fine_time'])
            else:
                # Transpose to the right order for sigproc to write out filterbanks
                if args.verbose:
                    PrintStuffBlock(fb_gpu)

                fb_gpu = bf.blocks.transpose(fb_gpu, ['time','fine_time','beam','freq'])
                fb_gpu = bf.views.merge_axes(fb_gpu, 'time', 'fine_time')
                fb_gpu = bf.views.add_axis(fb_gpu, axis=2, label='pol')

        # Produce the cassette beam filterbanks, if desired
        if args.individualcassettes:
            # ci8 output seems to have no problems - but we could do cf32 if preferred. The following detect block then correctly accumulates into cf32. 
            cb_gpu = apply_weights2(b_gpu, weights_callback=wgcassettebeam, output_dtype='ci8',
                                    update_frequency=2000000)
            cb_gpu = bf.views.split_axis(cb_gpu, 'station', 2, label='pol')
            #cb_gpu = byip.detect(cb_gpu, mode='coherence', axis='pol')
            cb_gpu = byip.detect(cb_gpu, mode='stokes_i', axis='pol')
            cb_gpu = bf.blocks.reduce(cb_gpu, 'fine_time', n_tavg)
            if args.verbose:
                PrintStuffBlock(cb_gpu)
            if args.cboutbuffer is not None:
                # Transpose the data on the GPU and delete the unit-length pol axis, so it is in the correct order ready to send off
                cb_gpu = bf.blocks.transpose(cb_gpu, ['time', 'station', 'freq', 'fine_time', 'pol'])
                cb_gpu = bf.views.delete_axis(cb_gpu, 4)

        # Now go back to the uncorrected voltage stream and form the tied array beams
        for i in range(len(args.tboutbuffer)):
            tb_gpus[i] = apply_weights2(b_gpu, weights_callback=wgtiedbeams[i], output_dtype='cf32',
                                        update_frequency=tiedbeam_update_frequency, core=args.core+1+i)
            if log_bandpasses and i == 0:
                #log_gpu =  byip.detect(tb_gpus[i], mode='scalar', axis=None) # Data shape is time, freq, fine_time, station
                #bandpass_gpu = bf.blocks.reduce(log_gpu, axis='fine_time')
                bandpass_gpu = bf.blocks.reduce(tb_gpus[i], axis='fine_time', op='pwrmean')
                #bandpass_cpu = bf.blocks.copy(bandpass_gpu, space='cuda_host', core=args.core+3+len(args.tboutbuffer))
                #PrintStuffBlock(bandpass_cpu, printlabel='bandpass_cpu')
                #bandpass_gpu = bf.blocks.transpose(bandpass_gpu, ['time', 'station', 'freq', 'fine_time'])
                bandpass_gpu = bf.views.delete_axis(bandpass_gpu, 2) # Now time, freq, station
                bandpass_gpu = bf.blocks.transpose(bandpass_gpu, ['time', 'station', 'freq'])
                bandpass_cpu = bf.blocks.copy(bandpass_gpu, space='cuda_host', core=args.core+3+len(args.tboutbuffer))
                if args.verbose:
                    PrintStuffBlock(bandpass_cpu, printlabel='bandpass_cpu')
                DumpBlock(bandpass_cpu, dobandpass=True)
                #time_gpu = bf.blocks.reduce(log_gpu, axis='freq')
                time_gpu = bf.blocks.reduce(tb_gpus[i], axis='freq', op='pwrmean')
                time_gpu = bf.views.delete_axis(time_gpu, 1) # Now time, fine_time, station
                time_gpu = bf.blocks.reduce(time_gpu, 'fine_time', 128) # Produce ~1ms samples
                time_gpu = bf.blocks.transpose(time_gpu, ['time', 'station', 'fine_time'])
                #time_gpu = bf.views.merge_axes(time_gpu, 'time', 'fine_time')
                time_cpu = bf.blocks.copy(time_gpu, space='cuda_host', core=args.core+4+len(args.tboutbuffer))
                if args.verbose:
                    PrintStuffBlock(bandpass_cpu, printlabel='bandpass_cpu')
                DumpBlock(time_cpu, dobandpass=False)
            tb_gpus[i] = bf.views.split_axis(tb_gpus[i], 'station', 2, label='pol')
            tb_gpus[i] = bf.blocks.reduce(tb_gpus[i], axis='station')
            # Deal with the tied beam output format
            if args.tbfilterbank:
                tb_gpus[i] = byip.detect(tb_gpus[i], mode='coherence', axis='pol')
                tb_gpus[i] = bf.blocks.reduce(tb_gpus[i], 'fine_time', n_tavg)
            else:
                # transpose to time, station, freq, pol for DSPSR to process
                tb_gpus[i] = bf.blocks.transpose(tb_gpus[i], ['time', 'fine_time', 'station', 'freq', 'pol'])
                tb_gpus[i] = bf.views.merge_axes(tb_gpus[i], 'time', 'fine_time')

    # Back to CPU, print out
    tb_cpus = {}
    tb_cpus0 = {}
    tb_dada = {}
    for i in range(len(args.tboutbuffer)):
        tb_cpus[i] = bf.blocks.copy(tb_gpus[i], space='cuda_host', core=args.core+1+i)
        if args.verbose:
            PrintStuffBlock(tb_cpus[i])
    if args.nfanbeam > 0:
        fb_cpu =  bf.blocks.copy(fb_gpu, space='cuda_host', core=args.core+1+len(args.tboutbuffer))
        if args.verbose:
            PrintStuffBlock(fb_cpu)
    if args.individualcassettes:
        cb_cpu = bf.blocks.copy(cb_gpu, space='cuda_host', core=args.core+2+len(args.tboutbuffer))
        if args.verbose:
            PrintStuffBlock(cb_cpu)

    # Send off the tied beams if necessary
    if args.tboutbuffer is not None:
        for i in range(len(args.tboutbuffer)):
            prefix = "TB" + str(i) + "_"
            tb_dada[i] = bf.blocks.psrdada.write_psrdada_buffer(tb_cpus[i], args.tboutbuffer[i], gulp_nframe=1, datasizefactor=8./144.)
            tb_dada[i].add_header_keywords( {"INSTRUMENT": "dspsr", "TELESCOPE": "MOST", "BEAM": str(i)} )
            tb_dada[i].sub_header_keywords( [prefix + "ENABLED"] )
            tb_dada[i].remap_prefixed_keywords( prefix, ["SOURCE", "PROC_FILE", "FLAGGED_INPUTS_FILE"] )

    elif not args.benchmark:
        if args.tbfilterbank: # Convert to filterbank
            for i in range(len(args.tboutbuffer)):
                tb_cpus[i] = bf.blocks.transpose(tb_cpus[i], ['time', 'fine_time', 'station', 'pol', 'freq'])
                tb_cpus0[i] = bf.views.merge_axes(tb_cpus[i], 'time', 'fine_time')
                tb_cpus0[i] = extract_antenna(tb_cpus0[i], ant_id=0)
                tb_cpus0[i] = RenameBlock(tb_cpus0[i], 'bf0_tb%i' % i)
                bf.blocks.write_sigproc(tb_cpus0[i], path=outdir)
        else: # Just write out voltages in h5 format
            for i in range(len(args.tboutbuffer)):
                h5write(tb_cpus[i], outdir=outdir, prefix=file_prefix + '_' + str(i), n_int_per_file=n_int_per_file, core=args.core+2+len(args.tboutbuffer)+i)
    else:
        print("Enabling benchmark mode...")

    # Send of the fan beams
    if args.nfanbeam > 0:
        if args.fboutbuffer is not None:
            fb_dada = bf.blocks.psrdada.write_psrdada_buffer(fb_cpu, args.fboutbuffer, gulp_nframe=1, datasizefactor=args.nfanbeam/(144*2.0*n_tavg))
        elif not args.benchmark:
            fb_cpus = {}
            for beam_id in range(args.nfanbeam):
                fb_cpus[beam_id] = extract_beam(fb_cpu, beam_id)
                fb_cpus[beam_id] = RenameBlock(fb_cpus[beam_id], 'bf0_fb%03i' % beam_id)
                bf.blocks.write_sigproc(fb_cpus[beam_id], path=outdir)

    # Send off / write out the individual cassette beams as needed
    if args.individualcassettes:
        if args.cboutbuffer is not None:
            cb_dada = bf.blocks.psrdada.write_psrdada_buffer(cb_cpu, args.cboutbuffer, gulp_nframe=1, datasizefactor=0.5/n_tavg)
        elif not args.benchmark:
            cb_cpu = bf.blocks.transpose(cb_cpu, ['time', 'fine_time', 'station', 'pol', 'freq'])
            cb_cpu = bf.views.merge_axes(cb_cpu, 'time', 'fine_time')
            cb_cpus = {}
            for antenna in range(nant):
                cb_cpus[antenna] = extract_antenna(cb_cpu, ant_id=antenna)
                cb_cpus[antenna] = RenameBlock(cb_cpus[antenna], 'cb_a%i' % antenna)
                bf.blocks.write_sigproc(cb_cpus[antenna], path=outdir)

    print("Running pipeline")
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()
