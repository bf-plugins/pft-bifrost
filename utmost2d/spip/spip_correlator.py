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
from blocks import h5write
from blocks.dada_header_spip2 import dada_dict_to_bf_dict
import hickle as hkl
import os
import h5py
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
        #pprint(iseq.header)i
        pprint(iseq.header)
        self.n_iter = 0

    def on_data(self, ispan):
        now = datetime.now()
        if self.n_iter % self.n_gulp_per_print == 0 and self.print_on_data:
            d = ispan.data
            #d = np.array(d).astype('float32')
            print("[%s] %s %s" % (now, str(ispan.data.shape), str(ispan.data.dtype)))
        self.n_iter += 1

if __name__ == "__main__":
    from argparse import ArgumentParser
    p = ArgumentParser(description='SNAP small-N correlator')
    p.add_argument('-c', '--core', default=8, type=str, help="core: cpu core to start scheduling processes on")
    p.add_argument('-m', '--mode', default='lo', type=str, help="mode: 'lo' (1 sec) or 'hi' (1 ms) time resolution.")
    p.add_argument('-b', '--buffer', default=0xBABA, type=lambda x: int(x,0), help="PSRDADA buffer to connect to")
    p.add_argument('-n', '--nintperfile', default=None, type=int, help="Number of integrations in each file (default 8192 HTR, 256 1s)")
    p.add_argument('-f', '--filename', default=None, type=str, help="If set, will read from file instead of PSRDADA.")
    p.add_argument('-o', '--outdir',  default='/data/dprice', type=str, help="Path to output data to.")
    p.add_argument('-p', '--processframe',  default=1, type=int, help="Only process 1 frame out of every processframe frames. Default 1 = all")
    p.add_argument('-t', '--temperaturefile', default='/home/npsr/temperature_logs/current_temp.txt', help='File continuously updated with current temperature')
    
    args = p.parse_args()

    htr_mode = True if args.mode == 'hi' else False

    file_prefix = 'spip_xcor'
    hdr_callback    = dada_dict_to_bf_dict

    if htr_mode:
        print("Starting in HTR mode")
        n_tavg          = 128              # 128=1.3ms. TSAMP=10.24us
        n_int_per_file  = 8192             # about 10.7s
        nframe_to_avg   = 1
    else:
        print("Starting in 1s xcor mode")
        #n_tavg          = 8192             # 83.8 ms. TSAMP=10.24us
        n_tavg          = 4096             # 41.94 ms. TSAMP=10.24us
        n_int_per_file  = 128               # about 300s
        nframe_to_avg   = 128               # 5.3 sec
    
    if args.nintperfile is not None:
        n_int_per_file = args.nintperfile

    n_subheap = n_tavg / 8       # 8 time steps in packet frame
 
    # First DADA buffer
    if args.filename is None:
        b_dada = bf.blocks.psrdada.read_psrdada_buffer(args.buffer, hdr_callback, 1, single=True, core=args.core)
    else:
        b_dada = bf.blocks.read_dada_file([args.filename,], hdr_callback, gulp_nframe=1, core=args.core)
    PrintStuffBlock(b_dada)
    
    
    b_gpu = bf.blocks.copy(b_dada, space='cuda', core=args.core+1, gpu=0)
    with bf.block_scope(fuse=False, gpu=0):
        b_gpu = bf.views.merge_axes(b_gpu, 'station', 'pol')
        b_gpu = bf.views.split_axis(b_gpu, 'heap', n_subheap, label='subheap')
        #PrintStuffBlock(b_gpu)
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'heap', 'subband', 'freq', 'snap', 'station', 'subheap', 'frame'])
        b_gpu = bf.views.merge_axes(b_gpu, 'subband', 'freq')
        b_gpu = bf.views.merge_axes(b_gpu, 'snap', 'station')
        b_gpu = bf.views.merge_axes(b_gpu, 'subheap', 'frame')
        b_gpu = bf.views.merge_axes(b_gpu, 'time', 'heap')
        b_gpu = bf.views.rename_axis(b_gpu, 'subheap', 'fine_time')
        b_gpu = bf.views.rename_axis(b_gpu, 'subband', 'freq')
        b_gpu = bf.views.rename_axis(b_gpu, 'snap', 'station')
        b_gpu = bf.blocks.correlate_dp4a(b_gpu, nframe_to_avg=nframe_to_avg, process_frame=args.processframe)
    b_cpu = bf.blocks.copy(b_gpu, space='cuda_host', core=args.core+2)
    PrintStuffBlock(b_cpu)
    h5write(b_cpu, outdir=args.outdir, prefix=file_prefix, n_int_per_file=n_int_per_file, temperature_file=args.temperaturefile,core=args.core+3)
    

    print("Running pipeline")
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()
