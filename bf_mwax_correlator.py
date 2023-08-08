""" 
# bf_mwax_correlator.py

A pipeline to run tensor core correlator on MWAX data
"""
import bifrost as bf
from blocks.read_vcs_mwalib import read_vcs_block
from blocks.detect import DetectBlock
from blocks.print_stuff import print_stuff_block
from blocks.quantize import quantize
from blocks.tcc_correlator import tensor_core_correlator
from blocks.h5write import h5write_block
from logger import setup_logger
import glob

setup_logger(filter="blocks.read_vcs_mwalib", level="DEBUG")

if __name__ == "__main__":
    metafits = "/datax2/users/dancpr/2023.jun-tian-bifrost-mwa/1369756816.metafits"
    filelist = sorted(glob.glob('/datax2/users/dancpr/2023.jun-tian-bifrost-mwa/1369756816_*.sub'))

    filelist = [
        [metafits, filelist], 
        ]
    
    # Hardcoded values 
    coarse_chan_bw   = 1.28e6
    N_samp_per_block = 64000
    scale_factor = 1.0 / 2**13

    # Set desired channel and time integration
    # note 64000 = 2^9 x 5^3
    N_chan  = 125        # 10.24 kHz
    N_int   = 4          # 4 / 10.24 kHz = 0.390625 ms
    
    try:
        assert N_samp_per_block % (N_chan * N_int) == 0
    except AssertionError:
        print("Error: N_chan x N_int must equally divide N_samp_per_block")
    

    # Data arrive as ['time', 'coarse_channel', 'station', 'pol', 'sample']
    b_vcs    = read_vcs_block(filelist, coarse_chan=0, space='cuda_host')
    b_vcs    = bf.views.merge_axes(b_vcs, 'time', 'block', label='time')

    b_gpu    = bf.blocks.copy(b_vcs, space='cuda', gulp_nframe=1)

    with bf.block_scope(fuse=True, gpu=0):
        b_gpu = bf.views.split_axis(b_gpu, 'sample', n=N_chan, label='fine_time')
        b_gpu = bf.views.split_axis(b_gpu, 'sample', n=16, label='tcc_block')
        b_gpu = bf.blocks.fft(b_gpu, axes='fine_time', axis_labels='fine_channel', apply_fftshift=True)
        b_gpu = quantize(b_gpu, 'ci8', scale=scale_factor)
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'coarse_channel', 'fine_channel', 'sample', 'station', 'pol', 'tcc_block'])
        b_gpu = bf.views.merge_axes(b_gpu, 'coarse_channel', 'fine_channel', label='freq')
        b_gpu = tensor_core_correlator(b_gpu)
        #print_stuff_block(b_gpu, n_gulp_per_print=1)

    b_cpu = bf.blocks.copy(b_gpu, space='system')

    # Write to disk
    h5write_block(b_cpu, outdir='./data', n_int_per_file=48)

    
    print("Running pipeline")
    pipeline = bf.get_default_pipeline()
    pipeline.shutdown_on_signals()
    pipeline.dot_graph().render('bf_mwax_incoherent.log')
    pipeline.run()


