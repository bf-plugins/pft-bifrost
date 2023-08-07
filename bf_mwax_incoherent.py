""" 
# bf_mwax_incoherent.py

A pipeline to convert MWAX VCS data into filterbank files.
"""
import bifrost as bf
from blocks.read_vcs_mwalib import read_vcs_block
from blocks.detect import DetectBlock
from blocks.print_stuff import print_stuff_block
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
    #print_stuff_block(b_vcs, n_gulp_per_print=10)
    b_gpu    = bf.blocks.copy(b_vcs, space='cuda', gulp_nframe=1)

    with bf.block_scope(fuse=True, gpu=0):
        b_gpu = bf.views.split_axis(b_gpu, 'sample', n=N_chan, label='fine_time')
        b_gpu = bf.blocks.fft(b_gpu, axes='fine_time', axis_labels='fine_channel', apply_fftshift=True)
        b_gpu = DetectBlock(b_gpu, mode='stokes_i')
        b_gpu = bf.blocks.reduce(b_gpu, axis='station')
        b_gpu = bf.blocks.reduce(b_gpu, factor=N_int, axis='sample')
        b_gpu = bf.views.delete_axis(b_gpu, axis='station')
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'sample', 'pol', 'coarse_channel', 'fine_channel'])
    b_cpu = bf.blocks.copy(b_gpu, space='system')
    b_cpu = bf.views.merge_axes(b_cpu, 'time', 'sample', label='time')
    b_cpu = bf.views.merge_axes(b_cpu, 'coarse_channel', 'fine_channel', label='freq')
    
    print_stuff_block(b_cpu, n_gulp_per_print=1)
    
    # Write to sigproc. Need [time pol freq] axes
    bf.blocks.write_sigproc(b_cpu, path='./data/')

    
    print("Running pipeline")
    pipeline = bf.get_default_pipeline()
    pipeline.shutdown_on_signals()
    pipeline.dot_graph().render('bf_mwax_incoherent.log')
    pipeline.run()


