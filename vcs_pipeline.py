""" 
# vcs_pipeline.py

A pipeline to convert MWA VCS data into filterbank files.
"""
import bifrost as bf
from blocks.read_vcs import read_vcs_block
from blocks.print_stuff import print_stuff_block
from blocks.h5write import h5write_block

from logger import setup_logger

setup_logger(filter="blocks.read_vcs", level="INFO")

if __name__ == "__main__":
    fn = '../fast-imaging-test/vcs/1164110416_metafits.fits'
    filelist = [fn, ]
    scale_factor = 1.0 / 2**13
    print(f"SCALE FACTOR: {scale_factor}")
    
    # Data arrive as ['time', 'coarse_channel',  'frame', 'fine_channel', 'station', 'pol']
    b_vcs    = read_vcs_block(filelist, space='cuda_host')
    b_gpu    = bf.blocks.copy(b_vcs, space='cuda')
    
    with bf.block_scope(fuse=False, gpu=0):
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'frame', 'station', 'coarse_channel', 'fine_channel', 'pol'])
        b_gpu = bf.views.merge_axes(b_gpu, 'coarse_channel', 'fine_channel', label='channel')
        b_gpu = bf.blocks.detect(b_gpu, mode='stokes_i')
        b_gpu = bf.blocks.reduce(b_gpu, axis='station')
        b_gpu = bf.blocks.reduce(b_gpu, factor=10, axis='frame')
        b_gpu = bf.blocks.quantize(b_gpu, dtype='u8', scale=scale_factor)
        b_gpu = bf.views.delete_axis(b_gpu, axis='station')
        b_gpu = bf.views.delete_axis(b_gpu, axis='pol')
        #b_gpu = bf.blocks.transpose(b_gpu, ['time', 'frame', 'pol', 'channel'])
        b_gpu = bf.views.rename_axis(b_gpu, 'channel', 'freq')
    b_cpu = bf.blocks.copy(b_gpu, space='system')
    b_cpu = bf.views.add_axis(b_cpu, 'frame', label='pol')
    b_cpu = bf.views.merge_axes(b_cpu, 'time', 'frame', label='time')
    print_stuff_block(b_cpu, n_gulp_per_print=10)
    bf.blocks.write_sigproc(b_cpu, path='./data/')
    
    print("Running pipeline")
    pipeline = bf.get_default_pipeline()
    pipeline.shutdown_on_signals()
    pipeline.dot_graph().render('vcs_pipeline_graph.log')
    pipeline.run()


