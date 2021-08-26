""" 
# vcs_pipeline.py

A pipeline to convert MWA VCS data into filterbank files.
"""
import bifrost as bf
from blocks.read_vcs_sub import read_vcs_block
from blocks.print_stuff import print_stuff_block
from blocks.h5write import h5write_block

from logger import setup_logger
import time

setup_logger(filter="blocks.read_vcs", level="INFO")

if __name__ == "__main__":
    t0 = time.time()
    fn = '../fast-imaging-test/vcs/1164110416_metafits.fits'
    filelist = [fn, ]
    
    # Data arrive as ['time', 'coarse_channel',  'frame', 'fine_channel', 'station', 'pol']
    b_vcs    = read_vcs_block(filelist, space='cuda_host', gulp_nframe=1)
    b_gpu    = bf.blocks.copy(b_vcs, space='cuda')
    
    with bf.block_scope(fuse=True, gpu=0):
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'coarse_channel', 'fine_channel', 'station', 'pol', 'frame'])
        b_gpu = bf.views.merge_axes(b_gpu, 'coarse_channel', 'fine_channel', label='freq')
        b_gpu = bf.views.merge_axes(b_gpu, 'station', 'pol', label='station')
        b_gpu = bf.views.rename_axis(b_gpu, 'frame', 'fine_time')
        b_gpu = bf.blocks.correlate_dp4a(b_gpu, nframe_to_avg=10)
    b_cpu = bf.blocks.copy(b_gpu, space='system')
    print_stuff_block(b_cpu, n_gulp_per_print=10)
    h5write_block(b_cpu, prefix='correlator_test', n_int_per_file=10)
 
    print("Running pipeline")
    pipeline = bf.get_default_pipeline()
    pipeline.shutdown_on_signals()
    pipeline.dot_graph().render('vcs_pipeline_graph.log')
    pipeline.run()


