import bifrost as bf
from blocks.read_vcs import read_vcs_block
from blocks.print_stuff import print_stuff_block
from blocks.h5write import h5write_block

from logger import setup_logger

setup_logger(filter="blocks.read_vcs", level="INFO")

if __name__ == "__main__":
    fn = '../fast-imaging-test/vcs/1164110416_metafits.fits'
    filelist = [fn, ]
    
    # Data arrive as ['time', 'coarse_channel',  'frame', 'fine_channel', 'station', 'pol']
    b_vcs    = read_vcs_block(filelist, space='cuda_host')
    b_gpu    = bf.blocks.copy(b_vcs, space='cuda')
    
    with bf.block_scope(fuse=False, gpu=0):
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'frame', 'station', 'coarse_channel', 'fine_channel', 'pol'])
        b_gpu = bf.views.merge_axes(b_gpu, 'coarse_channel', 'fine_channel', label='channel')
        b_gpu = bf.blocks.detect(b_gpu, mode='stokes')
        b_gpu = bf.blocks.reduce(b_gpu, axis='station')
        b_gpu = bf.views.delete_axis(b_gpu, axis='station')
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'frame', 'pol', 'channel'])
        b_gpu = bf.views.rename_axis(b_gpu, 'channel', 'freq')
    b_cpu = bf.blocks.copy(b_gpu, space='system')
    b_cpu = bf.views.merge_axes(b_cpu, 'time', 'frame', label='time')
    print_stuff_block(b_cpu, n_gulp_per_print=10)
    bf.blocks.write_sigproc(b_cpu, path='./data/')
    
    print("Running pipeline")
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()


