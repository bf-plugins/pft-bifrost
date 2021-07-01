import bifrost as bf
from blocks.read_vcs import read_vcs_block
from blocks.print_stuff import print_stuff_block
from blocks.h5write import h5write_block

if __name__ == "__main__":
    fn = '../fast-imaging-test/vcs/1164110416_metafits.fits'
    filelist = [fn, fn]
    
    # Data arrive as ['time', 'coarse_channel',  'frame', 'fine_channel', 'station', 'pol']
    b_vcs    = read_vcs_block(filelist, space='cuda_host')
    #b_pinned = bf.blocks.copy(b_vcs, space='cuda_host')
    b_gpu    = bf.blocks.copy(b_vcs, space='cuda')
    
    with bf.block_scope(fuse=False, gpu=0):
        b_gpu = bf.blocks.transpose(b_gpu, ['time', 'frame', 'station', 'coarse_channel', 'fine_channel', 'pol'])
        b_gpu = bf.views.merge_axes(b_gpu, 'coarse_channel', 'fine_channel', label='channel')
        b_gpu = bf.blocks.detect(b_gpu, mode='stokes')
        b_gpu = bf.blocks.reduce(b_gpu, axis='station')
        b_gpu = bf.views.delete_axis(b_gpu, axis='station')
    b_cpu = bf.blocks.copy(b_gpu, space='system')
    print_stuff_block(b_cpu, n_gulp_per_print=10)
    h5write_block(b_cpu, n_int_per_file=100)
    
    print("Running pipeline")
    bf.get_default_pipeline().shutdown_on_signals()
    bf.get_default_pipeline().run()


