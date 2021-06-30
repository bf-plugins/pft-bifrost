from blocks import read_vcs

fn = '../fast-imaging-test/vcs/1164110416_metafits.fits'
fn = [fn, fn]

with read_vcs.MwaVcsReader(fn) as mwa:
    for ii in range(201):
        d = mwa.read_frame()
        print(ii, d.shape)