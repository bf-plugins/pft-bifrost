# pft-bifrost

Bifrost pipeline blocks for the Pulsar and Fast Transient group at CIRA.

### Reading VCS data

A bifrost block to read MWAX VCS data is supplied in `blocks/read_vcs_mwalib.py`. This block uses [pymwalib](https://github.com/MWATelescope/pymwalib) to read data, and then reinterprets it as a bifrost array. 

### Current apps

* `bf_mwax_incoherent.py` - An incoherent beamformer to create filterbank (.fil) files. Applies an FFT and integrates to desired frequency / time resolution.
