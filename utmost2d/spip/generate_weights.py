import numpy as np
import hickle as hkl

w = np.zeros((512, 32, 1, 12), dtype=[('re', 'int8'), ('im', 'int8')])


# Check linear scaling
#for beamid in range(32):
#   w['re'][:, beamid] = beamid
#   w['im'][:, beamid] = beamid

for antid in range(12):
    w['re'][:, antid, 0, antid] = 1 
    w['im'][:, antid, 0, antid] = 1 


w['re'][:, 12] = 1
w['im'][:, 12] = 1




hkl.dump(w, 'beam_weights.hkl')

