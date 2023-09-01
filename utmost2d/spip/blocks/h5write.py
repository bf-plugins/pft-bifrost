import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import numpy as np
import os
import h5py

class H5Write(bf.pipeline.SinkBlock):
    def __init__(self, iring, outdir='./', prefix='h5w',  n_int_per_file=100, *args, **kwargs):
        super(H5Write, self).__init__(iring, *args, **kwargs)
        self.outdir = outdir
        self.seq_idx = 0
        self.data_idx = 0
        self.current_fh = None
        self.n_int_per_file = n_int_per_file
        self.fh = None
        self.prefix = prefix
        self._meta = {}

    def on_sequence(self, iseq):
        self.seq_idx += 1
        self.data_idx = 0
        
        # Setup metadata
        ihdr = iseq.header
        self._meta['source_name'] = ihdr['source_name']
        self._meta['ra']          = ihdr['RA']
        self._meta['dec']         = ihdr['DEC']
        self._meta['labels']      = map(str, ihdr['_tensor']['labels'])
        self._meta['units']       = map(str, ihdr['_tensor']['units'])
        self._meta['scales']      = map(list, ihdr['_tensor']['scales'])
    
        self.dshape = [self.n_int_per_file,] + iseq.header['_tensor']['shape'][1:]
        self.dtype = bf.dtype.string2numpy(iseq.header['_tensor']['dtype'])
        self._create_new_h5()

    def on_sequence_end(self, iseq):
        if self.fh is not None:
            self.fh.close()

    def _create_new_h5(self):
        if self.fh is not None:
            self.fh.close()
              
        now = datetime.utcnow()        
        now_str  = now.strftime("%Y-%m-%d-%H%M%S")
        fn = '%s_%s_%i_%i_%s.h5' % (self.prefix, self._meta['source_name'], self.seq_idx, self.data_idx, now_str)
        self.filename = os.path.join(self.outdir, fn)
        print("Creating %s" % self.filename)
        self.fh = h5py.File(self.filename, 'w') 
        self.fh.create_dataset('data', shape=self.dshape, dtype=self.dtype)
        
        file_metadata = self.fh.attrs
        
        # Add time metadata at time of creation
        self._meta['utc_start']   = str(Time(now).iso)
        self._meta['mjd_start']   = float(Time(now).mjd)
        self._meta['data_idx']    = self.data_idx
        
        pprint(self._meta)
        # Copy metadata over to file
        for key, val in self._meta.items():
            file_metadata[key] = val

    def on_data(self, ispan):
        if self.data_idx % self.n_int_per_file == 0 and self.data_idx > 0:
            self._create_new_h5()
        if self.dtype is np.complex64:
            d_c64 =  np.array(ispan.data).view('complex64')
        else:
            d_c64 = np.array(ispan.data).astype('float32')

        start_idx = self.data_idx % self.n_int_per_file
        stop_idx  = start_idx + ispan.data.shape[0]
        self.fh['data'][start_idx:stop_idx] = d_c64
        self.data_idx += ispan.data.shape[0]
        # print(start_idx, stop_idx, ispan.data.shape)

def h5write(iring, outdir='./', n_int_per_file=100, prefix='h5w', *args, **kwargs):
    return H5Write(iring, outdir, prefix, n_int_per_file, *args, **kwargs)
