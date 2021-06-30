from astropy.time import Time
from astropy.io import fits as pf
import glob
import numpy as np
import numbits
import time
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import string2numpy
import numbits

def parse_metafits(filename):
    """ Parse a metafits file and return a python dictionary 
    
    Args:
        filename (str): Path to metafits file
    
    Returns:
        hdr (dict): Dictionary of metafits header:value cards. 
    
    Notes:
        Combines COMMENT and HISTORY lines, and ignores FITS-specific keywords.
    """
    mf = pf.open(filename)
    mf_hdr = mf[0].header
    hdr = {}
    comment, history = '', ''
    for key, val in mf_hdr.items():
        if key in ('SIMPLE', 'BITPIX', 'NAXIS', 'EXTEND'):
            pass
        elif key == 'COMMENT':
            comment += val  + ' '
        elif key == 'HISTORY':
            history += val  + ' '
        else:
            hdr[key] = val 
    hdr['COMMENT'] = comment
    hdr['HISTORY'] = history
    return hdr

def generate_filelist_from_metafits(filename):
    """ Generate a list of data files from metafits file 
    
    Args:
        filename (str): Path to metafits file
    
    Returns:
        filelist (list): list of dat files corresponding to metafits file. 
    
    """
    hdr = parse_metafits(filename)
    n_coarse_chan = len(hdr['CHANNELS'].split(','))
    fl = sorted(glob.glob(filename.replace('_metafits.fits', '*.dat')))
    
    if len(fl) != n_coarse_chan:
        print("Warning: Number of coarse channels does not match number of dat files.")
    return fl

class MwaVcsReader(object):
    """ A file reader for MWA VCS data.
    
    Args:
        filename_or_filelist (str/list): Name of metafits file to open, or list of metafits files
        N_time_per_frame (int): Number of time samples to read in each frame. 
                                Should evenly divide N_time_per_file
        N_time_per_file (int): Total number of time samples in file (default is 10,000)
    """
    def __init__(self, filename_or_filelist):
        
        super(MwaVcsReader, self).__init__()
        
        if isinstance(filename_or_filelist, str):
            self.obs_list = [filename_or_filelist,]
            self.n_obs    = 1
        else:
            self.obs_list = filename_or_filelist
            self.n_obs    = len(filename_or_filelist)
        
        self.obs_count = 0
        self.frame_count = 0
        self.n_frame_per_obs = 100
        
        self._open_next_obs()
        
        self.header = self._read_header()
        
        itensor = self.header['_tensor']
        
        self.dtype          = string2numpy(itensor['dtype'])
        self.frame_shape    = np.copy(itensor['shape'])
        self.frame_shape[0] = np.abs(self.frame_shape[0])
        self.frame_size = np.prod(self.frame_shape) * (self.dtype.itemsize)
        
        # Here 'gulp' is how much to read from a single file
        # Whereas 'frame' is how big the full array is (includes N_coarse_chan axis)
        self.dat_gulp_shape = list(self.frame_shape[2:]) + [2,]
        self.dat_gulp_size  = np.prod(self.dat_gulp_shape) * (self.dtype.itemsize // 2) // 2 # We will unpack 4-bit ci4 to 8-bit ci8
        
        # Create a temporary contiguous array to reuse.
        # Unpacks using numbits to int8, extra axis for real/imag
        self._data  = np.ndarray(shape=list(self.frame_shape[1:]) + [2,], dtype='int8')

    def _open_dat_files(self, idx):
        # Generate a list of datafiles
        self.current_datlist = generate_filelist_from_metafits(self.obs_list[idx])
        
        # Create filehandlers
        self.current_datobjs = []
        for fn in self.current_datlist:
            self.current_datobjs.append(open(fn, 'rb'))
    
    def _close_dat_files(self):
        for fh in self.current_datobjs:
            fh.close()    
        self.frame_count = 0
            
    def _read_header(self):
        """ Read metafits header and convert to bifrost header.
        
        Specifically, need to generate the '_tensor' from the FITS header.
        Currently there are some hardcoded values -- could use header callback to avoid this.
        """
        
        mf_header = parse_metafits(self.obs_list[0])
        N_coarse_chan = len(self.current_datlist)
        N_fine_chan   = 128
        N_station     = mf_header['NINPUTS'] // 2
        N_pol         = 2
        N_frame       = self.n_frame_per_obs
        
        t0 = Time(mf_header['GPSTIME'], format='gps')
        dt = 100e-6
        
        f0_coarse = mf_header['FREQCENT'] - mf_header['BANDWDTH'] / 2
        df_coarse = 1.28   # 1.28 MHz coarse channel 
        df_fine   = 0.01   # 10 kHz fine channel
           
        self.header = {
            '_tensor': {'dtype':  'ci8',  # Note data are 4-bit, but bifrost doesn't support ci4
                         'shape':  [-1,     N_coarse_chan,    N_frame,    N_fine_chan,    N_station, N_pol],
                         'labels': ['time', 'coarse_channel', 'frame',   'fine_channel', 'station', 'pol'],
                         'units':  ['s',    'MHz',             's',       'MHz',   '',     ''],
                         'scales': [[t0, dt * N_frame],
                                    [f0_coarse, df_coarse], 
                                    [0, dt],
                                    [0, df_fine],
                                    [0,0], 
                                    [0,0]]
                         },
             'name': mf_header['FILENAME'],
             'telescope': mf_header['TELESCOP'],       
             'ra': mf_header['RA'],
             'dec': mf_header['DEC'],
             'metafits': mf_header
        }
        return self.header
    
    def _open_next_obs(self):
        print("%i/%i: opening %s" % (self.obs_count+1, self.n_obs, self.obs_list[self.obs_count]))
        if self.obs_count > 0:
            self._close_dat_files()
        self._open_dat_files(idx=self.obs_count)            
        self.obs_count += 1

    def _read_data(self):
        #print("here", self.frame_count)
        
        if self.frame_count < self.n_frame_per_obs:
            for ii, fh in enumerate(self.current_datobjs):
                d_packed = np.fromfile(fh,  count=self.dat_gulp_size, dtype='int8')
                d_unpacked = numbits.unpack(d_packed, 4).reshape(self.dat_gulp_shape)
                self._data[ii] = d_unpacked
            self.frame_count += 1
            return self._data
        else:
            return np.ndarray(shape=0, dtype='int8')
    
    def read_frame(self):
        #print("Reading...")
        d = self._read_data()
        if d.size == 0 and self.frame_count == self.n_frame_per_obs:
            if self.obs_count == self.n_obs:
                print("EoDS")
                d = np.array([0]) # End of data stream
            else:
                print("Opening next obs")
                self._open_next_obs()
                d = self._read_data()
        return d
    
    def read(self):
        d = self.read_frame()
        return d

    def __enter__(self):
        return self

    def close(self):
        self._close_dat_files()

    def __exit__(self, type, value, tb):
        self.close()