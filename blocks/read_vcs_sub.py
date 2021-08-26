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

from loguru import logger

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
    fl = fl[:12]
 
    if len(fl) != n_coarse_chan:
        logger.warning("Warning: Number of coarse channels does not match number of dat files.")
    return fl


class MwaVcsReader(object):
    """ A file reader for MWA VCS data.
    
    Args:
        filename_or_filelist (str/list): Name of metafits file to open, or list of metafits files
                                         This will treat all of these as one 'sequence' and concatenate
                                         together as if one big stream.
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
        
        # Currently hardcoded values, 10,000 timesteps per file (one second)
        self.n_frame_per_obs = 100
        self.n_time_per_frame = 100
        
        self._open_next_obs()
        
        itensor = self.header['_tensor']
        
        self.dtype          = string2numpy(itensor['dtype'])
        self.frame_shape    = np.copy(itensor['shape'])
        self.frame_shape[0] = np.abs(self.frame_shape[0])
        self.frame_size = np.prod(self.frame_shape) * (self.dtype.itemsize)
        
        # Here 'gulp' is how much to read from a single file
        # Whereas 'frame' is how big the full array is (includes N_coarse_chan axis)
        self.dat_gulp_shape = list(self.frame_shape[2:])
        self.dat_gulp_size  = np.prod(self.dat_gulp_shape) * (self.dtype.itemsize // 2) # We will unpack 4-bit ci4 to 8-bit ci8
        
        # Create a temporary contiguous array to reuse.
        # Unpacks using numbits to int8, extra axis for real/imag
        self._data  = bf.ndarray(np.ndarray(shape=list(self.frame_shape[1:]), dtype=self.dtype), dtype='ci8')

    def _open_dat_files(self, idx):
        """ Internal method to open file handlers for dat file list"""
        # Generate a list of datafiles
        self.current_datlist = generate_filelist_from_metafits(self.obs_list[idx])
        
        # Create filehandlers
        self.current_datobjs = []
        for fn in self.current_datlist:
            self.current_datobjs.append(open(fn, 'rb'))
    
    def _close_dat_files(self):
        """ Internal method to close any open dat files """
        for fh in self.current_datobjs:
            fh.close()    
        self.frame_count = 0
            
    def _read_header(self):
        """ Read metafits header and convert to bifrost header.
        
        Specifically, need to generate the '_tensor' from the FITS header.
        Currently there are some hardcoded values -- could use header callback to avoid this.
        """
        
        mf_header = parse_metafits(self.obs_list[self.obs_count])
        N_coarse_chan = len(self.current_datlist)
        N_fine_chan   = 128
        N_station     = mf_header['NINPUTS'] // 2
        N_pol         = 2
        N_time        = self.n_time_per_frame
        
        t0 = Time(mf_header['GPSTIME'], format='gps').unix
        dt = 100e-6
        
        f0_coarse = mf_header['FREQCENT'] - mf_header['BANDWDTH'] / 2
        df_coarse = 1.28   # 1.28 MHz coarse channel 
        df_fine   = 0.01   # 10 kHz fine channel
           
        self.header = {
            '_tensor': {'dtype':  'ci8',  # Note raw data are 4-bit, but bifrost doesn't support ci4 so we use ci9=8
                         'shape':  [-1,     N_coarse_chan,    N_time,    N_fine_chan,    N_station, N_pol],
                         'labels': ['time', 'coarse_channel', 'frame',   'fine_channel', 'station', 'pol'],
                         'units':  ['s',    'MHz',             's',       'MHz',   '',     ''],
                         'scales': [[t0, dt * N_time],
                                    [f0_coarse, df_coarse], 
                                    [0, dt],
                                    [0, df_fine],
                                    [0,0], 
                                    [0,0]]
                         },
             'name': mf_header['FILENAME'] + '_'+ str(self.obs_count),
             'source_name': mf_header['FILENAME'], 
             'telescope': mf_header['TELESCOP'],       
             'ra': mf_header['RA'],
             'dec': mf_header['DEC'],
             'metafits': mf_header
        }
        return self.header
    
    def _open_next_obs(self):
        """ Internal method to open next observation """
        logger.info("%i/%i: opening %s" % (self.obs_count+1, self.n_obs, self.obs_list[self.obs_count]))
        if self.obs_count > 0:
            self._close_dat_files()
        self._open_dat_files(idx=self.obs_count)  
        self.header = self._read_header()
        self.obs_count += 1

    def _read_data(self):
        """ Internal method to read next data frame """
        
        if self.frame_count < self.n_frame_per_obs:
            for ii, fh in enumerate(self.current_datobjs):
                d_packed = np.fromfile(fh,  count=self.dat_gulp_size, dtype='int8')
                d_unpacked = numbits.unpack(d_packed, 4).view(self.dtype).reshape(self.dat_gulp_shape)
                self._data[ii] = d_unpacked
            self.frame_count += 1
            return self._data
        else:
            return np.ndarray(shape=0, dtype=self.dtype)
    
    def read_frame(self):
        """ Read next frame of data """
        logger.debug("Reading frame %i ..." % self.frame_count)
        d = self._read_data()
        if d.size == 0 and self.frame_count == self.n_frame_per_obs:
            if self.obs_count == self.n_obs:
                logger.info("End of file data stream")
                d = np.array([0]) # End of data stream
            else:
                logger.debug("Opening next observation")
                self._open_next_obs()
                d = self._read_data()
        return d
    
    def read(self):
        d = self.read_frame()
        return d

    def close(self):
        """ Close all open files """
        self._close_dat_files()
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

        
class MwaVcsReadBlock(bfp.SourceBlock):
    def __init__(self, filelist, gulp_nframe=1,  *args, **kwargs):
        super(MwaVcsReadBlock, self).__init__(filelist, gulp_nframe, *args, **kwargs)

    def create_reader(self, filename):
        logger.info(f"Reading {filename}...")
        return MwaVcsReader(filename)

    def on_sequence(self, ireader, filename):
        ohdr = ireader.header
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()
        odata  = ospans[0].data
        logger.debug("MWA VCS reader on_data called, reading data block")
        if np.prod(indata.shape) == np.prod(odata.shape[1:]):
            ospans[0].data[0] = indata
            return [1]
        else:
            # EOF or truncated block
            return [0]


def read_vcs_block(filename, *args, **kwargs):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline
    Args:
        filenames (list): A list of filenames to open
    """
    return MwaVcsReadBlock(filename, *args, **kwargs)
