from astropy.time import Time
from astropy.io import fits as pf
import glob
import os
import numpy as np
import time
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import string2numpy


from loguru import logger

# https://wiki.mwatelescope.org/display/MP/MWA+High+Time+Resolution+Voltage+Capture+System


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


def generate_filelist_from_metafits(filename, start_chan=None, stop_chan=None):
    """ Generate a list of data files from metafits file 
    
    Args:
        filename (str): Path to metafits file
    
    Returns:
        filelist (list): list of sub files corresponding to metafits file. 
    
    """
    hdr = parse_metafits(filename)
    n_coarse_chan = len(hdr['CHANNELS'].split(','))
    # Seems some files are called _metafits.fits and some are called .metafits
    file_start_timestamp = os.path.basename(filename).split('.')[0]
    dir_root             = os.path.dirname(filename)
    
    file_search_base  = f"{dir_root}/{file_start_timestamp}_{file_start_timestamp}_*.sub"
    logger.debug(file_search_base)
    fl = sorted(glob.glob(file_search_base))
    
    fl_sel = []
    for fn in fl:
        file_chan = int(os.path.basename(fn).split('_')[-1].replace('.sub', ''))

        add_file = True
        if start_chan is not None:
            if file_chan < start_chan:
                add_file = False
        if stop_chan is not None:
            if file_chan > stop_chan:
                add_file = False
        if add_file:
            fl_sel.append(fn)
    
    fl = fl_sel
    
    if len(fl) != n_coarse_chan:
        logger.warning("Warning: Number of coarse channels does not match number of sub files.")
    return fl


class MwaVcsReader(object):
    """ A file reader for MWA VCS data.
    
    Args:
        filename_or_filelist (str/list): Name of metafits file to open, or list of metafits files
                                         This will treat all of these as one 'sequence' and concatenate
                                         together as if one big stream.
    """
    def __init__(self, filename_or_filelist, start_chan=None, stop_chan=None):
        
        super(MwaVcsReader, self).__init__()
        
        if isinstance(filename_or_filelist, str):
            self.obs_list = [filename_or_filelist,]
            self.n_obs    = 1
        else:
            self.obs_list = filename_or_filelist
            self.n_obs    = len(filename_or_filelist)
        
        self.obs_count = 0
        self.frame_count = 0

        self.start_chan = start_chan
        self.stop_chan  = stop_chan
        
        # Currently hardcoded values
        # From https://wiki.mwatelescope.org/display/MP/MWA+High+Time+Resolution+Voltage+Capture+System
        self.n_block_per_obs  = 160
        self.n_samp_per_block = 64000
        self.coarse_chan_bw   = 1.28e6

        self.subfile_hdr_len  = 4096
        self.subfile_delay_md_len  = 0  # This needs to be computed with 2 * 128000 * N_station
        self.subfile_delay_md_base = 2 * 128000 

        self._open_next_obs()
        
        itensor = self.header['_tensor']
        
        self.dtype          = string2numpy(itensor['dtype'])
        self.frame_shape    = np.copy(itensor['shape'])
        self.frame_shape[0] = np.abs(self.frame_shape[0])
        self.frame_size = np.prod(self.frame_shape) * (self.dtype.itemsize)
        
        # Here 'gulp' is how much to read from a single file
        # Whereas 'frame' is how big the full array is (includes N_coarse_chan axis)
        self.sub_gulp_shape = list(self.frame_shape[2:])
        self.sub_gulp_size  = np.prod(self.sub_gulp_shape) * (self.dtype.itemsize) 
        
        # Create a temporary contiguous array to reuse.
        # Unpacks using numbits to int8, extra axis for real/imag
        self._data  = bf.ndarray(np.ndarray(shape=list(self.frame_shape[1:]), dtype=self.dtype), dtype='ci8')

    def _open_sub_files(self, idx):
        """ Internal method to open file handlers for sub file list"""
        # Generate a list of datafiles
        self.current_sublist = generate_filelist_from_metafits(self.obs_list[idx], self.start_chan, self.stop_chan)
        
        # Create filehandlers
        self.current_subobjs = []
        for fn in self.current_sublist:
            sub_fh = open(fn, 'rb')
            sub_fh.seek(self.subfile_hdr_len + self.subfile_delay_md_len) # Skip headers and delay metadata
            self.current_subobjs.append(sub_fh)
    
    def _close_sub_files(self):
        """ Internal method to close any open sub files """
        for fh in self.current_subobjs:
            fh.close()    
        self.frame_count = 0
            
    def _read_header(self):
        """ Read metafits header and convert to bifrost header.
        
        Specifically, need to generate the '_tensor' from the FITS header.
        Currently there are some hardcoded values -- could use header callback to avoid this.
        """
        
        mf_header = parse_metafits(self.obs_list[self.obs_count])
        N_coarse_chan = len(self.current_sublist)
        N_station     = mf_header['NINPUTS'] // 2
        N_pol         = 2
        N_samp        = self.n_samp_per_block

        # Setup start time and sample period        
        t0 = Time(mf_header['GPSTIME'], format='gps').unix
        dt = 1/ self.coarse_chan_bw
        
        # Setup start frequency and coarse channel step
        f0_coarse = mf_header['FREQCENT'] - mf_header['BANDWDTH'] / 2
        df_coarse = self.coarse_chan_bw / 1e6
        
        # update delay md calc
        self.subfile_delay_md_len = self.subfile_delay_md_base * N_station 

        #(Nblock,  Nant,  Npol, Nsamp, Ncplx)
        self.header = {
            '_tensor': {'dtype':  'ci8',  # Note raw data are 5-bit padded too 8-bit
                         'shape':  [-1,      N_coarse_chan,    N_station,   N_pol,   N_samp],  #N_cplx = 2  not needed for ci8 data
                         'labels': ['time', 'coarse_channel', 'station',   'pol',   'sample'],
                         'units':  ['s',    'MHz',             's',        '',      's'],
                         'scales': [[t0, dt * N_samp],
                                    [f0_coarse, df_coarse], 
                                    [0, 1],
                                    [0, 1], 
                                    [0, dt]]
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
            self._close_sub_files()
        self._open_sub_files(idx=self.obs_count)  
        self.header = self._read_header()
        self.obs_count += 1

    def _read_data(self):
        """ Internal method to read next data frame """
        
        if self.frame_count < self.n_block_per_obs:
            for ii, fh in enumerate(self.current_subobjs):
                d = np.fromfile(fh,  count=self.sub_gulp_size, dtype='int8').view(self.dtype).reshape(self.sub_gulp_shape)
                self._data[ii] = d
            self.frame_count += 1
            return self._data
        else:
            return np.ndarray(shape=0, dtype=self.dtype)
    
    def read_frame(self):
        """ Read next frame of data """
        logger.debug("Reading frame %i ..." % self.frame_count)
        d = self._read_data()
        if d.size == 0 and self.frame_count == self.n_block_per_obs:
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
        self._close_sub_files()
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

        
class MwaVcsReadBlock(bfp.SourceBlock):
    def __init__(self, filelist, gulp_nframe=1, start_chan=None, stop_chan=None, *args, **kwargs):
        super(MwaVcsReadBlock, self).__init__(filelist, gulp_nframe, *args, **kwargs)
        self.start_chan = start_chan
        self.stop_chan  = stop_chan

    def create_reader(self, filename):
        logger.info(f"Reading {filename}...")
        return MwaVcsReader(filename, start_chan=self.start_chan, stop_chan=self.stop_chan)

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