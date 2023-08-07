from astropy.time import Time
from astropy.io import fits as pf
import glob
import os
import numpy as np
import time
import bifrost as bf
import bifrost.pipeline as bfp
from bifrost.dtype import string2numpy

from pymwalib.metafits_context import MetafitsContext
from pymwalib.voltage_context import VoltageContext

from loguru import logger

# https://wiki.mwatelescope.org/display/MP/MWA+High+Time+Resolution+Voltage+Capture+System


class MwaVcsReader(object):
    """ A file reader for MWA VCS data.
    
    Args:
        filename_or_filelist (str/list): Name of metafits file to open, or list of metafits files
                                         This will treat all of these as one 'sequence' and concatenate
                                         together as if one big stream.
    """
    def __init__(self, metafits_filename, sub_filelist, coarse_chan=0):
        
        super(MwaVcsReader, self).__init__()
        
        
        self.vcs   = VoltageContext(metafits_filename, sub_filelist)
        self.meta  = self.vcs.metafits_context
        self.coarse_channel_idx = coarse_chan

        vcs = self.vcs
        obs_len = vcs.common_duration_ms / 1e3
        self.N_frame    = int(obs_len)  # 1 second frames
 
        self.N_coarse_chan     = 1 #self.vcs.num_coarse_chans     # Only load one coarse channel at a time
        self.N_station         = self.vcs.metafits_context.num_ants
        self.N_pol             = self.vcs.metafits_context.num_ant_pols
        self.N_samp            = self.vcs.num_samples_per_voltage_block
        self.N_block           = vcs.num_voltage_blocks_per_second
        self.N_cplx            = 2

        self._data_read_shape = (self.N_block, self.N_coarse_chan, self.N_station, self.N_pol, self.N_samp, 2)
    
        self.t0_gps = int(self.vcs.timesteps[0].gps_time_ms / 1e3)
        
        self.obs_count = 0
        self.frame_count = 0

        
        # Currently hardcoded values
        # From https://wiki.mwatelescope.org/display/MP/MWA+High+Time+Resolution+Voltage+Capture+System
        self.n_block_per_obs  = vcs.num_voltage_blocks_per_timestep
        self.n_samp_per_block = vcs.num_samples_per_voltage_block
        self.coarse_chan_bw   = vcs.coarse_chan_width_hz

        self._create_header()
        
        itensor = self.header['_tensor']
        
        self.dtype          = string2numpy(itensor['dtype'])
        self.frame_shape    = np.copy(itensor['shape'])
        self.frame_shape[0] = np.abs(self.frame_shape[0])
        self.frame_size = np.prod(self.frame_shape) * (self.dtype.itemsize)
        
        # Here 'gulp' is how much to read from a single file
        # Whereas 'frame' is how big the full array is (includes N_coarse_chan axis)
        self.sub_gulp_shape = list(self.frame_shape[2:])
        self.sub_gulp_size  = np.prod(self.sub_gulp_shape) * (self.dtype.itemsize) 
        
        # Create a temporary contiguous array to reuse
        self._data  = bf.ndarray(np.ndarray(shape=list(self.frame_shape[1:]), dtype=self.dtype), dtype='ci8')

            
    def _create_header(self):
        """ Read metafits header and convert to bifrost header.
        
        Specifically, need to generate the '_tensor' from the FITS header.
        Currently there are some hardcoded values -- could use header callback to avoid this.
        """
        
        # Setup start time and sample period        
        t0 = self.vcs.common_start_unix_time_ms / 1e3
        dt = 1/ self.vcs.coarse_chan_width_hz
        
        # Setup start frequency and coarse channel step
        f0_coarse = (self.meta.centre_freq_hz - self.meta.coarse_chan_width_hz / 2) / 1e6 # in MHz
        df_coarse = self.meta.coarse_chan_width_hz / 1e6

        #(Nblock,  Nchan, Nant,  Npol, Nsamp, Ncplx)
        self.header = {
            '_tensor': {'dtype':  'ci8',  # Note raw data are 5-bit padded to 8-bit
                         'shape':  [-1,      self.N_block, self.N_coarse_chan, self.N_station, self.N_pol, self.N_samp],  #N_cplx = 2  not needed for ci8 data
                         'labels': ['time', 'block',       'coarse_channel', 'station',   'pol',   'sample'],
                         'units':  ['s',    's',            'MHz',             's',        '',      's'],
                         'scales': [[t0, dt * self.N_samp * self.N_block],
                                    [0, dt * self.N_samp],
                                    [f0_coarse, df_coarse], 
                                    [0, 1],
                                    [0, 1], 
                                    [0, dt]]
                         },
             'name': self.meta.obs_name + '_'+ str(self.obs_count),
             'source_name': self.meta.obs_name, 
             'telescope': 'MWA',       
             'ra': self.meta.ra_phase_center_deg,
             'dec': self.meta.dec_phase_center_deg
        }
        return self.header
    
    def read_frame(self):
        """ Read next frame of data """
        logger.debug(f"Reading frame {self.frame_count + 1}  / {self.N_frame}...")
        if self.frame_count < self.N_frame:
            d = self.vcs.read_second(self.t0_gps + self.frame_count, 1, self.coarse_channel_idx)
            d = bf.ndarray(d.reshape(self._data_read_shape)).view('ci8').reshape(self._data_read_shape[:-1]) 
            self.frame_count += 1
        if self.frame_count == self.N_frame:
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
        #self._close_sub_files()
        pass
        
    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()

        
class MwaVcsReadBlock(bfp.SourceBlock):
    def __init__(self, filelist, coarse_chan, gulp_nframe=1, *args, **kwargs):
        super(MwaVcsReadBlock, self).__init__(filelist, gulp_nframe, *args, **kwargs)
        self.coarse_chan = coarse_chan

    def create_reader(self, filelist):
        metafits_filename, sub_filelist = filelist[0], filelist[1]
        return MwaVcsReader(metafits_filename, sub_filelist, self.coarse_chan)

    def on_sequence(self, ireader, filename):
        ohdr = ireader.header
        return [ohdr]

    def on_data(self, reader, ospans):
        indata = reader.read()
        odata  = ospans[0].data
        logger.debug("MWA VCS reader on_data called, reading data block")
        #logger.debug(f"Input data: {indata.shape} {indata.dtype}, Output: {odata.shape} {odata.dtype}")
        if np.prod(indata.shape) == np.prod(odata.shape[1:]):
            ospans[0].data[0] = indata
            return [1]
        else:
            # EOF or truncated block
            return [0]


def read_vcs_block(filelist, coarse_chan, *args, **kwargs):
    """ Block for reading binary data from file and streaming it into a bifrost pipeline
    Args:
        filelist (list): A list of metafits + corresponding files to open. Should be like
            [
                [metafits_filename_0: str, list_of_files_0: list],
                [metafits_filename_1: str, list_of_files_1: list],
            ]
        coarse_chan (int): Coarse channel ID to read
    """
    return MwaVcsReadBlock(filelist, coarse_chan, *args, **kwargs)