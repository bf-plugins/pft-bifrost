import bifrost as bf
from copy import deepcopy
from datetime import datetime
from pprint import pprint
import time
from astropy.time import Time
import numpy as np

def _angle_str_to_sigproc(ang):
    aparts = ang.split(':')
    if len(aparts) == 2:
        sp_ang = int(aparts[0])*10000 + int(aparts[1])*100 + 0.0
    elif len(aparts) == 3:
        sp_ang = int(aparts[0])*10000 + int(aparts[1])*100 + float(aparts[2])
    else:
        raise RuntimeError("Cannot parse: {ang} does not match XX:YY:ZZ.zz".format(ang=ang))
    return sp_ang

def dada_dict_to_bf_dict(hdr):

    nant        = int(hdr['NANT'])
    nchan       = int(hdr['NCHAN'])
    nbit        = int(hdr['NBIT'])
    npol        = int(hdr['NPOL'])
    FS          = float(hdr['ADC_CLK'])
    STARTCHAN   = int(hdr['START_CHANNEL'])
    nsubband    = int(hdr['NSUBBAND'])
    nframe      = int(hdr['NFRAME'])
    
    nchan_per_sub = int(nchan / nsubband)

    db_size      = int(hdr['DADA_BUF_SIZE'])
    nheap        = int(db_size / int(hdr['RESOLUTION']) / 2)
    telescope_id = 4

    bw_chan      = float(hdr['CHAN_BW'])
    f_mhz0       = float(hdr['FREQ'])
    bw_subband   = nchan_per_sub * bw_chan
    ibeam        = 1
    nbeams       = 1
    barycentric  = 1
    is_complex  = True

    if nbit == 8:
        if is_complex:
            dtype = 'ci8'
        else:
            dtype = 'i8'
    if nbit == 32:
        if is_complex:
            dtype = 'cf32'
        else:
            dtype = 'f32'
    
    utc_str = hdr['UTC_START']
    utc_str = utc_str[:10] + " " + utc_str[11:]
    ts = Time(utc_str, format='iso').unix
    dt = float(1.0/bw_chan) * 1e-6
    name = 'bf_{ts}'.format(ts=Time(utc_str, format='iso').isot)

    h = {
        '_tensor': {'dtype':  dtype,
                     'shape':  [-1,     nheap,  nsubband,  nframe,  nchan_per_sub,  nant,      npol],
                     'labels': ['time', 'heap', 'subband', 'frame', 'freq', 'station', 'pol'],
                     'units':  ['s',    's',    'MHz',     's',     'MHz',  '',        ''],
                     'scales': [[ts, dt * nheap * nframe],
                                [0, dt * nframe],
                                [f_mhz0, bw_subband], 
                                [0, dt], 
                                [0, bw_chan],
                                [0,0], 
                                [0,0]]
                     },
         'name': name,
         'source_name':  hdr['SOURCE'],
         'telescope_id': telescope_id,       
         'raj': _angle_str_to_sigproc(hdr['RA']),
         'dej': _angle_str_to_sigproc(hdr['DEC']),
         'ibeam': ibeam,
         'nbeams': nbeams,
         'barycentric': barycentric,
    }

    for key, val in hdr.items():
        h[key] = val

    return h 
