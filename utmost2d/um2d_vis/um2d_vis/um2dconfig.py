import numpy as np
import os

def read_antenna_positions(filename):
    """ Read antenna positions from file
    
    Args: filename
    Returns numpy array XYZ antenna positions, plus antenna name 
    """
    d = np.genfromtxt(filename, skip_header=1, 
                     dtype={'names': ('X', 'Y', 'Z', 'ANT_ID', 'DESC_NAME'), 'formats': (float, float, float, '<U8', '<U8')})
    return d

def get_cassette_dictionary(filename):
    """ Read antenna positions from file
    
    Args: filename
    Returns dictionary keyed on cassette name, containing xyz position
    """
    d = np.genfromtxt(filename, skip_header=1,
                     dtype={'names': ('X', 'Y', 'Z', 'ANT_ID', 'DESC_NAME'), 'formats': (float, float, float, '<U8', '<U8')})
    toreturn = {}
    for row in d:
        toreturn[row["ANT_ID"]] = [row["X"], row["Y"], row["Z"]]
    return toreturn


def read_snap_mapping(filename):
    """ Read SNAP mapping from file
    Args: filename
    Returns array of (SNAP_ID, ADC_INPUT, CASSETTE_NAME)
    """
    d = np.genfromtxt(filename, dtype={'names': ('SNAP_ID', 'ADC_INPUT', 'ANT_ID'), 'formats':(int, int, '<U8')})
    return d

def get_antsnapmap(antposfile, snapmapfile):
    """ Read antenna positions and snap mapping
    Creates a snap--antenna mapping to use to apply delays
    
    Args:
        antposfile (str): Path to antenna positions file
        snapmapfile (str): Path to SNAP mapping file

    Returns a dictionary mapping SNAP inputs to antennas
    """
    ant_pos  = read_antenna_positions(antposfile)
    ant_ids  = ant_pos['ANT_ID']
    antennas = np.column_stack((ant_pos['X'], ant_pos['Y'], ant_pos['Z'])) 
    
    sm = read_snap_mapping(snapmapfile)
    antsnapmap = {}
      
    for ii, ant_id in enumerate(ant_ids):
        try:
            s_hp = sm[sm['ANT_ID'] == ant_id + 'H'][0]            
            s_vp = sm[sm['ANT_ID'] == ant_id + 'V'][0]
            antsnapmap[ant_id + 'H'] = (s_hp['SNAP_ID'], s_hp['ADC_INPUT'])
            antsnapmap[ant_id + 'V'] = (s_vp['SNAP_ID'], s_vp['ADC_INPUT'])
        except IndexError:
            #print(f"Warning: cannot find {ant_id} in snap mapping file {snapmapfile}")
            print("Warning: cannot find {ant_id} in snap mapping file {0}".format(snapmapfile))

    return antsnapmap

def get_delays(delayfile):
    """ Read a delay file and store in an array indexed by SNAP id

    Args:
         delayfile (str): Path to the delay file
 
    Returns: numpy array of dimensions max(SNAP_ID), 12, containing the cable delays for the 12 SNAP inputs of each SNAP
    """
    if not os.path.exists(delayfile):
        raise RuntimeError("Delay file " + delayfile + " doesn't exist")
    d = np.loadtxt(delayfile) 
    print(d.size, d[0])
    numsnaps = int(max(d[:,0])) + 1
    print(numsnaps)
    numinputs = 12
    delays = np.zeros(numsnaps * numinputs).reshape(numsnaps, numinputs)
    for row in d:
        delays[int(row[0])] = np.copy(row[1:])
    return delays

def get_bandpass(bandpassfile):
    """ Read a bandpass file and store in an basic array (Nchan, Nant)

    Args:
         bandpassfile (str): Path to the bandpass file
 
    Returns: numpy array of dimensions Nchan, Nant, containing the voltage scale that should be divided out for each input
    """
    if not os.path.exists(bandpassfile):
        raise RuntimeError("Bandpass file " + bandpassfile + " doesn't exist")
    b = np.genfromtxt(bandpassfile, delimiter=',', autostrip=True)
    b = b[:,1:]
    b[b == 0] = 9e9
    b = b/20.0 # Do a rough scaling
    return b

def get_flagged_inputs(flaggedinputsfile):
    """ Read the list of flagged inputs from the configuration file

    Args:
        flaggedinputsfile (str): Path to the flagged inputs file

    Returns: map of inputs that are flagged as invalid
    """
    b = np.genfromtxt(flaggedinputsfile, delimiter=',', autostrip=True, dtype=int).tolist()
    return b

def get_recorded_snaps(recordedsnapsfile):
    """ Read the list of recorded snaps from the configuration file

    Args:
        recordedsnapsfile (str): Path to the flagged inputs file

    Returns: map of inputs that are flagged as invalid
    """
    b = np.genfromtxt(recordedsnapsfile, delimiter=',', autostrip=True, dtype=int).tolist()
    return b
