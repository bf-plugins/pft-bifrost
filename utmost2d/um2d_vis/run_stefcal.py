#!/home/npsr/miniconda2/envs/aoflagger_all/bin/python
"""
run_stefcal.py
==============
Apply stefcal to UTMOST-2D visibility data.

Basic gist:
    * Load observed visibilities
    * Generate sky model, then simulate model visibilities
    * Apply stefcal to solve V_obs = G^H V_model G
    * Save and plot complex gain solutions

"""

import numpy as np
from numpy import sin
import pylab as plt
import time
from copy import deepcopy

# Ignore irritating warnings
import warnings
from matplotlib.cbook import mplDeprecation

from um2d_vis.source_db import source_db

warnings.filterwarnings("ignore",category=mplDeprecation)


from astropy.time import Time, TimeDelta
from astropy import units as u
import astropy.constants as c

from um2d_vis.stefcal import stefcal
from um2d_vis.ant_array import make_antenna_array
from um2d_vis.sky_model import make_sky_model
from um2d_vis.visibility import Visibility
from um2d_vis.um2dconfig import read_antenna_positions, get_cassette_dictionary, read_snap_mapping, get_antsnapmap
from um2d_vis.plotting import plot_matrix, plot_amp_phs, plot_average_bandpass

LIGHT_SPEED = c.c.value

def make_metadata(v):
    """ Generate metadata dictionary 
    
    Args:
        v (um2d_vis.Visibility): UTMOST-2D visibility object
    
    Returns:
        md (dict): Metadata dictionary, with keys (time, freq, source_name, nant, nchan).
                   time is an astropy.Time array, and freq is an astropy Quantity array.
    """
    freq_axis = v.header['labels'].index('freq')
    time_axis = v.header['labels'].index('time')
    antj_axis = v.header['labels'].index('station_j')

    f0, fd = v.header['scales'][freq_axis]
    _t0, td = v.header['scales'][time_axis]
    t0 = Time(v.header['mjd_start'], format='mjd')
    
    nt, nf = v.data.shape[time_axis], v.data.shape[freq_axis]

    t = t0 + TimeDelta(np.arange(nt) * td, format='sec')
    f = u.Quantity(f0 + np.arange(nf) * fd, unit=v.header['units'][freq_axis]) 
    #f = u.Quantity(f0 + 2*fd + np.arange(nf) * fd, unit=v.header['units'][freq_axis]) #WAR: MONKEY PATCH FREQ AXIS
    #f = 900 * u.MHz - (f - 800 * u.MHz)  # WAR: MONKEY PATCH FREQ AXIS

    nchan, nant = v.data.shape[freq_axis], v.data.shape[antj_axis]
    return {'time': t, 'freq': f, 'nant': nant, 'nchan': nchan, 'source_name': v.header['source_name']}


def derive_tracking_stats(tracking_vis_matrix, obs_metadata, activeinputs):
    """ Do some averaging on the source tracking visibilities and get stats on the amplitudes

    Args:
        tracking_vis_matrix (array): The visibilities, rotated to track the source
        obs_metadata (dict): Observation metadata, from make_metadata() function
        activeinputs (list): List of cassette names
    """

    NCHAN = len(obs_metadata['freq'])
    NANT  = tracking_vis_matrix.shape[-1]
    averaged_vis_matrix = np.abs(np.mean(np.reshape(np.transpose(tracking_vis_matrix), (NANT, NANT, -1, 2)), axis=3))
    nonzero = averaged_vis_matrix[np.nonzero(averaged_vis_matrix)]
    medianamplitude = np.median(nonzero)
    for a in range(NANT):
      badbaselinecount = 0
      for aj in range(NANT):
        if np.median(averaged_vis_matrix[a,aj]) > 1.5*medianamplitude:
          badbaselinecount += 1
      if badbaselinecount > 0:
        print("Antenna",activeinputs[a],"had",badbaselinecount,"potentially problematic baselines")

def simulate_visibilities(ant_arr, sky_model, obs_metadata, applieddelays):
    """ Simulate model visibilities for an antenna array 
    
    Args:
        ant_arr (AntArray): Antenna array to use
        sky_model (sky_model.SkyModel): Sky model to use
        obs_metadata (dict): Observation metadata, from make_metadata() function
        applieddelays (np.array): Delays that were already applied pre-correlation in sec (Nant)

    Returns:
        model_vis_matrix (np.array): Model visibilities that should be expected given the known applied delays, (Nchan, Nant, Nant)
    """
    NCHAN = len(obs_metadata['freq'])
    NANT  = len(ant_arr.antennas)
    model_vis_matrix = np.zeros((NCHAN, NANT, NANT), dtype='complex128')

    # Generate XYZ phase offsets
    w_ang = 2 * np.pi * obs_metadata['freq'].to('Hz').value

    print("WARNING WARNING WARNING need to check the sign of the applieddelay correction!")
    for idx, (i, j) in enumerate(ant_arr.baselines):
        p = np.zeros_like(w_ang, dtype='complex128')
        for src in sky_model.sources:
            #print("Baseline xyz:",ant_arr.xyz[idx, 0])
            #print("Src details:",src.alt, src.mag)
            t_g = (ant_arr.xyz[idx, 0] * sin(np.pi/2 - src.alt)  * np.cos(src.az) + ant_arr.xyz[idx, 2] * sin(src.alt) ) / LIGHT_SPEED
            tau = t_g - (applieddelays[i-1] - applieddelays[j-1])
            p   += np.exp(1j * w_ang *  tau) * src.mag
        model_vis_matrix[:, i-1, j-1] = p
        model_vis_matrix[:, j-1, i-1] = np.conj(p)
    return model_vis_matrix


def compute_gains(obs_vis_matrix, model_vis_matrix, obs_metadata, numprocesses=8, max_iter=100, verbose=False):
    """ Compute per-channel gain calibration solutions using stefcal algorithm
    
    Args:
        obs_vis_matrix (np.array):   Observed visibilites (Nchan, Nant, Nant)
        model_vis_matrix (np.array): Simulated Model visibilities (Nchan, Nant, Nant)
        obs_metadata (dict): Observation metadata, from make_metadata() function
        num_processes (int): Number of parallel threads to use
        max_iter (int): Maximum number of iterations of stefcal, default 100
        verbose (bool): Print verbose info if True

    Returns:
        gain_matrix (np.array), converged (np.array): Gain calibration solutions (Nchan, Nant) and
                                                      bool mask if they converged (likely bad if not)
    """
    gain_matrix = np.zeros((obs_metadata['nchan'], obs_metadata['nant']), dtype='complex128')
    n_iter      = np.zeros(obs_metadata['nchan'], dtype='int32')    

    p = multiprocessing.Pool(numprocesses)
    multilist = []
    for i in range(obs_metadata['nchan']):
        chanfreq = obs_metadata['freq'].to('MHz').value[i]
        multilist.append((obs_vis_matrix[i,:,:], model_vis_matrix[i,:,:], obs_metadata['nant'], chanfreq, max_iter, verbose))

    r = np.array(p.starmap(compute_channel_gain, multilist))
    gains = np.array([i[0] for i in r])
    converged = np.array([i[1] for i in r])
    return gains, converged

def compute_channel_gain(obs_vis_channel, model_vis_channel, numant, freq,  max_iter=100, verbose=False):
    """
    Args:
        obs_vis_matrix (np.array):   Observed visibilites (Nant, Nant)
        model_vis_matrix (np.array): Simulated Model visibilities (Nant, Nant)
        numant (int): The number of antennas in the visibilities
        freq (float): The frequency of this channel (purely for verbose printing purposes)
        max_iter (int): Maximum number of iterations of stefcal, default 100
        verbose (bool): Print verbose info if True

    Returns:
        gain_matrix (np.array), converged (bool): Gain calibration solutions (Nant) and
                                                  bool for if they converged (likely bad if not)
    """
    gstart = np.ones(numant, dtype='complex128')
    if np.sum(np.abs(obs_vis_channel)) == 0: # channel was flagged, can skip
        g = gstart*np.NaN
        converged = False
    else:
        g, nit, dg = stefcal(obs_vis_channel, model_vis_channel, tol=1.0e-7, niter=max_iter, gstart=gstart)
        if verbose:
            #print(f"f {freq:2.2f}MHz N_iter {nit} dG {dg:2.5f}")
            print("f {0:2.2f}MHz N_iter {1} dG {2:2.5f}".format(freq, nit, dg))
        converged = nit < max_iter
    return g, converged

def apply_gains(vis_matrix, gains):
    """ Apply gains to visibility matrix 
    
    Args:
        vis_matrix (np.array): Visibility matrix to calibrate, (Nchan, Nant, Nant)
        gains (np.array): Gain solutions, (Nchan, Nant)
    
    Returns:
        cal_vis_matrix (np.array): Calibrated visibility matrix, (Nchan, Nant, Nant)
    """
    cal_vis_matrix   = np.zeros_like(vis_matrix)
    nant = vis_matrix.shape[-1]
    for ii in range(nant): 
        for jj in range(nant): 
            g = np.conj(gains[:, ii]) * gains[:, jj]
            cal_vis_matrix[:, ii, jj] = obs_vis_matrix[:, ii, jj] / g   
    return cal_vis_matrix

def apply_delays(vis_matrix, delays, ant_arr, sky_model, obs_metadata):
    """ Apply gains to visibility matrix 
    
    Args:
        vis_matrix (np.array): Visibility matrix to calibrate, (Nchan, Nant, Nant)
        delays (np.array): Calibration delays, (Nant)
        sky_model (sky_model.SkyModel): Sky model to use
        obs_metadata (dict): Observation metadata, from make_metadata() function

    Returns:
        cal_vis_matrix (np.array): Calibrated visibility matrix, (Nchan, Nant, Nant)
    """
    cal_vis_matrix   = np.zeros_like(vis_matrix)
    nant = vis_matrix.shape[-1]
    w_ang = 2 * np.pi * obs_metadata['freq'].to('Hz').value
    for idx, (i, j) in enumerate(ant_arr.baselines):
        corr = np.zeros(vis_matrix.shape[0])
        t_g = (ant_arr.xyz[idx, 0] * sin(np.pi/2 - sky_model.sources[0].alt) * np.cos(sky_model.sources[0].az) + ant_arr.xyz[idx, 2] * np.sin(sky_model.sources[0].az)) / LIGHT_SPEED
        total_baseline_delay = t_g + (delays[i-1] - delays[j-1])
        corr = np.exp(-1j * w_ang *  total_baseline_delay)
        cal_vis_matrix[:, i-1, j-1] = vis_matrix[:,i-1,j-1]*corr
        cal_vis_matrix[:, j-1, i-1] = np.conj(cal_vis_matrix[:, i-1, j-1])
    return cal_vis_matrix

def rereference(gains, indices, refindex):
    """ References a subset of a gains array back to a given input

    Args:
        gains: The gains array (Nchan, Nant)
        indices: Reference this subset of antennas
        refindex: Reference to this index

    Returns:
        new_gains (np.array): Modified gains array (NChan, Nant)
    """
    new_gains = gains.copy()
    nant = gains.shape[-1]
    for ii in range(nant):
        if ii in indices:
            new_gains[:,ii] = new_gains[:,ii] * np.exp(-1j*np.angle(gains[:,refindex]))
    return new_gains

def delays2text(delays, recorded_snaps, outputfile):
    """ Writes delays out into a text file suitable for use in SNAPs/beamformer/correlator

    Args:
        delays (np.array): Array containing fitted delays (Nant)
        recorded_snaps: List of SNAP id in order
        outputfile: File to write to
    """

    outdata = np.zeros(13 * (max(recorded_snaps)+1)).reshape(max(recorded_snaps)+1, 13)
    for i, r in enumerate(recorded_snaps):
        np.copyto(outdata[r][1:], 1e9*delays[i*12:(i+1)*12])
    for i in range(max(recorded_snaps)+1):
        outdata[i][0] = i
    fmt = ["%d"]
    for i in range(12):
        fmt.append("%11.6f")
    with open(outputfile, "w") as output:
        headerstring = "SNAP_ID   D0  D1  D2  D3  D4  D5  D6  D7  D8  D9  D10 D11"
        np.savetxt(outputfile, outdata, header=headerstring, fmt=fmt)

def scales2text(scales, recorded_snaps, outputfile):
    """ Writes scales out into a text file suitable for use in SNAPs/beamformer/correlator

    Args:
        sca;es (np.array): Array containing fitted scales (Nant)
        recorded_snaps: List of SNAP id in order
        outputfile: File to write to
    """

    outdata = np.ones(13 * (max(recorded_snaps)+1)).reshape(max(recorded_snaps)+1, 13)
    for i, r in enumerate(recorded_snaps):
        np.copyto(outdata[r][1:], scales[i*12:(i+1)*12])
    for i in range(max(recorded_snaps)+1):
        outdata[i][0] = i
    fmt = ["%d"]
    for i in range(12):
        fmt.append("%8.3f")
    with open(outputfile, "w") as output:
        headerstring = "SNAP_ID   G0  G1  G2  G3  G4  G5  G6  G7  G8  G9  G10 G11"
        np.savetxt(outputfile, outdata, header=headerstring, fmt=fmt)


def testdelay(tofit, delay, freqs_mhz):
    """ Evaluates a trial delay, to see how good the fit is to a set of gains

    Args:
        tofit: The complex gains array (Nchan)
        delay: The delay to trial in seconds
        freqs_mhz: The frequencies of each channel in MHz

    Returns:
        signal: The sum of real component of gains after applying delay correction
    """
    phases = 2 * np.pi * delay * 1e6 * freqs_mhz
    correction = np.exp(1j*phases)
    return np.real(np.average(tofit * correction))

def estimate_sensitivity(tracking_vis, sky_model, chanrange, chanavg):
    """ Uses the measured visibilities and the sky model flux density to estimate sensitivity

    Args: 
        tracking_vis: The calibrated data, phased at the location of first source in the sky model
        sky_model: The sky model used
        chanrange: The channels to use to estimate sensitivity
        chanavg: The number of channels to average together

    Returns: Estimate of average system temperature for every baseline
    """
    nant = tracking_vis.shape[-1]
    baselinetemps = []
    for i in range(nant):
        for j in range(nant):
            baselineamps = np.abs(np.mean(tracking_vis[chanrange[0]:chanrange[1],i,j].reshape(-1, chanavg), axis=1))
            #print(baselineamps)
            if np.mean(baselineamps) > 0:
                # An SEFD of 25000 Jy corresponds to Tsys 100 K and 65% efficiency
                # Just assuming first source dominates for now, will do properly later
                baselinetemp = 100*(sky_model.sources[0].mag / (np.median(baselineamps)*25000))
                #print(baselinetemp)
                baselinetemps.append(baselinetemp)
    print("Median estimated system temperature (assuming 65 percent aperture efficiency) is", np.median(np.array(baselinetemps)), "K")
    print("Standard deviation was ", np.std(np.array(baselinetemps)), "K")
    return baselinetemps


def singleantenna2delay(antenna, gains, freqs_mhz, verbose=False, forcestartdelays=None):
    """ Fits delays to the phases from gain solutions for a single antenna

    Args:
        antenna: The antenna being solved for
        gains: The gains array for one antenna (Nchan)
        freqs_mhz: The frequencies of each channel in MHz
        verbose: print debug information if true
        forcestartdelays: array of coarse starting delays (Nant)

    Returns:
        delay: Float value of fitted delay for this antenna
    """
    ntrials = 12001
    bandwidth = (freqs_mhz[1] - freqs_mhz[0])*len(freqs_mhz)
    if verbose:
        print("Bandwidth (MHz): ", bandwidth, ", 1/bandwidth (s): ", 1/(bandwidth*1e6))

    phases = np.angle(gains)
    tofit = np.nan_to_num(np.exp(1j*phases))
    # Below is not actually needed, as nan_to_num takes care of it.
    #tofit[np.abs(gains) == 0] = 0 + 0j
    if not (forcestartdelays is None):
        coarsedelay = forcestartdelays[antenna]
    else:
        lags = np.abs(np.fft.ifft(tofit))
        if np.argmax(lags) > len(lags)//2:
            coarsedelay = (np.argmax(lags) - len(lags)) / (bandwidth*1e6)
        else:
            coarsedelay = np.argmax(lags) / (bandwidth*1e6)
    trialsnrs = np.zeros(ntrials)
    trialdelays = np.zeros(ntrials)
    multilist = []
    for jj, trialdelay in enumerate(np.linspace(coarsedelay - 1.5/(bandwidth*1e6), coarsedelay + 1.5/(bandwidth*1e6), ntrials)):
        trialsnrs[jj] = testdelay(tofit, trialdelay, freqs_mhz)
        trialdelays[jj] = trialdelay
    delay = trialdelays[np.argmax(trialsnrs)]
    if verbose:
        print("Best delay for {0} was {1}, after best coarse delay was {2}".format(antenna, delay, np.argmax(lags)))
    return delay

def gains2delays(gains, freqs_mhz, verbose=False, numprocesses=12, forcestartdelays=None):
    """ Fits delays to the phases from gain solutions. Runs in parallel.

    Args:
        gains: The gains array (Nchan, Nant)
        freqs_mhz: The frequencies of each channel in MHz
        verbose: print debug information if true
        numprocesses: The number of processes to run in parallel
        forcestartdelays: array of coarse starting delays (Nant)

    Returns:
        delays (np.array): Array containing fitted delays (Nant)
    """
    nant = gains.shape[-1]
    p = multiprocessing.Pool(numprocesses)
    multilist = []
    for ii in range(nant):
        multilist.append((ii, gains[:,ii], freqs_mhz, verbose, forcestartdelays))
    delays = np.array(p.starmap(singleantenna2delay, multilist))
    return delays
    #delays = np.zeros(nant)
    #for ii in range(nant):
    #    phases = np.angle(gains[:,ii])
    #    tofit = np.nan_to_num(np.exp(1j*phases))
    #    lags = np.abs(np.fft.ifft(tofit))
    #    if np.argmax(lags) > len(lags)//2:
    #        coarsedelay = (np.argmax(lags) - len(lags)) / (bandwidth*1e6)
    #    else:
    #        coarsedelay = np.argmax(lags) / (bandwidth*1e6)
    #    trialsnrs = np.zeros(ntrials)
    #    trialdelays = np.zeros(ntrials)
    #    multilist = []
    #    for jj, trialdelay in enumerate(np.linspace(coarsedelay - 1.5/(bandwidth*1e6), coarsedelay + 1.5/(bandwidth*1e6), ntrials)):
    #        trialsnrs[jj] = testdelay(tofit, trialdelay, freqs_mhz)
    #        trialdelays[jj] = trialdelay
    #    delays[ii] = trialdelays[np.argmax(trialsnrs)]
    #    if verbose:
    #        print("Best delay for {0} was {1}, after best coarse delay was {2}".format(ii, delays[ii], np.argmax(lags)))
    #return delays

def rotategains(gains, freqs_mhz, applieddelays):
    """ Fits delays to the phases from gain solutions

    Args:
        gains: The gains array (Nchan, Nant)
        freqs_mhz: The frequencies of each channel in MHz
        applieddelays (np.array): The delays that had been applied already prior to or during correlation

    Returns:
        rotatedgains (np.array): Rotated gains array (NChan, Nant)
    """
    return gains * np.exp(1j*np.outer(2*np.pi*freqs_mhz, np.transpose(applieddelays*1e6)))

def readdelays(visibilities, recorded_snaps, inputfile):
    """ returns an array that contains the delays already applied during or prior to the correlator

    Args:
        visibilities: the h5 file containing the visibilities and the applied SNAP delays in the metadata
        recorded_snaps (list): The indices of the SNAPs that have been recorded
        inputfile: the file to read in

    Returns:
        applieddelays (np.array): The delays that had been applied already by the SNAPs (Nant) in s
    """

    # Until the SNAP delays are stored in the the h5 files, this is a hack to read them in from a hard-coded file
    alldelays = np.loadtxt(inputfile)
    applieddelays = np.zeros(len(recorded_snaps)*12)
    for idx,r in enumerate(recorded_snaps):
        for i in range(alldelays.shape[0]):
            if int(alldelays[i][0]) == r:  # This is the matching row
                for j in range(12): # 12 inputs per SNAP
                    applieddelays[idx*12 + j] = alldelays[r][j+1] / 1e9
   
    return applieddelays


def printelapsedsince(lasttime, activitystring):
    """ Prints elapsed time since last time a checkpoint was done

    Args:
        lasttime: the last time something was printed
        activitystring: the activities that have been undertaken since last time

    Returns:
        newtime: the time right now
    """
    newtime = time.time()
    print("Time to", activitystring,":",newtime-lasttime,"seconds")
    return newtime

if __name__ == "__main__":

    import argparse, os, multiprocessing

    p = argparse.ArgumentParser(description='Compute complex gain solutions for UTMOST-2D')
    p.add_argument('filename', help='Name of visibility file(s) [multiple files will be concatenated in frequency]', nargs='+')
    p.add_argument('-o', '--outfile', default=None, help='Save gains to file', type=str)
    p.add_argument('--plot_gains',      action='store_true', help='plot complex gain solutions')
    p.add_argument('--plot_model',      action='store_true', help='Plot model visibilities')
    p.add_argument('--plot_observed',   action='store_true', help='Plot observed visibilities')
    p.add_argument('--plot_lags',       action='store_true', help='Plot lags from observed visibilities')
    p.add_argument('--plot_calibrated', action='store_true', help='Plot calibrated visibilities')
    p.add_argument('--plot_ratio',      action='store_true', help='Plot calibrated / model visibilities')
    p.add_argument('--plot_tracking',   action='store_true', help='Plot tracking visibilities (fringe stopped to sky position)')
    p.add_argument('--plot_aoflagger',  action='store_true', help='Plot the aoflagger waterfall plots')
    p.add_argument('--plot_all',        action='store_true', help='Plot all')
    p.add_argument('--plot_onlyreference',  action='store_true', help='Plot only baselines to the reference antenna')
    p.add_argument('--calibration_location', default=None, help='If set, rotate calibrated data to this position (otherwise use first source in the sky model')
    p.add_argument('--configdir',       default='/home/adeller/packages/src/utmost2d_snap/config/', help='Directory in which config files (antenna, snap etc) are kept')
    p.add_argument('--antposfile',      default='2020.08.27.txt', help='antenna position file name')
    p.add_argument('--snapmapfile',     default='2020.08.25.txt', help='snap mapping file name')
    p.add_argument('--recordedsnaps',   default='0,2', help='Comma-separated list of SNAP ids that were recorded')
    p.add_argument('--forcestartdelays',default='', help='Set to a filename containing starting coarse delays if desired')
    p.add_argument('--flagfreqfile',    default='', help='File with a list of frequencies to flag between')
    p.add_argument('--flaginputs',      default='', help='comma-separated list of input indices to flag')
    p.add_argument('--flagmodulespacingbelow',default=3, type=int, help="Flag baselines shorter than this many modules apart")
    p.add_argument('--zapphonebands',   default='', help="Comma-separated list of phone bands to zap (f1,f2,f3,f4 = 842.5/837.5/832.5/827.5 MHz central freqs)")
    p.add_argument('--dumpnumpy',       default=False, action='store_true', help='Write out the selected data as a numpy array')
    p.add_argument('--verbose',         action='store_true', help='Spew out an enormous amount of debugging info')
    p.add_argument('--referencecassette',default="M100C5", help="The cassette to set as having zero phases")
    p.add_argument('--targetaz',        default=180, type=float, help='Centre of azimuth range to include in solution (degrees, 180 is transit)')
    p.add_argument('--maxazoffset',     default=2, type=float, help='Maximum deviation of azimuth in degrees from targetaz to include in solution')
    p.add_argument('--useaoflagger',    default=False, action='store_true', help='Use aoflagger rather than frequency blacklist')
    
    # Parse arguments
    args = p.parse_args()

    # Check filenames
    for f in args.filename:
        if not os.path.exists(f):
             p.error("Input file " + f + "doesn't exist")

    # Get starting time
    lasttime = time.time()

    # Get the list of inputs to discard entirely
    if args.flaginputs.strip() == "":
        flaginputs = []
    else:
        flaginputs = [int(x) for x in args.flaginputs.rstrip(',').split(',')]
        flaginputs.sort()
    if args.verbose:
        print("Flaginputs:",flaginputs)

    # Load up the observed visibilities
    print("Loading observed visibilities...")
    v = Visibility(args.filename[0], flaginputs=flaginputs)
    if len(args.filename) > 1:
        for count, f in enumerate(args.filename[1:], 1):
            if args.verbose:
                print("Appending file",count,"/",len(args.filename[1:]))
            v.append_data_freq(f, flaginputs=flaginputs, verbose=args.verbose)
    v.determine_validity()
    lasttime = printelapsedsince(lasttime, "load visibilities")

    # Get antenna and SNAP info from config files
    antposfile = args.configdir + '/antenna_positions/' + args.antposfile
    snapmapfile = args.configdir + '/snap_mapping/' + args.snapmapfile
    recorded_snaps = [int(r) for r in args.recordedsnaps.split(',')]
    print("WARNING WARNING presently assuming that SNAP IDs", recorded_snaps, "are being recorded based on command line input, this needs to be read from h5 file once available")
    cassettepositions = get_cassette_dictionary(antposfile)
    snapmap = read_snap_mapping(snapmapfile)
    snapinputs = [""] * 12 * len(recorded_snaps)
    for row in snapmap:
        if row[0] in recorded_snaps:
            snapinputs[12*recorded_snaps.index(row[0]) + row[1]] = row[2]
    
    # Remove the wholly-flagged inputs
    activeinputs = deepcopy(snapinputs)
    for f in flaginputs:
        activeinputs.remove(snapinputs[f])

    # Get the polarisation info
    if args.verbose:
        print("SNAP inputs:", activeinputs)
    firstpol = activeinputs[0][-1]
    if firstpol == "H":
        secondpol = "V"
    else:
        secondpol = "H"

    # Set the reference input numbers
    firstrefindex = activeinputs.index(args.referencecassette + firstpol)
    secondrefindex = activeinputs.index(args.referencecassette + secondpol)
    if args.verbose:
        print("The two reference inputs are",firstrefindex,secondrefindex)
    if firstrefindex < 0 or secondrefindex < 0:
        print("Couldn't find the reference cassette",args.referencecassette,"- aborting")

    # Fill in the antennas list
    antennas = []
    for a in activeinputs:
        antennas.append(cassettepositions[a[:-1]])

    # Get the delays that were applied at the SNAP (np.array of length Nant)
    applieddelays = readdelays(v, recorded_snaps, "appliedsnapdelays.txt")
    #applieddelays = -1 * readdelays(v, recorded_snaps, "appliedsnapdelays.txt")
    activeapplieddelays = np.delete(applieddelays, flaginputs)

    # Metadata setup from h5 file header
    obs_metadata = make_metadata(v)
    if args.verbose:
        print("obs_metadata: number of antennas:", obs_metadata["nant"])
    freqs_mhz    = obs_metadata['freq'].to('MHz').value
    t_start      = obs_metadata['time'][0].to_datetime()
    t_end        = obs_metadata['time'][-1].to_datetime()
    if args.verbose:
        print("Frequency info:",freqs_mhz)

    # Observatory setup
    lat, lon, elev  = '-35:22:15', '149:25:26', 750
    um = make_antenna_array(lat, lon, elev, t_start, antennas)
    if args.verbose:
        um.report()

    # Sky model setup
    srclist   = source_db[obs_metadata['source_name']]
    sky_model = make_sky_model(srclist)
    sky_model.compute_ephemeris(um)
    if args.verbose:
        print("sky_model at beginning of observation:")
        sky_model.report()

    # Update the observatory date to be the end of the observation
    um.update(t_end)
    sky_model.compute_ephemeris(um)
    if args.verbose:
        print("sky_model at end of observation:")
        sky_model.report()
        print("Start and end time of observation:",t_start, t_end)

    # Decide what times we want to add into the sum
    azimuths = np.zeros(len(obs_metadata['time']))
    elevations =  np.zeros(len(obs_metadata['time']))
    for i, t in enumerate(obs_metadata['time']):
        if args.verbose:
            print(t.to_datetime())
        um.update(t.to_datetime())
        sky_model.compute_ephemeris(um)
        elevations[i], azimuths[i] = sky_model.getaltaz()
    if sky_model.sources[0].dec > um.lat:
        azimuths -= 180
        azimuths[np.where(azimuths < 0)] += 360
    excludetimes = np.abs(azimuths - args.targetaz) > args.maxazoffset
    maskedelevations = np.ma.array(elevations, mask=excludetimes)
    meanelevation = maskedelevations.mean()
    meanelevationindex = (np.abs(elevations-meanelevation)).argmin()
    referencetime = obs_metadata['time'][meanelevationindex].to_datetime()
    if args.verbose:
        print("Elevations:",elevations)
        print("Azimuths:", azimuths)
        print(excludetimes)
        print("Masked elevations:", maskedelevations)
        print("Masked elevations:", maskedelevations)
        print("Mean elevation index:",meanelevationindex)
        print(referencetime)
    um.update(referencetime)
    sky_model.compute_ephemeris(um)
    if args.verbose:
        print("sky_model for mean elevation:")
        sky_model.report()

    # Calculate the first and last time index that will be included in the calibration
    validtimes = np.where(excludetimes == False)[0]
    if len(validtimes) == 0:
        print("Azimuths:",azimuths)
        raise RuntimeError("No times close enough to targetaz ({0} +/- {1})  in this file! Find another file or change the az limits".format(args.targetaz, args.maxazoffset))
    firsttimeindex = validtimes[0]
    lasttimeindex = validtimes[-1]
    # Check against times that actually have non-zero data
    nonzerodataindices = np.where(v.valid == True)[0]
    nonzerostarttimeindex = nonzerodataindices[0]
    nonzeroendtimeindex = nonzerodataindices[-1]
    if nonzerostarttimeindex > firsttimeindex:
        print("Elevation limit is satisfied from", firsttimeindex, ", but non-zero data only available from", nonzerostarttimeindex, ", updating start time index")
        firsttimeindex = nonzerostarttimeindex
    if nonzeroendtimeindex < lasttimeindex:
        print("Elevation limit is satisfied until", lasttimeindex, ", but non-zero data only available until", nonzeroendtimeindex, ", updating last time index")
        lasttimeindex = nonzeroendtimeindex
    if args.verbose:
        print("First and last time index that will be included in calibration:", firsttimeindex, lasttimeindex)
    lasttime = printelapsedsince(lasttime, "set up antenna/source metadata")

    # Dump out the numpy array if requested
    if args.dumpnumpy:
        if args.verbose:
            print("Dumping numpy data: shape is ", v.data[firsttimeindex:lasttimeindex+1,:,:,:].shape)
        np.save("corrdata.npy",v.data[firsttimeindex:lasttimeindex+1,:,:,:])

    # Flag some known bad channels
    flagchans = []
    flagfreqs = []
    if args.useaoflagger:
        print("Running aoflagger")
        plotinput = -1
        if args.plot_aoflagger or args.plot_all:
            plotinput = firstrefindex
        v.aoflag(nonzerostarttimeindex, nonzeroendtimeindex, plotinput)

    if args.flagfreqfile=="":
        if not args.useaoflagger:
            flagfreqs.append([805.0, 809.0])
            flagfreqs.append([812.25, 812.4])
            flagfreqs.append([816.05, 816.65])
            flagfreqs.append([825.0, 829.8])
            flagfreqs.append([830.1, 838.0])
            flagfreqs.append([839.1, 844.8])
            flagfreqs.append([846.6, 846.9])
            flagfreqs.append([848.3, 848.5])
            flagfreqs.append([849.0, 849.9])
            flagfreqs.append([851.6, 859.9])
        else:
            flagfreqs.append([805,808.5])
            flagfreqs.append([851.5, 859.9])
    else:
        if os.path.exists(args.flagfreqfile):
            flagfreqlines = open(args.flagfreqfile).readlines()
            for line in flagfreqlines:
                if len(line) > 0 and not line[0] == "#":
                    line = line.split('#')[0]
                    if not len(line.split(',')) == 2:
                        print("Ignoring bad flag frequency line", line)
                    else:
                        flagfreqs.append(list(map(float, line.split(','))))
        elif args.flagfreqfile == "none":
            pass
        else:
            print("Received flag file",args.flagfreqfile,"which does not exist - aborting")
            sys.exit()
    for z in args.zapphonebands.split(','):
        if z == "f1" or z == "F1":
            flagfreqs.append([840.0, 845.0])
        elif z == "f2" or z == "F2":
            flagfreqs.append([835.0, 840.0])
        elif z == "f3" or z == "F3":
            flagfreqs.append([830.0, 835.0])
        elif z == "f4" or z == "F4":
            flagfreqs.append([825.0, 830.0])
        else:
            if not z == "":
                print("Ignoring unknown phone band",z)
    for i in range(v.data.shape[1]):
        for f in flagfreqs:
            if v.freqs[i].value > f[0] and v.freqs[i].value < f[1] and not i in flagchans:
                flagchans.append(i)
    if args.verbose:
        print("Flagging channels", flagchans)
        print("Channel 120, before flagging:",v.data.mean(axis=0)[120])
    v.flagchans(flagchans)
    if args.verbose:
        print("Channel 120, before normalisation:",v.data.mean(axis=0)[120])
        print("Channel 180 autocorr 1, all times:",v.data[:,180,1,1])
        print("Channel 60 autocorr 1, all times:",v.data[:,60,1,1])
        print("Channel 120 autocorr 1, all times:",v.data[:,120,1,1])
        print("Channel 120 autocorr 55, all times:",v.data[:,120,55,55])
        print("Channel 120 2-20, all times:",v.data[:,120,2,20])

    lasttime = printelapsedsince(lasttime, "flag frequency blacklist or aoflagger")
    ## Write out a scaling array 
    #scalearray = np.nanmean(np.abs(v.data[firsttimeindex:lasttimeindex+1]), axis=0)
    #scales = np.ones(v.data.shape[2])
    #scales[0] = 1.0
    #for i in range(1,v.data.shape[2]):
    #    gainratio = scalearray[:,i,i] / scalearray[:,0,0]
    #    if args.verbose:
    #        print("Average scale factor for antenna",i,"relative to 0 is ",np.median(gainratio[gainratio != 1.0]))
    #    scales[i] = 1./np.sqrt(np.median(gainratio[gainratio != 1.0]))
    #if args.verbose:
    #    print("Writing out scaling solutions to scales.txt")
    #scales2text(scales, recorded_snaps, "scales.txt")    

    # Normalise the data and average it in time (dropping the last integration in case it is partial
    v.normalise()
    obs_vis_matrix = np.nanmean(v.data[firsttimeindex:lasttimeindex], axis=0)
    if args.verbose:
        print("NaN indices:", np.argwhere(np.isnan(obs_vis_matrix)))
        print("Channel 120, after normalisation:",obs_vis_matrix[120])

    # Flag all the intra-cassette baselines
    flagbaselines = []
    for i, an1 in enumerate(activeinputs):
        for j, an2 in enumerate(activeinputs):
            if an1[:4] == an2[:4] and i != j:
                flagbaselines.append([i,j])
    #for i in range(v.data.shape[2]//12):
    #    for j in range(12):
    #        for k in range(12):
    #            if j != k:
    #                toappend = []
    #                toappend.append(i*12 + j)
    #                toappend.append(i*12 + k)
    #                flagbaselines.append(toappend)

    # And flag all cross-polarisation baselines
    for i, an1 in enumerate(activeinputs):
        for j, an2 in enumerate(activeinputs):
            if an1[-1] != an2[-1] and i != j:
                toappend = [i, j]
                #print("Flagging {0},{1} [{2},{3}] for cross pol".format(i,j,an1,an2))
                if not toappend in flagbaselines:
                    flagbaselines.append(toappend)
    #for i in range(v.data.shape[2]):
    #    for j in range(v.data.shape[2]):
    #        if i%2 != j%2:
    #            toappend = [i, j]
    #            if not toappend in flagbaselines:
    #                flagbaselines.append(toappend)

    # Flag all the too-short baselines
    for i,an1 in enumerate(activeinputs):
        for j,an2 in enumerate(activeinputs):
            if i == j: continue
            if abs(int(an1[1:-3]) - int(an2[1:-3])) < args.flagmodulespacingbelow: # Too short a baseline, remove
                toappend = [i, j]
                #print("Flagging {0},{1} [{2},{3}] for baseline length: ({4},{5})".format(i,j,an1,an2,an1[1:-3],an2[1:-3]))
                if not toappend in flagbaselines:
                    flagbaselines.append(toappend)

    # Flag any inputs that were specifically requested to be zapped
    #if args.flaginputs.strip() == "":
    #    flaginputs = []
    #else:
    #    flaginputs =  map(int, args.flaginputs.rstrip(',').split(','))

    ## THIS IS NOW TAKEN CARE OF AT DATA LOADING TIME
    #for i in range(v.data.shape[2]):
    #    for j in range(v.data.shape[2]):
    #        if (i in flaginputs or j in flaginputs) and not (i in flaginputs and j in flaginputs): # Leave baselines between bad inputs alone, to avoid NaNs everywhere
    #            toappend = [i, j]
    #            flagbaselines.append(toappend)

    # And finally the autocorrelations
    for i in range(v.data.shape[2]):
        toappend = [i, i]
        if not toappend in flagbaselines:
            flagbaselines.append(toappend)

    # Print the flagbaselines - what is left is what we actually want to use in stefcal
    if args.verbose:
        print("Flagbaselines:",flagbaselines)

    # If desired, make a plot of each autocorrelation amplitude
    #if args.verbose
    #    for i in range(v.data.shape[2]):
    #        print("Median autocorrelation for antenna",i,"was",np.median

    # Actually apply the flagging to obs_vis_matrix
    for fb in flagbaselines:
        obs_vis_matrix[:,fb[0],fb[1]] = 0.0j + 0
    if args.verbose:
        print("Channel 120, after normalisation and flagging:",obs_vis_matrix[120])
    lasttime = printelapsedsince(lasttime, "flag intra-module baselines, other too-short baselines, and cross-pol data")

    # Simulate model visibilities
    model_vis_matrix = simulate_visibilities(um, sky_model, obs_metadata, activeapplieddelays)
    if args.verbose:
        print("obs_vis_matrix size", obs_vis_matrix.size, "model_vis_matrix size", model_vis_matrix.size)

    # Flag the appropriate baselines in the model
    for fb in flagbaselines:
        model_vis_matrix[:,fb[0],fb[1]] = 0j + 0
    if args.verbose:
        print("Channel 120, after normalisation and flagging:",obs_vis_matrix[120])
    lasttime = printelapsedsince(lasttime, "simulate model visibilities")

    # Compute calibration solutions, which will include the fixed/cable delays
    print("Running stefcal...")
    gains, converged = compute_gains(obs_vis_matrix, model_vis_matrix, obs_metadata, numprocesses=8, max_iter=500, verbose=args.verbose)
    lasttime = printelapsedsince(lasttime, "runstefcal")
    if args.verbose:
        print("Observed matrix:",obs_vis_matrix[120][1][2])
        print("Model matrix:",model_vis_matrix[120][1][2])
        print("Gains:",gains[120][1])
        print("Converged:",converged[120])
        print("Gains shape (prior to flagged insertion", gains.shape)

    # Rereference the second polarisation to the first input that has that second polarisation
    newgains = rereference(gains, [i for i, x in enumerate(activeinputs) if x[-1] == secondpol], secondrefindex)
    newgains = rereference(newgains, [i for i, x in enumerate(activeinputs) if x[-1] == firstpol], firstrefindex)

    # Insert zeros for all the fully flagged inputs
    for f in flaginputs:
        newgains = np.insert(newgains, f, [1 + 0j], axis=1)

    # If requested, load in forced starting delays to search around
    if args.forcestartdelays != "":
        forcestartdelays = readdelays(v, recorded_snaps, args.forcestartdelays)
    else:
        forcestartdelays = None

    # Solve for delays
    numparallelprocesses = 12
    delays = gains2delays(newgains, freqs_mhz, args.verbose, numparallelprocesses, forcestartdelays)

    # Rotate the gains to account for the already-applied delays (to give a correction matrix that can be applied to the observed visibilities)
    newrotatedgains = rotategains(newgains, freqs_mhz, applieddelays)

    # Print out the average gain per antenna if verbose is on
    if args.verbose:
        for ii in range(newgains.shape[1]):
            antgain = newgains[:, ii]
            antgain = antgain[~np.isnan(antgain)]
            print("Median gain amplitude for input {0} was {1}".format(ii, np.median(np.abs(antgain))))

    # Save gains to file
    if args.outfile:
        import hickle as hkl
        #print(f"Saving gain solutions to {args.outfile}")
        print("Saving gain solutions to", args.outfile)
        hkl.dump(newgains, args.outfile)

    # Write out the delays
    print("Writing out delay solutions to cable_delays.txt")
    delays2text(delays, recorded_snaps, "cable_delays.txt")

    # Apply calibration solutions (leaving the geometric delays in)
    print("Applying calibration")
    cal_vis_matrix = apply_gains(obs_vis_matrix, newrotatedgains)

    # Update the sky model for the desired target location (rather than first source), if specified
    if args.calibration_location != None:
        sky_model = make_sky_model([args.calibration_location])
        sky_model.compute_ephemeris(um)

    # Then apply delay-only calibration, including geometric delays, taking into account the already-applied delays
    trackingdelays = np.delete(delays-applieddelays, flaginputs)
    if args.verbose:
        print("Shape of the tracking delays:", trackingdelays.shape)
    tracking_vis_matrix = apply_delays(obs_vis_matrix, trackingdelays, um, sky_model, obs_metadata)

    # Finally apply goeometric delays to the cal_vis_matrix to see what it looks like when tracking
    tracking_vis_matrix_bandpasscorrected = apply_delays(cal_vis_matrix, np.zeros(obs_vis_matrix.shape[0]), um, sky_model, obs_metadata)
    derive_tracking_stats(tracking_vis_matrix, obs_metadata, activeinputs)
    lasttime = printelapsedsince(lasttime, "apply calibration and write out cable delays")

    # Define the plotting reference
    plotreference = None
    alternateplotreference = None
    if args.plot_onlyreference:
        plotreference = firstrefindex
        alternateplotreference = secondrefindex

    # Produce some lag plots if desired
    if args.plot_lags:
        plot_average_bandpass(obs_vis_matrix, obs_metadata, fig_id='Lags', individual_prefix="visibilities.lags", referenceantenna=plotreference, plotlag=True, activeinputs=activeinputs)
        if not alternateplotreference == None:
            plot_average_bandpass(obs_vis_matrix, obs_metadata, fig_id='Lags', individual_prefix="visibilities.lags", referenceantenna=alternateplotreference, plotlag=True, activeinputs=activeinputs)

    # Plot model, observed + calibrated
    if args.plot_model or args.plot_all:
        plot_matrix(model_vis_matrix, obs_metadata, phase=True, fig_id='model visibilites - phase')
        plt.savefig("modelvis.phase.png")
    if args.plot_observed or args.plot_all:
        if plotreference == None:
            plot_matrix(obs_vis_matrix, obs_metadata, phase=True, fig_id='Observed visibilities - phase')
            plt.savefig("observedvis.phase.png")
        plot_average_bandpass(obs_vis_matrix, obs_metadata, fig_id='Uncalibrated Visibilities', individual_prefix="visibilities.uncalibrated", referenceantenna=plotreference)
    if args.plot_calibrated or args.plot_all:
        if plotreference == None:
            plot_matrix(cal_vis_matrix, obs_metadata, phase=True, fig_id='Calibrated visibilities - phase')
            plt.savefig("calibratedvis.phase.png")
        plot_average_bandpass(cal_vis_matrix, obs_metadata, fig_id='Calibrated Visibilities', individual_prefix="visibilities.calibrated", referenceantenna=plotreference)
    if args.plot_ratio or args.plot_all:
        plot_matrix(cal_vis_matrix / model_vis_matrix, obs_metadata, phase=True, fig_id='Cal / model - phase')
    #if args.plot_gains or args.plot_all:
    #    plot_amp_phs(gains, obs_metadata, fig_id='Gain solutions', individual_prefix="individualgains")
    #    plt.savefig("gainsolutions.png")
    if args.plot_gains or args.plot_all:
        plot_amp_phs(newgains, obs_metadata, fig_id='Gain solutions', individual_prefix="individualgains.new", delays=delays, snapinputs=snapinputs)
        plt.savefig("newgainsolutions.png")
    if args.plot_tracking or args.plot_all:
        if plotreference == None:
            plot_matrix(tracking_vis_matrix, obs_metadata, phase=True, fig_id='Tracking visibilities - phase')
            plt.savefig("trackingvis.phase.png")
        plot_average_bandpass(tracking_vis_matrix, obs_metadata, fig_id='Tracking Visibilities', individual_prefix="visibilities.tracking", referenceantenna=plotreference, activeinputs=activeinputs)
        if not alternateplotreference == None:
            plot_average_bandpass(tracking_vis_matrix, obs_metadata, fig_id='Tracking Visibilities', individual_prefix="visibilities.tracking", referenceantenna=alternateplotreference, activeinputs=activeinputs)
        #plot_average_bandpass(tracking_vis_matrix_bandpasscorrected, obs_metadata, fig_id='Tracking Visibilities', individual_prefix="visibilities.tracking.fullbandpass", referenceantenna=plotreference, activeinputs=activeinputs)
    #plt.show()
    lasttime = printelapsedsince(lasttime, "do all plotting")

    # Get a sensitivity estimate using a good part of the band rarely affected by phone calls
    chanrange = [80,160]
    chanavg = 4
    baselinetemps = estimate_sensitivity(tracking_vis_matrix, sky_model, chanrange, chanavg)
    np.savetxt("sefd_estimates.txt", baselinetemps)
