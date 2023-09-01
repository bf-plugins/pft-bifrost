import numpy as np
import pylab as plt


def plot_matrix(mat, obs_metadata, phase=False, fig_id=None, figsize=(6, 6)):
    """ Plot a (Nchan x Nant x Nant) matrix

    Args:
        mat (np.array): Matrix to plot, (Nchan, Nant, Nant)
        phase (bool): Plot magnitude or phase
        obs_metadata (dict): Observation metadata, from make_metadata() function
    """
    if fig_id:
        plt.figure(fig_id, figsize=figsize)
    ufunc = np.angle if phase else np.abs
    NANT = obs_metadata['nant']
    f_mhz = obs_metadata['freq'].to('MHz').value
    for ii in range(NANT):
        for jj in range(NANT):
            plt.subplot(NANT, NANT, ii+1 + NANT*(jj))
            plt.plot(f_mhz, ufunc(mat[:, ii, jj]))
            if phase:
                plt.ylim(-3.2, 3.2)


def plot_average_bandpass(data, obs_metadata, fig_id=None, figsize=(10, 6), individual_prefix=None, referenceantenna=None, activeinputs=None, plotlag=False):
    """ Plot (Nchan x Nant x Nant) complex amplitude + phase

    Args:
        data (np.array):     Complex data to plot, (Nchan, Nant, Nant)
        obs_metadata (dict):  Observation metadata, from make_metadata() function
    """
    if fig_id:
        plt.figure(fig_id, figsize=figsize)

    f_mhz = obs_metadata['freq'].to('MHz').value
    if not individual_prefix == None:
        for ii in range(obs_metadata['nant']):
            for jj in range(obs_metadata['nant']):
                if referenceantenna == None or ii == referenceantenna:
                    plt.clf()
                    if activeinputs == None:
                        inplabel = str(ii) + "-" + str(jj)
                    else:
                        inplabel = activeinputs[ii] + '-' + activeinputs[jj]
                    if plotlag:
                        ax1 = plt.subplot(1,1,1)
                        phases = np.angle(data[:, ii, jj])
                        tofit = np.nan_to_num(np.exp(1j*phases))
                        tofit[np.abs(data[:, ii, jj]) == 0] = 0 + 0j
                        lags = np.abs(np.fft.ifft(tofit))
                        plt.plot(f_mhz, lags)
                        plt.savefig(individual_prefix + "." + inplabel + ".bandpass.png")
                    else:
                        ax1 = plt.subplot(2,1,1)
                        plt.plot(f_mhz, np.abs(data[:, ii, jj]))
                        plt.subplot(2,1,2,sharex=ax1)
                        plt.plot(f_mhz, np.angle(data[:, ii, jj]))
                        plt.savefig(individual_prefix + "." + inplabel + ".bandpass.png")

    else:
        print("This function only works for plotting individual baselines")

def plot_amp_phs(data, obs_metadata, fig_id=None, figsize=(10, 6), individual_prefix=None, delays=None, snapinputs=None):
    """ Plot (Nchan x Nant) complex amplitude + phase

    Args:
        data (np.array):     Complex data to plot, (Nchan, Nant)
        obs_metadata (dict):  Observation metadata, from make_metadata() function
    """
    if fig_id:
        plt.figure(fig_id, figsize=figsize)

    f_mhz = obs_metadata['freq'].to('MHz').value
    if not individual_prefix == None:
        #for ii in range(obs_metadata['nant']):
        for ii in range(data.shape[-1]):
            plt.clf()
            ax1 = plt.subplot(2,1,1)    
            plt.plot(f_mhz, np.abs(data[:, ii]))
            plt.subplot(2,1,2,sharex=ax1)
            plt.plot(f_mhz, np.angle(data[:, ii]))
            if not delays is None:
                phases = -2*np.pi*(f_mhz*1e6*delays[ii] - np.around(f_mhz*1e6*delays[ii]))
                plt.plot(f_mhz, phases)
            if snapinputs == None:
                inplabel = str(ii)
            else:
                inplabel = snapinputs[ii]
            plt.savefig(individual_prefix + "." + inplabel + ".gaincal.png")

    plt.clf()
    #for ii in range(obs_metadata['nant']):
    for ii in range(data.shape[-1]):
            #plt.subplot(obs_metadata['nant'], 2, 2*ii + 1)
            plt.subplot(data.shape[-1], 2, 2*ii + 1)
            plt.plot(f_mhz, np.abs(data[:, ii]))
            #plt.subplot(obs_metadata['nant'], 2, 2*ii + 2)
            plt.subplot(data.shape[-1], 2, 2*ii + 2)
            plt.plot(f_mhz, np.angle(data[:, ii]))
            plt.ylim(-3.2, 3.2)
