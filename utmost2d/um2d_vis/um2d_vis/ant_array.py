"""
Basic antenna array geometry class
"""
import ephem
import numpy as np


class AntArray(ephem.Observer):
    """ Antenna array

    Based on pyEphem's Observer class.

    Args:
        lat (str):       latitude of array centre, e.g. 44:31:24.88
        long (str):      longitude of array centre, e.g. 11:38:45.56
        elev (float):    elevation in metres of array centre
        date (datetime): datetime object, date and time of observation
        antennas (np.array): numpy array of antenna positions, in xyz coordinates in meters, relative to the array centre.
    """
    def __init__(self, lat, long, elev, date, antennas):
        super(AntArray, self).__init__()
        self.lat = lat
        self.long = long
        self.elev = elev
        self.date = date
        self.antennas = np.array(antennas)
        self.n_ant    = len(antennas)
        self.baselines = self._generate_baseline_ids()
        self.xyz       = self._compute_baseline_vectors()

    def _compute_baseline_vectors(self, autocorrs=True):
        """ Compute all the baseline vectors (XYZ) for antennas

        Args:
            autocorrs (bool): baselines should contain autocorrs (zero-length baselines)

        Returns:
            xyz (np.array): XYZ array of all antenna combinations, in ascending antenna IDs.
        """
        xyz   = self.antennas
        n_ant = self.n_ant

        if autocorrs:
            bls = np.zeros([n_ant * (n_ant - 1) // 2 + n_ant, 3])
        else:
            bls = np.zeros([n_ant * (n_ant - 1) // 2, 3])

        bls_idx = 0
        for ii in range(n_ant):
            for jj in range(n_ant):
                if jj >= ii:
                    if autocorrs is False and ii == jj:
                        pass
                    else:
                        bls[bls_idx] = xyz[ii] - xyz[jj]
                        bls_idx += 1
        return bls

    def _generate_baseline_ids(self, autocorrs=True):
        """ Generate a list of unique baseline IDs and antenna pairs

        Args:
            autocorrs (bool): baselines should contain autocorrs (zero-length baselines)

        Returns:
            ant_arr (list): List of antenna pairs that correspond to baseline vectors
        """
        ant_arr = []
        for ii in range(1, self.n_ant + 1):
            for jj in range(1, self.n_ant + 1):
                if jj >= ii:
                    if autocorrs is False and ii == jj:
                        pass
                    else:
                        ant_arr.append((ii, jj))
        return ant_arr

    def update(self, date):
        """ Update antenna with a new datetime """
        self.date = date

    def report(self):
        print(self)
        print(self.xyz)


def make_antenna_array(lat, lon, elev, date, antennas):
    """ Generate a new AntArray object

    Args:
        lat (str):       latitude of array centre, e.g. 44:31:24.88
        lon (str):      longitude of array centre, e.g. 11:38:45.56
        elev (float):    elevation in metres of array centre
        date (datetime): datetime object, date and time of observation
        antennas (np.array): numpy array of antenna positions, in xyz coordinates in meters,
                             relative to the array centre.

    Returns:
        ant_arr (AntArray): New Antenna Array object
    """
    return AntArray(lat, lon, elev, date, antennas)
