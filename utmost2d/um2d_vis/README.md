## um2d_vis

**Requires Py 3.7**

Utilities for UTMOST-2D visibility viewing and calibration. 

### run_stefcal.py

Apply stefcal to data to derive complex gain calibration solutions.

```python
./run_stefcal.py -h
usage: run_stefcal.py [-h] [-o OUTFILE] [--plot_gains] [--plot_model]
                      [--plot_observed] [--plot_calibrated] [--plot_ratio]
                      [--plot_all]
                      filename

Compute complex gain solutions for UTMOST-2D

positional arguments:
  filename              Name of visibility file

optional arguments:
  -h, --help            show this help message and exit
  -o OUTFILE, --outfile OUTFILE
                        Save gains to file
  --plot_gains          plot complex gain solutions
  --plot_model          Plot model visibilities
  --plot_observed       Plot observed visibilities
  --plot_calibrated     Plot calibrated visibilities
  --plot_ratio          Plot calibrated / model visibilities
  --plot_all            Plot all
```

### visview.py

View data in visibility files.

```python
./visview.py -h
usage: visview.py [-h] [-l] [-e EXTRACT [EXTRACT ...]] [-g GAINS] [-w] [-p]
                  [-A]
                  filename

SNAP-12x cross correlation plotter

positional arguments:
  filename              Name of file to plot

optional arguments:
  -h, --help            show this help message and exit
  -l, --logged          Plot in dB scale
  -e EXTRACT [EXTRACT ...], --extract EXTRACT [EXTRACT ...]
                        List of antennas to extract
  -g GAINS, --gains GAINS
                        Apply gains from filename
  -w, --waterfall       Plot waterfall instead of bandpass
  -p, --phase           Plot phase instead of power
  -A, --plotall         Plot all possible (4x) plots
```

### Example usage

Run stefcal on CJ1935 data:

```python
./run_stefcal.py /data/dprice/spip_xcor_CJ1935-4620_48_1280_2020-05-17-175638.h5 -o cal-CJ1935-4620.gains --plot_all
```

Apply gains, extract antennas of interest, and view:

```python
./visview.py /data/dprice/spip_xcor_3C273_51_1280_2020-05-18-104636.h5 -e 1 3 7 11 -p -w -g cal-CJ1935-4620.gains
```