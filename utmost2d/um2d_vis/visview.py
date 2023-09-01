#!/usr/bin/env python
from um2d_vis.visibility import load_vis
import pylab as plt

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="SNAP-12x cross correlation plotter")
    p.add_argument('filename', help='Name of file to plot')
    p.add_argument('-l', '--logged', help='Plot in dB scale', default=False, action='store_true')
    p.add_argument('-e', '--extract', help='List of antennas to extract', default=None, nargs='+', type=int)
    p.add_argument('-g', '--gains', help='Apply gains from filename', default=None, type=str)
    p.add_argument('-w', '--waterfall', help='Plot waterfall instead of bandpass',
                   default=False, action='store_true')
    p.add_argument('-p', '--phase', help='Plot phase instead of power',
                   default=False, action='store_true')
    p.add_argument('-A', '--plotall', help='Plot all possible (4x) plots',
                   default=False, action='store_true')
    args = p.parse_args()

    V = load_vis(args.filename)

    if args.extract:
        print(f"Extracting antennas {args.extract}")
        V.extract_antennas(args.extract)

    if args.gains:
        print("Applying gain calibration")
        V.apply_gains(args.gains)

    if args.plotall:
        V.plot_matrix(logged=args.logged, phase=False, waterfall=False)
        V.plot_matrix(logged=False, phase=True, waterfall=False)
        V.plot_matrix(logged=args.logged, phase=False, waterfall=True)
        V.plot_matrix(logged=False, phase=True, waterfall=True)
    else:
        V.plot_matrix(logged=args.logged, phase=args.phase, waterfall=args.waterfall)
    plt.show()
