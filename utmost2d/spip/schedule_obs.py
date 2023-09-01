#!/usr/bin/env python

import time
import os
from datetime import datetime
import numpy as np
import sys

from astropy.coordinates import EarthLocation, SkyCoord, Angle
from astroplan import Observer, FixedTarget
from astropy.time import Time, TimeDelta

from source_db import srcs, um_obs

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Scheduler to control SNAP data capture and beamformer pointing')
    p.add_argument('sourcelist', type=str, nargs='*',
                    help='list of source names to observe. Type `all` to observe all sources in source DB.')
    p.add_argument('-l', '--obslen', type=float, default=5.0, 
                   help='Length of observation, in minutes.')
    p.add_argument('-c', '--capture', action='store_true', default=False,
                   help='Trigger data capture (send spip START command).')
    p.add_argument('-b', '--beamformer', action='store_true', default=False,
                   help='Control beamformer (send pointing commands).')
    args = p.parse_args()

    OBS_LEN_MIN = args.obslen
    OBS_LEN_HR = OBS_LEN_MIN/ 60
    OBS_LEN_SEC = OBS_LEN_MIN * 60
    data_capture_active = False

    now = Time(datetime.utcnow())
    print("Starting observation scheduler")
    print("UTC Now: {}".format(now))
    print("Obs len:               {} min".format(OBS_LEN_MIN))
    print("Beamformer control:    {}".format(args.beamformer))
    print("Trigger data capture:  {}".format(args.capture))

    if args.capture is False and args.beamformer is False:
        print("Warning: data capture AND beamformer control are deactivated. "
              "Scheduler will run, but nothing is being controlled.")
    if args.sourcelist is False:
        print("Error: need to give a list of sources, or pass 'all' keyword.")
        exit()

    try:
        while True:
            now = Time(datetime.utcnow())
            transit_times = []
            if args.sourcelist[0] == 'all':
                src_names = srcs.keys()
            else:
                src_names = args.sourcelist
            
            for name in src_names:
                src = srcs[name]  # Read from source_db
                t_transit = um_obs.target_meridian_transit_time(time=now, target=src, which="next")
                #print("T_transit for source {} is {}".format(name, t_transit))
                tdelt = (t_transit - now).to('s').value
                if tdelt < 0:
                    tdelt += 86400
                transit_times.append(tdelt)
                #src_names.append(src.name)
                print("SRC: {:16} UTC TRANSIT: {:28} WAIT TIME: {:2.2f}s".format(name, t_transit.iso, tdelt))

            idx_next = np.argmin(transit_times)
            next_src = src_names[idx_next]
            t_next_transit = transit_times[idx_next]
            
            ra_str_next = srcs[next_src].ra.to_string(unit='hourangle', sep=':')
            dec_str_next = srcs[next_src].dec.to_string(unit='degree', sep=':')
            
            print("Next transit is {} in {:2.1f}s".format(next_src, t_next_transit))
            print("RA DEC: {} {}".format(ra_str_next, dec_str_next))
            sleep_dur = t_next_transit - OBS_LEN_SEC/2
            if sleep_dur > 0:

#                print("Sleeping for {:2.2f} seconds".format(sleep_dur))
#                time.sleep(sleep_dur)
#                #time.sleep(0)

                # CF hack to display a countdown of the time remaining 03/11/2020
                for remaining in range(np.int(sleep_dur), 0, -1):
                    sys.stdout.write("\r")
                    sys.stdout.write("{:2.2f} seconds remaining.".format(remaining))
                    sys.stdout.flush()
                    time.sleep(1)

            else:
                print("Source is about to transit, starting to record immediately")

            if args.beamformer:
                #------------
                #VG/CF adding a line to point at the source before starting to observe
                dec_next_source = srcs[next_src].dec.value    #degrees
                ns_next_source = int(np.round(dec_next_source - um_obs.location.lat.value)) #Because beam_point cannot handle floats

                #cmd = "/home/dprice/utmost/scripts/beam_point_redis.py {}".format(ns_next_source)
                #print("Pointing the cassettes to the source's ({}) NS angle ({})".format(next_src, ns_next_source))
                #os.system(cmd)

                cmd = "./home/dprice/utmost/scripts/beam_point.py {}".format(ns_next_source)
                print("Pointing the cassettes to the source's ({}) NS angle ({})".format(next_src, ns_next_source))
                os.system(cmd)

                #-----------

            if args.capture:
                now = Time(datetime.utcnow()) 
                print("[{}] Starting data capture for {}".format(now, next_src)) 
                data_capture_active = True
                if args.beamformer:
                    # first point the beamformer
                    cmd = "./beam_point.py {}".format(ns_next_source)
                    print("Pointing the cassettes to the source's ({}) NS angle ({})".format(next_src, ns_next_source))
                    os.system(cmd)
                    # sleep for a second
                    time.sleep(1)
                # now capture data
                os.system("./tcp_start.py {} {} {}".format(next_src, ra_str_next, dec_str_next))
                
            time.sleep(OBS_LEN_SEC)
            
            if args.capture:
                now = Time(datetime.utcnow()) 
                print("[{}] Stopping obs".format(now))
                os.system("./tcp_stop.py")
                data_capture_active = False

    except KeyboardInterrupt:
        print("Exiting...")
        exit()
    finally:
        if args.capture and data_capture_active:
            os.system("./tcp_stop.py")

