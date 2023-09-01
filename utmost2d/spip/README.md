# SPIP control script overview

Control scripts for the new system are located in 
`/home/dprice/snap_r3/utmost2d_snap/spip`. 

## Script walkthrough

Create PSRDADA ring buffer:
`./spip_create_db.py [s01_bf09_mcast_50mhz.cfg]`
The cfg file contains info to make sure buffers are correct size.

Startup UDP capture code:
`./spip_udp_capture.py`
This wraps `snap_udpdb`, loads it using correct libvma acceleator library, and 
starts the capture pipeline into a READY state, but does not trigger capture.
Capture and reconfig are controlled over a TCP socket.

Startup correlator backend:
`./spip_correlator.py`
To startup in hi time res mode: `./spip_correlator.py -m lo`

Trigger observation and data capture:
`./tcp_start [SOURCE] [RA] [DEC]`
This sends TCP messages to `snap_udpdb` to trigger data capture.
NOTE: This does not control beamformer.

Stop observation:
`./tcp_stop.py`

## Scheduling observations

The scheduler, `schedule_obs.py`, now takes a list of sources to observe:

```
[dprice@mpsr-bf09 spip]$ ./schedule_obs.py -h
usage: schedule_obs.py [-h] [-l OBSLEN] [-c] [-b]
                       [sourcelist [sourcelist ...]]

Scheduler to control SNAP data capture and beamformer pointing

positional arguments:
  sourcelist            list of source names to observe. Type `all` to observe
                        all sources in source DB.

optional arguments:
  -h, --help            show this help message and exit
  -l OBSLEN, --obslen OBSLEN
                        Length of observation, in minutes.
  -c, --capture         Trigger data capture (send spip START command).
  -b, --beamformer      Control beamformer (send pointing commands).
```

Basically add `-b` to control beamformer, and `-c` to control UDP capture.

Example usage:

```
[dprice@mpsr-bf09 spip]$ ./schedule_obs.py -b -c 3C273 SUN 3C353 3C348 -l 60
Starting observation scheduler
UTC Now: 2020-04-19 03:03:46.271343
Obs len:               60.0 min
Beamformer control:    True
Trigger data capture:  True

SRC: 3C273            UTC TRANSIT: 2020-04-19 12:40:09.705      WAIT TIME: 34583.43s
SRC: SUN              UTC TRANSIT: 2020-04-20 01:39:12.283      WAIT TIME: 81326.01s
SRC: 3C353            UTC TRANSIT: 2020-04-19 17:30:45.616      WAIT TIME: 52019.34s
SRC: 3C348            UTC TRANSIT: 2020-04-19 17:01:25.868      WAIT TIME: 50259.60s

Next transit is CJ0408-6545 in 4563.3s
RA DEC: 4:08:20.3 -65:45:08.5
Sleeping for 2763.28 seconds
```

## Example tmux startup

On bf09:

```
./spip_create_db.py s01_bf09_mcast_50mhz.cfg

tmux new -s spip_udp_capture
    ./spip_udp_capture.py
    [ctrl+B, then D, to detach from screen]

tmux new -s spip_gpu_backend
    ./spip_correlator.py
    [ctrl+B, D]

tmux new -s spip_scheduler
    ./schedule_obs.py -l 10 -b -cc 3C273 3C353 3C348
    [ctrl+B, D]

[Use ctrl+B, S, to enter screen selection in tmux]
```

