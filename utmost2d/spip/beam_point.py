#!/usr/bin/env python

import sys
import subprocess
import time
from datetime import datetime
import numpy as np

# create the command to send the pointing angle via ssh to a pole on a module 
def ssh_command(module, pole, angle):
    cmd = 'ssh -i ~/.ssh/id_rsa_pi pi@ut2d-'+\
        str(module)+\
        ' "sudo /mnt/utmost2d/beam_control_local/beam_bidir_v2/RPI_BIDIRECTIONAL_V2.py --point 0 '+\
        str(pole)+' '+\
        str(angle)+'"'
    return cmd

##############################################################################

# process command line arguments

nargs = len(sys.argv) - 1

NS = False
dec = False
wait = False

if (nargs>=1):
    if nargs==1:
        NSangle = np.int(sys.argv[1])
        decangle = NSangle - 35
        NS = True
    if (nargs==2) or (nargs==4):
        if sys.argv[1]=="-d":
            decangle = np.int(sys.argv[2])
            dec = True
        else:
            print("Incorrect arguments")
            sys.exit()
    if nargs==4:    
        if sys.argv[3]=="-w":
            waittime = np.int(sys.argv[4])
            wait = True
        else:
            print("Incorrect arguments")
            sys.exit()
else:
    print("Requires a NS angle or declination, and an optinal wait time")
    print("e.g.")
    print("beam_point.py -11              : point to NS angle -11 deg")
    print("beam_point.py -d -45           : point to dec angle -45 deg")
    print("beam_point.py -d -45 -w 3600   : point to dec angle -45 deg, wait 3600 seconds before pointing")
    sys.exit()

# figure out the pointing angle based on input arguments
if NS:
    angle = NSangle
if dec:
    angle = decangle + 35

print("NS pointing angle is "+str(angle)+" deg")

if wait:
    print("Waiting for "+str(waittime)+" seconds before the pointing")
    time.sleep(waittime)
    
# open the log file in append mode
f = open("/home/dprice/utmost/scripts/pointing_log.txt","a")

# loop over modules
for module in ("s","m1","m2","dc1","dc2","n"):

    # loop over poles
    for pole in ("vp", "hp"):

        print ("Pointing module "+str(module)+" on "+pole)

        # assemble the ssh command
        cmd = ssh_command(module, pole, angle)

        # echo the command
        print(cmd)

        # run the command as a subprocess
        proc = subprocess.Popen(cmd, stderr=subprocess.PIPE, shell=True)

        # get the date and time in a suitable format
        now = datetime.now().strftime('%a %b %d %Y - %H:%M:%S')+" "+str(time.tzname[0])
        
        # append date, time, timezone and the pointing angle to the log file
        print(str(now)+"  NS="+str(angle)+" dec="+str(decangle)+"\n")
        f.write(str(now)+"  NS="+str(angle)+" dec="+str(decangle)+"\n")
        
        # sleep for 1 second -- be kind to our BFs!
        time.sleep(1)

f.close()

print("NS angle : "+str(angle)+" deg")
print("DEC      : "+str(angle-35)+" deg")







