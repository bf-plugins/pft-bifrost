#!/bin/bash

# Check for correct number of arguments
if [ "$#" -ne 2 ]; then
  echo "Usage: $0 <source name> <date>"
  exit 1
fi

# OK, we have the right number of arguments.  Set up and get data.
cd /data/npsr/calsolutions/
caldir=xcor-${2}-${1}
if [ -e "$caldir" ]; then
  echo "$caldir already exists"
  exit 1
fi
mkdir $caldir
cd $caldir
for bfnode in 00 01 02 03 04 05 06 07
do
    scp mpsr-bf${bfnode}:/data/npsr/*${1}*${2}*.h5 bf${bfnode}.h5
done
cp ../flagfreqs.txt .
cp ../appliedsnapdelays.txt .
#echo ". ~/setup_stefcal" > runcal
#echo "run_stefcal.py bf00.h5 bf01.h5 bf02.h5 bf03.h5 bf04.h5 bf05.h5 bf06.h5 bf07.h5 --recordedsnaps=0,1,2,3,4,5,6,7,8,9,10,11 --configdir=/home/npsr/software/utmost2d_snap/config/ --snapmapfile=active.txt --antposfile=active.txt --flagmodulespacingbelow=6 --flaginputs=120,121,122,123,124,125,126,127,128,129,130,131 --useaoflagger --referencecassette=M122C3" >> runcal
echo "run_stefcal.py bf00.h5 bf01.h5 bf02.h5 bf03.h5 bf04.h5 bf05.h5 bf06.h5 bf07.h5 --recordedsnaps=0,1,2,3,4,5,6,7,8,9,10,11 --configdir=/home/npsr/software/utmost2d_snap/config/ --snapmapfile=active.txt --antposfile=active.txt --flagmodulespacingbelow=6 --flaginputs=68,69,70,71,73,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127,128,129,130,131,134,135,141 --useaoflagger --referencecassette=M122C3" >> runcal
chmod 775 runcal
./runcal
