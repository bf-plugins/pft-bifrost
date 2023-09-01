#!/usr/bin/env python
import os
import socket

import argparse
p = argparse.ArgumentParser('Startup UDP packet capture')
p.add_argument('cfg', help='Path to cfg file')
args = p.parse_args()

host = socket.gethostname().split('.')[0].split('-')[1]
env = 'LD_PRELOAD=libvma.so VMA_RING_ALLOCATION_LOGIC=0 VMA_THREAD_MODE=0 VMA_INTERNAL_THREAD_AFFINITY=15 VMA_MEM_ALOC_TYPE=1 VMA_TRACELEVEL=WARNING VMA_STATS_SHMEM_DIR=""'
cmd = '{env} snap_udpdb {cfg} -k baba -c 8999 -b 15'.format(env=env, cfg=args.cfg, host=host)

print(cmd)
os.system(cmd)

