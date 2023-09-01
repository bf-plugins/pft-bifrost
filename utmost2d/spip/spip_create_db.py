#!/usr/bin/env python

import os, sys

def parse_config(filename):
    conf = {}
    with open(filename, 'r') as fh:
        d  = fh.readlines()
    for line in d:
        l   = line.strip().split()
        conf[l[0].strip()] = l[1].strip()
    return conf

try:
    fn = sys.argv[1]
except:
    print("Usage: ./spip_create_db.py [config_file.cfg]")

try:
    N = int(sys.argv[2])
except:
    N = 64

conf = parse_config(fn)
db_size = int(conf['DADA_BUF_SIZE'])

print("Recreating UDP capture ring buffer BABA...")
# os.system("dada_db -d -k baba; sleep 1; dada_db -b {db_size} -k baba -n {N}; sleep 1;".format(db_size=db_size, N=N))
os.system("dada_db -b {db_size} -k baba -n {N} -c 1 -p -l -w".format(db_size=db_size, N=N))
