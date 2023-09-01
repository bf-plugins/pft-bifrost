#!/usr/bin/env python
from generate_spip_cfg import generate_spip_utc_start, generate_spip_obs_cfg
import socket
import sys
PORT = 8999
IP   = 'localhost'

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Start pipeline by sending TCP start message',
                                prefix_chars='@')  # this allows -ve signs not to be parsed as prefix
    p.add_argument('source_ra_dec', type=str, help='Name of source to observe, RA, and DEC', nargs='*')
    
    args = p.parse_args()
    if len(args.source_ra_dec) == 3:
        source, RA, DEC = args.source_ra_dec
        source, RA, DEC = source.strip(), RA.strip(), DEC.strip()
        UTC = None
    elif len(args.source_ra_dec) == 4:
        source, RA, DEC, UTC = args.source_ra_dec
        source, RA, DEC, UTC = source.strip(), RA.strip(), DEC.strip(), UTC.strip()
    else:
        print("Error: you need to specify RA, DEC, and optionally UTC start")
        exit()

    # Create a TCP/IP socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (IP, PORT)
        sock.connect(server_address)


        hdr = generate_spip_obs_cfg(source, RA, DEC)
        hdr += generate_spip_utc_start(UTC)
        hdr += "COMMAND             START\n"

        print(hdr)
        sock.sendall(hdr)
    except Exception as E:
        print('Exception -- TCP start did not send.\n{}'.format(E))
    finally:
        sock.close()
