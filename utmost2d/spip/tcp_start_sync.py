#!/usr/bin/env python
from generate_spip_cfg import generate_spip_utc_start, generate_spip_obs_cfg
import socket
import sys
PORT = 8999
#IP   = ['mpsr-bf09'] 
##IP   = ['mpsr-bf08','mpsr-bf09'] 
IP   = ['mpsr-bf00','mpsr-bf01','mpsr-bf02','mpsr-bf03','mpsr-bf04','mpsr-bf05','mpsr-bf06','mpsr-bf07'] 

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Start pipeline by sending TCP start message',
                                prefix_chars='@')  # this allows -ve signs not to be parsed as prefix
    p.add_argument('source_ra_dec', type=str, help='Name of source to observe, RA, and DEC', nargs='*')
    #p.add_argument('-p', '--port', type=int, default=PORT, help='Port to send control message to')
    #p.add_argument('-i', '--host', type=str, default=IP, help='comma separated list of hostnames')
    
    args = p.parse_args()
    #hosts = args.host.split(",")
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
    socks = []
    #for h in hosts:
    for h in IP:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #print("opening socket [" + str(sock) + "] to  " + h + ":" + str(args.port))
            #print("opening socket [" + str(sock) + "] to  " + h + ":" + str(PORT))
            #server_address = (h, args.port)
            server_address = (h, PORT)
            sock.connect(server_address)
            socks.append(sock)
        except Exception as E:
            #print('Could not open TCP socket to {}:{} {}'.format(h, args.port, E))
            print('Could not open TCP socket to {}:{} {}'.format(h, PORT, E))
            sock.close()

    if len(socks) > 0:
        hdr = generate_spip_obs_cfg(source, RA, DEC)
        hdr += generate_spip_utc_start(UTC)
        hdr += "COMMAND             START\n"
        print(hdr)

    for sock in socks:
        #print("sending hdr to " + str(sock))
        try:
            sock.sendall(hdr)
        except Exception as E:
            print('Exception -- TCP start did not send: {}'.format(E))
        finally:
            #print("closing " + str(sock))
            sock.close()
