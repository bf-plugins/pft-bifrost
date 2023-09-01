#!/usr/bin/env python
from generate_spip_cfg import generate_spip_utc_start, generate_spip_obs_cfg
import socket
import sys
PORT = 8999
#IP   = ['mpsr-bf09']
#IP   = ['mpsr-bf08','mpsr-bf09']
IP   = ['mpsr-bf00','mpsr-bf01','mpsr-bf02','mpsr-bf03','mpsr-bf04','mpsr-bf05','mpsr-bf06','mpsr-bf07']

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Stop pipeline by sending TCP start message')
    #p.add_argument('-p', '--port', type=int, default=PORT, help='Port to send control message to')
    #p.add_argument('-i', '--host', type=str, default=IP, help='comma separated list of hostnames')
    
    args = p.parse_args()
    #hosts = args.host.split(",")
    hosts = IP

    # Create a TCP/IP socket
    socks = []
    for h in hosts:
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            #server_address = (h, args.port)
            server_address = (h, PORT)
            sock.connect(server_address)
            socks.append(sock)
        except Exception as E:
            #print('Could not open TCP socket to {}:{} {}'.format(h, args.port, E))
            print('Could not open TCP socket to {}:{} {}'.format(h, PORT, E))
            sock.close()

    if len(socks) > 0:
        msg = '{cmd} {val}\n'.format(cmd='COMMAND'.ljust(19), val='STOP')
        print msg

    for sock in socks:
        try:
            sock.sendall(msg)
        except Exception as E:
            print('Exception -- TCP stop did not send: {}'.format(E))
        finally:
            sock.close()
