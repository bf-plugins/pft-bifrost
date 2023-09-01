#!/usr/bin/env python
from generate_spip_cfg import generate_spip_xml_configure, generate_spip_xml_start
import socket
import sys
import xmltodict


PORT = 14100
IP   = 'mpsr-srv0'

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
        xml = generate_spip_xml_configure(source, RA, DEC)
        print("Sending configure command")
        sock.sendall(xml + "\r\n")
        reply = sock.recv(4096)
        xml_reply = xmltodict.parse(reply)
        print(xml_reply["tcs_response"])

        xml = generate_spip_xml_start(UTC)
        print("Sending start command")
        sock.sendall(xml + "\r\n")
        reply = sock.recv(4096)
        xml_reply = xmltodict.parse(reply)
        print(xml_reply["tcs_response"])


    except Exception as E:
        print('Exception -- TCP start did not send.\n{}'.format(E))
    finally:
        sock.close()
