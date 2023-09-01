#!/usr/bin/env python
import socket
import sys
import xmltodict

from generate_spip_cfg import generate_spip_xml_stop


PORT = 14100
IP   = 'mpsr-srv0'

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description='Stop SPIP pipeline by sending TCP stopmessage')
    p.add_argument('--utc', type=str, help='UTC timestamp at which to stop the observation')
    args = p.parse_args()

    # Create a TCP/IP socket
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_address = (IP, PORT)
        sock.connect(server_address)
        xml = generate_spip_xml_stop(args.utc)
        print("Sending STOP command")
        sock.sendall(xml + "\r\n")
        reply = sock.recv(4096)
        xml_reply = xmltodict.parse(reply)
        print(xml_reply["tcs_response"])

    except Exception as E:
        print('Exception -- XML Stop error.\n{}'.format(E))
    finally:
        sock.close()
