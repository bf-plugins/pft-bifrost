#!/usr/bin/env python
import socket
import sys
PORT = 8999

# Create a TCP/IP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_address = ('localhost', PORT)

sock.connect(server_address)

msg = '{cmd} {val}\n'.format(cmd='COMMAND'.ljust(19), val='STOP')
print msg
sock.sendall(msg)

sock.close()
