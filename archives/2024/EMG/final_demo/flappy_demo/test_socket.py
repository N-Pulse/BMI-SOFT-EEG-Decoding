import pygame
import socket
import sys
from threading import Thread
from multiprocessing import Value
from ctypes import c_char_p

def udp_listener(shared_label, port=5005):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while True:
        data, _ = sock.recvfrom(1024)

        shared_label.value=data



UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005     # Must match the sender's port

print(f"Listening for UDP messages on port {UDP_PORT}...")

labels=[]
shared_label = Value(c_char_p, b"\x00")
labels.append(shared_label)

thread = Thread(target=udp_listener, args=(shared_label,), daemon=True)
thread.start()

while True:
    label = shared_label.value.decode()
    if label!=labels[-1]:
        print(label)

    labels.append(label)
    if len(labels)>3:
        labels.pop(0)


# sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# sock.bind((UDP_IP, UDP_PORT))

# while True:
#     data, addr = sock.recvfrom(1024)
#     print(f"{data.decode().strip()}")