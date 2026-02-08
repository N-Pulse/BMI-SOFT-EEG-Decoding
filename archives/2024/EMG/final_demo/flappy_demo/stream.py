# from spikerbox_serial import read_arduino, process_data, init_serial
import numpy as np
import pygame
from pong import Pong
import sys

from multiprocessing import Process, Value
from ctypes import c_double

from flappy import Flappy

from threading import Thread
import socket

UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 5005     # Must match the sender's port

def udp_listener(shared_label, port=5005):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((UDP_IP, UDP_PORT))
    while True:
        data, _ = sock.recvfrom(1024)
        # shared_label.value = data.strip()
        # print(shared_label.value)
        if data == b'\x06' :
            shared_label.value=b"Fist"
        else :
            shared_label.value=b"Rest"


### SET SOME VARIABLES ###
cport = "/dev/cu.usbserial-DM89OND5"  # set the correct port before you run it
inputBufferSize = 2000  # keep between 2000-20000
movementThreshold = 1  # start with 1 std above running mean
playerPaddle = 200  # refers to paddle size, default is 100
cpuPlayStyle = "random"  # options are 'following' or 'random'
game_choice = "flappy"  # Choose your game! Options are "flappy" or "pong"
use_emg = True
###


##### If you want to modify the p1_handle_event function, go into the pong script and modify the p1_handle_event(running_mean_tmp)


def main():
    from multiprocessing import Value
    from ctypes import c_char_p
    # Select game
    # game_choice = "flappy"  # or "pong"
    if game_choice == "flappy":
        game = Flappy()
    elif game_choice == "pong":
        game = Pong(cpuPlayStyle="random")
        game.set_new_paddle(playerPaddle)
    else:
        raise ValueError("Invalid game_choice")

    # Shared EMG signal value

    # labels : {"Rest": "0", "Fist" : "6"}
    labels = []
    space_pressed = []

    shared_label = Value(c_char_p, b"Rest")
    labels.append(shared_label)
    space_pressed.append(False)

    thread = Thread(target=udp_listener, args=(shared_label,), daemon=True)
    thread.start()

    game.draw()
    # input("Press key to begin")

    running = True
    # Main game loop
    while running:

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            keys = pygame.key.get_pressed()
            space = keys[pygame.K_SPACE]
            # print("Space pressed:", space)

            if hasattr(game, "p2_handle_event"):  # only Pong has this
                game.p2_handle_event(event)
        
        label = shared_label.value.decode()
        if label!=labels[-1]:
            print(label)

        if (label=="Fist" and labels[-1]=="Rest") or (space==True and space_pressed[-1]==False):
            emg_val = movementThreshold + 10
        else:
            emg_val = 0

        game.handle_input(emg_val, movementThreshold)

        game.update()
        game.draw()

        labels.append(label)
        if len(labels)>3:
            labels.pop(0)
        
        space_pressed.append(space)
        if len(space_pressed)>3:
            space_pressed.pop(0)

        if getattr(game, "done", False):
            print("Game over!")
            running = False
            


if __name__ == "__main__":
    main()
