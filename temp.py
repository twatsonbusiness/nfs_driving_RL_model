import time
import threading

import pydirectinput
from pynput import keyboard


running = False


def macro1():
    global running

    try:
        pydirectinput.press('0')
        for _ in range(8):
            pydirectinput.press('tab', presses=2, interval=0.1)
            pydirectinput.press('0')
    finally:
        running = False


def macro2():
    global running

    try:
        pydirectinput.press('0')

        for _ in range(10):
            pydirectinput.press('tab', presses=2, interval=0.1)
            pydirectinput.press('0')

        pydirectinput.press('tab', presses=3)
        pydirectinput.press('enter')
    finally:
        running = False


def controller(key):
    global running

    if running:
        return

    # Normal character key
    if key == keyboard.Key.left:
        running = True
        threading.Thread(
            target=macro1,
            daemon=True
        ).start()

    # Special key
    elif key == keyboard.Key.right:
        running = True
        threading.Thread(
            target=macro2,
            daemon=True
        ).start()


listener = keyboard.Listener(on_press=controller)
listener.start()

print("Ready")
print("left         -> macro1")
print("right        -> macro2")

while True:
    time.sleep(1)