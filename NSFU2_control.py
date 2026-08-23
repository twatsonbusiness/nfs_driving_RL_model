import time
import cv2
import numpy as np
import pydirectinput
from mss import MSS
import threading

from torch._C._jit_tree_views import While

sct = MSS()
ACTION_INTERVAL = 1

monitor = {
    "top": 0,
    "left": 0,
    "width": 800,
    "height": 625,
}



quit_keys = [ord("q"), ord("Q")]

def release_steering():
    pydirectinput.keyUp('a')
    pydirectinput.keyUp('d')

def release_throttle():
    pydirectinput.keyUp('w')
    pydirectinput.keyUp('s')

def action(action_number):
    release_steering()

    if action_number == 0:
        pydirectinput.keyDown('w')
    elif action_number == 1:
        pydirectinput.keyDown('s')
    elif action_number == 2:
        pydirectinput.keyDown('a')
    elif action_number == 3:
        pydirectinput.keyDown('d')
    elif action_number >= 4:
        release_steering()

print('Starting in 3 seconds...')
time.sleep(3)


def action_chooser():
    start = time.perf_counter()

    while True:
        pydirectinput.keyDown('w')

        elapsed = time.perf_counter() - start
        remaining = ACTION_INTERVAL - elapsed

        print(f"Remaining: {remaining}")

        if remaining <= 0.2:
            start = time.perf_counter()
            action(np.random.randint(2, 5))

def screen_capture():

    while True:
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        small = cv2.resize(frame, (400, 313))
        cv2.imshow("NSFU2 View", small)

        if cv2.waitKey(1) & 0xFF in quit_keys:
            break

action_thread = threading.Thread(target=action_chooser)
capture_thread = threading.Thread(target=screen_capture)
capture_thread.start()
action_thread.start()

cv2.destroyAllWindows()
pydirectinput.keyUp('w')







