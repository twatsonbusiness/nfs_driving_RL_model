import time
import cv2
import numpy as np
import pydirectinput
from mss import MSS
import threading
from pynput import keyboard
import pytesseract
pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")

sct = MSS()
ACTION_INTERVAL = 1

monitor = {
    "top": 0,
    "left": 0,
    "width": 800,
    "height": 625,
}

stop_event = threading.Event()
def abort(key):
    exit_keys = ['q', '-', '0']
    try:
        if key.char in exit_keys:
            stop_event.set()
            release_steering()
            release_throttle()
            print('\n\nABORTING...\n\n')
    except AttributeError:
        pass

listener = keyboard.Listener(on_press=abort)
listener.start()



quit_keys = [ord("q"), ord("Q")]

def steering_adjustment(action_number):
    time_short = 0.05
    time_long = 0.15
    match action_number:
        case 1:
            pydirectinput.keyDown('a')
            time.sleep(time_long)
            pydirectinput.keyUp('a')
        case 2:
            pydirectinput.keyDown('a')
            time.sleep(time_short)
            pydirectinput.keyUp('a')
        case 3:
            release_steering()
            time.sleep(1)
        case 4:
            pydirectinput.keyDown('d')
            time.sleep(time_short)
            pydirectinput.keyUp('d')
        case 5:
            pydirectinput.keyDown('d')
            time.sleep(time_long)
            pydirectinput.keyUp('d')

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

def read_speed(speedometer):
    gray_speedometer = cv2.cvtColor(
        speedometer, cv2.COLOR_BGR2GRAY
    )
    speed = pytesseract.image_to_string(
        gray_speedometer, config="--psm 7 -c tessedit_char_whitelist=0123456789"

    )
    if int(speed) > 120:
        return False
    else:
        print(f"Speed: {speed}")
        return True

def action_chooser():
    start = time.perf_counter()

    while not stop_event.is_set():
        pydirectinput.keyDown('w')

        elapsed = time.perf_counter() - start
        remaining = ACTION_INTERVAL - elapsed

        print(f"Remaining: {remaining:.2f}")

        if remaining <= 0.2:
            start = time.perf_counter()
            steering_adjustment(np.random.randint(1, 6))

def screen_capture():
    frame_count =  0

    while not stop_event.is_set():
        frame_count += 1
        frame = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
        small = cv2.resize(frame, (400, 313))
        cv2.imshow("NSFU2 View", small)
        speedometer = frame[515:550, 700:775]
        cv2.imshow("Speedometer", speedometer)
        if frame_count % 60 == 0:
            read_speed_thread = threading.Thread(target=read_speed, args=(speedometer,))
            read_speed_thread.start()
        cv2.waitKey(1)

action_thread = threading.Thread(target=action_chooser)
capture_thread = threading.Thread(target=screen_capture)
capture_thread.start()
action_thread.start()

capture_thread.join()
action_thread.join()

listener.stop()

pydirectinput.keyUp('w')
release_throttle()
release_steering()

cv2.destroyAllWindows()
print('End of program.')
