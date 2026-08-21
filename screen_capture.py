import cv2
import numpy as np
from mss import MSS

sct = MSS()

monitor = {
    "top": 0,
    "left": 0,
    "width": 800,
    "height": 625,
}

quit_keys = [ord("q"), ord("Q")]

while True:
    frame = np.array(sct.grab(monitor))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    small = cv2.resize(frame, (400, 313))
    cv2.imshow("NSFU2 View", small)

    if cv2.waitKey(1) & 0xFF in quit_keys:
        break

cv2.destroyAllWindows()