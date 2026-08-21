import time
import pydirectinput

print("Switch to NSFU2...")
time.sleep(3)

pydirectinput.keyDown('w')
time.sleep(2)

pydirectinput.keyDown('a')
time.sleep(0.5)
pydirectinput.keyUp('a')

time.sleep(1)

pydirectinput.keyDown('d')
time.sleep(0.5)
pydirectinput.keyUp('d')

time.sleep(1)

pydirectinput.keyUp('w')


















