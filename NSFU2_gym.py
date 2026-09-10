import time
import cv2
import re
import numpy as np
import pydirectinput
from mss import MSS
import threading
from pynput import keyboard
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback, CallbackList
import gymnasium as gym
from gymnasium import spaces
import pytesseract

from NSFU2_control import stop_event

pytesseract.pytesseract.tesseract_cmd = (r"C:\Program Files\Tesseract-OCR\tesseract.exe")


class NFSU2Env(gym.Env):
    metadata = {"render_models": ['human']}

    def __init__(self):
        super().__init__()
        self.sct = MSS()
        self.monitor = {
            "top": 0,
            "left": 0,
            "width": 800,
            "height": 625,
        }
        self.step_time = 0.1

        """
        Actions:
        0 = Nitrous
        1 = back + left
        2 = back
        3 = back + right
        4 = left
        5 = nothing/coast
        6 = right
        7 = forward + left
        8 = forward
        9 = forward + right
        """
        self.action_space = spaces.Discrete(10)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
            ),
            "speed": spaces.Box(
                low=0.0, high=1.0, shape=(1,), dtype=np.float32
            ),
        })
        self.last_speed = 0
        self.stuck_steps = 0
        self.episode_steps = 0

        self.max_episode_steps = 600

    def release_all(self):
        pydirectinput.keyUp('w')
        pydirectinput.keyUp('s')
        pydirectinput.keyUp('a')
        pydirectinput.keyUp('d')
        pydirectinput.keyUp('altleft')

    def set_action(self, action):
        self.release_all()

        match action:
            case 0:
                pydirectinput.keyDown('altleft')
            case 1:
                pydirectinput.keyDown('s')
                pydirectinput.keyDown('a')
            case 2:
                pydirectinput.keyDown('s')
            case 3:
                pydirectinput.keyDown('s')
                pydirectinput.keyDown('d')
            case 4:
                pydirectinput.keyDown('a')
            case 5:
                pass
            case 6:
                pydirectinput.keyDown('d')
            case 7:
                pydirectinput.keyDown('w')
                pydirectinput.keyDown('a')
            case 8:
                pydirectinput.keyDown('w')
            case 9:
                pydirectinput.keyDown('w')
                pydirectinput.keyDown('d')
    def capture_frame(self):
        frame = np.array(self.sct.grab(self.monitor))

        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        return frame

    def get_observation(self, frame=None, speed=None):
        if frame is None:
            frame = self.capture_frame()
        gameplay = frame[40:500, 40:760]
        gray = cv2.cvtColor(gameplay, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)
        gray = gray[np.newaxis, :, :].astype(np.uint8)

        if speed is None:
            speed = self.read_speed(frame)

        return {
            "image":gray,
            "speed":np.array([np.clip(speed/200.0, 0.0, 1.0)], dtype=np.float32)
        }



    def read_speed(self, frame):
        speedometer = frame[515:550, 700:775]
        gray = cv2.cvtColor(speedometer, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        _, binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        text = pytesseract.image_to_string(
            binary, config=(
            "--psm 7 "
            "-c tessedit_char_whitelist=0123456789"
            ),
        )
        try:
            speed = int(text)
        except ValueError:
            return self.last_speed

        if speed > 200 or speed < 0:
            return self.last_speed

        return speed

    def restart_race(self):
        self.release_all()
        pydirectinput.keyDown('s')
        time.sleep(3)
        self.release_all()
        pydirectinput.keyDown('space')
        time.sleep(5)
        pydirectinput.keyUp('space')

        return True
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.release_all()

        self.episode_steps = 0
        self.stuck_steps = 0
        self.last_speed = 0

        self.restart_race()

        frame = self.capture_frame()
        speed = self.read_speed(frame)
        self.last_speed = speed
        observation = self.get_observation(frame, speed)
        return observation, {}

    def calculate_reward(self, speed):
        speed_reward = speed/200.0
        acceleration = speed - self.last_speed
        if acceleration>0:
            acceleration_reward = acceleration/50
        elif acceleration<0:
            acceleration_reward = acceleration/20
        else:
            acceleration_reward = 0.0

        reward = (speed_reward + acceleration_reward)

        if speed < 5:
            reward -= 0.25

        return float(reward)

    def step(self, action):
        self.episode_steps += 1
        self.set_action(int(action))
        time.sleep(self.step_time)
        frame = self.capture_frame()
        speed = self.read_speed(frame)
        observation = self.get_observation(frame, speed)
        reward = self.calculate_reward(speed)
        if speed < 3:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0

        stuck = self.stuck_steps >= 40
        terminated = stuck

        truncated = (self.episode_steps >= self.max_episode_steps)
        info = {
            "speed":speed,
            "stuck":stuck
        }
        self.last_speed = speed

        return observation, reward, terminated, truncated, info

    def render(self):
        frame = self.capture_frame()
        cv2.imshow("NFSU2", cv2.resize(frame,(400,313)))
        cv2.waitKey(1)
    def close(self):
        self.release_all()
        cv2.destroyAllWindows()



if __name__ == "__main__":

    stop_event = threading.Event()

    def abort(key):
        try:
            if key.char.lower() in ['q', '-', '0']:
                print('Stopping Training...')
                stop_event.set()
        except AttributeError:
            pass

    listener = keyboard.Listener(on_press=abort)
    listener.start()

    class StopTrainingCallback(BaseCallback):
        def __init__(self, stop_event):
            super().__init__()
            self.stop_event = stop_event

        def _on_step(self) -> bool:
            if self.stop_event.is_set():
                print('Stopping model.learn()...')
                return False
            return True

    env = NFSU2Env()

    checkpoint_callback = CheckpointCallback(
        save_freq=5000,
        save_path="./checkpoints/",
        name_prefix='nfsu2',
        save_replay_buffer=True
    )
    stop_callback = StopTrainingCallback(stop_event)
    callbacks = CallbackList([
        checkpoint_callback, stop_callback
    ])
    resume = False
    try:
        if resume:
            model = DQN.load(
                "./checkpoints/nfsu2_50000_steps.zip",
                env=env
            )
            model.load_replay_buffer("./checkpoints/nfsu2_50000_steps.pkl", )
            model.learn(
                total_timesteps=1_000_000, reset_num_timesteps=False, callback=callbacks
            )
        else:
            model = DQN(
                "MultiInputPolicy",
                env,
                verbose=1,
            )

            model.learn(total_timesteps=1_000_000, callback=callbacks)

    finally:
        print('Saving current state...')
        model.save("./checkpoints/nfsu2_manual")
        model.save_replay_buffer("./checkpoints/nfsu2_manual_replay_buffer.pkl")




