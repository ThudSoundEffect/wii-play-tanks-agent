"""Gymnasium environment for Wii Play Tanks running on Dolphin emulator."""
import os
import signal
import subprocess
import time
from collections import deque
from pathlib import Path

import cv2
import dolphin_memory_engine as dme
import gymnasium as gym
import numpy as np
import mss
import pydirectinput as pdi
import pygetwindow as gw
from gymnasium import spaces

import config
import memory_addresses as mem

pdi.FAILSAFE = False
pdi.PAUSE = 0

CAPTURE_LEFT_OFFSET = 80
CAPTURE_TOP_OFFSET = 85

AIM_DIRECTIONS = 24

_AIM_BIN_COS = np.cos(np.deg2rad(np.arange(AIM_DIRECTIONS) * 360.0 / AIM_DIRECTIONS))
_AIM_BIN_SIN = np.sin(np.deg2rad(np.arange(AIM_DIRECTIONS) * 360.0 / AIM_DIRECTIONS))

MOVE_HOLD = 7

MAX_LEVEL_STEPS = 3600
DOLPHIN_BOOT_WAIT = 5
SAVESTATE_LOAD_WAIT = .25

_MOVEMENT_MAP = {
    0: ("right",),
    1: ("right", "up"),
    2: ("up",),
    3: ("up", "left"),
    4: ("left",),
    5: ("left", "down"),
    6: ("down",),
    7: ("right", "down"),
}


def _press_key(key: str, pause_time: float = 0.017) -> None:
    pdi.keyDown(key)
    time.sleep(pause_time)
    pdi.keyUp(key)


class TankEnv(gym.Env):
    """Single-process Gymnasium environment for Wii Play Tanks via Dolphin."""

    def __init__(self, config: dict, weights: dict) -> None:
        super().__init__()

        self.window_name = config["window_name"]
        self.dolphin_path = config["dolphin_path"]
        self.state_path = config["state_path"]
        self.rom_path = config["rom_path"]
        self.num_frames = config["num_frames"]
        self.num_features = config["num_features"]
        self.reward_weights = weights

        self._features = np.zeros(self.num_features, dtype=np.float32)
        self._movement_actions = [
            ["right", False],
            ["left",  False],
            ["up",    False],
            ["down",  False],
        ]
        self._button_keys = ["b"]

        parent_dir = Path(__file__).parent
        process = subprocess.Popen(
            [
                str(parent_dir / config["dolphin_path"]),
                "-e", str(parent_dir / config["rom_path"]),
                "-s", str(parent_dir / config["state_path"]),
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.dolphin_pid = process.pid
        print(f"[Emulator] Dolphin started with PID {self.dolphin_pid}.")

        time.sleep(DOLPHIN_BOOT_WAIT)

        dme.hook()
        if not dme.is_hooked():
            os.kill(self.dolphin_pid, signal.SIGTERM)
            raise RuntimeError("Could not hook Dolphin. Close all other instances.")

        self.window = self._get_dolphin_window()
        self._get_geometry()
        self.sct = mss.MSS()

        self.action_space = spaces.Dict({
            "buttons": spaces.MultiBinary(1),
            "dpad":    spaces.Discrete(8),
            "pointer": spaces.Discrete(AIM_DIRECTIONS),
        })

        self.observation_space = spaces.Dict({
            "frames": spaces.Box(
                low=0, high=255,
                shape=(self.height, self.width, self.num_frames),
                dtype=np.uint8,
            ),
            "features": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(self.num_features,),
                dtype=np.float32,
            ),
            "buttons": spaces.MultiBinary(1),
            "dpad":    spaces.Discrete(8),
            "pointer": spaces.Discrete(AIM_DIRECTIONS),
        })

        self.max_steps = MAX_LEVEL_STEPS
        self.lives = 3
        self.bullets_remaining = 5
        self.level = 1
        self.enemies_destroyed = 0
        self.user_x = 0.0
        self.user_y = 0.0
        self.user_x_norm = 0.0
        self.user_y_norm = 0.0
        self.steps = 0

        self._previous_enemies = 0
        self._previous_lives = 3
        self._previous_level = 1

        self._capture_w, self._capture_h = self._capture_dims()

        self.frames: deque = deque(
            (np.zeros((self._capture_h, self._capture_w), dtype=np.uint8) for _ in range(self.num_frames)),
            maxlen=self.num_frames,
        )

        _press_key("f1")
        time.sleep(SAVESTATE_LOAD_WAIT)

    def reset(self, seed=None, options: dict = None):
        super().reset(seed=seed)

        self._get_geometry()
        self._capture_w, self._capture_h = self._capture_dims()
        self.steps = 0
        self._previous_enemies = 0
        self._previous_lives = 3
        self._previous_level = 1

        for entry in self._movement_actions:
            key, pressed = entry
            if pressed:
                pdi.keyUp(key)
            entry[1] = False

        target_key = "f1"
        if options and "curriculum_stage" in options:
            sampled_stage = self._sample_stage(options["curriculum_stage"])
            target_key = f"f{sampled_stage}"
            self.max_steps = options["curriculum_stage"] * MAX_LEVEL_STEPS
            self._previous_enemies = config.STAGE_ENEMY_NORM[sampled_stage - 1]
            self._previous_level = sampled_stage

        _press_key(target_key, 0.005)

        time.sleep(SAVESTATE_LOAD_WAIT)

        frame = self._capture_frame()
        for _ in range(self.num_frames):
            self.frames.append(frame)

        self._update_agent_position()
        return self._get_obs(np.zeros(1, dtype=np.int8), 0, 0), {}

    def step(self, action: dict):
        while not dme.read_byte(mem.ACTIVE_LEVEL) and dme.read_byte(mem.LIVES) > 0:
            time.sleep(0.005)

        buttons, dpad, pointer = self._act(action)
        self.steps += 1

        self.frames.append(self._capture_frame())

        observation = self._get_obs(buttons, dpad, pointer)
        reward_components = self._compute_reward(observation["features"])
        reward = sum(reward_components.values())
        terminated = self._is_terminated(observation["features"])
        truncated = self.steps > self.max_steps
        info = {"reward_components": reward_components}
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        print(f"[Emulator] Ending Dolphin Emulator with PID {self.dolphin_pid}.")
        os.kill(self.dolphin_pid, signal.SIGTERM)

    def _capture_dims(self) -> tuple[int, int]:
        return self.width - 160, self.height - 140

    def _get_geometry(self) -> None:
        self.x = self.window.left
        self.y = self.window.top
        self.width = self.window.width
        self.height = self.window.height

    def _update_agent_position(self) -> None:
        """Read the agent's real in-game position from memory and derive
        both the capture-pixel position (used for mouse aiming) and the
        normalized [0, 1] position (used in the observation). This is the
        single source of truth for the agent's position — downstream
        consumers (e.g. the SB3 wrapper) should use the normalized value
        directly rather than re-deriving it from memory or raw pixels.
        """
        mem_x = dme.read_float(mem.USER_X)
        mem_y = dme.read_float(mem.USER_Y)

        self.user_x = self._capture_w / 2 + mem_x
        self.user_y = self._capture_h / 2 - mem_y

        self.user_x_norm = float(np.clip(self.user_x / self._capture_w, 0.0, 1.0))
        self.user_y_norm = float(np.clip((self.user_y - 35) / self._capture_h, 0.0, 1.0))

    def _get_obs(self, buttons: np.ndarray, dpad: int, pointer: int) -> dict:

        self.lives = int(dme.read_byte(mem.LIVES))
        self.bullets_remaining = 5 - int(dme.read_word(mem.BULLETS))
        self.level = int(dme.read_word(mem.LEVEL))
        self.enemies_destroyed = int(dme.read_word(mem.TANKS_DESTROYED))

        self._features[0] = self.lives
        self._features[1] = self.bullets_remaining
        self._features[2] = self.level
        self._features[3] = self.enemies_destroyed
        self._features[4] = self.user_x_norm
        self._features[5] = self.user_y_norm

        return {
            "frames":   np.stack(self.frames, axis=-1),
            "features": self._features,
            "buttons":  buttons,
            "dpad":     dpad,
            "pointer":  int(pointer),
        }

    def _act(self, action: dict) -> tuple[np.ndarray, int, int]:
        dpad = int(action["dpad"])
        self._movement_press(dpad)

        self._update_agent_position()

        pointer = int(action["pointer"])
        self._point_mouse(pointer)

        buttons = np.array(action["buttons"], dtype=np.int8, copy=True)
        self._button_press(buttons)

        return buttons, dpad, pointer

    def _compute_reward(self, features: np.ndarray) -> dict:
        lives, bullets, level, enemies_destroyed, _, _ = features

        weights = self.reward_weights
        components = {
            "step_penalty": weights["step_penalty"],
            "enemy_destroyed": weights["enemy_destroyed"] * float(enemies_destroyed - self._previous_enemies),
            "life_lost": weights["life_lost"] * float(max(0, self._previous_lives - lives)),
            "level_advance": weights["level_advance"] * float(max(0, level - self._previous_level)),
        }

        self._previous_enemies = enemies_destroyed
        self._previous_lives = lives
        self._previous_level = level
        return components

    def _is_terminated(self, features: np.ndarray) -> bool:
        return int(features[0]) == 0

    def _sample_stage(self, max_stage: int) -> int:
        weights = 0.5 ** np.arange(max_stage, 0, -1)
        weights /= weights.sum()
        return int(np.random.choice(np.arange(1, max_stage + 1), p=weights))

    def _get_dolphin_window(self):
        for window in gw.getAllWindows():
            if window.title and self.window_name in window.title:
                if window.isMinimized:
                    window.restore()
                window.activate()
                window.moveTo(window.left, window.top)
                return window
        raise RuntimeError(
            f"Could not find a window titled '{self.window_name}'. "
            "Make sure Dolphin is running and the window name matches config."
        )

    def _capture_frame(self) -> np.ndarray:
        monitor = {
            "top":    self.y + CAPTURE_TOP_OFFSET,
            "left":   self.x + CAPTURE_LEFT_OFFSET,
            "height": self._capture_h,
            "width":  self._capture_w,
        }

        screenshot = self.sct.grab(monitor)
        rgb = np.frombuffer(screenshot.rgb, dtype=np.uint8).reshape(
            (screenshot.height, screenshot.width, 3)
        )
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    def _movement_press(self, dpad: int) -> None:
        if self.steps % MOVE_HOLD != 0:
            return

        keys_to_press = set(_MOVEMENT_MAP.get(dpad, ()))
        for i, (key, pressed) in enumerate(self._movement_actions):
            should_press = key in keys_to_press
            if pressed and not should_press:
                pdi.keyUp(key)
                self._movement_actions[i][1] = False
            elif not pressed and should_press:
                pdi.keyDown(key)
                self._movement_actions[i][1] = True

    def _button_press(self, buttons: np.ndarray) -> None:
        for i, active in enumerate(buttons):
            if active:
                _press_key(self._button_keys[i])

    def _point_mouse(self, aim_bin: int) -> None:
        screen_x = self.x + CAPTURE_LEFT_OFFSET + int(round(self.user_x + 70 * _AIM_BIN_COS[aim_bin]))
        screen_y = self.y + CAPTURE_TOP_OFFSET  + int(round(self.user_y + 70 * _AIM_BIN_SIN[aim_bin]))

        pdi.moveTo(screen_x, screen_y)