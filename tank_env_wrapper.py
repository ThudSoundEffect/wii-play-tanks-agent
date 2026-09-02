"""SB3 wrapper: observation preprocessing, shaped rewards, motion-aware aim forecasting, scheduled aim guidance, and action masking."""
import math
import os

import cv2
import gymnasium as gym
import numpy as np
from gymnasium import spaces
from numba import njit

import config
from tank_env import AIM_DIRECTIONS, TankEnv

FRAME_H = 107
FRAME_W = 207

AGENT_RADAR_MAX = 30
AGENT_RAY_COUNT = 40

AIM_RADAR_MAX = 12
AIM_RAY_COUNT = 20

AIM_FORECAST_MAX = FRAME_W + FRAME_H

CURRICULUM_CAP_BONUS = 50.0
WALL_MAX = 10
FIELD_MIN = 245

THREAT_GRAY_MIN = 33
THREAT_GRAY_MAX = 222

THREAT_VALUES = np.array([42, 61, 80, 99, 118, 137, 156, 175, 194, 213], dtype=np.float32)
THREAT_TOLERANCE = 9
NUM_THREAT_TYPES = len(THREAT_VALUES)

BULLET_THREAT_TYPE = 0

THREAT_TYPE_SCORE = np.linspace(0.3, 1.0, NUM_THREAT_TYPES, dtype=np.float32)

DETECT_WALL = 1.0
DETECT_THREAT_BASE = 2.0
NUM_DETECT_CLASSES = int(DETECT_THREAT_BASE) + NUM_THREAT_TYPES


FEATURE_NORM = np.array([3.0, 5.0, 20.0, 100.0, 1.0, 1.0], dtype=np.float32)

_AGENT_ANGLES = np.deg2rad(np.arange(0, 360, 360 / AGENT_RAY_COUNT))
_AGENT_COS = np.cos(_AGENT_ANGLES)
_AGENT_SIN = np.sin(_AGENT_ANGLES)

_AIM_ANGLES = np.deg2rad(np.arange(0, 360, 360 / AIM_RAY_COUNT))
_AIM_COS = np.cos(_AIM_ANGLES)
_AIM_SIN = np.sin(_AIM_ANGLES)

_AIM_BIN_COS = np.cos(np.deg2rad(np.arange(AIM_DIRECTIONS) * 360.0 / AIM_DIRECTIONS))
_AIM_BIN_SIN = np.sin(np.deg2rad(np.arange(AIM_DIRECTIONS) * 360.0 / AIM_DIRECTIONS))

SUB_RAYS_PER_BIN = 3
SUB_ANGLE_OFFSETS_RAD = np.deg2rad(np.linspace(-5.0, 5.0, SUB_RAYS_PER_BIN))


SELF_CLEARANCE_STEPS = 5


@njit(cache=True)
def _march_single_ray(
    user_x: float, user_y: float, cos_a: float, sin_a: float,
    frame: np.ndarray, wall_mask: np.ndarray, threat_type_map: np.ndarray, lead_threat_mask: np.ndarray,
    frame_w: int, frame_h: int, forecast_max: int,
    field_min: float,
):
    x = user_x
    y = user_y
    has_bounce = False
    bounced = False
    bounce_ix = 0
    bounce_iy = 0
    forecast_len = 0
    direct_hit = False
    lead_hit = False
    direct_hit_type = -1
    lead_hit_type = -1
    min_self_dist = 1e9

    for r in range(forecast_max):
        forecast_len = r
        nx = x + cos_a
        ny = y + sin_a

        if not (0.0 <= nx < frame_w and 0.0 <= ny < frame_h):
            break

        ix = int(min(max(nx, 0.0), frame_w - 1))
        iy = int(min(max(ny, 0.0), frame_h - 1))
        pixel = frame[iy, ix]

        if bounced and r >= 5:
            dx = nx - user_x
            dy = ny - user_y
            self_dist = math.sqrt(dx * dx + dy * dy)
            if self_dist < min_self_dist:
                min_self_dist = self_dist
            if self_dist <= 5.0:
                break

        skip_self_hit_test = (not bounced) and (r < SELF_CLEARANCE_STEPS)

        if not skip_self_hit_test:
            if wall_mask[iy, ix]:
                if bounced:
                    break
                bounced = True
                has_bounce = True
                bounce_ix = ix
                bounce_iy = iy

                iy_prev = int(min(max(round(y), 0.0), frame_h - 1))
                ix_prev = int(min(max(round(x), 0.0), frame_w - 1))

                hit_x = wall_mask[iy_prev, ix]
                hit_y = wall_mask[iy, ix_prev]
                if hit_x:
                    cos_a = -cos_a
                if hit_y:
                    sin_a = -sin_a
                if (not hit_x) and (not hit_y):
                    cos_a = -cos_a
                    sin_a = -sin_a
            else:
                threat_type = threat_type_map[iy, ix]
                if threat_type >= 0:
                    direct_hit = True
                    direct_hit_type = threat_type
                if lead_threat_mask[iy, ix] > 0:
                    lead_hit = True
                    if threat_type >= 0:
                        lead_hit_type = threat_type

                if direct_hit or (pixel < field_min and not wall_mask[iy, ix]):
                    break

        x = nx
        y = ny

    end_x = int(min(max(round(x + 2 * cos_a), 0.0), frame_w - 1))
    end_y = int(min(max(round(y + 2 * sin_a), 0.0), frame_h - 1))

    if not has_bounce:
        bounce_ix = end_x
        bounce_iy = end_y

    return (forecast_len, end_x, end_y, bounce_ix, bounce_iy, direct_hit, lead_hit,
            direct_hit_type, lead_hit_type, min_self_dist)


@njit(cache=True)
def _march_aim_bin_jit(
    user_x: float, user_y: float, base_angle: float, sub_offsets: np.ndarray,
    frame: np.ndarray, wall_mask: np.ndarray, threat_type_map: np.ndarray, lead_threat_mask: np.ndarray,
    frame_w: int, frame_h: int, forecast_max: int,
    field_min: float,
    primary_idx: int,
):
    best_direct_hit = False
    best_lead_hit = False
    best_direct_type = -1
    best_lead_type = -1
    best_min_self_dist = 1e9

    primary_f_len = 0
    primary_ep_x = 0
    primary_ep_y = 0
    primary_bp_x = 0
    primary_bp_y = 0

    n_sub = sub_offsets.shape[0]
    for sub_idx in range(n_sub):
        angle = base_angle + sub_offsets[sub_idx]
        cos_a = math.cos(angle)
        sin_a = math.sin(angle)

        (f_len, ep_x, ep_y, bp_x, bp_y,
         direct_hit, lead_hit, direct_type, lead_type, min_self_dist) = _march_single_ray(
            user_x, user_y, cos_a, sin_a,
            frame, wall_mask, threat_type_map, lead_threat_mask,
            frame_w, frame_h, forecast_max,
            field_min,
        )

        if sub_idx == primary_idx:
            primary_f_len = f_len
            primary_ep_x = ep_x
            primary_ep_y = ep_y
            primary_bp_x = bp_x
            primary_bp_y = bp_y

        if direct_hit:
            best_direct_hit = True
            if direct_type >= 0:
                best_direct_type = direct_type
        if lead_hit:
            best_lead_hit = True
            if lead_type >= 0:
                best_lead_type = lead_type
        if min_self_dist < best_min_self_dist:
            best_min_self_dist = min_self_dist

    return (primary_f_len, primary_ep_x, primary_ep_y, primary_bp_x, primary_bp_y,
            best_direct_hit, best_lead_hit, best_direct_type, best_lead_type, best_min_self_dist)


SELF_AIM_THRESHOLD = AGENT_RADAR_MAX

_BULLET_LIMITS = [1, 1, 2, 3, 4, 5, 5]

_WALL_DILATE_KERNEL = np.ones((3, 3), np.uint8)
_THREAT_CLEAN_KERNEL = np.ones((3, 3), np.uint8)
_BULLET_CLEAN_KERNEL = np.ones((1, 1), np.uint8)

_THREAT_TYPE_DEBUG_COLORS = [
    (0, 0, 255),      # type 0 (bullet) - red
    (0, 128, 255),    # type 1 - orange
    (0, 255, 255),    # type 2 - yellow
    (0, 255, 0),      # type 3 - green
    (255, 255, 0),    # type 4 - teal
    (255, 128, 0),    # type 5 - light blue
    (255, 0, 0),      # type 6 - blue
    (255, 0, 255),    # type 7 - purple
    (128, 0, 255),    # type 8 - pink
    (203, 192, 255),  # type 9 - light pink
]


class TankEnvSB3Wrapper(gym.Wrapper):
    """Preprocessing and shaped-reward wrapper for use with Stable-Baselines3."""

    def __init__(
        self,
        env: gym.Env,
        save_frames: bool = False,
        save_aim_debug: bool = False,
        curriculum_stage: int = 1
    ) -> None:
        super().__init__(env)

        self.curriculum_stage = curriculum_stage
        self.save_frames = save_frames
        self.save_aim_debug = save_aim_debug

        self._feature_norm = FEATURE_NORM.copy()
        self._curriculum_weights = config.CURRICULUM_WEIGHTS
        self._set_weights(self._curriculum_weights[self.curriculum_stage])

        num_frames = env.num_frames
        num_features = env.num_features
        capture_w, capture_h = env._capture_dims()
        self._capture_w = capture_w
        self._capture_h = capture_h

        self.observation_space = spaces.Dict({
            "frames": spaces.Box(
                low=0, high=255,
                shape=(num_frames, FRAME_H, FRAME_W),
                dtype=np.uint8,
            ),
            "features": spaces.Box(low=0.0, high=1.0, shape=(num_features,), dtype=np.float32),
            "buttons": spaces.MultiBinary(1),
            "dpad": spaces.Discrete(8),
            "pointer": spaces.Discrete(AIM_DIRECTIONS),
            "agent_radar": spaces.Box(
                low=0.0, high=AGENT_RADAR_MAX,
                shape=(AGENT_RAY_COUNT, 2),
                dtype=np.float32,
            ),
            "aim_radar": spaces.Box(
                low=0.0, high=AIM_RADAR_MAX,
                shape=(AIM_RAY_COUNT, 2),
                dtype=np.float32,
            ),
            "aim_bin_scores": spaces.Box(
                low=-1.0, high=1.0,
                shape=(AIM_DIRECTIONS,),
                dtype=np.float32,
            ),
            "best_aim_bin": spaces.Discrete(AIM_DIRECTIONS),
            "aim_delta": spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32),
        })

        self.action_space = spaces.MultiDiscrete([2, 8, AIM_DIRECTIONS])

        self._num_frames = num_frames
        self._aim_radars = np.zeros((AIM_DIRECTIONS, AIM_RAY_COUNT, 2), dtype=np.float32)
        self._aim_accuracies = np.zeros(AIM_DIRECTIONS, dtype=np.float32)
        self._aim_threat_detected = np.zeros(AIM_DIRECTIONS, dtype=bool)
        self._aim_self_danger = np.zeros(AIM_DIRECTIONS, dtype=bool)
        self._aim_lens = np.zeros(AIM_DIRECTIONS, dtype=np.float32)
        self._aim_ends = np.zeros((AIM_DIRECTIONS, 2), dtype=np.float32)
        self._aim_bounces = np.zeros((AIM_DIRECTIONS, 2), dtype=np.float32)
        self._agent_radar = np.zeros((AGENT_RAY_COUNT, 2), dtype=np.float32)
        self._aim_radar = np.zeros((AIM_RAY_COUNT, 2), dtype=np.float32)
        self._agent_radar_origin = (0.0, 0.0)
        self._aim_radar_origin = (0.0, 0.0)
        self._bounce_origin = (0.0, 0.0)
        self.best_aim = 0

        self._action_mask = np.ones(2 + 8 + AIM_DIRECTIONS, dtype=bool)
        self._movement_mask = np.ones(8, dtype=bool)

        self._prev_danger = 0
        self._prev_bullets = 5
        self._prev_lives = 3
        self._prev_enemies = 0

        self.episode_reward = 0.0
        self.episode_reward_breakdown: dict = {}
        self.window_reward_breakdown: dict = {}

        self.window_reward_sum = 0.0
        self.window_episode_count = 0
        self.window_step_sum = 0
        self.window_successes = 0
        self.window_count = 1

        self.window_aim_steps = 0
        self.window_shots_fired = 0

        self.total_episodes = 0
        self.total_steps_lifetime = 0

    def set_curriculum(self, level: int) -> None:
        print(f"[Wrapper] Curriculum level set from {self.curriculum_stage} to {level}")
        self.curriculum_stage = level
        self._set_weights(self._curriculum_weights[self.curriculum_stage])

    def _set_weights(self, weights: dict) -> None:
        print(f"[Wrapper] Reward weights updated for stage {self.curriculum_stage}")
        self.reward_weights = weights
        self.weights = weights
        self.env.reward_weights = weights

    def _compute_enemy_norm(self) -> float:
        level = self.env.level
        if level <= 7:
            return float(config.STAGE_ENEMY_NORM[level])
        return float(config.STAGE_ENEMY_NORM[7] + (level - 7) * 5.6)

    def _reset_window(self) -> None:
        self._print_reward_distribution()
        self.window_reward_sum = 0.0
        self.window_reward_breakdown = {}
        self.window_episode_count = 0
        self.window_successes = 0
        self.window_step_sum = 0
        self.window_aim_steps = 0
        self.window_shots_fired = 0
        self.window_count += 1

    def reset(self, **kwargs) -> tuple[dict, dict]:
        if self.total_episodes > 0:
            episode_steps = self.env.steps
            self.window_reward_sum += self.episode_reward
            self.window_episode_count += 1
            self.window_step_sum += episode_steps
            self.total_steps_lifetime += episode_steps

            for category, value in self.episode_reward_breakdown.items():
                self.window_reward_breakdown[category] = (
                    self.window_reward_breakdown.get(category, 0.0) + value
                )

            window_average_reward = self.window_reward_sum / self.window_episode_count
            window_average_steps = self.window_step_sum / self.window_episode_count
            print(
                f"[Wrapper] Window: {self.window_count} | "
                f"Window timestep: {self.window_step_sum:,} | "
                f"Total training timestep: {self.total_steps_lifetime:,} | "
                f"Window success rate: {self.window_successes / self.window_episode_count * 100:.2f}%"
            )
            print(
                f"[Wrapper] Window ep: {self.window_episode_count} | "
                f"Ep reward: {self.episode_reward:.2f} | "
                f"Window avg reward: {window_average_reward:.2f} | "
                f"Window avg ep length: {window_average_steps:.1f} steps"
            )

        self.total_episodes += 1
        self.episode_reward = 0.0
        self.episode_reward_breakdown = {}

        if "options" not in kwargs or kwargs["options"] is None:
            kwargs["options"] = {}
        kwargs["options"]["curriculum_stage"] = self.curriculum_stage
        print(f"[Wrapper] Resetting env | Curriculum level: {self.curriculum_stage}")

        observation, info = self.env.reset(**kwargs)

        self.env._get_geometry()
        self._capture_w, self._capture_h = self.env._capture_dims()
        self.env._capture_w, self.env._capture_h = self._capture_w, self._capture_h

        self._prev_danger = 0
        self._prev_bullets = self.env.bullets_remaining
        self._prev_lives = self.env.lives
        self._prev_enemies = self.env.enemies_destroyed

        return self._process_obs(observation), info

    def step(self, action: np.ndarray) -> tuple[dict, float, bool, bool, dict]:
        prev_bullets = self._prev_bullets
        prev_lives = self._prev_lives

        decoded_action = self._decode_action(action)

        observation, reward, terminated, truncated, info = self.env.step(decoded_action)
        processed_observation = self._process_obs(observation)

        chosen_pointer = processed_observation["pointer"]

        reward_components = dict(info.get("reward_components", {}))
        reward_components.setdefault("curriculum_cap_bonus", 0.0)

        current_level = int(observation["features"][2])
        if current_level > self.curriculum_stage:
            truncated = True
            reward_components["curriculum_cap_bonus"] += CURRICULUM_CAP_BONUS
            self.window_successes += 1
            print(
                f"[Wrapper] Level exceeded curriculum cap ({current_level} > {self.curriculum_stage}) "
                f"truncating. +{CURRICULUM_CAP_BONUS:.1f} bonus."
            )

        reward_components["radar_change"] = self._radar_change_penalty(processed_observation)

        aim_accuracy_value = float(self._aim_accuracies[chosen_pointer])
        aim_choice_penalty_value = self._aim_choice_penalty(
            chosen_pointer, processed_observation["best_aim_bin"]
        )

        fired = bool(processed_observation["buttons"][0])

        if fired:
            if aim_accuracy_value > 0.0:
                shot_value = aim_accuracy_value * self.weights.get("fire_hit_scale", 3.0)
            else:
                shot_value = self.weights.get("fire_miss_scale", 0.0)
            self.window_shots_fired += 1
        else:
            shot_value = aim_accuracy_value

        reward_components["aim_accuracy"] = shot_value
        reward_components["aim_choice_penalty"] = aim_choice_penalty_value
        self.window_aim_steps += 1

        wall_block = self._wall_detect()
        self._movement_mask = ~wall_block

        self._prev_bullets = self.env.bullets_remaining
        self._prev_lives = self.env.lives
        self._prev_enemies = self.env.enemies_destroyed

        reward = sum(reward_components.values())
        info["reward_components"] = reward_components

        self.episode_reward += reward
        for category, value in reward_components.items():
            self.episode_reward_breakdown[category] = (
                self.episode_reward_breakdown.get(category, 0.0) + value
            )

        if terminated:
            print(f"[Wrapper] Episode terminated at step {self.env.steps:,}")
        elif truncated:
            print(f"[Wrapper] Episode truncated at step {self.env.steps:,}")

        return processed_observation, reward, terminated, truncated, info

    def action_masks(self) -> np.ndarray:
        self._action_mask[:] = True
        stage_index = self.curriculum_stage - 1

        if 0 <= stage_index < len(_BULLET_LIMITS):
            if (5 - self.env.bullets_remaining) >= _BULLET_LIMITS[stage_index]:
                self._action_mask[1] = False

        self._action_mask[2:10] = self._movement_mask

        return self._action_mask

    def _reward_category_components(self) -> dict:
        rewards = self.window_reward_breakdown

        core = {
            "enemy_destroyed": rewards.get("enemy_destroyed", 0.0),
            "curriculum_cap_bonus": rewards.get("curriculum_cap_bonus", 0.0),
            "level_advance": rewards.get("level_advance", 0.0),
        }
        shaping = {
            "aim_accuracy": rewards.get("aim_accuracy", 0.0),
            "radar_change": rewards.get("radar_change", 0.0),
        }
        penalty = {
            "aim_choice_penalty": rewards.get("aim_choice_penalty", 0.0),
            "step_penalty": rewards.get("step_penalty", 0.0),
        }
        guardrail = {
            "life_lost": rewards.get("life_lost", 0.0),
        }

        return {"core": core, "shaping": shaping, "penalty": penalty, "guardrail": guardrail}

    def _print_reward_distribution(self) -> None:
        components = self._reward_category_components()
        sums = {category: abs(sum(v for v in values.values())) for category, values in components.items()}
        total = sums["core"] + sums["shaping"] + sums["penalty"]

        print(f"[Wrapper] Window {self.window_count} reward distribution:")

        for category, abs_value in sums.items():
            share = (abs_value / total * 100) if total else 0.0
            print(f"{category}: {abs_value:>.2f} ({share:.1f}%)")
            for name, component_value in components[category].items():
                print(f"    {name}: {component_value:>+.2f}")

        aim_steps = self.window_aim_steps
        aim_avg = (
            self.window_reward_breakdown.get("aim_accuracy", 0.0) / aim_steps
            if aim_steps else float("nan")
        )
        print(f"[Wrapper] Aim accuracy per step: {aim_avg:.4f} ({aim_steps} steps)")
        print(f"[Wrapper] Shots fired this window: {self.window_shots_fired}")

    def _extract_threat_motion(self, stacked_frames: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        current_frame = stacked_frames[-1]
        prev_frame = stacked_frames[0]

        current_threats = (current_frame >= THREAT_GRAY_MIN) & (current_frame <= THREAT_GRAY_MAX)
        prev_threats = (prev_frame >= THREAT_GRAY_MIN) & (prev_frame <= THREAT_GRAY_MAX)

        curr_y, curr_x = np.where(current_threats)
        prev_y, prev_x = np.where(prev_threats)

        if len(curr_x) > 0 and len(prev_x) > 0:
            dx = np.mean(curr_x) - np.mean(prev_x)
            dy = np.mean(curr_y) - np.mean(prev_y)
            motion_vec = np.clip(np.array([dx, dy], dtype=np.float32), -10.0, 10.0)
        else:
            motion_vec = np.array([0.0, 0.0], dtype=np.float32)

        return current_threats, motion_vec

    def _create_lead_threat_mask(self, threat_mask: np.ndarray, motion_vec: np.ndarray) -> np.ndarray:
        speed = float(np.linalg.norm(motion_vec))
        if speed < 0.5:
            return threat_mask.astype(np.uint8)

        dx, dy = motion_vec
        angle = np.arctan2(dy, dx)

        lead_distance = int(np.clip(speed * 2.5, 3, 15))
        kernel_size = max(3, lead_distance)
        kernel = np.zeros((kernel_size, kernel_size), dtype=np.uint8)

        center = kernel_size // 2
        end_x = int(round(center + (kernel_size / 2) * np.cos(angle)))
        end_y = int(round(center + (kernel_size / 2) * np.sin(angle)))
        cv2.line(kernel, (center, center), (end_x, end_y), 1, thickness=2)

        return cv2.dilate(threat_mask.astype(np.uint8), kernel, iterations=1)

    def _aim_forecast_with_subangles(
        self,
        user_x: float,
        user_y: float,
        aim_bin: int,
        frame: np.ndarray,
        wall_mask: np.ndarray,
        threat_type_map: np.ndarray,
        lead_threat_mask: np.ndarray,
    ) -> tuple[int, tuple[int, int], tuple[int, int], bool, bool, int, int, float]:
        base_angle = aim_bin * 2.0 * math.pi / AIM_DIRECTIONS

        (f_len, ep_x, ep_y, bp_x, bp_y,
         direct_hit, lead_hit, direct_hit_type, lead_hit_type, min_self_dist) = _march_aim_bin_jit(
            float(user_x), float(user_y), base_angle, SUB_ANGLE_OFFSETS_RAD,
            frame, wall_mask, threat_type_map, lead_threat_mask,
            FRAME_W, FRAME_H, AIM_FORECAST_MAX,
            float(FIELD_MIN),
            SUB_RAYS_PER_BIN // 2,
        )

        return f_len, (ep_x, ep_y), (bp_x, bp_y), direct_hit, lead_hit, direct_hit_type, lead_hit_type, min_self_dist

    def _aim_accuracy_reward(
        self, aim_bin: int, direct_hit: bool, lead_hit: bool, hit_type: int, lead_hit_type: int,
        min_self_dist: float,
    ) -> tuple[float, bool, bool]:
        """Returns (reward, threat_detected, self_danger). threat_detected is True only
        when a real, non-bullet threat is actually in the firing line (direct or lead).
        self_danger is True if the bin's endpoint OR any point along a post-bounce path
        comes back within the self-aim safety radius — either signal alone means the
        shot can hit the agent, so both must be checked, not just the final endpoint.
        """
        aim_x, aim_y = self._aim_ends[aim_bin]
        agent_x, agent_y = self._agent_radar_origin

        self_danger = (
            math.dist((aim_x, aim_y), (agent_x, agent_y)) < SELF_AIM_THRESHOLD / 2
            or min_self_dist < SELF_AIM_THRESHOLD / 2
        )

        scale = self.weights.get("aim_reward_scale", 1.0)

        valid_direct = direct_hit and hit_type >= 0 and hit_type != BULLET_THREAT_TYPE
        valid_lead = lead_hit and lead_hit_type >= 0 and lead_hit_type != BULLET_THREAT_TYPE
        threat_detected = valid_direct or valid_lead

        if self_danger:
            return -1 * scale, threat_detected, self_danger

        if valid_lead and not valid_direct:
            type_score = float(THREAT_TYPE_SCORE[lead_hit_type])
            return scale * type_score * self.weights.get("lead_shot_bonus_scale", 1.5), threat_detected, self_danger
        elif valid_direct:
            type_score = float(THREAT_TYPE_SCORE[hit_type])
            return scale * type_score, threat_detected, self_danger

        return 0.0, threat_detected, self_danger

    def _filter_threat_components(self, threat_mask: np.ndarray) -> np.ndarray:
        """Drop connected components shaped like wall-edge bleed (thin, elongated,
        low fill-ratio slivers that trace a wall boundary) and keep components
        shaped like real entity sprites (compact, blob-like). This catches bleed
        that survives the wall dilation + MORPH_OPEN because it happened to be
        thicker than one pixel in every direction.
        """
        mask_u8 = threat_mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return threat_mask

        clean = np.zeros_like(threat_mask)
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area < 1:
                continue
            fill_ratio = area / float(w * h)
            aspect = max(w, h) / float(max(1, min(w, h)))
            if fill_ratio >= 0.5 and aspect <= 3.0:
                clean |= (labels == i)
        return clean

    def _process_obs(self, observation: dict) -> dict:
        enemy_norm = self._compute_enemy_norm()
        if enemy_norm != self._feature_norm[3]:
            self._feature_norm[3] = enemy_norm

        raw_features = observation["features"]
        features = np.clip(raw_features / self._feature_norm, 0.0, 1.0)

        stacked_frames = np.stack([
            cv2.resize(
                observation["frames"][:, :, i],
                (FRAME_W, FRAME_H),
                interpolation=cv2.INTER_NEAREST_EXACT,
            )
            for i in range(self._num_frames)
        ], axis=0)

        last_frame = stacked_frames[-1]

        wall_mask = cv2.dilate(
            (last_frame <= WALL_MAX).astype(np.uint8),
            _WALL_DILATE_KERNEL,
        ).astype(bool)

        _, motion_vec = self._extract_threat_motion(stacked_frames)

        diffs = np.abs(last_frame.astype(np.float32)[..., None] - THREAT_VALUES)
        nearest_type = np.argmin(diffs, axis=-1)
        nearest_diff = np.min(diffs, axis=-1)
        raw_classified = (nearest_diff <= THREAT_TOLERANCE) & (~wall_mask)

        threat_mask = np.zeros_like(wall_mask)
        threat_type_map = np.full(last_frame.shape, -1, dtype=np.int8)
        for t in range(NUM_THREAT_TYPES):
            type_mask = raw_classified & (nearest_type == t)
            kernel = _BULLET_CLEAN_KERNEL if t == BULLET_THREAT_TYPE else _THREAT_CLEAN_KERNEL
            type_mask = cv2.morphologyEx(
                type_mask.astype(np.uint8), cv2.MORPH_OPEN, kernel
            ).astype(bool)
            type_mask = self._filter_threat_components(type_mask)
            threat_mask |= type_mask
            threat_type_map[type_mask] = t

        lead_threat_mask = self._create_lead_threat_mask(threat_mask, motion_vec)

        agent_x = float(np.clip(raw_features[4] * FRAME_W, 0, FRAME_W - 1))
        agent_y = float(np.clip(raw_features[5] * FRAME_H, 0, FRAME_H - 1))

        aim_bin = int(observation["pointer"])
        self._agent_radar_origin = (agent_x, agent_y)

        self._make_radar(
            last_frame, agent_x, agent_y,
            AGENT_RADAR_MAX, AGENT_RAY_COUNT,
            self._agent_radar, _AGENT_COS, _AGENT_SIN, wall_mask, threat_type_map,
        )

        for i in range(AIM_DIRECTIONS):
            f_len, ep, bp, direct_hit, lead_hit, hit_type, lead_hit_type, min_self_dist = self._aim_forecast_with_subangles(
                agent_x, agent_y, i, last_frame, wall_mask, threat_type_map, lead_threat_mask
            )
            self._aim_lens[i] = f_len
            self._aim_ends[i] = ep
            self._aim_bounces[i] = bp

            aim_x, aim_y = self._aim_ends[i]

            self._make_radar(
                last_frame, aim_x, aim_y,
                AIM_RADAR_MAX, AIM_RAY_COUNT,
                self._aim_radars[i, :, :], _AIM_COS, _AIM_SIN, wall_mask, threat_type_map,
            )

            aim_reward, threat_detected, self_danger = self._aim_accuracy_reward(
                i, direct_hit, lead_hit, hit_type, lead_hit_type, min_self_dist
            )
            self._aim_accuracies[i] = aim_reward
            self._aim_threat_detected[i] = threat_detected
            self._aim_self_danger[i] = self_danger

        if np.max(self._aim_accuracies) > 0.0:
            self.best_aim = int(np.argmax(self._aim_accuracies))
        else:
            min_clearances = [
                -1.0 if self._aim_self_danger[i] else min(
                    math.dist(self._aim_ends[i], (agent_x, agent_y)),
                    math.dist(self._aim_bounces[i], (agent_x, agent_y))
                )
                for i in range(AIM_DIRECTIONS)
            ]
            self.best_aim = int(np.argmax(min_clearances))

        self._aim_radar = self._aim_radars[aim_bin].copy()
        self._aim_radar_origin = tuple(self._aim_ends[aim_bin])
        self._bounce_origin = tuple(self._aim_bounces[aim_bin])

        raw_delta = (self.best_aim - aim_bin + 12) % AIM_DIRECTIONS - 12
        aim_delta_norm = np.array([float(raw_delta / 12.0)], dtype=np.float32)

        if self.save_frames:
            self._save_framestack(stacked_frames)

        if self.save_aim_debug:
            self._save_aim_debug_framestack(last_frame, wall_mask, threat_type_map, lead_threat_mask)

        return {
            "frames": stacked_frames,
            "features": features,
            "buttons": np.asarray(observation["buttons"], dtype=np.int8),
            "dpad": int(observation["dpad"]),
            "pointer": aim_bin,
            "agent_radar": self._agent_radar.copy(),
            "aim_radar": self._aim_radar.copy(),
            "aim_bin_scores": self._aim_accuracies.copy(),
            "best_aim_bin": self.best_aim,
            "aim_delta": aim_delta_norm,
        }

    def _make_radar(
            self,
            frame: np.ndarray,
            origin_x: float,
            origin_y: float,
            r_max: int,
            ray_count: int,
            radar: np.ndarray,
            cos_lut: np.ndarray,
            sin_lut: np.ndarray,
            wall_mask: np.ndarray,
            threat_type_map: np.ndarray,
    ) -> None:
        radii = np.arange(0, r_max)

        xs = np.round(origin_x + np.outer(cos_lut, radii)).astype(np.int32)
        ys = np.round(origin_y + np.outer(sin_lut, radii)).astype(np.int32)

        in_bounds = (xs >= 0) & (xs < FRAME_W) & (ys >= 0) & (ys < FRAME_H)
        clipped_xs = np.clip(xs, 0, FRAME_W - 1)
        clipped_ys = np.clip(ys, 0, FRAME_H - 1)
        pixels = frame[clipped_ys, clipped_xs]

        hit_mask = in_bounds & (pixels < FIELD_MIN)

        any_hit = hit_mask.any(axis=1)
        first_hit_index = hit_mask.argmax(axis=1)

        oob_mask = ~in_bounds
        any_oob = oob_mask.any(axis=1)
        first_oob_index = oob_mask.argmax(axis=1)

        oob_first = any_oob & (~any_hit | (first_oob_index < first_hit_index))

        ray_idx = np.arange(ray_count)
        hit_xs = clipped_xs[ray_idx, first_hit_index]
        hit_ys = clipped_ys[ray_idx, first_hit_index]
        hit_is_wall = wall_mask[hit_ys, hit_xs]
        threat_type_idx = threat_type_map[hit_ys, hit_xs]

        is_classified_threat = (~hit_is_wall) & (threat_type_idx >= 0)

        detect_vals = np.where(
            is_classified_threat,
            DETECT_THREAT_BASE + threat_type_idx,
            DETECT_WALL,
        )

        radar[:, 0] = np.where(
            oob_first,
            radii[first_oob_index].astype(float),
            np.where(any_hit, radii[first_hit_index].astype(float), float(r_max)),
        )

        radar[:, 1] = np.where(
            oob_first,
            DETECT_WALL,
            np.where(any_hit, detect_vals, 0.0),
        )

    def _wall_detect(self) -> np.ndarray:
        stride = AGENT_RAY_COUNT // 8
        cardinal_rays = (-stride * np.arange(8)) % AGENT_RAY_COUNT

        offsets = np.arange(-stride // 2, stride // 2 + 1)
        bucket_rays = (cardinal_rays[:, None] + offsets[None, :]) % AGENT_RAY_COUNT

        bucket_types = self._agent_radar[bucket_rays, 1]
        bucket_dists = self._agent_radar[bucket_rays, 0]

        threshold = 13.0

        wall_in_bucket = (bucket_types == DETECT_WALL) & (bucket_dists <= threshold)
        return np.any(wall_in_bucket, axis=1)

    def _radar_change_penalty(self, observation: dict) -> float:
        radar = observation["agent_radar"]
        distances = radar[:, 0]
        detect_type = radar[:, 1]

        threat_distances = np.where(detect_type >= DETECT_THREAT_BASE, distances, float(AGENT_RADAR_MAX))
        danger_per_ray = np.clip(
            (AGENT_RADAR_MAX - threat_distances) / AGENT_RADAR_MAX,
            0.0,
            1.0,
        )

        max_weight = self.weights["radar_penalty_max_weight"]
        combined_danger = (
            max_weight * float(np.max(danger_per_ray))
            + (1.0 - max_weight) * float(np.mean(danger_per_ray))
        )

        current_danger = -combined_danger
        change_danger = current_danger - self._prev_danger
        self._prev_danger = current_danger

        if self.env.lives < self._prev_lives:
            change_danger = -1
        return change_danger * self.weights["radar_penalty_scale"]

    def _aim_choice_penalty(self, chosen_bin: int, best_bin: int) -> float:
        angular_dist = min(
            (chosen_bin - best_bin) % AIM_DIRECTIONS,
            (best_bin - chosen_bin) % AIM_DIRECTIONS,
        )
        if angular_dist == 0:
            return 0.0

        distance_factor = angular_dist / (AIM_DIRECTIONS / 2.0)
        scale = abs(self.weights.get("aim_choice_penalty_scale", 1.0))
        return -distance_factor * scale

    def _decode_action(self, action: np.ndarray) -> dict:
        return {
            "buttons": np.array([int(action[0])], dtype=np.int8),
            "dpad": int(action[1]),
            "pointer": int(action[2]),
        }

    def _draw_radar_rays(
        self,
        canvas: np.ndarray,
        radar: np.ndarray,
        origin: tuple,
        ray_count: int,
        hit_color: tuple,
        wall_color: tuple,
        miss_color: tuple,
        cos_lut: np.ndarray,
        sin_lut: np.ndarray,
    ) -> None:
        origin_x, origin_y = float(origin[0]), float(origin[1])
        origin_pt = (int(round(origin_x)), int(round(origin_y)))

        end_xs = np.round(origin_x + radar[:, 0] * cos_lut).astype(int)
        end_ys = np.round(origin_y + radar[:, 0] * sin_lut).astype(int)

        for i in range(ray_count):
            endpoint = (end_xs[i], end_ys[i])
            detection_type = radar[i, 1]
            color = hit_color if detection_type > 1.5 else (wall_color if detection_type > 0.5 else miss_color)
            cv2.line(canvas, origin_pt, endpoint, color, 1)
            if detection_type > 0.5:
                cv2.circle(canvas, endpoint, 2, color, -1)

        cv2.circle(canvas, origin_pt, 2, (255, 255, 0), -1)

    def _draw_radars(self, gray_frame: np.ndarray) -> np.ndarray:
        canvas = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)

        if self._agent_radar is None or self._aim_radar is None:
            return canvas

        self._draw_radar_rays(
            canvas, self._agent_radar, self._agent_radar_origin, AGENT_RAY_COUNT,
            hit_color=(0, 0, 255), wall_color=(0, 255, 255), miss_color=(0, 255, 0),
            cos_lut=_AGENT_COS, sin_lut=_AGENT_SIN,
        )
        self._draw_radar_rays(
            canvas, self._aim_radar, self._aim_radar_origin, AIM_RAY_COUNT,
            hit_color=(255, 0, 255), wall_color=(255, 0, 0), miss_color=(255, 255, 0),
            cos_lut=_AIM_COS, sin_lut=_AIM_SIN,
        )
        self._draw_radar_rays(
            canvas, self._aim_radars[self.best_aim], self._aim_ends[self.best_aim], AIM_RAY_COUNT,
            hit_color=(255, 0, 0), wall_color=(0, 255, 0), miss_color=(0, 0, 255),
            cos_lut=_AIM_COS, sin_lut=_AIM_SIN,
        )

        agent_x, agent_y = self._agent_radar_origin
        aim_x, aim_y = self._aim_radar_origin
        bounce_x, bounce_y = self._bounce_origin
        best_aim_x, best_aim_y = self._aim_ends[self.best_aim]
        best_bounce_x, best_bounce_y = self._aim_bounces[self.best_aim]

        cv2.line(canvas, (int(round(agent_x)), int(round(agent_y))), (int(round(bounce_x)), int(round(bounce_y))), (0, 0, 0), 1)
        cv2.line(canvas, (int(round(bounce_x)), int(round(bounce_y))), (int(round(aim_x)), int(round(aim_y))), (0, 0, 0), 1)

        cv2.line(canvas, (int(round(agent_x)), int(round(agent_y))),
                 (int(round(best_bounce_x)), int(round(best_bounce_y))),
                 (0, 255, 255), 1)
        cv2.line(canvas, (int(round(best_bounce_x)), int(round(best_bounce_y))),
                 (int(round(best_aim_x)), int(round(best_aim_y))),
                 (0, 255, 255), 1)

        cv2.drawMarker(canvas, (int(round(aim_x)), int(round(aim_y))), (0, 165, 255), markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1)

        return canvas

    def _save_framestack(self, frames: np.ndarray) -> None:
        save_dir = "pics"
        os.makedirs(save_dir, exist_ok=True)

        bgr_frames = [
            cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR) if frame.ndim == 2 else frame
            for frame in frames[:-1]
        ]
        bgr_frames.append(self._draw_radars(frames[-1]))

        grid = np.concatenate(bgr_frames, axis=1)
        filename = f"stack_{self.env.steps:06d}.png" if self.env.steps is not None else "stack.png"
        cv2.imwrite(os.path.join(save_dir, filename), grid)

    @staticmethod
    def _label_panel(panel: np.ndarray, label: str) -> np.ndarray:
        cv2.rectangle(panel, (0, 0), (panel.shape[1], 12), (0, 0, 0), -1)
        cv2.putText(
            panel, label, (2, 9), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, cv2.LINE_AA
        )
        return panel

    def _save_aim_debug_framestack(
        self,
        last_frame: np.ndarray,
        wall_mask: np.ndarray,
        threat_type_map: np.ndarray,
        lead_threat_mask: np.ndarray,
    ) -> None:
        save_dir = "pics_aim_debug"
        os.makedirs(save_dir, exist_ok=True)

        base_bgr = cv2.cvtColor(last_frame, cv2.COLOR_GRAY2BGR)
        panels = [self._label_panel(base_bgr.copy(), "frame")]

        wall_panel = base_bgr.copy()
        wall_panel[wall_mask] = (0, 255, 255)
        panels.append(self._label_panel(wall_panel, "wall_mask"))

        lead_panel = base_bgr.copy()
        lead_panel[lead_threat_mask] = (255, 0, 255)
        panels.append(self._label_panel(lead_panel, "lead_threat_mask"))

        for t in range(NUM_THREAT_TYPES):
            panel = base_bgr.copy()
            this_type_mask = threat_type_map == t
            color = _THREAT_TYPE_DEBUG_COLORS[t % len(_THREAT_TYPE_DEBUG_COLORS)]
            panel[this_type_mask] = color
            label = f"threat_type_{t}" + (" (bullet)" if t == BULLET_THREAT_TYPE else "")
            panels.append(self._label_panel(panel, label))

        forecast_panel = base_bgr.copy()
        agent_x, agent_y = self._agent_radar_origin
        agent_pt = (int(round(agent_x)), int(round(agent_y)))

        for i in range(AIM_DIRECTIONS):
            bounce_x, bounce_y = self._aim_bounces[i]
            end_x, end_y = self._aim_ends[i]
            bounce_pt = (int(round(bounce_x)), int(round(bounce_y)))
            end_pt = (int(round(end_x)), int(round(end_y)))

            if self._aim_self_danger[i]:
                color = (0, 0, 255)
            elif self._aim_threat_detected[i]:
                color = (0, 255, 0)
            else:
                color = (160, 160, 160)

            thickness = 2 if i == self.best_aim else 1
            cv2.line(forecast_panel, agent_pt, bounce_pt, color, thickness)
            cv2.line(forecast_panel, bounce_pt, end_pt, color, thickness)
            cv2.circle(forecast_panel, end_pt, 2, color, -1)

        if self.best_aim is not None:
            best_x, best_y = self._aim_ends[self.best_aim]
            cv2.drawMarker(
                forecast_panel, (int(round(best_x)), int(round(best_y))),
                (0, 255, 255), markerType=cv2.MARKER_STAR, markerSize=8, thickness=1,
            )
        cv2.circle(forecast_panel, agent_pt, 2, (255, 255, 0), -1)

        panels.append(self._label_panel(forecast_panel, "aim forecasts (all bins)"))

        grid = np.concatenate(panels, axis=1)
        filename = f"aimdebug_{self.env.steps:06d}.png" if self.env.steps is not None else "aimdebug.png"
        cv2.imwrite(os.path.join(save_dir, filename), grid)


def make_tank_env(
    emulator_config: dict,
    save_frames: bool = False,
    save_aim_debug: bool = False,
    curriculum_stage: int = 1,
) -> TankEnvSB3Wrapper:
    return TankEnvSB3Wrapper(
        TankEnv(config=emulator_config, weights=config.CURRICULUM_WEIGHTS[1]),
        save_frames=save_frames,
        save_aim_debug=save_aim_debug,
        curriculum_stage=curriculum_stage,
    )