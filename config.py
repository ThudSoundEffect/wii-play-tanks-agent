"""Configuration settings for emulator, stages, and curriculum weights."""

EMULATOR_CONFIG = {
    "window_name": "Dolphin scripting-preview3-2361 | JIT64 SC | Direct3D 12 | HLE | Wii Play (RHAE01)",
    "num_frames": 8,
    "num_features": 6,
    "dolphin_path": r"dolphin/Dolphin.exe",
    "rom_path": r"Wii Play (USA, Canada) (Rev 1).wbfs",
    "state_path": r"tanks1.sav",
}

STAGE_ENEMY_NORM = {0: 0.0, 1: 1.0, 2: 2.0, 3: 5.0, 4: 9.0, 5: 11.0, 6: 15.0, 7: 20.0}

CURRICULUM_WEIGHTS = {
    1: {
        "enemy_destroyed": 50.0,
        "life_lost": -10.0,
        "level_advance": 25.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.01,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.001,
        "aim_choice_penalty_scale": 0.02,
        "fire_hit_scale": 5.0,
        "fire_miss_scale": -0.01,
    },

    2: {
        "enemy_destroyed": 50.0,
        "life_lost": -20.0,
        "level_advance": 50.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.05,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.005,
        "aim_choice_penalty_scale": 0.01,
        "fire_hit_scale": 3.0,
        "fire_miss_scale": -0.03,
    },

    3: {
        "enemy_destroyed": 50.0,
        "life_lost": -20.0,
        "level_advance": 50.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.05,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.01,
        "aim_choice_penalty_scale": 0.02,
        "fire_hit_scale": 3.0,
        "fire_miss_scale": -0.04,
    },

    4: {
        "enemy_destroyed": 50.0,
        "life_lost": -20.0,
        "level_advance": 50.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.05,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.01,
        "aim_choice_penalty_scale": 0.02,
        "fire_hit_scale": 3.0,
        "fire_miss_scale": -0.04,
    },

    5: {
        "enemy_destroyed": 50.0,
        "life_lost": -20.0,
        "level_advance": 50.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.05,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.01,
        "aim_choice_penalty_scale": 0.02,
        "fire_hit_scale": 3.0,
        "fire_miss_scale": -0.04,
    },

    6: {
        "enemy_destroyed": 50.0,
        "life_lost": -20.0,
        "level_advance": 50.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.05,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.01,
        "aim_choice_penalty_scale": 0.02,
        "fire_hit_scale": 3.0,
        "fire_miss_scale": -0.04,
    },

    7: {
        "enemy_destroyed": 50.0,
        "life_lost": -20.0,
        "level_advance": 50.0,

        "radar_penalty_max_weight": 0.9,
        "radar_penalty_scale": 0.5,
        "aim_reward_scale": 0.05,
        "lead_shot_bonus_scale": 1.5,

        "step_penalty": -0.01,
        "aim_choice_penalty_scale": 0.02,
        "fire_hit_scale": 3.0,
        "fire_miss_scale": -0.04,
    },
}