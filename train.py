"""Entry point for training the MaskablePPO tank agent."""
import os

import numpy as np
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.monitor import Monitor

from config import EMULATOR_CONFIG
from curriculum import CurriculumCallback, CurriculumCheckpointCallback, EmulatorPauseCallback
from custom_ppo import AuxAimMaskablePPO
from feature_extractor import TankCNN
from tank_env_wrapper import make_tank_env

MODELS_DIR = "./models"

SAVE_FREQ = 50_000
EVAL_FREQ = 60
TOTAL_TIMESTEPS = 10_000_000

# Set None to train from scratch.
MODEL_PATH = None
_POLICY_KWARGS = dict(
    features_extractor_class=TankCNN,
    features_extractor_kwargs=dict(
        shared_dim=128,
        head_dim=64,
    ),
)


def mask_fn(env) -> np.ndarray:
    """Helper function to extract action masks from the environment wrapper."""
    return env.action_masks()


def main() -> None:
    """Build environment, construct or reload model, and run training."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    curriculum_callback = CurriculumCallback(
        eval_freq=EVAL_FREQ,
    )

    if MODEL_PATH is not None:
        curriculum_callback.load_state(MODEL_PATH.replace(".zip", "_curriculum.json"))

    checkpoint_callback = CurriculumCheckpointCallback(
        curriculum_callback=curriculum_callback,
        save_freq=SAVE_FREQ,
        save_path=MODELS_DIR,
        name_prefix="tank_model",
    )

    pause_callback = EmulatorPauseCallback()

    print("[Train] Building environment...")
    raw_env = make_tank_env(
        EMULATOR_CONFIG,
        curriculum_stage=curriculum_callback.current_stage,
        save_frames=False,
        save_aim_debug=False,
    )
    masked_env = ActionMasker(raw_env, mask_fn)
    env = Monitor(masked_env)
    print("[Train] Environment ready.")

    print("[Train] Initialising model...")
    if MODEL_PATH is None:
        model = AuxAimMaskablePPO(
            "MultiInputPolicy",
            env,
            verbose=1,
            n_steps=8192,
            batch_size=512,
            ent_coef=0.1,
            policy_kwargs=_POLICY_KWARGS,
            gamma=0.995,
            gae_lambda=0.998,
            target_kl=0.02,
            n_epochs=10,
            aux_loss_coef=0.1,
            learning_rate=6e-4,
        )
    else:
        model = AuxAimMaskablePPO.load(MODEL_PATH, env=env)

    print(
        f"[Train] Starting training for {TOTAL_TIMESTEPS:,} timesteps | "
        f"Current curriculum stage: {curriculum_callback.current_stage} | "
        f"Checkpoint every {SAVE_FREQ:,} steps | "
        f"Evaluation every {EVAL_FREQ} episodes"
    )

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=[checkpoint_callback, curriculum_callback, pause_callback],
        reset_num_timesteps=False,
    )

    print("[Train] Training complete.")


if __name__ == "__main__":
    main()