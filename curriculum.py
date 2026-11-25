"""Curriculum learning callbacks for staged reward progression."""
import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import tank_env


class CurriculumCallback(BaseCallback):
    """Advance curriculum stage when mean win rate clears a threshold for
    two consecutive evaluation windows. No masking systems are involved --
    the agent always sees the raw action space (aside from the physical
    ammo/wall constraints enforced in action_masks()), so promotion is
    gated purely by observed success rate.

    Entropy is annealed toward min_entropy as win_rate approaches
    threshold, and reset to max_entropy on stage advance so the freshly
    harder stage gets full exploration again.

    Args:
        eval_freq: Number of episodes between curriculum evaluations.
        current_stage: Stage to begin training from.
        max_stage: Highest available curriculum stage.
        max_entropy: Maximum bound for entropy coefficient.
        min_entropy: Minimum bound for entropy coefficient.
        threshold: Target success rate threshold to advance curriculum stages.
        verbose: Stable-Baselines3 verbosity level.
    """

    def __init__(
            self,
            eval_freq: int,
            current_stage: int = 1,
            max_stage: int = 7,
            max_entropy: float = 0.001,
            min_entropy: float = 0.0001,
            threshold: float = 0.60,
            verbose: int = 0,
    ) -> None:
        super().__init__(verbose)
        self.eval_freq = eval_freq
        self.current_stage = current_stage
        self.max_stage = max_stage
        self.threshold_streak = 0
        self.threshold = threshold
        self.previous_episode_count = 0
        self.max_entropy = max_entropy
        self.min_entropy = min_entropy
        self.ent_coef = max_entropy

        self.stage_window_count = 0

    def _compute_target_entropy(self, win_rate: float) -> float:
        """Entropy relaxes smoothly as win_rate approaches threshold -- the
        only performance signal left now that masking is gone. Clipped so
        exceeding threshold doesn't push entropy below min_entropy before
        the streak-gated promotion actually happens.
        """
        progress = float(np.clip(win_rate / self.threshold, 0.0, 1.0)) if self.threshold > 0 else 1.0
        return self.max_entropy - progress * (self.max_entropy - self.min_entropy)

    def save_state(self, filename: str) -> None:
        """Save curriculum progress to a JSON file."""
        state = {
            "current_stage": self.current_stage,
            "threshold_streak": self.threshold_streak,
            "entropy": self.ent_coef,
            "stage_window_count": self.stage_window_count,
        }
        with open(filename, "w") as f:
            json.dump(state, f)

    def load_state(self, filename: str) -> None:
        """Load curriculum progress from a JSON file."""
        with open(filename, "r") as f:
            state = json.load(f)
        self.current_stage = state["current_stage"]
        self.threshold_streak = state["threshold_streak"]
        self.ent_coef = state.get("entropy", self.max_entropy)
        self.stage_window_count = state.get("stage_window_count", 0)

    def _init_callback(self) -> None:
        """Executed automatically by SB3 when model.learn() starts."""
        super()._init_callback()

        if self.training_env is not None:
            self.training_env.env_method("set_curriculum", self.current_stage)

            self.model.ent_coef = self.ent_coef

            print(
                f"[Curriculum] Initialized Env -> Stage: {self.current_stage} | "
                f"Entropy: {self.model.ent_coef:.5f}"
            )

    def _on_step(self) -> bool:
        episode_count = sum(self.training_env.get_attr("window_episode_count"))
        success_rate = 0.0

        if episode_count > 0:
            success_rate = sum(self.training_env.get_attr("window_successes")) / episode_count
        if episode_count <= self.previous_episode_count or episode_count < self.eval_freq:
            self.previous_episode_count = episode_count
            return True

        self.stage_window_count += 1

        target_entropy = self._compute_target_entropy(success_rate)

        self.model.ent_coef = float(np.clip(
            (0.9 * self.model.ent_coef + 0.1 * target_entropy),
            self.min_entropy,
            self.max_entropy
        ))
        self.ent_coef = float(self.model.ent_coef)

        print(
            f"[Curriculum] Success Rate: {success_rate * 100:.2f}% | "
            f"Stage Threshold: {self.threshold * 100:.2f}%"
        )

        print(f"[Curriculum] Entropy coefficient set to {self.model.ent_coef:.5f}")

        print(
            f"[Curriculum] Step {self.n_calls:,} | Stage {self.current_stage}/{self.max_stage} | "
            f"Window win rate: {success_rate * 100:.2f}% ({episode_count} eps)"
        )

        win_threshold_met = success_rate >= self.threshold

        if win_threshold_met:
            self.threshold_streak += 1
            if self.threshold_streak >= 2:
                if self.current_stage < self.max_stage:
                    previous_stage = self.current_stage
                    self.current_stage += 1
                    self.threshold_streak = 0
                    self.stage_window_count = 0
                    self.training_env.env_method("set_curriculum", self.current_stage)

                    self.model.ent_coef = self.max_entropy
                    self.ent_coef = float(self.max_entropy)

                    print(
                        f"[Curriculum] Current success rate: {success_rate * 100:.2f}% | "
                        f"Threshold success rate: {self.threshold * 100:.2f}% | "
                        f"Threshold met (streak 2/2) | "
                        f"Stage advanced from {previous_stage} to {self.current_stage}"
                    )
                else:
                    print(f"[Curriculum] Max curriculum level {self.max_stage} already mastered.")
            else:
                print(
                    f"[Curriculum] Current success rate: {success_rate * 100:.2f}% | "
                    f"Threshold success rate: {self.threshold * 100:.2f}% | "
                    f"Threshold met (streak 1/2)"
                )
        else:
            self.threshold_streak = 0
            print(
                f"[Curriculum] Current success rate: {success_rate * 100:.2f}% | "
                f"Threshold success rate: {self.threshold * 100:.2f}% | "
                f"Promotion blocked (streak 0/2): Win rate too low "
                f"({success_rate * 100:.2f}% < {self.threshold * 100:.2f}%)"
            )

        self.training_env.env_method("_reset_window")
        self.previous_episode_count = 0
        return True


class CurriculumCheckpointCallback(CheckpointCallback):
    """CheckpointCallback that saves curriculum state alongside model weights."""

    def __init__(self, curriculum_callback: CurriculumCallback, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.curriculum_callback = curriculum_callback

    def _on_step(self) -> bool:
        result = super()._on_step()

        if self.n_calls % self.save_freq == 0:
            model_name = f"{self.name_prefix}_{self.num_timesteps}_steps"
            state_path = Path(self.save_path) / f"{model_name}_curriculum.json"
            self.curriculum_callback.save_state(state_path)

        return result


class EmulatorPauseCallback(BaseCallback):
    def __init__(self) -> None:
        super().__init__()
        self.emulator_paused = False

    def _on_rollout_start(self) -> None:
        if self.emulator_paused:
            tank_env._press_key("f10", 0.05)
            self.emulator_paused = False
            print("[Emulator] Resumed")

    def _on_rollout_end(self) -> None:
        if not self.emulator_paused:
            tank_env._press_key("f10", 0.05)
            self.emulator_paused = True
            print("[Emulator] Paused")

    def _on_step(self) -> bool:
        return True
