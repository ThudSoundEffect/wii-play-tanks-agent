"""MaskablePPO subclass that adds a supervised auxiliary loss predicting
the privileged best_aim_bin label from the pointer-head embedding.
"""
from typing import Optional

import numpy as np
import torch as th
import torch.nn.functional as F
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import explained_variance


class AuxAimMaskablePPO(MaskablePPO):
    """MaskablePPO + an auxiliary cross-entropy loss on the pointer head's
    aim-bin prediction.

    Args:
        aux_loss_coef: Weight on the auxiliary loss relative to the
            standard PPO loss (policy + entropy + value).
    """

    def __init__(
        self,
        *args,
        aux_loss_coef: float = 0.1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.aux_loss_coef = aux_loss_coef

    def _get_aim_aux_logits(self) -> Optional[th.Tensor]:
        policy = self.policy
        for attr in ("features_extractor", "pi_features_extractor"):
            extractor = getattr(policy, attr, None)
            logits = getattr(extractor, "_last_aim_aux_logits", None)
            if logits is not None:
                return logits
        return None

    def train(self) -> None:
        self.policy.set_training_mode(True)

        self._update_learning_rate(self.policy.optimizer)

        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]

        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses = [], []
        clip_fractions = []
        aux_losses = []

        continue_training = True

        for epoch in range(self.n_epochs):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size):
                actions = rollout_data.actions
                if isinstance(self.action_space, spaces.Discrete):
                    actions = rollout_data.actions.long().flatten()

                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    action_masks=rollout_data.action_masks,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                if self.normalize_advantage:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                ratio = th.exp(log_prob - rollout_data.old_log_prob)

                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    values_pred = values
                else:
                    values_pred = rollout_data.old_values + th.clamp(
                        values - rollout_data.old_values, -clip_range_vf, clip_range_vf
                    )
                value_loss = F.mse_loss(rollout_data.returns, values_pred)
                value_losses.append(value_loss.item())

                if entropy is None:
                    entropy_loss = -th.mean(-log_prob)
                else:
                    entropy_loss = -th.mean(entropy)

                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss


                aux_logits = self._get_aim_aux_logits()
                if aux_logits is not None and "best_aim_bin" in rollout_data.observations:
                    aux_target = rollout_data.observations["best_aim_bin"].long().flatten()

                    if aux_logits.shape[0] == aux_target.shape[0]:
                        aux_loss = F.cross_entropy(aux_logits, aux_target)
                        aux_losses.append(aux_loss.item())
                        loss = loss + self.aux_loss_coef * aux_loss

                with th.no_grad():
                    log_ratio = log_prob - rollout_data.old_log_prob
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    if self.verbose >= 1:
                        print(f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}")
                    break

                self.policy.optimizer.zero_grad()
                loss.backward()

                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            self._n_updates += 1
            if not continue_training:
                break
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if aux_losses:
            self.logger.record("train/aim_aux_loss", np.mean(aux_losses))  # <-- added
        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)
        if self.clip_range_vf is not None:
            self.logger.record("train/clip_range_vf", clip_range_vf)
