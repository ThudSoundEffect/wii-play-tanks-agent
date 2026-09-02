import torch
import torch.nn as nn
import torch.nn.functional as F
from gymnasium import spaces
from gymnasium.spaces.utils import flatdim
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

NUM_DETECT_CLASSES = 12


class TankCNN(BaseFeaturesExtractor):
    """Refactored multi-branch feature extractor for Wii Play Tanks.

    Extracts a shared contextual representation with
    three embeddings for movement, aim, and buttons.
    """

    def __init__(
        self,
        observation_space: spaces.Dict,
        shared_dim: int = 128,
        head_dim: int = 64,
    ) -> None:
        total_features_dim = shared_dim + (head_dim * 3)
        super().__init__(observation_space, features_dim=total_features_dim)

        num_frames, height, width = observation_space["frames"].shape

        self.register_buffer(
            "_agent_radar_max",
            torch.tensor(float(observation_space["agent_radar"].high.flat[0])),
        )
        self.register_buffer(
            "_aim_radar_max",
            torch.tensor(float(observation_space["aim_radar"].high.flat[0])),
        )

        self.cnn = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((6, 6)),
            nn.Flatten(),
        )

        with torch.no_grad():
            cnn_output_dim = self.cnn(
                torch.zeros(1, num_frames, height, width)
            ).shape[1]

        self.cnn_proj = nn.Sequential(
            nn.Linear(cnn_output_dim, 128),
            nn.ReLU(inplace=True),
        )

        scalar_in_dim = flatdim(observation_space["features"])
        self.shared_encoder = nn.Sequential(
            nn.Linear(128 + scalar_in_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Linear(128, shared_dim),
            nn.ReLU(inplace=True),
        )

        agent_radar_rays = observation_space["agent_radar"].shape[0]
        agent_radar_encoded_dim = agent_radar_rays * (1 + NUM_DETECT_CLASSES)

        dpad_in_dim = flatdim(observation_space["dpad"]) + agent_radar_encoded_dim
        self.dpad_head = nn.Sequential(
            nn.Linear(dpad_in_dim + shared_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, head_dim),
            nn.ReLU(inplace=True),
        )

        aim_radar_rays = observation_space["aim_radar"].shape[0]
        aim_radar_encoded_dim = aim_radar_rays * (1 + NUM_DETECT_CLASSES)

        pointer_in_dim = (
            flatdim(observation_space["pointer"])
            + aim_radar_encoded_dim
            + flatdim(observation_space["aim_bin_scores"])
            + flatdim(observation_space["best_aim_bin"])
            + flatdim(observation_space["aim_delta"])
        )
        self.pointer_head = nn.Sequential(
            nn.Linear(pointer_in_dim + shared_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, head_dim),
            nn.ReLU(inplace=True),
        )

        buttons_in_dim = flatdim(observation_space["buttons"])
        self.buttons_head = nn.Sequential(
            nn.Linear(buttons_in_dim + shared_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, head_dim),
            nn.ReLU(inplace=True),
        )

    @staticmethod
    def _encode_radar(
        radar: torch.Tensor,
        max_dist: torch.Tensor,
    ) -> torch.Tensor:
        """Normalize the distance channel to [0, 1] and one-hot encode the
        detect-type channel (miss / wall / one of 10 threat types).
        """
        dist = (radar[..., 0] / max_dist).unsqueeze(-1)

        type_idx = radar[..., 1].round().long().clamp(0, NUM_DETECT_CLASSES - 1)
        type_onehot = F.one_hot(type_idx, num_classes=NUM_DETECT_CLASSES).float()

        encoded = torch.cat([dist, type_onehot], dim=-1)
        return encoded.reshape(radar.shape[0], -1)

    def forward(self, obs: dict[str, torch.Tensor]) -> torch.Tensor:
        frames = obs["frames"].float() / 255.0
        cnn_features = self.cnn_proj(self.cnn(frames))
        scalar_features = obs["features"].flatten(1)

        shared_ctx = self.shared_encoder(
            torch.cat([cnn_features, scalar_features], dim=1)
        )

        agent_radar = self._encode_radar(
            obs["agent_radar"], self._agent_radar_max
        )
        aim_radar = self._encode_radar(
            obs["aim_radar"], self._aim_radar_max
        )

        dpad = obs["dpad"].float().flatten(1)
        pointer = obs["pointer"].float().flatten(1)
        buttons = obs["buttons"].float().flatten(1)
        best_aim = obs["best_aim_bin"].float().flatten(1)
        aim_scores = obs["aim_bin_scores"].flatten(1)
        aim_delta = obs["aim_delta"].float().flatten(1)

        dpad_in = torch.cat([shared_ctx, dpad, agent_radar], dim=1)
        dpad_embed = self.dpad_head(dpad_in)

        pointer_in = torch.cat(
            [shared_ctx, pointer, aim_radar, aim_scores, best_aim, aim_delta],
            dim=1,
        )
        pointer_embed = self.pointer_head(pointer_in)

        buttons_in = torch.cat([shared_ctx, buttons], dim=1)
        buttons_embed = self.buttons_head(buttons_in)

        return torch.cat(
            [shared_ctx, dpad_embed, pointer_embed, buttons_embed], dim=1
        )