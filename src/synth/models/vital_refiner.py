from __future__ import annotations

from typing import NamedTuple

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn


class VitalRefinerConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    control_count: int = Field(default=903, gt=0)
    input_channels: int = Field(default=2, gt=0)
    mel_bins: int = Field(default=128, gt=0)
    frames: int = Field(default=690, gt=0)
    width: int = Field(default=256, gt=0)
    audio_layers: int = Field(default=4, gt=0)
    attention_heads: int = Field(default=8, gt=0)
    feedforward_width: int = Field(default=1024, gt=0)
    ensemble_heads: int = Field(default=3, gt=1)
    dropout: float = Field(default=0.05, ge=0.0, lt=1.0)


class EditProposal(NamedTuple):
    relevance_logits: Tensor
    delta_mean: Tensor
    delta_log_scale: Tensor
    stop_logit: Tensor


class TransitionPrediction(NamedTuple):
    next_latents: Tensor
    distances: Tensor


class VitalRefiner(nn.Module):
    """Audio-conditioned sparse editor with an action-conditioned latent world model."""

    def __init__(self, config: VitalRefinerConfig) -> None:
        super().__init__()
        self.config = config
        self.patch = nn.Conv2d(
            config.input_channels,
            config.width,
            kernel_size=(16, 20),
            stride=(16, 20),
        )
        patch_count = (config.mel_bins // 16) * (config.frames // 20)
        self.audio_cls = nn.Parameter(torch.zeros(1, 1, config.width))
        self.audio_position = nn.Parameter(torch.randn(1, patch_count + 1, config.width) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=config.width,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_width,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.audio_encoder = nn.TransformerEncoder(layer, config.audio_layers)
        self.audio_norm = nn.LayerNorm(config.width)
        self.audio_summary_head = nn.Linear(
            config.width, config.input_channels * config.mel_bins * 2
        )

        self.preset_encoder = nn.Sequential(
            nn.Linear(config.control_count * 2, config.feedforward_width),
            nn.GELU(),
            nn.Linear(config.feedforward_width, config.width),
            nn.LayerNorm(config.width),
        )
        policy_input = config.width * 3
        self.policy = nn.Sequential(
            nn.Linear(policy_input, config.feedforward_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_width, config.control_count * 3 + 1),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(config.control_count * 2, config.feedforward_width),
            nn.GELU(),
            nn.Linear(config.feedforward_width, config.width),
        )
        self.world_trunk = nn.Sequential(
            nn.Linear(config.width * 3, config.feedforward_width),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.feedforward_width, config.width),
            nn.GELU(),
        )
        self.world_heads = nn.ModuleList(
            nn.Linear(config.width, config.width) for _ in range(config.ensemble_heads)
        )
        self.distance_heads = nn.ModuleList(
            nn.Linear(config.width, 1) for _ in range(config.ensemble_heads)
        )

    def encode_audio(self, feature: Tensor) -> tuple[Tensor, Tensor]:
        expected = (
            self.config.input_channels,
            self.config.mel_bins,
            self.config.frames,
        )
        if feature.ndim != 4 or tuple(feature.shape[1:]) != expected:
            raise ValueError(f"Expected audio feature [batch, {expected}], got {tuple(feature.shape)}")
        tokens = self.patch(feature).flatten(2).transpose(1, 2)
        cls = self.audio_cls.expand(feature.shape[0], -1, -1)
        tokens = torch.cat((cls, tokens), dim=1)
        encoded = self.audio_norm(self.audio_encoder(tokens + self.audio_position))
        latent = encoded[:, 0]
        return latent, self.audio_summary_head(latent)

    def encode_preset(self, controls: Tensor, mask: Tensor) -> Tensor:
        expected = (controls.shape[0], self.config.control_count)
        if controls.ndim != 2 or tuple(controls.shape) != expected:
            raise ValueError(f"Expected controls [batch, {self.config.control_count}]")
        if tuple(mask.shape) != expected:
            raise ValueError(f"Expected mask shape {expected}")
        return self.preset_encoder(torch.cat((controls, mask.to(controls.dtype)), dim=-1))

    def propose(
        self,
        target_audio_latent: Tensor,
        current_audio_latent: Tensor,
        preset_latent: Tensor,
    ) -> EditProposal:
        hidden = self.policy(
            torch.cat((target_audio_latent, current_audio_latent, preset_latent), dim=-1)
        )
        relevance, mean, scale, stop = torch.split(
            hidden,
            (self.config.control_count,) * 3 + (1,),
            dim=-1,
        )
        return EditProposal(relevance, mean, scale.clamp(-5.0, 2.0), stop)

    def predict_transition(
        self,
        current_audio_latent: Tensor,
        preset_latent: Tensor,
        action_delta: Tensor,
        action_mask: Tensor,
    ) -> TransitionPrediction:
        expected = (action_delta.shape[0], self.config.control_count)
        if action_delta.ndim != 2 or tuple(action_delta.shape) != expected:
            raise ValueError(f"Expected action delta [batch, {self.config.control_count}]")
        if tuple(action_mask.shape) != expected:
            raise ValueError(f"Expected action mask shape {expected}")
        action = self.action_encoder(
            torch.cat((action_delta, action_mask.to(action_delta.dtype)), dim=-1)
        )
        hidden = self.world_trunk(
            torch.cat((current_audio_latent, preset_latent, action), dim=-1)
        )
        next_latents = torch.stack(
            tuple(current_audio_latent + head(hidden) for head in self.world_heads),
            dim=1,
        )
        distances = torch.cat(tuple(head(hidden) for head in self.distance_heads), dim=1)
        return TransitionPrediction(next_latents, distances)

    @staticmethod
    def audio_summary(feature: Tensor) -> Tensor:
        if feature.ndim != 4:
            raise ValueError("Expected a four-dimensional audio feature batch")
        return torch.cat((feature.mean(dim=-1), feature.std(dim=-1)), dim=-1).flatten(1)
