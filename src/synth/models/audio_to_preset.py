from __future__ import annotations

from typing import Literal

import torch
from pydantic import BaseModel, ConfigDict, Field
from torch import Tensor, nn


FeatureName = Literal["dac", "log_mel", "multi_stft", "waveform"]


class AudioToPresetConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    feature_name: FeatureName
    template_count: int = Field(gt=0)
    control_count: Literal[903] = 903
    audio_embedding_size: Literal[128] = 128
    model_width: Literal[384] = 384
    layers: Literal[10] = 10
    attention_heads: Literal[6] = 6
    feedforward_size: Literal[1536] = 1536
    dropout: float = Field(default=0.1, ge=0.0, lt=1.0)


def feature_patch_contract(name: FeatureName) -> tuple[int, tuple[int, int]]:
    if name == "log_mel":
        return 2, (16, 10)
    if name == "multi_stft":
        return 6, (16, 10)
    if name == "dac":
        return 1, (8, 5)
    if name == "waveform":
        return 1, (2, 32)
    raise AssertionError(name)


class AudioToPresetTransformer(nn.Module):
    def __init__(self, config: AudioToPresetConfig) -> None:
        super().__init__()
        self.config = config
        input_channels, patch = feature_patch_contract(config.feature_name)
        self.patch = nn.Conv2d(
            input_channels,
            config.model_width,
            kernel_size=patch,
            stride=patch,
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, config.model_width))
        self.positions = nn.Parameter(torch.randn(1, 1025, config.model_width) * 0.01)
        layer = nn.TransformerEncoderLayer(
            d_model=config.model_width,
            nhead=config.attention_heads,
            dim_feedforward=config.feedforward_size,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            layer,
            num_layers=config.layers,
            norm=nn.LayerNorm(config.model_width),
        )
        self.control_head = nn.Linear(config.model_width, config.control_count)
        self.template_head = nn.Linear(config.model_width, config.template_count)
        self.audio_head = nn.Linear(config.model_width, config.audio_embedding_size)

    def _image(self, feature: Tensor) -> Tensor:
        if self.config.feature_name in {"dac", "waveform"}:
            if feature.ndim != 3:
                raise ValueError(f"Expected [batch, height, width], got {feature.shape}")
            return feature.unsqueeze(1)
        if feature.ndim != 4:
            raise ValueError(f"Expected [batch, channels, height, width], got {feature.shape}")
        return feature

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        patches = self.patch(self._image(feature)).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(feature.shape[0], -1, -1)
        tokens = torch.cat((class_token, patches), dim=1)
        if tokens.shape[1] > self.positions.shape[1]:
            raise ValueError(f"Token count {tokens.shape[1]} exceeds positional capacity")
        encoded = self.encoder(tokens + self.positions[:, : tokens.shape[1]])[:, 0]
        return (
            self.control_head(encoded),
            self.template_head(encoded),
            self.audio_head(encoded),
        )


class PresetAudioSurrogate(nn.Module):
    def __init__(
        self,
        template_count: int,
        control_count: int = 903,
        template_width: int = 128,
        audio_embedding_size: int = 128,
        hidden_width: int = 1024,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        self.template_embedding = nn.Embedding(template_count, template_width)
        layers: list[nn.Module] = [
            nn.Linear(control_count + template_width, hidden_width),
            nn.GELU(),
        ]
        for _ in range(hidden_layers - 1):
            layers.extend((nn.Linear(hidden_width, hidden_width), nn.GELU()))
        layers.append(nn.Linear(hidden_width, audio_embedding_size))
        self.network = nn.Sequential(*layers)

    def forward(self, controls: Tensor, template_index: Tensor) -> Tensor:
        template = self.template_embedding(template_index)
        return self.network(torch.cat((controls, template), dim=-1))

    def forward_soft(self, controls: Tensor, template_probabilities: Tensor) -> Tensor:
        template = template_probabilities @ self.template_embedding.weight
        return self.network(torch.cat((controls, template), dim=-1))
