from __future__ import annotations

import pytest
import torch

from synth.models.vital_refiner import (
    VitalHybridRefiner,
    VitalRefiner,
    VitalRefinerConfig,
)


def compact_config() -> VitalRefinerConfig:
    return VitalRefinerConfig(
        control_count=17,
        mel_bins=32,
        frames=40,
        width=64,
        audio_layers=2,
        attention_heads=4,
        feedforward_width=128,
    )


def test_refiner_shapes() -> None:
    config = compact_config()
    model = VitalRefiner(config)
    feature = torch.randn(3, 2, 32, 40)
    controls = torch.randn(3, 17)
    mask = torch.ones(3, 17, dtype=torch.bool)
    audio, summary = model.encode_audio(feature)
    preset = model.encode_preset(controls, mask)
    proposal = model.propose(audio, audio, preset)
    transition = model.predict_transition(audio, preset, controls, mask)
    assert audio.shape == (3, 64)
    assert summary.shape == (3, 128)
    assert proposal.relevance_logits.shape == (3, 17)
    assert proposal.delta_mean.shape == (3, 17)
    assert transition.next_latents.shape == (3, 3, 64)
    assert transition.distances.shape == (3, 3)


def test_refiner_rejects_wrong_audio_contract() -> None:
    model = VitalRefiner(compact_config())
    with pytest.raises(ValueError, match="Expected audio feature"):
        model.encode_audio(torch.randn(2, 2, 31, 40))


def test_hybrid_goal_value_shapes() -> None:
    config = compact_config()
    model = VitalHybridRefiner(config)
    feature = torch.randn(3, 2, 32, 40)
    controls = torch.randn(3, 17)
    mask = torch.ones(3, 17, dtype=torch.bool)
    latent, _ = model.refiner.encode_audio(feature)
    preset = model.refiner.encode_preset(controls, mask)
    value = model.score_action(latent, latent, preset, controls, mask)
    assert value.distance.shape == (3, 3)
    assert value.improvement.shape == (3, 3)
