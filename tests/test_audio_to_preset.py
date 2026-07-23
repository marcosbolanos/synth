import torch

from synth.models.audio_to_preset import AudioToPresetConfig, AudioToPresetTransformer


def test_log_mel_model_contract() -> None:
    config = AudioToPresetConfig(feature_name="log_mel", template_count=11)
    model = AudioToPresetTransformer(config)
    controls, templates, audio = model(torch.zeros(2, 2, 128, 690))

    assert controls.shape == (2, 903)
    assert templates.shape == (2, 11)
    assert audio.shape == (2, 128)
