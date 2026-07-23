import torch

from synth.audio_features import AudioFeatureKind, extract_feature
from synth.experiment import RENDER_FRAME_COUNT


def test_audio_feature_shapes() -> None:
    audio = torch.zeros(2, RENDER_FRAME_COUNT)

    assert extract_feature(audio, AudioFeatureKind.LOG_MEL).shape == (2, 128, 690)
    assert extract_feature(audio, AudioFeatureKind.MULTI_STFT).shape == (6, 128, 690)
    assert extract_feature(audio, AudioFeatureKind.WAVEFORM).shape == (2, 4096)
