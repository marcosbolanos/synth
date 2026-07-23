from __future__ import annotations

from enum import StrEnum
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
import torchaudio
from pedalboard.io import AudioFile
from torch import Tensor

from synth.experiment import RENDER_CONFIG, RENDER_FRAME_COUNT


class AudioFeatureKind(StrEnum):
    DAC = "dac"
    LOG_MEL = "log_mel"
    MULTI_STFT = "multi_stft"
    WAVEFORM = "waveform"


def read_render(path: Path) -> Tensor:
    with AudioFile(str(path.resolve(strict=True))) as source:
        if source.samplerate != RENDER_CONFIG.sample_rate:
            raise ValueError(f"Expected {RENDER_CONFIG.sample_rate} Hz, got {source.samplerate}")
        audio = source.read(source.frames)
    if audio.shape != (2, RENDER_FRAME_COUNT):
        raise ValueError(f"Expected stereo shape (2, {RENDER_FRAME_COUNT}), got {audio.shape}")
    return torch.from_numpy(np.asarray(audio, dtype=np.float32))


def mid_side(audio: Tensor) -> Tensor:
    if audio.shape != (2, RENDER_FRAME_COUNT):
        raise ValueError(f"Unexpected audio shape {tuple(audio.shape)}")
    scale = 2.0**-0.5
    return torch.stack(((audio[0] + audio[1]) * scale, (audio[0] - audio[1]) * scale))


def log_mel(audio: Tensor) -> Tensor:
    transform = torchaudio.transforms.MelSpectrogram(
        sample_rate=RENDER_CONFIG.sample_rate,
        n_fft=2048,
        win_length=2048,
        hop_length=256,
        n_mels=128,
        f_min=20.0,
        f_max=20_000.0,
        power=2.0,
        center=True,
        normalized=False,
    ).to(audio.device)
    return torch.log1p(transform(mid_side(audio)))


def multi_stft(audio: Tensor) -> Tensor:
    channels = mid_side(audio)
    features = []
    for n_fft, hop in ((512, 128), (2048, 256), (8192, 512)):
        spectrum = torch.stft(
            channels,
            n_fft=n_fft,
            hop_length=hop,
            win_length=n_fft,
            window=torch.hann_window(n_fft, device=audio.device),
            return_complex=True,
        ).abs()
        resized = functional.interpolate(
            torch.log1p(spectrum).unsqueeze(0),
            size=(128, 690),
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)
        features.append(resized)
    return torch.cat(features, dim=0)


def waveform_features(audio: Tensor) -> Tensor:
    return functional.interpolate(
        mid_side(audio).unsqueeze(0), size=4096, mode="linear", align_corners=False
    ).squeeze(0)


def extract_feature(audio: Tensor, kind: AudioFeatureKind) -> Tensor:
    if kind is AudioFeatureKind.DAC:
        raise ValueError("DAC features require the frozen DAC model")
    if kind is AudioFeatureKind.LOG_MEL:
        return log_mel(audio)
    if kind is AudioFeatureKind.MULTI_STFT:
        return multi_stft(audio)
    if kind is AudioFeatureKind.WAVEFORM:
        return waveform_features(audio)
    raise AssertionError(kind)


def extract_dac(audio: Tensor, model: torch.nn.Module) -> tuple[Tensor, Tensor]:
    mono = audio.mean(dim=0, keepdim=True).unsqueeze(0)
    prepared = model.preprocess(mono, RENDER_CONFIG.sample_rate)
    _, codes, latents, _, _ = model.encode(prepared)
    return latents.squeeze(0), codes.squeeze(0)
