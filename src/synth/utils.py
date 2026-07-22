from __future__ import annotations

import hashlib
import platform
import re
from datetime import datetime
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pedalboard.io import AudioFile


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def create_output_directory(repo_root: Path, generator_name: str) -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_directory = (
        repo_root / "data" / "outputs" / f"{generator_name}_{timestamp}" / "files"
    )
    output_directory.mkdir(parents=True)
    return output_directory


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def resolve_vst3_binary(plugin_bundle: Path) -> Path:
    resolved_bundle = plugin_bundle.resolve(strict=True)
    if resolved_bundle.name != "Vital.vst3" or not resolved_bundle.is_dir():
        raise ValueError(
            f"Expected a Vital.vst3 bundle directory, received: {resolved_bundle}"
        )
    architecture_directory = {
        "x86_64": "x86_64-linux",
        "aarch64": "aarch64-linux",
    }.get(platform.machine())
    if architecture_directory is None:
        raise ValueError(f"Unsupported machine architecture: {platform.machine()}")
    binary_directory = resolved_bundle / "Contents" / architecture_directory
    binaries = tuple(path for path in binary_directory.iterdir() if path.is_file())
    if len(binaries) != 1:
        raise ValueError(
            f"Expected exactly one plugin binary in {binary_directory}, "
            f"received {len(binaries)}"
        )
    return binaries[0]


def write_wav(
    output_path: Path,
    audio: NDArray[np.float32],
    sample_rate: int,
) -> None:
    if audio.ndim != 2:
        raise ValueError(f"Expected channels-first audio, received shape {audio.shape}")
    channels = audio.shape[0]
    with AudioFile(str(output_path), "w", sample_rate, channels) as output:
        output.write(audio)
