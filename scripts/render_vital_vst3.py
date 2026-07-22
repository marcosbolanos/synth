#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import version
from pathlib import Path
from typing import Final

from synth import (
    VitalRenderConfig,
    VitalRenderer,
    create_output_directory,
    file_sha256,
    resolve_vst3_binary,
    write_wav,
)


NOTE: Final[int] = 60
NOTE_DURATION_SECONDS: Final[float] = 4.0
GENERATOR_NAME: Final[str] = "render_vital_vst3_v2"
RENDER_CONFIG: Final[VitalRenderConfig] = VitalRenderConfig(
    sample_rate=44_100,
    channels=2,
    buffer_size=512,
    velocity=100,
    tail_duration_seconds=2.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render Vital's typed init patch through its native Linux VST3."
    )
    parser.add_argument(
        "--vital-vst3",
        required=True,
        type=Path,
        help="Path to the official native Linux Vital.vst3 bundle.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plugin_bundle: Path = args.vital_vst3.resolve(strict=True)
    plugin_binary = resolve_vst3_binary(plugin_bundle)
    renderer = VitalRenderer(plugin_bundle, RENDER_CONFIG)
    preset = renderer.current_preset
    audio = renderer.render_vital_audio(
        note=NOTE,
        duration_seconds=NOTE_DURATION_SECONDS,
        preset=preset,
    )

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = create_output_directory(repo_root, GENERATOR_NAME)
    wav_path = output_dir / "vital_init_C4_6s.wav"
    write_wav(wav_path, audio, RENDER_CONFIG.sample_rate)

    metadata = {
        "generator": GENERATOR_NAME,
        "python_version": sys.version,
        "pedalboard_version": version("pedalboard"),
        "vital_vst3": str(plugin_bundle),
        "plugin_binary": str(plugin_binary),
        "plugin_binary_sha256": file_sha256(plugin_binary),
        "plugin_identity": renderer.identity.model_dump(),
        "architecture": platform.machine(),
        "preset_sha256": preset.sha256,
        "sample_rate": RENDER_CONFIG.sample_rate,
        "channels": RENDER_CONFIG.channels,
        "buffer_size": RENDER_CONFIG.buffer_size,
        "render_duration_seconds": (
            NOTE_DURATION_SECONDS + RENDER_CONFIG.tail_duration_seconds
        ),
        "midi_events": [
            {
                "type": "note_on",
                "note": NOTE,
                "velocity": RENDER_CONFIG.velocity,
                "time": 0.0,
            },
            {
                "type": "note_off",
                "note": NOTE,
                "velocity": 0,
                "time": NOTE_DURATION_SECONDS,
            },
        ],
        "frame_count": int(audio.shape[1]),
        "peak_absolute_sample": float(abs(audio).max()),
    }
    (output_dir / "render_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print(output_dir)


if __name__ == "__main__":
    main()
