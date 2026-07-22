#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Final

from synth import (
    VitalPreset,
    VitalRenderConfig,
    VitalRenderer,
    create_output_directory,
    file_sha256,
    resolve_vst3_binary,
    slugify,
    write_wav,
)


MIDI_NOTE: Final[int] = 36
NOTE_DURATION_SECONDS: Final[float] = 4.0
GENERATOR_NAME: Final[str] = "render_midtempo_vital_v1"
RENDER_CONFIG: Final[VitalRenderConfig] = VitalRenderConfig(
    sample_rate=44_100,
    channels=2,
    buffer_size=512,
    velocity=110,
    tail_duration_seconds=2.0,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render every native .vital preset in the BLA Midtempo pack."
    )
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--preset-dir", required=True, type=Path)
    parser.add_argument("--worker-preset", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def render_worker(plugin_path: Path, preset_path: Path, output_path: Path) -> None:
    preset = VitalPreset.from_file(preset_path)
    renderer = VitalRenderer(plugin_path, RENDER_CONFIG)
    audio = renderer.render_vital_audio(
        note=MIDI_NOTE,
        duration_seconds=NOTE_DURATION_SECONDS,
        preset=preset,
    )

    write_wav(output_path, audio, RENDER_CONFIG.sample_rate)
    print(
        json.dumps(
            {
                "preset_name": preset.preset_name,
                "preset_style": preset.preset_style,
                "author": preset.author,
                "comments": preset.comments,
                "preset_sha256": preset.sha256,
                "peak_absolute_sample": float(abs(audio).max()),
            }
        )
    )


def render_all(plugin_path: Path, preset_dir: Path) -> Path:
    presets = tuple(sorted(preset_dir.glob("*.vital"), key=lambda path: path.name.lower()))
    if len(presets) != 16:
        raise ValueError(f"Expected 16 Midtempo presets in {preset_dir}, received {len(presets)}")

    repo_root = Path(__file__).resolve().parent.parent
    output_dir = create_output_directory(repo_root, GENERATOR_NAME)
    rendered_presets = []

    for index, preset_path in enumerate(presets, start=1):
        output_path = output_dir / f"{index:02d}_{slugify(preset_path.stem)}.wav"
        result = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--vital-vst3",
                str(plugin_path),
                "--preset-dir",
                str(preset_dir),
                "--worker-preset",
                str(preset_path),
                "--worker-output",
                str(output_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        worker_metadata = json.loads(result.stdout)
        rendered_presets.append(
            {
                "index": index,
                "source_file": preset_path.name,
                "audio_file": output_path.name,
                **worker_metadata,
            }
        )
        print(f"[{index:02d}/{len(presets)}] {worker_metadata['preset_name']}")

    manifest = {
        "generator": GENERATOR_NAME,
        "vital_vst3": str(plugin_path),
        "plugin_binary_sha256": file_sha256(resolve_vst3_binary(plugin_path)),
        "sample_rate": RENDER_CONFIG.sample_rate,
        "channels": RENDER_CONFIG.channels,
        "buffer_size": RENDER_CONFIG.buffer_size,
        "midi_note": MIDI_NOTE,
        "velocity": RENDER_CONFIG.velocity,
        "note_duration_seconds": NOTE_DURATION_SECONDS,
        "render_duration_seconds": (
            NOTE_DURATION_SECONDS + RENDER_CONFIG.tail_duration_seconds
        ),
        "presets": rendered_presets,
    }
    (output_dir / "midtempo_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> None:
    args = parse_args()
    plugin_path: Path = args.vital_vst3.resolve(strict=True)
    preset_dir: Path = args.preset_dir.resolve(strict=True)
    if (args.worker_preset is None) != (args.worker_output is None):
        raise ValueError("Worker preset and output arguments must be supplied together")
    if args.worker_preset is not None:
        render_worker(
            plugin_path,
            args.worker_preset.resolve(strict=True),
            args.worker_output.resolve(strict=False),
        )
        return
    print(render_all(plugin_path, preset_dir))


if __name__ == "__main__":
    main()
