#!/usr/bin/env -S uv run --script

from __future__ import annotations

import subprocess
import tempfile
import wave
from datetime import datetime
from pathlib import Path


DURATION_SECONDS = 5
VITAL_RELEASE_RATIO = 0.3
MIDI_NOTE = "C3"
BPM = 120


def trim_wav(source: Path, destination: Path, duration_seconds: int) -> None:
    """Copy exactly duration_seconds of PCM audio into destination."""
    with wave.open(str(source), "rb") as reader:
        params = reader.getparams()
        frames = reader.readframes(params.framerate * duration_seconds)

    with wave.open(str(destination), "wb") as writer:
        writer.setparams(params)
        writer.writeframes(frames)


def main() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    vital = repo_root / "src/vital/headless/builds/linux/build/vital"
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S_%f")
    output_dir = (
        repo_root
        / "data/outputs"
        / f"generate_vital_snippet_v1_{timestamp}"
        / "files"
    )
    output_dir.mkdir(parents=True)
    output_file = output_dir / "vital_basic_C3_5s.wav"

    # Vital appends a 30% release tail. Shorten its requested note length so
    # the rendered note and tail together fill the five-second snippet.
    vital_length = DURATION_SECONDS / (1.0 + VITAL_RELEASE_RATIO)

    with tempfile.TemporaryDirectory(prefix="vital_snippet_") as temp_dir:
        raw_wav = Path(temp_dir) / "vital_render.wav"
        subprocess.run(
            [
                str(vital),
                "--output",
                str(raw_wav),
                "--length",
                str(vital_length),
                "--midi",
                MIDI_NOTE,
                "--bpm",
                str(BPM),
            ],
            cwd=repo_root,
            check=True,
        )
        trim_wav(raw_wav, output_file, DURATION_SECONDS)

    print(output_file)


if __name__ == "__main__":
    main()
