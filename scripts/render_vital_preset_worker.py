#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from synth import VitalPreset, VitalRenderer, write_wav
from synth.experiment import MIDI_NOTE, NOTE_DURATION_SECONDS, RENDER_CONFIG


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--preset", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    preset = VitalPreset.from_file(args.preset)
    renderer = VitalRenderer(args.vital_vst3, RENDER_CONFIG)
    audio = renderer.render_vital_audio(
        note=MIDI_NOTE,
        duration_seconds=NOTE_DURATION_SECONDS,
        preset=preset,
        allow_silence=True,
        verify_loaded_preset=False,
    )
    write_wav(args.output, audio, RENDER_CONFIG.sample_rate)
    print(
        json.dumps(
            {
                "preset_sha256": preset.sha256,
                "peak": float(abs(audio).max()),
                "frames": int(audio.shape[1]),
                "plugin": renderer.identity.model_dump(),
            }
        )
    )


if __name__ == "__main__":
    main()
