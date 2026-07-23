#!/usr/bin/env python

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from synth import VitalPreset, VitalRenderer, write_wav
from synth.experiment import MIDI_NOTE, NOTE_DURATION_SECONDS, RENDER_CONFIG


class RenderTask(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preset: Path
    output: Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--tasks", required=True, type=Path)
    args = parser.parse_args()
    tasks = tuple(
        RenderTask.model_validate(item)
        for item in json.loads(args.tasks.resolve(strict=True).read_text(encoding="utf-8"))
    )
    renderer = VitalRenderer(args.vital_vst3, RENDER_CONFIG)
    for index, task in enumerate(tasks, start=1):
        preset = VitalPreset.from_file(task.preset)
        audio = renderer.render_vital_audio(
            note=MIDI_NOTE,
            duration_seconds=NOTE_DURATION_SECONDS,
            preset=preset,
            allow_silence=True,
            verify_loaded_preset=False,
        )
        write_wav(task.output, audio, RENDER_CONFIG.sample_rate)
        print(f"{index}/{len(tasks)} {preset.preset_name or task.preset.stem}", flush=True)


if __name__ == "__main__":
    main()
