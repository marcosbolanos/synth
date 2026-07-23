#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from synth import VitalRuntime, create_output_directory
from synth.corpus import PresetCorpusEntry
from synth.experiment import (
    MIDI_NOTE,
    NOTE_DURATION_SECONDS,
    RENDER_CONFIG,
    RENDER_FRAME_COUNT,
)


GENERATOR_NAME = "build_vital_dataset_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus-jsonl", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def load_entries(path: Path) -> tuple[PresetCorpusEntry, ...]:
    return tuple(
        PresetCorpusEntry.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else create_output_directory(repo_root, GENERATOR_NAME)
    )
    output.mkdir(parents=True, exist_ok=True)
    audio_root = output / "audio"
    preset_root = output / "presets"
    audio_root.mkdir(exist_ok=True)
    preset_root.mkdir(exist_ok=True)

    corpus_path = args.corpus_jsonl.resolve(strict=True)
    entries = load_entries(corpus_path)
    runtime = VitalRuntime.from_repo_root(repo_root)
    worker = repo_root / "scripts" / "render_vital_batch_worker.py"
    plugin = args.vital_vst3.resolve(strict=True)

    def record(entry: PresetCorpusEntry) -> dict[str, object]:
        preset_output = preset_root / f"{entry.preset_sha256}.vital"
        audio_output = audio_root / f"{entry.preset_sha256}.wav"
        if not preset_output.exists():
            shutil.copyfile(entry.source_paths[0], preset_output)
        return {
            **entry.model_dump(mode="json"),
            "preset_file": str(preset_output.relative_to(output)),
            "audio_file": str(audio_output.relative_to(output)),
        }

    records = [record(entry) for entry in entries]
    pending = tuple(
        record
        for record in records
        if not (output / str(record["audio_file"])).exists()
    )
    shard_root = output / "render_shards"
    shard_root.mkdir(exist_ok=True)
    shards: list[list[dict[str, str]]] = [[] for _ in range(args.workers)]
    for index, item in enumerate(pending):
        shards[index % args.workers].append(
            {
                "preset": str(output / str(item["preset_file"])),
                "output": str(output / str(item["audio_file"])),
            }
        )

    def render_shard(index: int, tasks: list[dict[str, str]]) -> None:
        if not tasks:
            return
        task_path = shard_root / f"shard_{index:02d}.json"
        task_path.write_text(json.dumps(tasks), encoding="utf-8")
        runtime.run_worker(
            worker,
            "--vital-vst3",
            str(plugin),
            "--tasks",
            str(task_path),
        )

    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(render_shard, index, tasks)
            for index, tasks in enumerate(shards)
        ]
        for future in as_completed(futures):
            future.result()

    missing = tuple(
        item for item in records if not (output / str(item["audio_file"])).is_file()
    )
    if missing:
        raise RuntimeError(f"Missing {len(missing)} rendered audio files")

    records.sort(key=lambda record: str(record["preset_sha256"]))
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as destination:
        for record in records:
            destination.write(json.dumps(record, ensure_ascii=False) + "\n")
    manifest = {
        "generator": GENERATOR_NAME,
        "preset_count": len(records),
        "vital_vst3": str(plugin),
        "sample_rate": RENDER_CONFIG.sample_rate,
        "channels": RENDER_CONFIG.channels,
        "midi_note": MIDI_NOTE,
        "velocity": RENDER_CONFIG.velocity,
        "note_duration_seconds": NOTE_DURATION_SECONDS,
        "tail_duration_seconds": RENDER_CONFIG.tail_duration_seconds,
        "frame_count": RENDER_FRAME_COUNT,
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
