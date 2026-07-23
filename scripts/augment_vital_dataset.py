#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from synth import VitalPreset, VitalRuntime, create_output_directory
from synth.audio_features import read_render
from synth.augmentation import (
    AUGMENTATION_STRENGTHS,
    ControlVariationStatistics,
    variation_seed,
)
from synth.experiment import (
    MIDI_NOTE,
    NOTE_DURATION_SECONDS,
    RENDER_CONFIG,
    RENDER_FRAME_COUNT,
)


GENERATOR_NAME = "augment_vital_dataset_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--workers", type=int, default=8)
    return parser.parse_args()


def link(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not destination.exists():
        os.link(source.resolve(strict=True), destination)


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    source = args.dataset_dir.resolve(strict=True)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else create_output_directory(repo_root, GENERATOR_NAME)
    )
    output.mkdir(parents=True, exist_ok=True)
    preset_root = output / "presets"
    audio_root = output / "audio"
    log_mel_root = output / "features" / "log_mel"
    preset_root.mkdir(exist_ok=True)
    audio_root.mkdir(exist_ok=True)
    log_mel_root.mkdir(parents=True, exist_ok=True)

    original_rows = tuple(
        json.loads(line)
        for line in (source / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    training_rows = tuple(row for row in original_rows if row["split"] != "test")
    training_presets = tuple(
        VitalPreset.from_file(source / str(row["preset_file"]))
        for row in training_rows
    )
    statistics = ControlVariationStatistics.fit(training_presets)
    (output / "variation_statistics.json").write_text(
        statistics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    records: list[dict[str, object]] = []
    for row in original_rows:
        digest = str(row["preset_sha256"])
        preset_file = Path("presets") / f"{digest}.vital"
        audio_file = Path("audio") / f"{digest}.wav"
        link(source / str(row["preset_file"]), output / preset_file)
        link(source / str(row["audio_file"]), output / audio_file)
        link(
            source / "features" / "log_mel" / f"{digest}.safetensors",
            log_mel_root / f"{digest}.safetensors",
        )
        records.append(
            {
                **row,
                "preset_file": preset_file.as_posix(),
                "audio_file": audio_file.as_posix(),
                "is_augmented": False,
                "parent_preset_sha256": None,
                "template_preset_sha256": digest,
                "augmentation_index": None,
                "augmentation_strength": None,
                "changed_control_count": 0,
            }
        )

    variant_records: list[dict[str, object]] = []
    for row, preset in zip(training_rows, training_presets, strict=True):
        parent_digest = str(row["preset_sha256"])
        for variation_index, strength in enumerate(AUGMENTATION_STRENGTHS):
            varied, changed = statistics.vary(
                preset,
                strength=strength,
                seed=variation_seed(parent_digest, variation_index),
            )
            digest = varied.sha256
            preset_file = Path("presets") / f"{digest}.vital"
            audio_file = Path("audio") / f"{digest}.wav"
            varied.to_file(output / preset_file, overwrite=True)
            variant_records.append(
                {
                    **row,
                    "preset_sha256": digest,
                    "preset_name": varied.preset_name,
                    "preset_file": preset_file.as_posix(),
                    "audio_file": audio_file.as_posix(),
                    "json_bytes": len(varied.to_json_bytes()),
                    "is_augmented": True,
                    "parent_preset_sha256": parent_digest,
                    "template_preset_sha256": parent_digest,
                    "augmentation_index": variation_index,
                    "augmentation_strength": strength,
                    "changed_control_count": len(changed),
                }
            )

    runtime = VitalRuntime.from_repo_root(repo_root)
    worker = repo_root / "scripts" / "render_vital_batch_worker.py"
    plugin = args.vital_vst3.resolve(strict=True)
    pending = tuple(
        row
        for row in variant_records
        if not (output / str(row["audio_file"])).exists()
    )
    shard_root = output / "render_shards"
    shard_root.mkdir(exist_ok=True)
    shards: list[list[dict[str, str]]] = [[] for _ in range(args.workers)]
    for index, row in enumerate(pending):
        shards[index % args.workers].append(
            {
                "preset": str(output / str(row["preset_file"])),
                "output": str(output / str(row["audio_file"])),
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

    parent_audio = {
        str(row["preset_sha256"]): read_render(output / str(row["audio_file"]))
        for row in training_rows
    }
    waveform_distances = []
    silent_count = 0
    clipped_count = 0
    for row in variant_records:
        audio = read_render(output / str(row["audio_file"]))
        peak = float(audio.abs().max())
        silent_count += peak < 1e-5
        clipped_count += peak >= 0.999
        parent = parent_audio[str(row["parent_preset_sha256"])]
        distance = float((audio - parent).square().mean())
        row["parent_waveform_mse"] = distance
        waveform_distances.append(distance)

    records.extend(variant_records)
    records.sort(key=lambda row: str(row["preset_sha256"]))
    with (output / "dataset.jsonl").open("w", encoding="utf-8") as destination:
        for row in records:
            destination.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "generator": GENERATOR_NAME,
        "source_dataset": str(source),
        "original_count": len(original_rows),
        "original_training_count": len(training_rows),
        "test_count": sum(row["split"] == "test" for row in original_rows),
        "variants_per_training_preset": len(AUGMENTATION_STRENGTHS),
        "augmentation_strengths": AUGMENTATION_STRENGTHS,
        "augmented_count": len(variant_records),
        "total_count": len(records),
        "continuous_control_count": len(statistics.continuous_controls),
        "mean_changed_control_count": float(
            np.mean([row["changed_control_count"] for row in variant_records])
        ),
        "mean_parent_waveform_mse": float(np.mean(waveform_distances)),
        "median_parent_waveform_mse": float(np.median(waveform_distances)),
        "silent_variant_count": silent_count,
        "clipped_variant_count": clipped_count,
        "vital_vst3": str(plugin),
        "sample_rate": RENDER_CONFIG.sample_rate,
        "channels": RENDER_CONFIG.channels,
        "midi_note": MIDI_NOTE,
        "note_duration_seconds": NOTE_DURATION_SECONDS,
        "tail_duration_seconds": RENDER_CONFIG.tail_duration_seconds,
        "frame_count": RENDER_FRAME_COUNT,
    }
    (output / "augmentation_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
