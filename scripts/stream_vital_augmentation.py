#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from safetensors.torch import load_file

from synth import VitalPreset, VitalRuntime, create_output_directory
from synth.augmentation import (
    ControlVariationStatistics,
    VariationMode,
    variation_seed,
)
from synth.preset_representation import ControlStatistics


GENERATOR_NAME = "stream_vital_augmentation_v1"
VARIANTS_PER_PRESET = 40
CHUNK_SIZE = 512
RENDER_WORKERS = 32
LOCAL_STRENGTHS = (
    0.04,
    0.06,
    0.08,
    0.10,
    0.12,
    0.14,
    0.17,
    0.20,
    0.23,
    0.26,
    0.29,
    0.32,
    0.36,
    0.40,
    0.45,
    0.50,
)
CORRELATED_STRENGTHS = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.34, 0.40)
INTERPOLATION_STRENGTHS = (0.08, 0.12, 0.16, 0.20, 0.24, 0.28, 0.34, 0.40)
MODULATION_STRENGTHS = (0.10, 0.18, 0.28, 0.40)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-dataset-dir", required=True, type=Path)
    parser.add_argument("--reference-training-dir", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--max-chunks", type=int)
    return parser.parse_args()


def summarize(feature: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (feature.mean(axis=-1).ravel(), feature.std(axis=-1).ravel())
    )


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dataset = args.reference_dataset_dir.resolve(strict=True)
    training = args.reference_training_dir.resolve(strict=True)
    output = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else create_output_directory(repo_root, GENERATOR_NAME)
    )
    output.mkdir(parents=True, exist_ok=True)
    shard_root = output / "shards"
    metadata_root = output / "metadata"
    shard_root.mkdir(exist_ok=True)
    metadata_root.mkdir(exist_ok=True)

    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    training_rows = tuple(row for row in rows if row["split"] != "test")
    original_by_digest = {
        str(row["preset_sha256"]): row
        for row in training_rows
        if not bool(row["is_augmented"])
    }
    template_digests = tuple(
        dict.fromkeys(
            str(row["template_preset_sha256"])
            for row in training_rows
        )
    )
    template_rows = tuple(original_by_digest[digest] for digest in template_digests)
    templates = tuple(
        VitalPreset.from_file(dataset / str(row["preset_file"]))
        for row in template_rows
    )
    statistics = ControlVariationStatistics.fit(templates)
    control_statistics_path = training / "control_statistics.json"
    ControlStatistics.model_validate_json(
        control_statistics_path.read_text(encoding="utf-8")
    )

    projection = np.load(training / "audio_projection.npz")
    template_audio = []
    for row in template_rows:
        feature = load_file(
            dataset
            / "features"
            / "log_mel"
            / f"{row['preset_sha256']}.safetensors"
        )["feature"].float().numpy()
        summary = summarize(feature)
        scaled = (
            summary - projection["summary_mean"]
        ) / projection["summary_scale"]
        embedding = (
            scaled - projection["pca_mean"]
        ) @ projection["pca_components"].T
        embedding /= np.linalg.norm(embedding) + 1e-8
        template_audio.append(embedding)
    template_audio_array = np.stack(template_audio)
    similarities = template_audio_array @ template_audio_array.T
    np.fill_diagonal(similarities, -np.inf)
    compatible_donors = []
    for template_index, parent in enumerate(templates):
        candidates = []
        for donor_index in np.argsort(similarities[template_index])[::-1]:
            donor = templates[int(donor_index)]
            if any(
                name in donor.settings.controls
                and donor.settings.controls[name] != value
                for name, value in parent.settings.controls.items()
                if name in statistics.continuous_controls
            ):
                candidates.append(int(donor_index))
            if len(candidates) == 16:
                break
        if len(candidates) != 16:
            raise ValueError(
                f"Template {template_digests[template_index]} has only "
                f"{len(candidates)} compatible donors"
            )
        compatible_donors.append(tuple(candidates))

    specifications = tuple(
        (template_index, variation_index)
        for template_index in range(len(templates))
        for variation_index in range(VARIANTS_PER_PRESET)
    )
    total_chunks = math.ceil(len(specifications) / CHUNK_SIZE)
    state_path = output / "stream_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
    else:
        state = {
            "generator": GENERATOR_NAME,
            "status": "generating",
            "completed_chunks": [],
            "generated_count": 0,
            "retained_count": 0,
            "started_at_unix": time.time(),
        }

    runtime = VitalRuntime.from_repo_root(repo_root)
    render_worker = repo_root / "scripts" / "render_vital_batch_worker.py"
    compact_worker = repo_root / "scripts" / "compact_vital_stream_chunk.py"
    plugin = args.vital_vst3.resolve(strict=True)

    def variant(
        template_index: int,
        variation_index: int,
    ) -> tuple[VitalPreset, VariationMode, float, tuple[str, ...], str | None]:
        parent = templates[template_index]
        parent_digest = template_digests[template_index]
        seed = variation_seed(parent_digest, variation_index + 10_000)
        donor_digest = None
        if variation_index < 16:
            strength = LOCAL_STRENGTHS[variation_index]
            varied, changed = statistics.vary(parent, strength=strength, seed=seed)
            mode = VariationMode.LOCAL
        elif variation_index < 24:
            strength = CORRELATED_STRENGTHS[variation_index - 16]
            varied, changed = statistics.vary_correlated(
                parent, strength=strength, seed=seed
            )
            mode = VariationMode.CORRELATED
        elif variation_index < 32:
            strength = INTERPOLATION_STRENGTHS[variation_index - 24]
            donor_candidates = compatible_donors[template_index]
            donor_index = int(
                donor_candidates[seed % len(donor_candidates)]
            )
            donor_digest = template_digests[donor_index]
            varied, changed = statistics.interpolate(
                parent,
                templates[donor_index],
                strength=strength,
                seed=seed,
            )
            mode = VariationMode.INTERPOLATED
        elif variation_index < 36:
            strength = MODULATION_STRENGTHS[variation_index - 32]
            varied, changed = statistics.vary_modulation(
                parent, strength=strength, seed=seed
            )
            mode = VariationMode.MODULATION
        else:
            strength = 1.0
            varied, changed = statistics.vary_categorical(parent, seed=seed)
            mode = VariationMode.CATEGORICAL
        return (
            varied,
            mode,
            strength,
            tuple(name.value for name in changed),
            donor_digest,
        )

    def render_chunk(
        chunk_index: int,
        chunk_specs: tuple[tuple[int, int], ...],
        temporary_root: Path,
    ) -> Path:
        tasks = []
        for item_index, (template_index, variation_index) in enumerate(chunk_specs):
            varied, mode, strength, changed, donor_digest = variant(
                template_index,
                variation_index,
            )
            preset_path = temporary_root / "presets" / f"{item_index:04d}.vital"
            audio_path = temporary_root / "audio" / f"{item_index:04d}.wav"
            varied.to_file(preset_path)
            audio_path.parent.mkdir(parents=True, exist_ok=True)
            tasks.append(
                {
                    "preset": str(preset_path),
                    "audio": str(audio_path),
                    "template_index": template_index,
                    "metadata": {
                        "parent_preset_sha256": template_digests[template_index],
                        "variation_index": variation_index,
                        "mode": mode.value,
                        "strength": strength,
                        "changed_controls": changed,
                        "donor_preset_sha256": donor_digest,
                        "generated_preset_sha256": varied.sha256,
                    },
                }
            )
        render_shards: list[list[dict[str, str]]] = [
            [] for _ in range(RENDER_WORKERS)
        ]
        for index, task in enumerate(tasks):
            render_shards[index % RENDER_WORKERS].append(
                {"preset": task["preset"], "output": task["audio"]}
            )

        def run_render_shard(index: int, shard: list[dict[str, str]]) -> None:
            task_path = temporary_root / f"render_{index:02d}.json"
            task_path.write_text(json.dumps(shard), encoding="utf-8")
            try:
                runtime.run_worker(
                    render_worker,
                    "--vital-vst3",
                    str(plugin),
                    "--tasks",
                    str(task_path),
                    capture_output=True,
                )
            except subprocess.CalledProcessError as error:
                raise RuntimeError(
                    f"Render shard {index} failed:\n{error.stdout}\n{error.stderr}"
                ) from error

        with ThreadPoolExecutor(max_workers=RENDER_WORKERS) as executor:
            futures = [
                executor.submit(run_render_shard, index, shard)
                for index, shard in enumerate(render_shards)
                if shard
            ]
            for future in as_completed(futures):
                future.result()
        task_path = temporary_root / f"compact_{chunk_index:05d}.json"
        task_path.write_text(json.dumps(tasks), encoding="utf-8")
        return task_path

    def compact_chunk(chunk_index: int, task_path: Path) -> dict[str, object]:
        result = subprocess.run(
            [
                sys.executable,
                str(compact_worker),
                "--tasks",
                str(task_path),
                "--control-statistics",
                str(control_statistics_path),
                "--audio-projection",
                str(training / "audio_projection.npz"),
                "--output",
                str(shard_root / f"shard_{chunk_index:05d}.safetensors"),
                "--metadata-output",
                str(metadata_root / f"shard_{chunk_index:05d}.jsonl"),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    completed = set(int(index) for index in state["completed_chunks"])
    pending_feature: tuple[
        int,
        int,
        tempfile.TemporaryDirectory[str],
        Future[dict[str, object]],
    ] | None = None

    def finalize_pending() -> None:
        nonlocal pending_feature, state
        if pending_feature is None:
            return
        chunk_index, generated_count, temporary, future = pending_feature
        result = future.result()
        state["completed_chunks"].append(chunk_index)
        state["completed_chunks"].sort()
        state["generated_count"] += generated_count
        state["retained_count"] += int(result["retained_count"])
        temporary.cleanup()
        temporary_state = state_path.with_suffix(".partial.json")
        temporary_state.write_text(
            json.dumps(state, indent=2) + "\n", encoding="utf-8"
        )
        temporary_state.replace(state_path)
        print(
            f"chunk {chunk_index + 1}/{total_chunks}: "
            f"retained={result['retained_count']}/{generated_count}",
            flush=True,
        )
        pending_feature = None

    with ThreadPoolExecutor(max_workers=1) as feature_executor:
        generated_chunks = 0
        for chunk_index in range(total_chunks):
            if chunk_index in completed:
                continue
            if (
                args.max_chunks is not None
                and generated_chunks >= args.max_chunks
            ):
                break
            start = chunk_index * CHUNK_SIZE
            chunk_specs = specifications[start : start + CHUNK_SIZE]
            temporary = tempfile.TemporaryDirectory(
                prefix=f"vital-stream-{chunk_index:05d}-",
                dir="/dev/shm",
            )
            task_path = render_chunk(
                chunk_index,
                chunk_specs,
                Path(temporary.name),
            )
            finalize_pending()
            pending_feature = (
                chunk_index,
                len(chunk_specs),
                temporary,
                feature_executor.submit(compact_chunk, chunk_index, task_path),
            )
            generated_chunks += 1
        finalize_pending()

    if len(state["completed_chunks"]) != total_chunks:
        print(output)
        return
    state["status"] = "complete"
    state["completed_at_unix"] = time.time()
    state_path.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    manifest = {
        **state,
        "reference_dataset": str(dataset),
        "reference_training": str(training),
        "vital_vst3": str(plugin),
        "template_count": len(templates),
        "variants_per_preset": VARIANTS_PER_PRESET,
        "planned_count": len(specifications),
        "chunk_size": CHUNK_SIZE,
        "render_workers": RENDER_WORKERS,
        "feature_gpu_uuid": "GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea",
        "variation_counts_per_preset": {
            "local": 16,
            "correlated": 8,
            "interpolated": 8,
            "modulation": 4,
            "categorical": 4,
        },
        "raw_presets_and_audio_retained": False,
    }
    (output / "stream_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
