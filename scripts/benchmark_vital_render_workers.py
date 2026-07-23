#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from synth import VitalRuntime, create_output_directory


GENERATOR_NAME = "benchmark_vital_render_workers_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--examples", type=int, default=256)
    parser.add_argument("--worker-counts", nargs="+", type=int, default=(8, 16, 24, 32))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dataset = args.dataset_dir.resolve(strict=True)
    all_rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    rows = tuple(
        row for row in all_rows if row.get("is_augmented", False)
    )[: args.examples]
    if len(rows) != args.examples:
        raise ValueError(f"Expected {args.examples} augmented examples, found {len(rows)}")
    runtime = VitalRuntime.from_repo_root(repo_root)
    worker_script = repo_root / "scripts" / "render_vital_batch_worker.py"
    plugin = args.vital_vst3.resolve(strict=True)
    results = []

    with tempfile.TemporaryDirectory(prefix="vital-render-benchmark-") as temporary:
        temporary_root = Path(temporary)
        for worker_count in args.worker_counts:
            run_root = temporary_root / f"workers_{worker_count}"
            run_root.mkdir()
            shards: list[list[dict[str, str]]] = [
                [] for _ in range(worker_count)
            ]
            for index, row in enumerate(rows):
                shards[index % worker_count].append(
                    {
                        "preset": str(dataset / str(row["preset_file"])),
                        "output": str(run_root / f"{index:04d}.wav"),
                    }
                )

            def run_shard(index: int, tasks: list[dict[str, str]]) -> None:
                task_path = run_root / f"tasks_{index:02d}.json"
                task_path.write_text(json.dumps(tasks), encoding="utf-8")
                runtime.run_worker(
                    worker_script,
                    "--vital-vst3",
                    str(plugin),
                    "--tasks",
                    str(task_path),
                    capture_output=True,
                )

            started = time.monotonic()
            with ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = [
                    executor.submit(run_shard, index, shard)
                    for index, shard in enumerate(shards)
                    if shard
                ]
                for future in as_completed(futures):
                    future.result()
            seconds = time.monotonic() - started
            rendered = len(tuple(run_root.glob("*.wav")))
            if rendered != len(rows):
                raise RuntimeError(f"Rendered {rendered}/{len(rows)} examples")
            result = {
                "workers": worker_count,
                "examples": rendered,
                "seconds": seconds,
                "examples_per_second": rendered / seconds,
            }
            results.append(result)
            print(json.dumps(result), flush=True)

    output = create_output_directory(repo_root, GENERATOR_NAME)
    report = {
        "generator": GENERATOR_NAME,
        "dataset": str(dataset),
        "vital_vst3": str(plugin),
        "results": results,
        "recommended_workers": max(
            results,
            key=lambda item: float(item["examples_per_second"]),
        )["workers"],
    }
    (output / "render_benchmark.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
