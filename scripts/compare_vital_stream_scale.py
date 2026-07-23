#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synth import A6000_UUID, create_output_directory, pin_a6000


GENERATOR_NAME = "compare_vital_stream_scale_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-training-dir", required=True, type=Path)
    parser.add_argument("--four-variant-training-dir", required=True, type=Path)
    parser.add_argument("--stream-training-dir", required=True, type=Path)
    return parser.parse_args()


def metrics(path: Path) -> tuple[dict[str, float], ...]:
    with path.open(newline="", encoding="utf-8") as source:
        return tuple(
            {name: float(value) for name, value in row.items()}
            for row in csv.DictReader(source)
        )


def main() -> None:
    gpu = pin_a6000()
    import torch

    from synth.audio_features import AudioFeatureKind, extract_feature, read_render

    args = parse_args()
    roots = {
        "baseline": args.baseline_training_dir.resolve(strict=True),
        "four_variant": args.four_variant_training_dir.resolve(strict=True),
        "stream": args.stream_training_dir.resolve(strict=True),
    }
    reports = {
        name: json.loads((root / "training_report.json").read_text())
        for name, root in roots.items()
    }
    if any(report["status"] != "complete" for report in reports.values()):
        raise ValueError("All compared training reports must be complete")
    stream_manifest = json.loads(
        (
            Path(reports["stream"]["stream"]) / "stream_manifest.json"
        ).read_text(encoding="utf-8")
    )
    if stream_manifest["status"] != "complete":
        raise ValueError("Streaming augmentation manifest is incomplete")
    manifests = {
        name: {
            item["preset_sha256"]: item
            for item in json.loads((root / "comparisons.json").read_text())
        }
        for name, root in roots.items()
    }
    digest_sets = {frozenset(items) for items in manifests.values()}
    if len(digest_sets) != 1:
        raise ValueError("Comparison sets differ")

    comparisons = []
    with torch.inference_mode():
        for digest in sorted(manifests["stream"]):
            stream_item = manifests["stream"][digest]
            source = read_render(
                roots["stream"] / stream_item["input_audio"]
            ).to("cuda")
            source_mel = extract_feature(source, AudioFeatureKind.LOG_MEL)
            source_stft = extract_feature(source, AudioFeatureKind.MULTI_STFT)
            distances = {}
            for name, root in roots.items():
                item = manifests[name][digest]
                predicted = read_render(root / item["output_audio"]).to("cuda")
                predicted_mel = extract_feature(predicted, AudioFeatureKind.LOG_MEL)
                predicted_stft = extract_feature(predicted, AudioFeatureKind.MULTI_STFT)
                distances[name] = float(
                    (source_mel - predicted_mel).square().mean()
                    + (source_stft - predicted_stft).square().mean()
                )
            comparisons.append(
                {
                    "preset_sha256": digest,
                    "preset_name": stream_item["preset_name"],
                    "pack": stream_item["pack"],
                    "family": stream_item["family"],
                    "split": stream_item["split"],
                    "input_audio": str(
                        (roots["stream"] / stream_item["input_audio"]).resolve()
                    ),
                    **{
                        f"{name}_output_audio": str(
                            (root / manifests[name][digest]["output_audio"]).resolve()
                        )
                        for name, root in roots.items()
                    },
                    **{
                        f"{name}_preset": str(
                            (root / manifests[name][digest]["predicted_preset"]).resolve()
                        )
                        for name, root in roots.items()
                    },
                    **{
                        f"{name}_distance": distance
                        for name, distance in distances.items()
                    },
                }
            )

    output = create_output_directory(Path(__file__).resolve().parent.parent, GENERATOR_NAME)
    curves = {
        "baseline": metrics(roots["baseline"] / "metrics.csv"),
        "four_variant": metrics(roots["four_variant"] / "metrics.csv"),
        "stream": metrics(roots["stream"] / "metrics.csv"),
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    for name, label in (
        ("baseline", "baseline"),
        ("four_variant", "four variants"),
    ):
        values = curves[name]
        axis.plot(
            np.linspace(0.0, 1.0, len(values)),
            [row["test_sound_loss"] for row in values],
            label=label,
        )
    stream_values = curves["stream"]
    axis.plot(
        np.asarray(
            [row["processed_stream_examples"] for row in stream_values]
        )
        / float(reports["stream"]["processed_stream_examples"]),
        [row["test_sound_loss"] for row in stream_values],
        label="diverse stream",
    )
    axis.set_xlabel("training progress")
    axis.set_ylabel("untouched test sound loss")
    axis.set_title("Augmentation scale comparison")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "scale_test_sound_loss.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    four_distances = np.asarray(
        [item["four_variant_distance"] for item in comparisons]
    )
    stream_distances = np.asarray(
        [item["stream_distance"] for item in comparisons]
    )
    for split, marker in (("train", "o"), ("test", "s")):
        indexes = [
            index for index, item in enumerate(comparisons) if item["split"] == split
        ]
        axis.scatter(
            four_distances[indexes],
            stream_distances[indexes],
            marker=marker,
            alpha=0.75,
            label=split,
        )
    limit = float(max(four_distances.max(), stream_distances.max()))
    axis.plot((0.0, limit), (0.0, limit), linestyle="--", color="black")
    axis.set_xlabel("four-variant rendered distance")
    axis.set_ylabel("diverse-stream rendered distance")
    axis.set_title("Paired real-Vital reconstruction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "scale_paired_rendered_distance.png", dpi=160)
    plt.close(figure)

    def mean_distance(name: str, split: str) -> float:
        return float(
            np.mean(
                [
                    item[f"{name}_distance"]
                    for item in comparisons
                    if item["split"] == split
                ]
            )
        )

    def stream_wins(split: str) -> int:
        return sum(
            item["stream_distance"] < item["four_variant_distance"]
            for item in comparisons
            if item["split"] == split
        )

    report = {
        "generator": GENERATOR_NAME,
        "status": "complete",
        "gpu_uuid": A6000_UUID,
        "gpu": gpu.model_dump(),
        "baseline_training_dir": str(roots["baseline"]),
        "four_variant_training_dir": str(roots["four_variant"]),
        "stream_training_dir": str(roots["stream"]),
        "baseline_best_test_sound_loss": reports["baseline"]["best_test_sound_loss"],
        "four_variant_best_test_sound_loss": reports["four_variant"][
            "best_test_sound_loss"
        ],
        "stream_best_test_sound_loss": reports["stream"]["best_test_sound_loss"],
        "processed_stream_examples": reports["stream"]["processed_stream_examples"],
        "generated_stream_examples": stream_manifest["generated_count"],
        "rejected_stream_examples": (
            stream_manifest["generated_count"] - stream_manifest["retained_count"]
        ),
        "optimizer_steps": reports["stream"]["optimizer_steps"],
        "training_seconds": reports["stream"]["training_seconds"],
        "model_parameters": reports["stream"]["model_parameters"],
        "replay_fraction": reports["stream"]["replay_fraction"],
        "mode_counts": reports["stream"]["mode_counts"],
        "mean_changed_control_count": reports["stream"][
            "mean_changed_control_count"
        ],
        **{
            f"{name}_{split}_rendered_distance": mean_distance(name, split)
            for name in roots
            for split in ("train", "test")
        },
        "stream_train_wins_over_four_variant": stream_wins("train"),
        "stream_test_wins_over_four_variant": stream_wins("test"),
        "plots": [
            "scale_test_sound_loss.png",
            "scale_paired_rendered_distance.png",
        ],
        "comparisons": "comparisons.json",
    }
    (output / "stream_scale_demo.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
