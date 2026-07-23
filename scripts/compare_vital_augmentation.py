#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synth import A6000_UUID, create_output_directory, pin_a6000


GENERATOR_NAME = "compare_vital_augmentation_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-training-dir", required=True, type=Path)
    parser.add_argument("--augmented-training-dir", required=True, type=Path)
    parser.add_argument("--augmented-dataset-dir", required=True, type=Path)
    return parser.parse_args()


def metric_rows(path: Path) -> tuple[dict[str, float], ...]:
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
    baseline = args.baseline_training_dir.resolve(strict=True)
    augmented = args.augmented_training_dir.resolve(strict=True)
    dataset = args.augmented_dataset_dir.resolve(strict=True)
    baseline_report = json.loads((baseline / "training_report.json").read_text())
    augmented_report = json.loads((augmented / "training_report.json").read_text())
    augmentation = json.loads((dataset / "augmentation_manifest.json").read_text())
    if baseline_report["status"] != "complete":
        raise ValueError("Baseline training report is not complete")
    if augmented_report["status"] != "complete":
        raise ValueError("Augmented training report is not complete")

    baseline_comparisons = {
        item["preset_sha256"]: item
        for item in json.loads((baseline / "comparisons.json").read_text())
    }
    augmented_comparisons = {
        item["preset_sha256"]: item
        for item in json.loads((augmented / "comparisons.json").read_text())
    }
    if baseline_comparisons.keys() != augmented_comparisons.keys():
        raise ValueError("Baseline and augmented comparison sets differ")

    comparisons = []
    with torch.inference_mode():
        for digest in sorted(baseline_comparisons):
            baseline_item = baseline_comparisons[digest]
            augmented_item = augmented_comparisons[digest]
            source = read_render(augmented / augmented_item["input_audio"]).to("cuda")
            source_mel = extract_feature(source, AudioFeatureKind.LOG_MEL)
            source_stft = extract_feature(source, AudioFeatureKind.MULTI_STFT)

            distances: dict[str, float] = {}
            for name, root, item in (
                ("baseline", baseline, baseline_item),
                ("augmented", augmented, augmented_item),
            ):
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
                    "preset_name": augmented_item["preset_name"],
                    "pack": augmented_item["pack"],
                    "family": augmented_item["family"],
                    "split": augmented_item["split"],
                    "input_audio": str((augmented / augmented_item["input_audio"]).resolve()),
                    "baseline_output_audio": str(
                        (baseline / baseline_item["output_audio"]).resolve()
                    ),
                    "augmented_output_audio": str(
                        (augmented / augmented_item["output_audio"]).resolve()
                    ),
                    "baseline_preset": str(
                        (baseline / baseline_item["predicted_preset"]).resolve()
                    ),
                    "augmented_preset": str(
                        (augmented / augmented_item["predicted_preset"]).resolve()
                    ),
                    "baseline_distance": distances["baseline"],
                    "augmented_distance": distances["augmented"],
                }
            )

    output = create_output_directory(Path(__file__).resolve().parent.parent, GENERATOR_NAME)
    baseline_metrics = metric_rows(baseline / "metrics.csv")
    augmented_metrics = metric_rows(augmented / "metrics.csv")
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.plot(
        [row["epoch"] for row in baseline_metrics],
        [row["test_sound_loss"] for row in baseline_metrics],
        label="baseline",
    )
    axis.plot(
        [row["epoch"] for row in augmented_metrics],
        [row["test_sound_loss"] for row in augmented_metrics],
        label="four variants per preset",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("test sound loss")
    axis.set_title("Untouched test-set learning curves")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "test_sound_loss_comparison.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    baseline_distances = np.asarray(
        [item["baseline_distance"] for item in comparisons]
    )
    augmented_distances = np.asarray(
        [item["augmented_distance"] for item in comparisons]
    )
    for split, marker in (("train", "o"), ("test", "s")):
        indexes = [
            index
            for index, item in enumerate(comparisons)
            if item["split"] == split
        ]
        axis.scatter(
            baseline_distances[indexes],
            augmented_distances[indexes],
            marker=marker,
            alpha=0.75,
            label=split,
        )
    limit = float(max(baseline_distances.max(), augmented_distances.max()))
    axis.plot((0.0, limit), (0.0, limit), linestyle="--", color="black")
    axis.set_xlabel("baseline rendered distance")
    axis.set_ylabel("augmented rendered distance")
    axis.set_title("Paired listening-set reconstruction")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "paired_rendered_distance.png", dpi=160)
    plt.close(figure)

    def mean_distance(name: str, split: str) -> float:
        values = [
            float(item[f"{name}_distance"])
            for item in comparisons
            if item["split"] == split
        ]
        return float(np.mean(values))

    def augmented_wins(split: str) -> int:
        return sum(
            item["augmented_distance"] < item["baseline_distance"]
            for item in comparisons
            if item["split"] == split
        )

    report = {
        "generator": GENERATOR_NAME,
        "status": "complete",
        "gpu_uuid": A6000_UUID,
        "gpu": gpu.model_dump(),
        "baseline_training_dir": str(baseline),
        "augmented_training_dir": str(augmented),
        "augmented_dataset_dir": str(dataset),
        "original_training_examples": augmentation["original_training_count"],
        "augmented_examples": augmentation["augmented_count"],
        "total_training_examples": augmented_report["training_examples"],
        "variants_per_preset": augmentation["variants_per_training_preset"],
        "mean_changed_control_count": augmentation["mean_changed_control_count"],
        "baseline_best_test_sound_loss": baseline_report["best_test_sound_loss"],
        "augmented_best_test_sound_loss": augmented_report["best_test_sound_loss"],
        "baseline_train_rendered_distance": mean_distance("baseline", "train"),
        "augmented_train_rendered_distance": mean_distance("augmented", "train"),
        "baseline_test_rendered_distance": mean_distance("baseline", "test"),
        "augmented_test_rendered_distance": mean_distance("augmented", "test"),
        "augmented_train_wins": augmented_wins("train"),
        "augmented_test_wins": augmented_wins("test"),
        "plots": [
            "test_sound_loss_comparison.png",
            "paired_rendered_distance.png",
        ],
        "comparisons": "comparisons.json",
    }
    (output / "augmentation_demo.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    (output / "comparisons.json").write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
