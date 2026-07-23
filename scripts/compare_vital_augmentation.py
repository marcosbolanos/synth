#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synth import (
    A6000_UUID,
    create_output_directory,
    pin_a6000,
    publish_gallery_run,
    repository_relative_output_path,
)


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
    repo_root = Path(__file__).resolve().parent.parent

    def portable(path: Path) -> str:
        return str(repository_relative_output_path(repo_root, path))

    def training_reference(root: Path, value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if path.parts[:2] == ("data", "outputs"):
            return repo_root / path
        return root / path

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
            source = read_render(
                training_reference(augmented, augmented_item["input_audio"])
            ).to("cuda")
            source_mel = extract_feature(source, AudioFeatureKind.LOG_MEL)
            source_stft = extract_feature(source, AudioFeatureKind.MULTI_STFT)

            distances: dict[str, float] = {}
            for name, root, item in (
                ("baseline", baseline, baseline_item),
                ("augmented", augmented, augmented_item),
            ):
                predicted = read_render(
                    training_reference(root, item["output_audio"])
                ).to("cuda")
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
                    "input_audio": portable(
                        training_reference(augmented, augmented_item["input_audio"])
                    ),
                    "baseline_output_audio": portable(
                        training_reference(baseline, baseline_item["output_audio"])
                    ),
                    "augmented_output_audio": portable(
                        training_reference(augmented, augmented_item["output_audio"])
                    ),
                    "baseline_preset": portable(
                        training_reference(
                            baseline, baseline_item["predicted_preset"]
                        )
                    ),
                    "augmented_preset": portable(
                        training_reference(
                            augmented, augmented_item["predicted_preset"]
                        )
                    ),
                    "baseline_distance": distances["baseline"],
                    "augmented_distance": distances["augmented"],
                }
            )

    output = create_output_directory(repo_root, GENERATOR_NAME)
    sound_plot = output / "test_sound_loss_comparison.png"
    distance_plot = output / "paired_rendered_distance.png"
    report_path = output / "augmentation_demo.json"
    comparisons_path = output / "comparisons.json"
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
    figure.savefig(sound_plot, dpi=160)
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
    figure.savefig(distance_plot, dpi=160)
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
            portable(sound_plot),
            portable(distance_plot),
        ],
        "comparisons": str(comparisons_path.relative_to(repo_root)),
    }
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    comparisons_path.write_text(
        json.dumps(comparisons, indent=2) + "\n", encoding="utf-8"
    )
    comparison_sources = tuple(
        source
        for item in comparisons
        for source in (
            (repo_root / item["input_audio"], "audio", "input audio"),
            (
                repo_root / item["baseline_output_audio"],
                "audio",
                "baseline reconstruction",
            ),
            (
                repo_root / item["augmented_output_audio"],
                "audio",
                "augmented reconstruction",
            ),
            (
                repo_root / item["baseline_preset"],
                "preset",
                "baseline predicted preset",
            ),
            (
                repo_root / item["augmented_preset"],
                "preset",
                "augmented predicted preset",
            ),
        )
    )
    publish_gallery_run(
        repo_root,
        output,
        GENERATOR_NAME,
        "/vital-augmentation",
        (
            (report_path, "json", "augmentation report"),
            (comparisons_path, "json", "comparison index"),
            (sound_plot, "image", "test sound loss comparison"),
            (distance_plot, "image", "paired rendered distance"),
            *comparison_sources,
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
