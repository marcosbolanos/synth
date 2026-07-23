#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from synth import (
    create_output_directory,
    publish_gallery_run,
)
from synth.audio_features import AudioFeatureKind
from synth.preset_representation import CONTROL_NAMES


GENERATOR_NAME = "investigate_vital_architecture_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--corpus-investigation", required=True, type=Path)
    return parser.parse_args()


def summarize(feature: np.ndarray) -> np.ndarray:
    if feature.ndim == 3:
        return np.concatenate((feature.mean(axis=-1).ravel(), feature.std(axis=-1).ravel()))
    if feature.ndim == 2 and feature.shape[-1] >= 128:
        if feature.shape[-1] == 4096:
            spectrum = np.log1p(np.abs(np.fft.rfft(feature, axis=-1)))
            return spectrum[:, ::16].ravel()
        return np.concatenate((feature.mean(axis=-1), feature.std(axis=-1)))
    return feature.ravel()


def load_summary(dataset: Path, kind: AudioFeatureKind, digest: str) -> np.ndarray:
    tensor = load_file(dataset / "features" / kind.value / f"{digest}.safetensors")[
        "feature"
    ]
    return summarize(tensor.float().numpy())


def main() -> None:
    args = parse_args()
    dataset = args.dataset_dir.resolve(strict=True)
    corpus_report = json.loads(args.corpus_investigation.resolve(strict=True).read_text())
    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    train = tuple(row for row in rows if row["split"] == "train")
    development = tuple(row for row in rows if row["split"] == "development")
    results: dict[str, dict[str, object]] = {}
    reference_kinds = (AudioFeatureKind.LOG_MEL, AudioFeatureKind.MULTI_STFT)

    for kind in AudioFeatureKind:
        started = time.monotonic()
        train_raw = np.stack(
            [load_summary(dataset, kind, str(row["preset_sha256"])) for row in train]
        )
        development_raw = np.stack(
            [load_summary(dataset, kind, str(row["preset_sha256"])) for row in development]
        )
        scaler = StandardScaler().fit(train_raw)
        train_scaled = scaler.transform(train_raw)
        development_scaled = scaler.transform(development_raw)
        components = min(128, len(train) - 1, train_scaled.shape[1])
        pca = PCA(n_components=components, random_state=20_260_722).fit(train_scaled)
        train_embedding = pca.transform(train_scaled)
        development_embedding = pca.transform(development_scaled)
        train_embedding /= np.linalg.norm(train_embedding, axis=1, keepdims=True) + 1e-8
        development_embedding /= np.linalg.norm(development_embedding, axis=1, keepdims=True) + 1e-8
        nearest = np.argmax(development_embedding @ train_embedding.T, axis=1)

        reference_losses = []
        for reference_kind in reference_kinds:
            losses = []
            for dev_index, train_index in enumerate(nearest):
                dev = load_file(
                    dataset
                    / "features"
                    / reference_kind.value
                    / f"{development[dev_index]['preset_sha256']}.safetensors"
                )["feature"].float()
                candidate = load_file(
                    dataset
                    / "features"
                    / reference_kind.value
                    / f"{train[train_index]['preset_sha256']}.safetensors"
                )["feature"].float()
                losses.append(float((dev - candidate).square().mean()))
            reference_losses.append(float(np.mean(losses)))
        results[kind.value] = {
            "raw_dimension": int(train_raw.shape[1]),
            "embedding_dimension": components,
            "log_mel_mse": reference_losses[0],
            "multi_stft_mse": reference_losses[1],
            "combined_sound_distance": float(sum(reference_losses)),
            "seconds": time.monotonic() - started,
        }

    best_distance = min(float(item["combined_sound_distance"]) for item in results.values())
    eligible = tuple(
        (name, item)
        for name, item in results.items()
        if float(item["combined_sound_distance"]) <= best_distance * 1.02
    )
    selected_input, _ = min(
        eligible, key=lambda pair: (int(pair[1]["raw_dimension"]), float(pair[1]["seconds"]))
    )

    distributions = corpus_report["distributions"]
    fixed_controls = distributions["control_count"]["p95"] - distributions["control_count"]["minimum"] <= 8
    sparse_routes = distributions["active_modulation_count"]["median"] < 32
    large_language = distributions["externalized_json_bytes"]["p95"] > 16_384
    if not (fixed_controls and sparse_routes and large_language):
        raise ValueError("Corpus does not satisfy the hybrid representation contract")
    recommendation = {
        "audio_encoding": selected_input,
        "preset_representation": "hybrid_controls_template_retrieval_v1",
        "control_dimension": len(CONTROL_NAMES),
        "model_parameter_target": 20_000_000,
        "reason": (
            "The corpus has fixed-width controls, sparse modulation, compact graphs, "
            "and asset-dominated serialized presets."
        ),
    }
    repo_root = Path(__file__).resolve().parent.parent
    output = create_output_directory(repo_root, GENERATOR_NAME)
    report_path = output / "architecture_investigation.json"
    plot_path = output / "audio_encoding_comparison.png"
    report = {
        "generator": GENERATOR_NAME,
        "dataset": str(dataset),
        "development_examples": len(development),
        "training_examples": len(train),
        "input_results": results,
        "recommendation": recommendation,
        "plot": str(plot_path.relative_to(repo_root)),
    }
    figure, axis = plt.subplots(figsize=(8, 4.5))
    names = list(results)
    axis.bar(names, [results[name]["combined_sound_distance"] for name in names])
    axis.set_ylabel("development retrieval sound distance")
    axis.set_title("Audio encoding bake-off")
    figure.tight_layout()
    figure.savefig(plot_path, dpi=160)
    plt.close(figure)
    report_path.write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    publish_gallery_run(
        repo_root,
        output,
        GENERATOR_NAME,
        "/vital-investigation",
        (
            (report_path, "json", "architecture report"),
            (plot_path, "image", "audio encoding comparison"),
        ),
    )
    print(output)


if __name__ == "__main__":
    main()
