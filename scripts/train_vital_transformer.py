#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file, save_file
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from synth import A6000_UUID, VitalRuntime, create_output_directory, pin_a6000


BASE_GENERATOR_NAME = "train_vital_transformer_v1"
AUGMENTED_GENERATOR_NAME = "train_vital_augmented_transformer_v2"
SEED = 20_260_722


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--architecture-report", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--surrogate-epochs", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def load_rows(dataset: Path) -> tuple[dict[str, object], ...]:
    return tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )


def summarize(feature: np.ndarray) -> np.ndarray:
    if feature.ndim == 3:
        return np.concatenate((feature.mean(axis=-1).ravel(), feature.std(axis=-1).ravel()))
    if feature.shape[-1] == 4096:
        return np.log1p(np.abs(np.fft.rfft(feature, axis=-1)))[:, ::16].ravel()
    return np.concatenate((feature.mean(axis=-1), feature.std(axis=-1)))


def plot_metrics(metrics: list[dict[str, float]], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    epochs = [row["epoch"] for row in metrics]
    axes[0].plot(epochs, [row["train_total_loss"] for row in metrics], label="train")
    axes[0].plot(epochs, [row["test_total_loss"] for row in metrics], label="test")
    axes[0].set_title("Total sound-driven loss")
    axes[0].set_xlabel("epoch")
    axes[0].legend()
    axes[1].plot(epochs, [row["train_sound_loss"] for row in metrics], label="train")
    axes[1].plot(epochs, [row["test_sound_loss"] for row in metrics], label="test")
    axes[1].set_title("Surrogate audio reconstruction")
    axes[1].set_xlabel("epoch")
    axes[1].legend()
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    gpu = pin_a6000()
    import torch
    import torch.nn.functional as functional
    from torch import Tensor
    from torch.utils.data import DataLoader, TensorDataset

    from synth.models.audio_to_preset import (
        AudioToPresetConfig,
        AudioToPresetTransformer,
        PresetAudioSurrogate,
    )
    from synth.preset_representation import ControlStatistics, load_presets

    args = parse_args()
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")
    dataset = args.dataset_dir.resolve(strict=True)
    architecture = json.loads(args.architecture_report.resolve(strict=True).read_text())
    feature_name = architecture["recommendation"]["audio_encoding"]
    rows = load_rows(dataset)
    training_rows = tuple(row for row in rows if row["split"] != "test")
    test_rows = tuple(row for row in rows if row["split"] == "test")
    if not test_rows:
        raise ValueError("The experiment requires a non-empty test split")
    augmented_example_count = sum(bool(row.get("is_augmented", False)) for row in training_rows)
    generator_name = (
        AUGMENTED_GENERATOR_NAME
        if augmented_example_count
        else BASE_GENERATOR_NAME
    )
    row_by_digest = {
        str(row["preset_sha256"]): row
        for row in training_rows
        if not bool(row.get("is_augmented", False))
    }
    template_digests = tuple(
        dict.fromkeys(
            str(row.get("template_preset_sha256", row["preset_sha256"]))
            for row in training_rows
        )
    )
    missing_templates = tuple(
        digest for digest in template_digests if digest not in row_by_digest
    )
    if missing_templates:
        raise ValueError(f"Missing {len(missing_templates)} template presets")
    template_rows = tuple(row_by_digest[digest] for digest in template_digests)
    template_index_by_digest = {
        digest: index for index, digest in enumerate(template_digests)
    }

    training_presets = load_presets(dataset, training_rows)
    test_presets = load_presets(dataset, test_rows)
    template_presets = load_presets(dataset, template_rows)
    control_statistics = ControlStatistics.fit(training_presets)
    training_controls_np = np.stack(
        [control_statistics.encode(preset)[0] for preset in training_presets]
    )
    training_masks_np = np.stack(
        [control_statistics.encode(preset)[1] for preset in training_presets]
    )
    test_controls_np = np.stack(
        [control_statistics.encode(preset)[0] for preset in test_presets]
    )
    test_masks_np = np.stack(
        [control_statistics.encode(preset)[1] for preset in test_presets]
    )

    def feature_for(row: dict[str, object]) -> Tensor:
        return load_file(
            dataset
            / "features"
            / feature_name
            / f"{row['preset_sha256']}.safetensors"
        )["feature"]

    training_features = torch.stack([feature_for(row) for row in training_rows])
    test_features = torch.stack([feature_for(row) for row in test_rows])
    feature_mean = float(training_features.float().mean())
    feature_std = float(training_features.float().std())
    training_features = ((training_features.float() - feature_mean) / feature_std).half()
    test_features = ((test_features.float() - feature_mean) / feature_std).half()

    training_summaries = np.stack(
        [summarize(feature_for(row).float().numpy()) for row in training_rows]
    )
    test_summaries = np.stack(
        [summarize(feature_for(row).float().numpy()) for row in test_rows]
    )
    summary_scaler = StandardScaler().fit(training_summaries)
    pca = PCA(n_components=128, random_state=SEED).fit(
        summary_scaler.transform(training_summaries)
    )
    training_audio_np = pca.transform(summary_scaler.transform(training_summaries))
    test_audio_np = pca.transform(summary_scaler.transform(test_summaries))
    audio_scaler = StandardScaler().fit(training_audio_np)
    training_audio_np = audio_scaler.transform(training_audio_np).astype(np.float32)
    test_audio_np = audio_scaler.transform(test_audio_np).astype(np.float32)

    output = create_output_directory(Path(__file__).resolve().parent.parent, generator_name)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    control_statistics_path = output / "control_statistics.json"
    control_statistics_path.write_text(
        control_statistics.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    np.savez(
        output / "audio_projection.npz",
        summary_mean=summary_scaler.mean_,
        summary_scale=summary_scaler.scale_,
        pca_mean=pca.mean_,
        pca_components=pca.components_,
        audio_mean=audio_scaler.mean_,
        audio_scale=audio_scaler.scale_,
    )
    (output / "preprocessing.json").write_text(
        json.dumps(
            {
                "feature_name": feature_name,
                "feature_mean": feature_mean,
                "feature_standard_deviation": feature_std,
                "training_preset_sha256": [row["preset_sha256"] for row in training_rows],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    training_controls = torch.from_numpy(training_controls_np)
    training_masks = torch.from_numpy(training_masks_np)
    training_audio = torch.from_numpy(training_audio_np)
    test_controls = torch.from_numpy(test_controls_np)
    test_masks = torch.from_numpy(test_masks_np)
    test_audio = torch.from_numpy(test_audio_np)
    template_indexes = torch.tensor(
        [
            template_index_by_digest[
                str(row.get("template_preset_sha256", row["preset_sha256"]))
            ]
            for row in training_rows
        ],
        dtype=torch.long,
    )

    surrogate = PresetAudioSurrogate(
        len(template_rows),
        template_width=256 if augmented_example_count else 128,
        hidden_width=2048 if augmented_example_count else 1024,
        hidden_layers=3 if augmented_example_count else 2,
    ).to(device)
    surrogate_optimizer = torch.optim.AdamW(surrogate.parameters(), lr=1e-3, weight_decay=0.01)
    surrogate_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        surrogate_optimizer,
        T_max=args.surrogate_epochs,
        eta_min=1e-5,
    )
    surrogate_loader = DataLoader(
        TensorDataset(training_controls, template_indexes, training_audio),
        batch_size=256 if augmented_example_count else 128,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    surrogate_curve = []
    best_surrogate_loss = float("inf")
    best_surrogate_state: dict[str, Tensor] | None = None
    for epoch in range(1, args.surrogate_epochs + 1):
        total = 0.0
        for controls, indexes, target_audio in surrogate_loader:
            controls = controls.to(device)
            indexes = indexes.to(device)
            target_audio = target_audio.to(device)
            predicted = surrogate(controls, indexes)
            loss = functional.mse_loss(predicted, target_audio)
            surrogate_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            surrogate_optimizer.step()
            total += float(loss) * controls.shape[0]
        mean_loss = total / len(training_rows)
        surrogate_curve.append(mean_loss)
        if mean_loss < best_surrogate_loss:
            best_surrogate_loss = mean_loss
            best_surrogate_state = {
                name: value.detach().cpu().contiguous()
                for name, value in surrogate.state_dict().items()
            }
        surrogate_scheduler.step()
        if epoch % 10 == 0:
            print(f"surrogate {epoch}/{args.surrogate_epochs}: {mean_loss:.6f}", flush=True)
    required_ratio = 0.30 if augmented_example_count else 0.25
    if best_surrogate_loss >= surrogate_curve[0] * required_ratio:
        raise RuntimeError("Preset-to-audio surrogate did not pass its overfit gate")
    if best_surrogate_state is None:
        raise AssertionError("Surrogate training produced no checkpoint")
    surrogate.load_state_dict(best_surrogate_state)
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)
    surrogate.eval()

    config = AudioToPresetConfig(
        feature_name=feature_name,
        template_count=len(template_rows),
    )
    model = AudioToPresetTransformer(config).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    combined_parameters = parameter_count + sum(
        parameter.numel() for parameter in surrogate.parameters()
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    total_steps = args.epochs * math.ceil(len(training_rows) / args.batch_size)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=3e-4,
        total_steps=total_steps,
        pct_start=0.05,
    )
    scaler = torch.cuda.amp.GradScaler()
    train_loader = DataLoader(
        TensorDataset(
            training_features,
            training_controls,
            training_masks,
            training_audio,
            template_indexes,
        ),
        batch_size=args.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(SEED),
    )
    test_loader = DataLoader(
        TensorDataset(test_features, test_controls, test_masks, test_audio),
        batch_size=args.batch_size,
    )

    def losses(
        feature: Tensor,
        target_controls: Tensor,
        control_mask: Tensor,
        target_audio: Tensor,
        target_template: Tensor | None,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        predicted_controls, template_logits, direct_audio = model(feature)
        probabilities = functional.softmax(template_logits, dim=-1)
        surrogate_audio = surrogate.forward_soft(predicted_controls, probabilities)
        sound_loss = functional.mse_loss(surrogate_audio, target_audio)
        direct_loss = functional.mse_loss(direct_audio, target_audio)
        squared_controls = (predicted_controls - target_controls).square()
        control_loss = (squared_controls * control_mask).sum() / control_mask.sum()
        template_loss = (
            functional.cross_entropy(template_logits, target_template)
            if target_template is not None
            else torch.zeros((), device=feature.device)
        )
        total = sound_loss + 0.5 * direct_loss + 0.1 * control_loss + 0.05 * template_loss
        return total, sound_loss, control_loss, template_loss

    metrics: list[dict[str, float]] = []
    best_test = float("inf")
    started = time.monotonic()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_totals = np.zeros(4, dtype=np.float64)
        for feature, controls, mask, target_audio, target_template in train_loader:
            feature = feature.to(device)
            controls = controls.to(device)
            mask = mask.to(device)
            target_audio = target_audio.to(device)
            target_template = target_template.to(device)
            feature = feature + torch.randn_like(feature) * 0.01
            with torch.cuda.amp.autocast(dtype=torch.float16):
                batch_losses = losses(feature, controls, mask, target_audio, target_template)
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(batch_losses[0]).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            train_totals += np.asarray([float(item) for item in batch_losses]) * feature.shape[0]

        model.eval()
        test_totals = np.zeros(4, dtype=np.float64)
        with torch.inference_mode():
            for feature, controls, mask, target_audio in test_loader:
                feature = feature.to(device)
                controls = controls.to(device)
                mask = mask.to(device)
                target_audio = target_audio.to(device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    batch_losses = losses(feature, controls, mask, target_audio, None)
                test_totals += np.asarray([float(item) for item in batch_losses]) * feature.shape[0]
        train_totals /= len(training_rows)
        test_totals /= len(test_rows)
        row = {
            "epoch": float(epoch),
            "train_total_loss": float(train_totals[0]),
            "train_sound_loss": float(train_totals[1]),
            "train_control_loss": float(train_totals[2]),
            "train_template_loss": float(train_totals[3]),
            "test_total_loss": float(test_totals[0]),
            "test_sound_loss": float(test_totals[1]),
            "test_control_loss": float(test_totals[2]),
            "learning_rate": scheduler.get_last_lr()[0],
        }
        metrics.append(row)
        print(
            f"epoch {epoch}/{args.epochs}: train={row['train_total_loss']:.5f} "
            f"test={row['test_total_loss']:.5f}",
            flush=True,
        )
        state = {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()}
        save_file(state, checkpoints / "last.safetensors")
        if row["test_sound_loss"] < best_test:
            best_test = row["test_sound_loss"]
            save_file(state, checkpoints / "best.safetensors")

    save_file(
        {name: value.detach().cpu().contiguous() for name, value in surrogate.state_dict().items()},
        checkpoints / "surrogate.safetensors",
    )
    config_path = output / "model_config.json"
    config_path.write_text(config.model_dump_json(indent=2) + "\n", encoding="utf-8")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    plot_metrics(metrics, output / "loss.png")
    figure, axis = plt.subplots(figsize=(7, 4.5))
    axis.plot(range(1, len(surrogate_curve) + 1), surrogate_curve)
    axis.set_title("Preset-to-audio surrogate loss")
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE")
    figure.tight_layout()
    figure.savefig(output / "surrogate_loss.png", dpi=160)
    plt.close(figure)

    report_path = output / "training_report.json"
    report = {
        "generator": generator_name,
        "status": "rendering_comparisons",
        "dataset": str(dataset),
        "architecture_report": str(args.architecture_report.resolve(strict=True)),
        "feature_name": feature_name,
        "preset_representation": architecture["recommendation"]["preset_representation"],
        "gpu_uuid": A6000_UUID,
        "gpu": gpu.model_dump(),
        "model_parameters": parameter_count,
        "combined_parameters": combined_parameters,
        "training_examples": len(training_rows),
        "training_templates": len(template_rows),
        "augmented_examples": augmented_example_count,
        "test_examples": len(test_rows),
        "epochs": args.epochs,
        "surrogate_epochs": args.surrogate_epochs,
        "training_seconds": time.monotonic() - started,
        "best_test_sound_loss": best_test,
        "plots": ["loss.png", "surrogate_loss.png"],
        "comparisons": "comparisons.json",
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    model.load_state_dict(load_file(checkpoints / "best.safetensors", device="cuda:0"))
    model.eval()
    comparison_rows = tuple(
        row
        for row in rows
        if row["evaluation_family"] is not None
        and not bool(row.get("is_augmented", False))
    )
    comparison_root = output / "comparisons"
    render_requests = []
    comparison_manifest = []
    with torch.inference_mode():
        for row in comparison_rows:
            digest = str(row["preset_sha256"])
            feature = feature_for(row).float()
            feature = ((feature - feature_mean) / feature_std).half().unsqueeze(0).to(device)
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predicted_controls, template_logits, _ = model(feature)
            template_index = int(template_logits.argmax(dim=-1).item())
            template = template_presets[template_index]
            controls = predicted_controls.squeeze(0).float().cpu().numpy().clip(-6.0, 6.0)
            generated = control_statistics.apply(template, controls)
            split = str(row["split"])
            item_root = comparison_root / split / digest
            item_root.mkdir(parents=True, exist_ok=True)
            input_audio = item_root / "input.wav"
            output_audio = item_root / "output.wav"
            preset_path = item_root / "predicted.vital"
            shutil.copyfile(dataset / str(row["audio_file"]), input_audio)
            generated.to_file(preset_path)
            render_requests.append((preset_path, output_audio))
            comparison_manifest.append(
                {
                    "preset_sha256": digest,
                    "preset_name": row["preset_name"],
                    "pack": row["pack"],
                    "family": row["evaluation_family"],
                    "split": split,
                    "template_sha256": template_rows[template_index]["preset_sha256"],
                    "input_audio": str(input_audio.relative_to(output)),
                    "output_audio": str(output_audio.relative_to(output)),
                    "predicted_preset": str(preset_path.relative_to(output)),
                }
            )

    runtime = VitalRuntime.from_repo_root(Path(__file__).resolve().parent.parent)
    worker = Path(__file__).resolve().parent / "render_vital_preset_worker.py"
    plugin = args.vital_vst3.resolve(strict=True)

    def render(request: tuple[Path, Path]) -> None:
        preset_path, audio_path = request
        runtime.run_worker(
            worker,
            "--vital-vst3",
            str(plugin),
            "--preset",
            str(preset_path),
            "--output",
            str(audio_path),
            capture_output=True,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [executor.submit(render, request) for request in render_requests]
        for future in as_completed(futures):
            future.result()

    (output / "comparisons.json").write_text(
        json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
    )
    report["status"] = "complete"
    report["training_seconds"] = time.monotonic() - started
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
