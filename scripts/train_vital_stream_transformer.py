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

from synth import A6000_UUID, VitalRuntime, create_output_directory, pin_a6000


GENERATOR_NAME = "train_vital_stream_transformer_v3"
SEED = 20_260_723


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-dir", required=True, type=Path)
    parser.add_argument("--reference-dataset-dir", required=True, type=Path)
    parser.add_argument("--reference-training-dir", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--prebuffer-shards", type=int, default=145)
    parser.add_argument("--passes-per-shard", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    return parser.parse_args()


def summarize(feature: np.ndarray) -> np.ndarray:
    return np.concatenate(
        (feature.mean(axis=-1).ravel(), feature.std(axis=-1).ravel())
    )


def main() -> None:
    gpu = pin_a6000()
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    from torch import Tensor

    from synth.models.audio_to_preset import (
        AudioToPresetConfig,
        AudioToPresetTransformer,
        PresetAudioSurrogate,
    )
    from synth.preset_representation import ControlStatistics, load_presets

    args = parse_args()
    if args.batch_size % 4:
        raise ValueError("Batch size must be divisible by four")
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")
    stream = args.stream_dir.resolve(strict=True)
    dataset = args.reference_dataset_dir.resolve(strict=True)
    reference = args.reference_training_dir.resolve(strict=True)
    output = create_output_directory(Path(__file__).resolve().parent.parent, GENERATOR_NAME)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()

    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    training_rows = tuple(row for row in rows if row["split"] != "test")
    test_rows = tuple(row for row in rows if row["split"] == "test")
    original_by_digest = {
        str(row["preset_sha256"]): row
        for row in training_rows
        if not bool(row["is_augmented"])
    }
    template_digests = tuple(
        dict.fromkeys(str(row["template_preset_sha256"]) for row in training_rows)
    )
    template_rows = tuple(original_by_digest[digest] for digest in template_digests)
    template_presets = load_presets(dataset, template_rows)
    template_index_by_digest = {
        digest: index for index, digest in enumerate(template_digests)
    }
    replay_rows = template_rows
    replay_presets = template_presets
    test_presets = load_presets(dataset, test_rows)
    statistics = ControlStatistics.model_validate_json(
        (reference / "control_statistics.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads(
        (reference / "preprocessing.json").read_text(encoding="utf-8")
    )
    feature_mean = float(preprocessing["feature_mean"])
    feature_std = float(preprocessing["feature_standard_deviation"])
    projection = np.load(reference / "audio_projection.npz")

    def raw_feature(row: dict[str, object]) -> Tensor:
        return load_file(
            dataset
            / "features"
            / "log_mel"
            / f"{row['preset_sha256']}.safetensors"
        )["feature"]

    def audio_target(feature: Tensor) -> np.ndarray:
        summary = summarize(feature.float().numpy())
        scaled = (
            summary - projection["summary_mean"]
        ) / projection["summary_scale"]
        embedding = (
            scaled - projection["pca_mean"]
        ) @ projection["pca_components"].T
        return np.asarray(
            (embedding - projection["audio_mean"]) / projection["audio_scale"],
            dtype=np.float32,
        )

    replay_features = torch.stack([raw_feature(row) for row in replay_rows])
    replay_controls = torch.from_numpy(
        np.stack([statistics.encode(preset)[0] for preset in replay_presets])
    ).half()
    replay_masks = torch.from_numpy(
        np.stack([statistics.encode(preset)[1] for preset in replay_presets])
    )
    replay_audio = torch.from_numpy(
        np.stack([audio_target(raw_feature(row)) for row in replay_rows])
    ).half()
    replay_templates = torch.tensor(
        [template_index_by_digest[str(row["preset_sha256"])] for row in replay_rows],
        dtype=torch.long,
    )
    test_features = torch.stack([raw_feature(row) for row in test_rows])
    test_controls = torch.from_numpy(
        np.stack([statistics.encode(preset)[0] for preset in test_presets])
    ).half()
    test_masks = torch.from_numpy(
        np.stack([statistics.encode(preset)[1] for preset in test_presets])
    )
    test_audio = torch.from_numpy(
        np.stack([audio_target(raw_feature(row)) for row in test_rows])
    ).half()

    config = AudioToPresetConfig.model_validate_json(
        (reference / "model_config.json").read_text(encoding="utf-8")
    )
    if config.template_count != len(template_rows):
        raise ValueError("Reference model template order does not match the stream")
    model = AudioToPresetTransformer(config).to(device)
    model.load_state_dict(
        load_file(reference / "checkpoints" / "best.safetensors", device="cuda")
    )
    surrogate = PresetAudioSurrogate(
        len(template_rows),
        template_width=256,
        hidden_width=2048,
        hidden_layers=3,
    ).to(device)
    surrogate.load_state_dict(
        load_file(reference / "checkpoints" / "surrogate.safetensors", device="cuda")
    )
    surrogate.eval()
    for parameter in surrogate.parameters():
        parameter.requires_grad_(False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    planned_examples = len(template_rows) * 40
    stream_batch_size = args.batch_size * 3 // 4
    replay_batch_size = args.batch_size - stream_batch_size
    full_shards, final_shard_size = divmod(planned_examples, 512)
    batches_per_pass = full_shards * math.ceil(512 / stream_batch_size)
    if final_shard_size:
        batches_per_pass += math.ceil(final_shard_size / stream_batch_size)
    planned_steps = batches_per_pass * args.passes_per_shard
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=planned_steps,
        eta_min=2e-6,
    )
    scaler = torch.cuda.amp.GradScaler()

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

    def evaluate() -> tuple[float, float, float]:
        model.eval()
        with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
            values = losses(
                ((test_features.float() - feature_mean) / feature_std)
                .half()
                .to(device),
                test_controls.to(device),
                test_masks.to(device),
                test_audio.to(device),
                None,
            )
        return float(values[0]), float(values[1]), float(values[2])

    def save_model(path: Path) -> None:
        temporary = path.with_suffix(".partial.safetensors")
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            temporary,
        )
        temporary.replace(path)

    initial_total, initial_sound, initial_control = evaluate()
    best_sound = initial_sound
    save_model(checkpoints / "best.safetensors")
    report_path = output / "training_report.json"
    report = {
        "generator": GENERATOR_NAME,
        "status": "waiting_for_prebuffer",
        "stream": str(stream),
        "reference_dataset": str(dataset),
        "reference_training": str(reference),
        "gpu_uuid": A6000_UUID,
        "gpu": gpu.model_dump(),
        "feature_gpu_uuid": "GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea",
        "prebuffer_shards": args.prebuffer_shards,
        "passes_per_shard": args.passes_per_shard,
        "batch_size": args.batch_size,
        "replay_fraction": replay_batch_size / args.batch_size,
        "initial_test_total_loss": initial_total,
        "initial_test_sound_loss": initial_sound,
        "initial_test_control_loss": initial_control,
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    def producer_state() -> dict[str, object]:
        return json.loads((stream / "stream_state.json").read_text(encoding="utf-8"))

    while True:
        available = tuple((stream / "shards").glob("shard_*.safetensors"))
        state = producer_state()
        if (
            len(available) >= args.prebuffer_shards
            or state["status"] == "complete"
        ):
            break
        print(
            f"waiting for stream prebuffer: {len(available)}/{args.prebuffer_shards}",
            flush=True,
        )
        time.sleep(30)

    report["status"] = "training"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    started = time.monotonic()
    metrics = []
    processed_examples = 0
    optimizer_steps = 0
    mode_counts: dict[str, int] = {}
    changed_control_total = 0
    metadata_examples = 0
    next_chunk = 0
    replay_generator = torch.Generator().manual_seed(SEED)

    while True:
        shard_path = stream / "shards" / f"shard_{next_chunk:05d}.safetensors"
        metadata_path = stream / "metadata" / f"shard_{next_chunk:05d}.jsonl"
        if not shard_path.exists():
            state = producer_state()
            if state["status"] == "complete":
                break
            time.sleep(10)
            continue

        shard = load_file(shard_path)
        stream_features = shard["feature"]
        stream_controls = shard["controls"]
        stream_masks = shard["mask"]
        stream_audio = shard["audio_target"]
        stream_templates = shard["template_index"].long()
        metadata = tuple(
            json.loads(line)
            for line in metadata_path.read_text(encoding="utf-8").splitlines()
        )
        if len(metadata) != len(stream_features):
            raise ValueError(f"Shard {next_chunk} tensor/metadata count mismatch")
        for item in metadata:
            mode = str(item["mode"])
            mode_counts[mode] = mode_counts.get(mode, 0) + 1
            changed_control_total += len(item["changed_controls"])
            metadata_examples += 1

        model.train()
        shard_totals = np.zeros(4, dtype=np.float64)
        shard_seen = 0
        for pass_index in range(args.passes_per_shard):
            order = torch.randperm(
                len(stream_features),
                generator=torch.Generator().manual_seed(
                    SEED + next_chunk * args.passes_per_shard + pass_index
                ),
            )
            for start in range(0, len(order), stream_batch_size):
                indexes = order[start : start + stream_batch_size]
                replay_indexes = torch.randint(
                    len(replay_features),
                    (replay_batch_size,),
                    generator=replay_generator,
                )
                feature = torch.cat(
                    (stream_features[indexes], replay_features[replay_indexes])
                )
                controls = torch.cat(
                    (stream_controls[indexes], replay_controls[replay_indexes])
                )
                mask = torch.cat(
                    (stream_masks[indexes], replay_masks[replay_indexes])
                )
                target_audio = torch.cat(
                    (stream_audio[indexes], replay_audio[replay_indexes])
                )
                target_template = torch.cat(
                    (stream_templates[indexes], replay_templates[replay_indexes])
                )
                feature = (
                    ((feature.float() - feature_mean) / feature_std).half().to(device)
                )
                feature = feature + torch.randn_like(feature) * 0.01
                controls = controls.to(device)
                mask = mask.to(device)
                target_audio = target_audio.to(device)
                target_template = target_template.to(device)
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    batch_losses = losses(
                        feature,
                        controls,
                        mask,
                        target_audio,
                        target_template,
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(batch_losses[0]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                shard_totals += np.asarray(
                    [float(value) for value in batch_losses]
                ) * feature.shape[0]
                shard_seen += feature.shape[0]
                optimizer_steps += 1
        shard_totals /= shard_seen
        processed_examples += len(stream_features)

        if (next_chunk + 1) % 10 == 0:
            test_total, test_sound, test_control = evaluate()
            metric = {
                "chunk": float(next_chunk + 1),
                "processed_stream_examples": float(processed_examples),
                "optimizer_steps": float(optimizer_steps),
                "train_total_loss": float(shard_totals[0]),
                "train_sound_loss": float(shard_totals[1]),
                "train_control_loss": float(shard_totals[2]),
                "train_template_loss": float(shard_totals[3]),
                "test_total_loss": test_total,
                "test_sound_loss": test_sound,
                "test_control_loss": test_control,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            metrics.append(metric)
            if test_sound < best_sound:
                best_sound = test_sound
                save_model(checkpoints / "best.safetensors")
            print(
                f"chunk {next_chunk + 1}: train={metric['train_total_loss']:.5f} "
                f"test={test_total:.5f} sound={test_sound:.5f}",
                flush=True,
            )
        save_model(checkpoints / "latest.safetensors")
        shard_path.unlink()
        metadata_path.unlink()
        next_chunk += 1

    final_total, final_sound, final_control = evaluate()
    if final_sound < best_sound:
        best_sound = final_sound
        save_model(checkpoints / "best.safetensors")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(
        [row["processed_stream_examples"] for row in metrics],
        [row["test_total_loss"] for row in metrics],
    )
    axes[0].set_title("Untouched test total loss")
    axes[0].set_xlabel("unique stream examples consumed")
    axes[1].plot(
        [row["processed_stream_examples"] for row in metrics],
        [row["test_sound_loss"] for row in metrics],
    )
    axes[1].set_title("Untouched test sound loss")
    axes[1].set_xlabel("unique stream examples consumed")
    figure.tight_layout()
    figure.savefig(output / "stream_loss.png", dpi=160)
    plt.close(figure)

    model.load_state_dict(
        load_file(checkpoints / "best.safetensors", device="cuda")
    )
    model.eval()
    comparison_rows = tuple(
        row
        for row in rows
        if row["evaluation_family"] is not None
        and not bool(row["is_augmented"])
    )
    comparison_root = output / "comparisons"
    render_tasks = []
    comparison_manifest = []
    with torch.inference_mode():
        for row in comparison_rows:
            digest = str(row["preset_sha256"])
            feature = (
                ((raw_feature(row).float() - feature_mean) / feature_std)
                .half()
                .unsqueeze(0)
                .to(device)
            )
            with torch.cuda.amp.autocast(dtype=torch.float16):
                predicted_controls, template_logits, _ = model(feature)
            template_index = int(template_logits.argmax(dim=-1))
            generated = statistics.apply(
                template_presets[template_index],
                predicted_controls.squeeze(0).float().cpu().numpy().clip(-6.0, 6.0),
            )
            item_root = comparison_root / str(row["split"]) / digest
            item_root.mkdir(parents=True, exist_ok=True)
            input_audio = item_root / "input.wav"
            output_audio = item_root / "output.wav"
            preset_path = item_root / "predicted.vital"
            shutil.copyfile(dataset / str(row["audio_file"]), input_audio)
            generated.to_file(preset_path)
            render_tasks.append(
                {"preset": str(preset_path), "output": str(output_audio)}
            )
            comparison_manifest.append(
                {
                    "preset_sha256": digest,
                    "preset_name": row["preset_name"],
                    "pack": row["pack"],
                    "family": row["evaluation_family"],
                    "split": row["split"],
                    "template_sha256": template_digests[template_index],
                    "input_audio": str(input_audio.relative_to(output)),
                    "output_audio": str(output_audio.relative_to(output)),
                    "predicted_preset": str(preset_path.relative_to(output)),
                }
            )

    runtime = VitalRuntime.from_repo_root(Path(__file__).resolve().parent.parent)
    worker = Path(__file__).resolve().parent / "render_vital_batch_worker.py"
    plugin = args.vital_vst3.resolve(strict=True)
    shards: list[list[dict[str, str]]] = [[] for _ in range(32)]
    for index, task in enumerate(render_tasks):
        shards[index % len(shards)].append(task)

    def render_shard(index: int, tasks: list[dict[str, str]]) -> None:
        path = output / f"comparison_render_{index:02d}.json"
        path.write_text(json.dumps(tasks), encoding="utf-8")
        runtime.run_worker(
            worker,
            "--vital-vst3",
            str(plugin),
            "--tasks",
            str(path),
            capture_output=True,
        )
        path.unlink()

    with ThreadPoolExecutor(max_workers=32) as executor:
        futures = [
            executor.submit(render_shard, index, shard)
            for index, shard in enumerate(shards)
            if shard
        ]
        for future in as_completed(futures):
            future.result()
    (output / "comparisons.json").write_text(
        json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
    )

    report.update(
        {
            "status": "complete",
            "processed_stream_examples": processed_examples,
            "optimizer_steps": optimizer_steps,
            "best_test_sound_loss": best_sound,
            "final_test_total_loss": final_total,
            "final_test_sound_loss": final_sound,
            "final_test_control_loss": final_control,
            "training_seconds": time.monotonic() - started,
            "mode_counts": mode_counts,
            "mean_changed_control_count": changed_control_total / metadata_examples,
            "remaining_stream_shards": len(
                tuple((stream / "shards").glob("shard_*.safetensors"))
            ),
            "model_parameters": sum(
                parameter.numel() for parameter in model.parameters()
            ),
            "plots": ["stream_loss.png"],
            "comparisons": "comparisons.json",
        }
    )
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
