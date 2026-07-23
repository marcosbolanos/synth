#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from safetensors.torch import load_file, save_file

from synth import A6000_UUID, VitalRuntime, pin_a6000


SEED = 20_260_722


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--candidates", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    return parser.parse_args()


def rows(path: Path) -> tuple[dict[str, object], ...]:
    return tuple(json.loads(line) for line in path.read_text().splitlines())


def main() -> None:
    gpu = pin_a6000()
    import torch
    import torch.nn.functional as functional
    from torch.utils.data import DataLoader, TensorDataset

    from synth.audio_features import AudioFeatureKind, extract_feature, read_render
    from synth.models.audio_to_preset import AudioToPresetConfig, AudioToPresetTransformer
    from synth.preset_representation import ControlStatistics, load_presets

    args = parse_args()
    torch.manual_seed(SEED)
    dataset = args.dataset_dir.resolve(strict=True)
    training = args.training_dir.resolve(strict=True)
    preprocessing = json.loads((training / "preprocessing.json").read_text())
    feature_name = preprocessing["feature_name"]
    feature_mean = float(preprocessing["feature_mean"])
    feature_std = float(preprocessing["feature_standard_deviation"])
    all_rows = rows(dataset / "dataset.jsonl")
    train_rows = tuple(row for row in all_rows if row["split"] != "test")
    test_rows = tuple(row for row in all_rows if row["split"] == "test")
    presets = load_presets(dataset, train_rows)
    statistics = ControlStatistics.model_validate_json(
        (training / "control_statistics.json").read_text()
    )
    controls_np = np.stack([statistics.encode(preset)[0] for preset in presets])
    masks_np = np.stack([statistics.encode(preset)[1] for preset in presets])

    def load_feature(row: dict[str, object]) -> torch.Tensor:
        value = load_file(
            dataset / "features" / feature_name / f"{row['preset_sha256']}.safetensors"
        )["feature"].float()
        return ((value - feature_mean) / feature_std).half()

    features = torch.stack([load_feature(row) for row in train_rows])
    config = AudioToPresetConfig.model_validate_json((training / "model_config.json").read_text())
    model = AudioToPresetTransformer(config).to("cuda")
    model.load_state_dict(load_file(training / "checkpoints" / "best.safetensors", device="cuda"))
    runtime = VitalRuntime.from_repo_root(Path(__file__).resolve().parent.parent)
    worker = Path(__file__).resolve().parent / "render_vital_batch_worker.py"
    plugin = args.vital_vst3.resolve(strict=True)

    def render_tasks(tasks: list[dict[str, str]], root: Path) -> None:
        shards: list[list[dict[str, str]]] = [[] for _ in range(8)]
        for index, task in enumerate(tasks):
            shards[index % len(shards)].append(task)

        def run_shard(index: int, shard: list[dict[str, str]]) -> None:
            path = root / f"render_shard_{index:02d}.json"
            path.write_text(json.dumps(shard), encoding="utf-8")
            runtime.run_worker(
                worker,
                "--vital-vst3",
                str(plugin),
                "--tasks",
                str(path),
            )

        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = [executor.submit(run_shard, index, shard) for index, shard in enumerate(shards)]
            for future in as_completed(futures):
                future.result()

    preference_history = []
    for round_index in range(1, args.rounds + 1):
        round_root = training / f"preference_round_{round_index}"
        preset_root = round_root / "presets"
        audio_root = round_root / "audio"
        preset_root.mkdir(parents=True, exist_ok=True)
        audio_root.mkdir(parents=True, exist_ok=True)
        candidate_templates = []
        predicted_controls = []
        model.eval()
        loader = DataLoader(TensorDataset(features), batch_size=args.batch_size)
        with torch.inference_mode():
            for (feature,) in loader:
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    control, logits, _ = model(feature.to("cuda"))
                predicted_controls.append(control.float().cpu())
                candidate_templates.append(logits.topk(args.candidates, dim=-1).indices.cpu())
        predicted_control_tensor = torch.cat(predicted_controls)
        candidate_template_tensor = torch.cat(candidate_templates)

        tasks = []
        for item_index, row in enumerate(train_rows):
            digest = str(row["preset_sha256"])
            control = predicted_control_tensor[item_index].numpy().clip(-6.0, 6.0)
            for rank, template_index_tensor in enumerate(candidate_template_tensor[item_index]):
                template_index = int(template_index_tensor)
                generated = statistics.apply(presets[template_index], control)
                preset_path = preset_root / f"{digest}_{rank}.vital"
                audio_path = audio_root / f"{digest}_{rank}.wav"
                generated.to_file(preset_path, overwrite=True)
                tasks.append({"preset": str(preset_path), "output": str(audio_path)})
        render_tasks(tasks, round_root)

        preferences = []
        with torch.inference_mode():
            for item_index, row in enumerate(train_rows):
                digest = str(row["preset_sha256"])
                source_audio = read_render(dataset / str(row["audio_file"])).to("cuda")
                source_mel = extract_feature(source_audio, AudioFeatureKind.LOG_MEL)
                source_stft = extract_feature(source_audio, AudioFeatureKind.MULTI_STFT)
                scores = []
                for rank in range(args.candidates):
                    candidate_audio = read_render(audio_root / f"{digest}_{rank}.wav").to("cuda")
                    candidate_mel = extract_feature(candidate_audio, AudioFeatureKind.LOG_MEL)
                    candidate_stft = extract_feature(candidate_audio, AudioFeatureKind.MULTI_STFT)
                    score = float(
                        (source_mel - candidate_mel).square().mean()
                        + (source_stft - candidate_stft).square().mean()
                    )
                    scores.append(score)
                best_rank = int(np.argmin(scores))
                worst_rank = int(np.argmax(scores))
                preferences.append(
                    {
                        "preset_sha256": digest,
                        "best_template": int(candidate_template_tensor[item_index, best_rank]),
                        "worst_template": int(candidate_template_tensor[item_index, worst_rank]),
                        "best_score": scores[best_rank],
                        "worst_score": scores[worst_rank],
                    }
                )
                if (item_index + 1) % 100 == 0:
                    print(f"preference scoring {item_index + 1}/{len(train_rows)}", flush=True)
        (round_root / "preferences.json").write_text(
            json.dumps(preferences, indent=2) + "\n", encoding="utf-8"
        )

        best = torch.tensor([item["best_template"] for item in preferences])
        worst = torch.tensor([item["worst_template"] for item in preferences])
        controls = torch.from_numpy(controls_np)
        masks = torch.from_numpy(masks_np)
        preference_loader = DataLoader(
            TensorDataset(features, controls, masks, best, worst),
            batch_size=args.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED + round_index),
        )
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
        model.train()
        total_loss = 0.0
        for feature, target_controls, mask, preferred, rejected in preference_loader:
            feature = feature.to("cuda")
            target_controls = target_controls.to("cuda")
            mask = mask.to("cuda")
            preferred = preferred.to("cuda")
            rejected = rejected.to("cuda")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                control, logits, _ = model(feature)
                preferred_logit = logits.gather(1, preferred[:, None]).squeeze(1)
                rejected_logit = logits.gather(1, rejected[:, None]).squeeze(1)
                preference_loss = -functional.logsigmoid(preferred_logit - rejected_logit).mean()
                control_loss = ((control - target_controls).square() * mask).sum() / mask.sum()
                loss = preference_loss + 0.05 * control_loss
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += float(loss) * feature.shape[0]
        round_loss = total_loss / len(train_rows)
        preference_history.append(
            {
                "round": round_index,
                "loss": round_loss,
                "mean_best_score": float(np.mean([item["best_score"] for item in preferences])),
                "mean_worst_score": float(np.mean([item["worst_score"] for item in preferences])),
            }
        )
        save_file(
            {name: value.detach().cpu().contiguous() for name, value in model.state_dict().items()},
            training / "checkpoints" / f"preference_round_{round_index}.safetensors",
        )

    (training / "preference_metrics.json").write_text(
        json.dumps(preference_history, indent=2) + "\n", encoding="utf-8"
    )

    comparison_rows = tuple(
        row for row in all_rows if row["evaluation_family"] is not None
    )
    comparison_manifest = []
    final_render_tasks = []
    model.eval()
    with torch.inference_mode():
        for row in comparison_rows:
            digest = str(row["preset_sha256"])
            feature = load_feature(row).unsqueeze(0).to("cuda")
            with torch.cuda.amp.autocast(dtype=torch.float16):
                control, logits, _ = model(feature)
            template_index = int(logits.argmax(dim=-1))
            generated = statistics.apply(
                presets[template_index],
                control.squeeze(0).float().cpu().numpy().clip(-6.0, 6.0),
            )
            item_root = training / "comparisons" / str(row["split"]) / digest
            item_root.mkdir(parents=True, exist_ok=True)
            input_audio = item_root / "input.wav"
            output_audio = item_root / "output.wav"
            preset_path = item_root / "predicted.vital"
            if not input_audio.exists():
                import shutil

                shutil.copyfile(dataset / str(row["audio_file"]), input_audio)
            generated.to_file(preset_path, overwrite=True)
            final_render_tasks.append(
                {"preset": str(preset_path), "output": str(output_audio)}
            )
            comparison_manifest.append(
                {
                    "preset_sha256": digest,
                    "preset_name": row["preset_name"],
                    "pack": row["pack"],
                    "family": row["evaluation_family"],
                    "split": row["split"],
                    "template_sha256": train_rows[template_index]["preset_sha256"],
                    "input_audio": str(input_audio.relative_to(training)),
                    "output_audio": str(output_audio.relative_to(training)),
                    "predicted_preset": str(preset_path.relative_to(training)),
                }
            )
    final_root = training / "final_preference_renders"
    final_root.mkdir(exist_ok=True)
    render_tasks(final_render_tasks, final_root)
    with torch.inference_mode():
        for item in comparison_manifest:
            source = read_render(training / str(item["input_audio"])).to("cuda")
            predicted = read_render(training / str(item["output_audio"])).to("cuda")
            source_mel = extract_feature(source, AudioFeatureKind.LOG_MEL)
            predicted_mel = extract_feature(predicted, AudioFeatureKind.LOG_MEL)
            source_stft = extract_feature(source, AudioFeatureKind.MULTI_STFT)
            predicted_stft = extract_feature(predicted, AudioFeatureKind.MULTI_STFT)
            item["log_mel_mse"] = float((source_mel - predicted_mel).square().mean())
            item["multi_stft_mse"] = float((source_stft - predicted_stft).square().mean())
            item["sound_distance"] = item["log_mel_mse"] + item["multi_stft_mse"]
    (training / "comparisons.json").write_text(
        json.dumps(comparison_manifest, indent=2) + "\n", encoding="utf-8"
    )

    import matplotlib.pyplot as plt

    preference_figure, preference_axis = plt.subplots(figsize=(8, 4.5))
    rounds = [item["round"] for item in preference_history]
    preference_axis.plot(
        rounds,
        [item["mean_best_score"] for item in preference_history],
        marker="o",
        label="preferred candidate",
    )
    preference_axis.plot(
        rounds,
        [item["mean_worst_score"] for item in preference_history],
        marker="o",
        label="rejected candidate",
    )
    preference_axis.set_xticks(rounds)
    preference_axis.set_xlabel("preference round")
    preference_axis.set_ylabel("rendered sound distance")
    preference_axis.set_title("Real-render preference separation")
    preference_axis.legend()
    preference_figure.tight_layout()
    preference_figure.savefig(training / "preference_distance.png", dpi=160)
    plt.close(preference_figure)

    figure, axis = plt.subplots(figsize=(8, 4.5))
    train_scores = [
        item["sound_distance"] for item in comparison_manifest if item["split"] != "test"
    ]
    test_scores = [
        item["sound_distance"] for item in comparison_manifest if item["split"] == "test"
    ]
    axis.boxplot((train_scores, test_scores), tick_labels=("train", "test"))
    axis.set_ylabel("log-mel + multi-STFT MSE")
    axis.set_title("Real Vital-rendered reconstruction distance")
    figure.tight_layout()
    figure.savefig(training / "comparison_sound_distance.png", dpi=160)
    plt.close(figure)

    report_path = training / "training_report.json"
    report = json.loads(report_path.read_text())
    report["preference_rounds"] = args.rounds
    report["preference_checkpoint"] = f"checkpoints/preference_round_{args.rounds}.safetensors"
    report["plots"] = [
        *report["plots"],
        "preference_distance.png",
        "comparison_sound_distance.png",
    ]
    report["mean_train_rendered_sound_distance"] = float(np.mean(train_scores))
    report["mean_test_rendered_sound_distance"] = float(np.mean(test_scores))
    report["gpu_uuid"] = A6000_UUID
    report["gpu"] = gpu.model_dump()
    report["status"] = "complete"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
