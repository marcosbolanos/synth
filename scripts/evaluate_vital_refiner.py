#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors.torch import load_file

from synth import (
    RTX_2080_TI_NAME,
    STREAM_FEATURE_GPU_UUID,
    VitalRuntime,
    create_output_directory,
    pin_gpu,
    publish_gallery_run,
    repository_relative_output_path,
)
from synth.audio_features import AudioFeatureKind, extract_feature, read_render
from synth.preset_representation import ControlStatistics


GENERATOR = "evaluate_vital_refiner_v1"
SEED = 20_260_723


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--interventions-dir", required=True, type=Path)
    parser.add_argument("--training-dir", required=True, type=Path)
    parser.add_argument("--reference-training-dir", required=True, type=Path)
    parser.add_argument("--old-training-dir", required=True, type=Path)
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--renders-per-round", type=int, default=12)
    parser.add_argument("--proposal-pool", type=int, default=96)
    return parser.parse_args()


def select_demo_rows(rows: tuple[dict[str, object], ...]) -> tuple[dict[str, object], ...]:
    selected = []
    for split in ("train", "test"):
        for family in ("dubstep", "midtempo"):
            candidates = tuple(
                row
                for row in rows
                if row["split"] == split
                and row["evaluation_family"] == family
                and not bool(row["is_augmented"])
            )
            if not candidates:
                raise ValueError(f"No {split} {family} evaluation target")
            selected.append(candidates[SEED % len(candidates)])
    return tuple(selected)


def main() -> None:
    pin_gpu(STREAM_FEATURE_GPU_UUID, RTX_2080_TI_NAME)
    args = parse_args()
    if args.rounds <= 0 or args.renders_per_round <= 0:
        raise ValueError("Search budget must be positive")
    repo_root = Path(__file__).resolve().parent.parent
    dataset = args.dataset_dir.resolve(strict=True)
    interventions = args.interventions_dir.resolve(strict=True)
    training = args.training_dir.resolve(strict=True)
    reference = args.reference_training_dir.resolve(strict=True)
    old_training = args.old_training_dir.resolve(strict=True)
    plugin = args.vital_vst3.resolve(strict=True)

    from synth.models.vital_preset_model import VitalPreset
    from synth.models.vital_refiner import VitalRefiner, VitalRefinerConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(SEED)
    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    state_rows = tuple(
        json.loads(line)
        for line in (interventions / "states.jsonl").read_text(encoding="utf-8").splitlines()
    )
    tensors = load_file(interventions / "interventions.safetensors")
    controls = tensors["controls"].float()
    masks = tensors["masks"].bool()
    empirical_actions = tensors["action_delta"].float().numpy()
    config = VitalRefinerConfig.model_validate_json(
        (training / "model_config.json").read_text(encoding="utf-8")
    )
    model = VitalRefiner(config).to(device)
    model.load_state_dict(load_file(training / "best_model.safetensors", device=str(device)))
    model.eval()
    statistics = ControlStatistics.model_validate_json(
        (reference / "control_statistics.json").read_text(encoding="utf-8")
    )
    preprocessing = json.loads((reference / "preprocessing.json").read_text())
    feature_mean = float(preprocessing["feature_mean"])
    feature_std = float(preprocessing["feature_standard_deviation"])
    index_by_digest = {
        str(row["preset_sha256"]): index for index, row in enumerate(state_rows)
    }
    original_training_indexes = tuple(
        index
        for index, row in enumerate(state_rows)
        if row["split"] != "test" and not bool(row["is_augmented"])
    )
    target_rows = select_demo_rows(rows)
    output = create_output_directory(repo_root, GENERATOR)
    runtime = VitalRuntime.from_repo_root(repo_root)
    worker = Path(__file__).resolve().parent / "render_vital_preset_worker.py"

    def portable(path: Path) -> str:
        return str(repository_relative_output_path(repo_root, path))

    def standardized_feature(path: Path) -> torch.Tensor:
        feature = extract_feature(read_render(path).to(device), AudioFeatureKind.LOG_MEL)
        return (feature - feature_mean) / feature_std

    @torch.inference_mode()
    def latent(feature: torch.Tensor) -> torch.Tensor:
        value, _ = model.encode_audio(feature.unsqueeze(0))
        return value

    print("Encoding retrieval bank", flush=True)
    retrieval_latents = []
    with torch.inference_mode():
        for start in range(0, len(original_training_indexes), 32):
            indexes = original_training_indexes[start : start + 32]
            batch = torch.stack(
                tuple(
                    (
                        load_file(repo_root / state_rows[index]["feature_file"])["feature"].float()
                        - feature_mean
                    )
                    / feature_std
                    for index in indexes
                )
            ).to(device)
            encoded, _ = model.encode_audio(batch)
            retrieval_latents.append(encoded.cpu())
    retrieval_bank = torch.cat(retrieval_latents)

    def render_candidates(requests: tuple[tuple[Path, Path], ...]) -> None:
        def render_one(request: tuple[Path, Path]) -> None:
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

        with ThreadPoolExecutor(max_workers=min(24, len(requests))) as executor:
            futures = tuple(executor.submit(render_one, request) for request in requests)
            for future in as_completed(futures):
                future.result()

    def run_search(
        method: str,
        target_feature: torch.Tensor,
        target_latent: torch.Tensor,
        seed_preset: VitalPreset,
        seed_controls: np.ndarray,
        seed_audio: Path,
        root: Path,
    ) -> tuple[VitalPreset, Path, list[dict[str, float]]]:
        current_preset = seed_preset
        current_controls = seed_controls.copy()
        current_audio = seed_audio
        working_audio = root / f".{method}_working.wav"
        current_feature = standardized_feature(current_audio)
        current_distance = float((current_feature - target_feature).square().mean())
        curve = [{"renders": 0.0, "distance": current_distance}]
        scale = 1.0
        for round_index in range(args.rounds):
            preset_tensor = torch.from_numpy(current_controls).float().unsqueeze(0).to(device)
            current_mask = torch.from_numpy(
                statistics.encode(current_preset)[1]
            ).unsqueeze(0).to(device)
            current_latent = latent(current_feature)
            preset_latent = model.encode_preset(preset_tensor, current_mask)
            candidate_actions: list[np.ndarray] = []
            if method == "refiner":
                proposal = model.propose(target_latent, current_latent, preset_latent)
                relevance = proposal.relevance_logits.squeeze(0)
                mean = proposal.delta_mean.squeeze(0)
                action_scale = proposal.delta_log_scale.squeeze(0).exp()
                world_scores = []
                for _ in range(args.proposal_pool):
                    changed_count = int(rng.integers(1, 9))
                    noise = torch.from_numpy(rng.gumbel(size=config.control_count)).to(
                        device=device, dtype=relevance.dtype
                    )
                    changed = torch.topk(relevance + noise, changed_count).indices
                    action = torch.zeros(config.control_count, device=device)
                    action[changed] = (
                        mean[changed]
                        + action_scale[changed]
                        * torch.randn(changed_count, device=device)
                    ).clamp(-2.5, 2.5)
                    action_mask = action.abs() > 1e-6
                    transition = model.predict_transition(
                        current_latent, preset_latent, action.unsqueeze(0), action_mask.unsqueeze(0)
                    )
                    distance = (
                        transition.next_latents - target_latent.unsqueeze(1)
                    ).square().mean(dim=-1)
                    score = float(distance.mean() + 0.05 * distance.std())
                    world_scores.append((score, action.detach().cpu().numpy()))
                world_scores.sort(key=lambda item: item[0])
                candidate_actions = [
                    item[1] for item in world_scores[: args.renders_per_round]
                ]
            elif method == "cem":
                sampled = rng.choice(
                    len(empirical_actions), size=args.renders_per_round, replace=False
                )
                candidate_actions = [
                    empirical_actions[index] * scale * float(rng.uniform(0.6, 1.4))
                    for index in sampled
                ]
            else:
                raise ValueError(f"Unknown search method {method}")

            requests = []
            candidates = []
            for candidate_index, action in enumerate(candidate_actions):
                values = np.clip(current_controls + action, -6.0, 6.0).astype(np.float32)
                preset = statistics.apply(current_preset, values)
                candidate_root = root / method / f"round_{round_index + 1}"
                candidate_root.mkdir(parents=True, exist_ok=True)
                preset_path = candidate_root / f"candidate_{candidate_index:03d}.vital"
                audio_path = candidate_root / f"candidate_{candidate_index:03d}.wav"
                preset.to_file(preset_path)
                requests.append((preset_path, audio_path))
                candidates.append((preset, values, audio_path))
            render_candidates(tuple(requests))
            scored = []
            for preset, values, audio_path in candidates:
                feature = standardized_feature(audio_path)
                distance = float((feature - target_feature).square().mean())
                scored.append((distance, preset, values, audio_path, feature))
            best = min(scored, key=lambda item: item[0])
            if best[0] < current_distance:
                current_distance, current_preset, current_controls, best_audio, current_feature = best
                shutil.copyfile(best_audio, working_audio)
                current_audio = working_audio
            shutil.rmtree(candidate_root)
            curve.append(
                {
                    "renders": float((round_index + 1) * args.renders_per_round),
                    "distance": current_distance,
                }
            )
            scale *= 0.65
            print(
                f"{root.name} {method} round={round_index + 1} distance={current_distance:.5f}",
                flush=True,
            )
        final_audio = root / f"{method}_final.wav"
        final_preset = root / f"{method}_final.vital"
        shutil.copyfile(current_audio, final_audio)
        current_preset.to_file(final_preset)
        if working_audio.is_file():
            working_audio.unlink()
        return current_preset, final_audio, curve

    old_items = {
        str(item["preset_sha256"]): item
        for item in json.loads((old_training / "comparisons.json").read_text())
    }

    def old_reference(value: str) -> Path:
        path = Path(value)
        if path.is_absolute():
            return path
        if path.parts[:2] == ("data", "outputs"):
            return repo_root / path
        return old_training / path

    comparisons = []
    curves: dict[str, dict[str, list[dict[str, float]]]] = {}
    for target_number, row in enumerate(target_rows, start=1):
        digest = str(row["preset_sha256"])
        target_index = index_by_digest[digest]
        target_feature = (
            (
                load_file(repo_root / state_rows[target_index]["feature_file"])["feature"].float()
                - feature_mean
            )
            / feature_std
        ).to(device)
        target_audio_latent = latent(target_feature)
        distances = (retrieval_bank - target_audio_latent.cpu()).square().mean(dim=-1)
        candidate_order = torch.argsort(distances)
        seed_bank_index = next(
            int(value)
            for value in candidate_order
            if original_training_indexes[int(value)] != target_index
        )
        seed_index = original_training_indexes[seed_bank_index]
        seed_row = state_rows[seed_index]
        seed_preset = VitalPreset.from_file(repo_root / seed_row["preset_file"])
        seed_controls = controls[seed_index].numpy()
        item_root = output / "comparisons" / str(row["split"]) / digest
        item_root.mkdir(parents=True)
        target_audio = item_root / "target.wav"
        seed_audio = item_root / "seed.wav"
        shutil.copyfile(dataset / str(row["audio_file"]), target_audio)
        shutil.copyfile(repo_root / seed_row["audio_file"], seed_audio)
        _, cem_audio, cem_curve = run_search(
            "cem", target_feature, target_audio_latent, seed_preset, seed_controls, seed_audio, item_root
        )
        _, refiner_audio, refiner_curve = run_search(
            "refiner", target_feature, target_audio_latent, seed_preset, seed_controls, seed_audio, item_root
        )
        old_item = old_items.get(digest)
        old_audio = (
            old_reference(str(old_item["output_audio"]))
            if old_item is not None
            else seed_audio
        )
        curves[digest] = {"cem": cem_curve, "refiner": refiner_curve}
        comparisons.append(
            {
                "preset_sha256": digest,
                "preset_name": row["preset_name"],
                "pack": row["pack"],
                "family": row["evaluation_family"],
                "split": row["split"],
                "target_audio": portable(target_audio),
                "seed_audio": portable(seed_audio),
                "old_audio": portable(old_audio),
                "cem_audio": portable(cem_audio),
                "cem_preset": portable(item_root / "cem_final.vital"),
                "refiner_audio": portable(refiner_audio),
                "refiner_preset": portable(item_root / "refiner_final.vital"),
                "seed_distance": cem_curve[0]["distance"],
                "cem_distance": cem_curve[-1]["distance"],
                "refiner_distance": refiner_curve[-1]["distance"],
            }
        )
        print(f"completed target {target_number}/{len(target_rows)}", flush=True)

    curve_plot = output / "search_curves.png"
    figure, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True)
    for axis, item in zip(axes.ravel(), comparisons, strict=True):
        digest = item["preset_sha256"]
        for method, label in (("cem", "render-only CEM"), ("refiner", "learned refiner")):
            values = curves[digest][method]
            axis.plot(
                [row["renders"] for row in values],
                [row["distance"] for row in values],
                marker="o",
                label=label,
            )
        axis.set_title(f"{item['split']} · {item['family']} · {item['preset_name']}")
        axis.set_ylabel("log-mel distance")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    for axis in axes[-1]:
        axis.set_xlabel("real Vital renders")
    figure.tight_layout()
    figure.savefig(curve_plot, dpi=160)
    plt.close(figure)

    comparisons_path = output / "comparisons.json"
    comparisons_path.write_text(json.dumps(comparisons, indent=2) + "\n", encoding="utf-8")
    report_path = output / "refiner_report.json"
    report = {
        "schema_version": 1,
        "status": "complete",
        "generator": GENERATOR,
        "architecture": "shared audio encoder + sparse edit policy + 3-head latent world model",
        "seed_strategy": "nearest learned-audio-latent training preset",
        "rounds": args.rounds,
        "renders_per_round": args.renders_per_round,
        "real_render_budget_per_method": args.rounds * args.renders_per_round,
        "proposal_pool": args.proposal_pool,
        "comparison_count": len(comparisons),
        "mean_seed_distance": float(np.mean([item["seed_distance"] for item in comparisons])),
        "mean_cem_distance": float(np.mean([item["cem_distance"] for item in comparisons])),
        "mean_refiner_distance": float(np.mean([item["refiner_distance"] for item in comparisons])),
        "refiner_wins_over_cem": sum(
            item["refiner_distance"] < item["cem_distance"] for item in comparisons
        ),
        "comparisons": portable(comparisons_path),
        "plots": [portable(curve_plot), portable(training / "loss_curves.png")],
    }
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    sources = [
        (report_path, "json", "refiner evaluation report"),
        (comparisons_path, "json", "comparison index"),
        (curve_plot, "image", "bounded search curves"),
        (training / "loss_curves.png", "image", "training losses"),
    ]
    for item in comparisons:
        sources.extend(
            (
                (repo_root / item["target_audio"], "audio", "target audio"),
                (repo_root / item["seed_audio"], "audio", "retrieval seed audio"),
                (repo_root / item["old_audio"], "audio", "old one-shot output"),
                (repo_root / item["cem_audio"], "audio", "CEM output"),
                (repo_root / item["cem_preset"], "preset", "CEM preset"),
                (repo_root / item["refiner_audio"], "audio", "refiner output"),
                (repo_root / item["refiner_preset"], "preset", "refiner preset"),
            )
        )
    publish_gallery_run(
        repo_root,
        output,
        GENERATOR,
        "/vital-refiner",
        tuple(sources),
    )
    print(output)


if __name__ == "__main__":
    main()
