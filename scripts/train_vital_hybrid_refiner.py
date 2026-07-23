#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from synth import create_output_directory, pin_a6000, repository_relative_output_path


GENERATOR = "train_vital_hybrid_refiner_v2"
SEED = 20_260_723


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stream-dir", required=True, type=Path)
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--interventions-dir", required=True, type=Path)
    parser.add_argument("--reference-training-dir", required=True, type=Path)
    parser.add_argument("--passes-per-shard", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=96)
    parser.add_argument("--prebuffer-shards", type=int, default=4)
    parser.add_argument("--early-stopping-evaluations", type=int, default=12)
    parser.add_argument("--resume-checkpoint", type=Path)
    return parser.parse_args()


def main() -> None:
    gpu = pin_a6000()
    import torch
    import torch.nn.functional as functional
    from safetensors.torch import load_file, save_file
    from torch import Tensor

    from synth.models.vital_refiner import VitalHybridRefiner, VitalRefinerConfig

    args = parse_args()
    if args.batch_size < 8:
        raise ValueError("batch-size must be at least eight")
    repo_root = Path(__file__).resolve().parent.parent
    stream = args.stream_dir.resolve(strict=True)
    dataset = args.dataset_dir.resolve(strict=True)
    interventions = args.interventions_dir.resolve(strict=True)
    reference = args.reference_training_dir.resolve(strict=True)
    device = torch.device("cuda:0")
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    originals = tuple(
        row for row in rows if row["split"] != "test" and not bool(row["is_augmented"])
    )
    original_by_digest = {str(row["preset_sha256"]): row for row in originals}
    template_digests = tuple(
        dict.fromkeys(
            str(row["template_preset_sha256"])
            for row in rows
            if row["split"] != "test"
        )
    )
    template_rows = tuple(original_by_digest[digest] for digest in template_digests)
    preprocessing = json.loads((reference / "preprocessing.json").read_text())
    feature_mean = float(preprocessing["feature_mean"])
    feature_std = float(preprocessing["feature_standard_deviation"])
    projection = np.load(reference / "audio_projection.npz")
    summary_mean = torch.from_numpy(projection["summary_mean"].astype(np.float32))
    summary_scale = torch.from_numpy(projection["summary_scale"].astype(np.float32))

    print(f"Loading {len(template_rows):,} parent states", flush=True)
    parent_features = torch.stack(
        tuple(
            load_file(
                dataset
                / "features"
                / "log_mel"
                / f"{row['preset_sha256']}.safetensors"
            )["feature"]
            for row in template_rows
        )
    ).half()
    intervention_tensors = load_file(interventions / "interventions.safetensors")
    all_controls = intervention_tensors["controls"].float()
    all_masks = intervention_tensors["masks"].bool()
    state_rows = tuple(
        json.loads(line)
        for line in (interventions / "states.jsonl").read_text().splitlines()
    )
    state_index = {
        str(row["preset_sha256"]): index for index, row in enumerate(state_rows)
    }
    parent_state_indexes = torch.tensor(
        [state_index[digest] for digest in template_digests], dtype=torch.long
    )
    parent_controls = all_controls[parent_state_indexes].half()
    parent_masks = all_masks[parent_state_indexes]

    edge_current = intervention_tensors["current"].long()
    edge_next = intervention_tensors["next"].long()
    development_edges = torch.tensor(
        [
            index
            for index, current_index in enumerate(edge_current.tolist())
            if state_rows[current_index]["split"] == "development"
        ][:512],
        dtype=torch.long,
    )
    if len(development_edges) < 128:
        raise ValueError("At least 128 development transitions are required")
    validation_current_indexes = edge_current[development_edges]
    validation_next_indexes = edge_next[development_edges]
    validation_current_features = torch.stack(
        tuple(
            load_file(repo_root / state_rows[index]["feature_file"])["feature"]
            for index in validation_current_indexes.tolist()
        )
    ).half()
    validation_next_features = torch.stack(
        tuple(
            load_file(repo_root / state_rows[index]["feature_file"])["feature"]
            for index in validation_next_indexes.tolist()
        )
    ).half()
    validation_controls = all_controls[validation_current_indexes].half()
    validation_masks = all_masks[validation_current_indexes]
    validation_deltas = (
        all_controls[validation_next_indexes] - all_controls[validation_current_indexes]
    ).half()
    validation_action_masks = validation_deltas.abs() > 1e-6

    config = VitalRefinerConfig(control_count=all_controls.shape[1])
    model = VitalHybridRefiner(config).to(device)
    if args.resume_checkpoint is not None:
        model.load_state_dict(
            load_file(args.resume_checkpoint.resolve(strict=True), device="cuda:0")
        )
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.02)
    scaler = torch.cuda.amp.GradScaler()
    output = create_output_directory(repo_root, GENERATOR)
    checkpoints = output / "checkpoints"
    checkpoints.mkdir()
    (output / "model_config.json").write_text(
        config.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    summary_mean_device = summary_mean.to(device)
    summary_scale_device = summary_scale.to(device)

    def prepare(feature: Tensor) -> Tensor:
        return ((feature.float() - feature_mean) / feature_std).to(
            device, non_blocking=True
        )

    def summary(standardized: Tensor) -> Tensor:
        raw = standardized * feature_std + feature_mean
        return (
            model.refiner.audio_summary(raw) - summary_mean_device
        ) / summary_scale_device

    def goal_indexes(group_ids: Tensor, seed: int) -> Tensor:
        generator = torch.Generator().manual_seed(seed)
        permutation = torch.randperm(len(group_ids), generator=generator)
        result = torch.empty_like(group_ids)
        for group in torch.unique(group_ids):
            members = torch.nonzero(group_ids == group).flatten()
            candidate = next(
                (
                    int(index)
                    for index in permutation
                    if group_ids[int(index)] != group
                ),
                int(permutation[0]),
            )
            result[members] = candidate
        return result

    def losses(
        current_feature: Tensor,
        next_feature: Tensor,
        controls: Tensor,
        masks: Tensor,
        delta: Tensor,
        changed: Tensor,
        group_ids: Tensor,
        seed: int,
    ) -> dict[str, Tensor]:
        goal_index = goal_indexes(group_ids.cpu(), seed).to(device)
        goal_feature = next_feature[goal_index]
        current_latent, current_summary_prediction = model.refiner.encode_audio(
            current_feature
        )
        next_latent, next_summary_prediction = model.refiner.encode_audio(next_feature)
        goal_latent, goal_summary_prediction = model.refiner.encode_audio(goal_feature)
        current_summary = summary(current_feature)
        next_summary = summary(next_feature)
        goal_summary = summary(goal_feature)
        preset_latent = model.refiner.encode_preset(controls, masks)
        proposal = model.refiner.propose(next_latent, current_latent, preset_latent)
        transition = model.refiner.predict_transition(
            current_latent, preset_latent, delta, changed
        )
        value = model.score_action(
            goal_latent, current_latent, preset_latent, delta, changed
        )
        next_distance = (next_summary - goal_summary).square().mean(dim=-1)
        current_distance = (current_summary - goal_summary).square().mean(dim=-1)
        improvement = current_distance - next_distance

        summary_loss = (
            functional.mse_loss(current_summary_prediction, current_summary)
            + functional.mse_loss(next_summary_prediction, next_summary)
            + functional.mse_loss(goal_summary_prediction, goal_summary)
        )
        world_loss = functional.mse_loss(
            transition.next_latents,
            next_latent.detach().unsqueeze(1).expand_as(transition.next_latents),
        )
        relevance_loss = functional.binary_cross_entropy_with_logits(
            proposal.relevance_logits, changed.float()
        )
        changed_float = changed.float()
        delta_loss = (
            (proposal.delta_mean - delta).square() * changed_float
        ).sum() / changed_float.sum().clamp_min(1.0)
        value_loss = functional.mse_loss(
            value.distance, next_distance.unsqueeze(1).expand_as(value.distance)
        ) + functional.mse_loss(
            value.improvement,
            improvement.unsqueeze(1).expand_as(value.improvement),
        )
        ranking_terms = []
        predicted_distance = value.distance.mean(dim=1)
        for group in torch.unique(group_ids):
            members = torch.nonzero(group_ids == group).flatten().to(device)
            if len(members) < 2:
                continue
            left = members[:-1]
            right = members[1:]
            sign = torch.sign(next_distance[right] - next_distance[left])
            nonzero = sign != 0
            if nonzero.any():
                ranking_terms.append(
                    functional.softplus(
                        -sign[nonzero]
                        * (predicted_distance[right][nonzero] - predicted_distance[left][nonzero])
                    ).mean()
                )
        ranking_loss = (
            torch.stack(ranking_terms).mean()
            if ranking_terms
            else torch.zeros((), device=device)
        )
        ensemble_uncertainty = value.distance.var(dim=1).mean()
        total = (
            summary_loss
            + world_loss
            + 0.35 * delta_loss
            + 0.15 * relevance_loss
            + 0.5 * value_loss
            + 0.2 * ranking_loss
        )
        return {
            "total": total,
            "summary": summary_loss,
            "world": world_loss,
            "delta": delta_loss,
            "relevance": relevance_loss,
            "value": value_loss,
            "ranking": ranking_loss,
            "uncertainty": ensemble_uncertainty,
        }

    @torch.no_grad()
    def evaluate(step_seed: int) -> dict[str, float]:
        model.eval()
        totals: dict[str, float] = {}
        batch_size = 64
        for start in range(0, len(development_edges), batch_size):
            indexes = slice(start, start + batch_size)
            values = losses(
                prepare(validation_current_features[indexes]),
                prepare(validation_next_features[indexes]),
                validation_controls[indexes].float().to(device),
                validation_masks[indexes].to(device),
                validation_deltas[indexes].float().to(device),
                validation_action_masks[indexes].to(device),
                torch.arange(start, min(start + batch_size, len(development_edges))) // 4,
                step_seed + start,
            )
            count = len(validation_current_features[indexes])
            for name, value in values.items():
                totals[name] = totals.get(name, 0.0) + float(value) * count
        return {name: value / len(development_edges) for name, value in totals.items()}

    def save_model(destination: Path) -> None:
        temporary = destination.with_suffix(".partial.safetensors")
        save_file(
            {
                name: value.detach().cpu().contiguous()
                for name, value in model.state_dict().items()
            },
            temporary,
        )
        temporary.replace(destination)

    def producer_state() -> dict[str, object]:
        return json.loads((stream / "stream_state.json").read_text())

    while True:
        available = len(tuple((stream / "shards").glob("shard_*.safetensors")))
        state = producer_state()
        if available >= args.prebuffer_shards or state["status"] == "complete":
            break
        print(f"waiting for stream prebuffer: {available}/{args.prebuffer_shards}", flush=True)
        time.sleep(20)

    planned_examples = len(template_rows) * 40
    planned_steps = (
        math.ceil(planned_examples / args.batch_size) * args.passes_per_shard
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=planned_steps, eta_min=3e-6
    )
    metrics: list[dict[str, float]] = []
    best_validation = math.inf
    evaluations_without_improvement = 0
    processed = 0
    optimizer_steps = 0
    available_indexes = tuple(
        sorted(
            int(path.stem.removeprefix("shard_"))
            for path in (stream / "shards").glob("shard_*.safetensors")
        )
    )
    if not available_indexes:
        raise RuntimeError("No stream shard is available to start training")
    next_chunk = available_indexes[0]
    processed = next_chunk * 512
    started = time.monotonic()
    stopped_early = False

    while True:
        shard_path = stream / "shards" / f"shard_{next_chunk:05d}.safetensors"
        metadata_path = stream / "metadata" / f"shard_{next_chunk:05d}.jsonl"
        if not shard_path.is_file():
            state = producer_state()
            if state["status"] == "complete":
                break
            time.sleep(10)
            continue
        if stopped_early:
            shard_path.unlink()
            metadata_path.unlink()
            next_chunk += 1
            continue
        shard = load_file(shard_path)
        child_features = shard["feature"]
        child_controls = shard["controls"]
        child_masks = shard["mask"]
        template_indexes = shard["template_index"].long()
        current_features = parent_features[template_indexes]
        current_controls = parent_controls[template_indexes]
        current_masks = parent_masks[template_indexes]
        deltas = child_controls.float() - current_controls.float()
        changed = child_masks & current_masks & (deltas.abs() > 1e-6)

        model.train()
        shard_accumulated: dict[str, float] = {}
        shard_seen = 0
        for pass_index in range(args.passes_per_shard):
            order = torch.randperm(
                len(child_features),
                generator=torch.Generator().manual_seed(
                    SEED + next_chunk * args.passes_per_shard + pass_index
                ),
            )
            for start in range(0, len(order), args.batch_size):
                indexes = order[start : start + args.batch_size]
                with torch.cuda.amp.autocast(dtype=torch.float16):
                    values = losses(
                        prepare(current_features[indexes]),
                        prepare(child_features[indexes]),
                        current_controls[indexes].float().to(device),
                        current_masks[indexes].to(device),
                        deltas[indexes].float().to(device),
                        changed[indexes].to(device),
                        template_indexes[indexes],
                        SEED + optimizer_steps,
                    )
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(values["total"]).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                count = len(indexes)
                for name, value in values.items():
                    shard_accumulated[name] = (
                        shard_accumulated.get(name, 0.0) + float(value) * count
                    )
                shard_seen += count
                optimizer_steps += 1
        processed += len(child_features)
        shard_path.unlink()
        metadata_path.unlink()
        next_chunk += 1

        if next_chunk % 10 == 0:
            validation = evaluate(SEED)
            metric = {
                "chunk": float(next_chunk),
                "processed_examples": float(processed),
                "optimizer_steps": float(optimizer_steps),
                **{
                    f"train_{name}": value / shard_seen
                    for name, value in shard_accumulated.items()
                },
                **{f"validation_{name}": value for name, value in validation.items()},
                "learning_rate": scheduler.get_last_lr()[0],
            }
            metrics.append(metric)
            if validation["total"] < best_validation:
                best_validation = validation["total"]
                evaluations_without_improvement = 0
                save_model(checkpoints / "best_model.safetensors")
            else:
                evaluations_without_improvement += 1
            save_model(checkpoints / "latest_model.safetensors")
            print(
                f"chunk={next_chunk} examples={processed} "
                f"train={metric['train_total']:.4f} "
                f"validation={validation['total']:.4f}",
                flush=True,
            )
            if evaluations_without_improvement >= args.early_stopping_evaluations:
                stopped_early = True
                print(
                    "early stopping activated; draining future shards without training",
                    flush=True,
                )

    if not metrics:
        raise RuntimeError("No streaming evaluations were completed")
    if not (checkpoints / "best_model.safetensors").is_file():
        save_model(checkpoints / "best_model.safetensors")
    with (output / "metrics.csv").open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=metrics[0].keys())
        writer.writeheader()
        writer.writerows(metrics)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    examples = [row["processed_examples"] for row in metrics]
    for axis, name, title in (
        (axes[0], "total", "Held-out total"),
        (axes[1], "value", "Goal value"),
        (axes[2], "ranking", "Candidate ranking"),
    ):
        axis.plot(examples, [row[f"validation_{name}"] for row in metrics])
        axis.set_title(title)
        axis.set_xlabel("unique streamed transitions")
        axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output / "hybrid_training.png", dpi=160)
    plt.close(figure)
    report = {
        "schema_version": 2,
        "status": "complete",
        "generator": GENERATOR,
        "gpu": gpu.model_dump(),
        "stream": str(repository_relative_output_path(repo_root, stream)),
        "processed_stream_examples": processed,
        "optimizer_steps": optimizer_steps,
        "passes_per_shard": args.passes_per_shard,
        "best_validation_total": best_validation,
        "stopped_early": stopped_early,
        "training_seconds": time.monotonic() - started,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "feature_mean": feature_mean,
        "feature_standard_deviation": feature_std,
        "plots": ["hybrid_training.png"],
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
