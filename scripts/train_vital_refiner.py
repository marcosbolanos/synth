#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file, save_file

from synth import create_output_directory, pin_a6000, repository_relative_output_path


GENERATOR = "train_vital_refiner_v1"
SEED = 20_260_723


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interventions-dir", required=True, type=Path)
    parser.add_argument("--reference-training-dir", required=True, type=Path)
    parser.add_argument("--training-seconds", type=float, default=3600.0)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--evaluation-interval", type=int, default=250)
    parser.add_argument("--maximum-steps", type=int, default=100_000)
    return parser.parse_args()


def plot_metrics(rows: list[dict[str, float]], destination: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    steps = [row["step"] for row in rows]
    for split in ("train", "test"):
        selected = [row for row in rows if row["split"] == split]
        selected_steps = [row["step"] for row in selected]
        axes[0].plot(selected_steps, [row["total"] for row in selected], label=split)
        axes[1].plot(selected_steps, [row["world"] for row in selected], label=split)
        axes[2].plot(selected_steps, [row["delta"] for row in selected], label=split)
    axes[0].set_title("Total objective")
    axes[1].set_title("Latent transition")
    axes[2].set_title("Sparse edit delta")
    for axis in axes:
        axis.set_xlabel("optimizer step")
        axis.set_yscale("log")
        axis.grid(alpha=0.2)
        axis.legend()
    if not steps:
        raise ValueError("Cannot plot an empty metric history")
    figure.tight_layout()
    figure.savefig(destination, dpi=160)
    plt.close(figure)


def main() -> None:
    gpu = pin_a6000()
    import torch
    import torch.nn.functional as functional
    from torch import Tensor

    from synth.models.vital_refiner import VitalRefiner, VitalRefinerConfig

    args = parse_args()
    if args.training_seconds <= 0:
        raise ValueError("training-seconds must be positive")
    repo_root = Path(__file__).resolve().parent.parent
    interventions_dir = args.interventions_dir.resolve(strict=True)
    reference_dir = args.reference_training_dir.resolve(strict=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = torch.device("cuda:0")

    states = tuple(
        json.loads(line)
        for line in (interventions_dir / "states.jsonl").read_text(encoding="utf-8").splitlines()
    )
    tensors = load_file(interventions_dir / "interventions.safetensors")
    preprocessing = json.loads(
        (reference_dir / "preprocessing.json").read_text(encoding="utf-8")
    )
    projection = np.load(reference_dir / "audio_projection.npz")
    summary_mean = torch.from_numpy(projection["summary_mean"].astype(np.float32))
    summary_scale = torch.from_numpy(projection["summary_scale"].astype(np.float32))
    feature_mean = float(preprocessing["feature_mean"])
    feature_std = float(preprocessing["feature_standard_deviation"])

    print(f"Loading {len(states):,} log-mel states into host memory", flush=True)
    features = torch.stack(
        tuple(load_file(repo_root / state["feature_file"])["feature"] for state in states)
    ).half()
    controls = tensors["controls"].float()
    masks = tensors["masks"].bool()
    current = tensors["current"].long()
    following = tensors["next"].long()
    action_delta = tensors["action_delta"].float()
    action_mask = tensors["action_mask"].bool()
    split_by_state = tuple(str(state["split"]) for state in states)
    train_edges = torch.tensor(
        [
            index
            for index, state_index in enumerate(current.tolist())
            if split_by_state[state_index] == "train"
        ],
        dtype=torch.long,
    )
    test_edges = torch.tensor(
        [
            index
            for index, state_index in enumerate(current.tolist())
            if split_by_state[state_index] == "development"
        ],
        dtype=torch.long,
    )
    if len(test_edges) == 0:
        raise ValueError("The refiner requires held-out transitions")

    config = VitalRefinerConfig(control_count=controls.shape[1])
    model = VitalRefiner(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.02)
    scaler = torch.cuda.amp.GradScaler()
    output = create_output_directory(repo_root, GENERATOR)
    (output / "model_config.json").write_text(
        config.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    generator = torch.Generator().manual_seed(SEED)

    def batch(edge_indexes: Tensor) -> tuple[Tensor, ...]:
        edge_indexes = edge_indexes.cpu()
        current_indexes = current[edge_indexes]
        next_indexes = following[edge_indexes]
        current_feature = (
            (features[current_indexes].float() - feature_mean) / feature_std
        ).to(device, non_blocking=True)
        next_feature = (
            (features[next_indexes].float() - feature_mean) / feature_std
        ).to(device, non_blocking=True)
        return (
            current_feature,
            next_feature,
            controls[current_indexes].to(device, non_blocking=True),
            masks[current_indexes].to(device, non_blocking=True),
            action_delta[edge_indexes].to(device, non_blocking=True),
            action_mask[edge_indexes].to(device, non_blocking=True),
        )

    summary_mean_device = summary_mean.to(device)
    summary_scale_device = summary_scale.to(device)

    def losses(items: tuple[Tensor, ...]) -> dict[str, Tensor]:
        current_feature, next_feature, current_controls, current_mask, delta, changed = items
        current_latent, current_summary_prediction = model.encode_audio(current_feature)
        next_latent, next_summary_prediction = model.encode_audio(next_feature)
        current_summary = (
            model.audio_summary(current_feature * feature_std + feature_mean)
            - summary_mean_device
        ) / summary_scale_device
        next_summary = (
            model.audio_summary(next_feature * feature_std + feature_mean)
            - summary_mean_device
        ) / summary_scale_device
        preset_latent = model.encode_preset(current_controls, current_mask)
        proposal = model.propose(next_latent, current_latent, preset_latent)
        transition = model.predict_transition(current_latent, preset_latent, delta, changed)
        summary_loss = functional.mse_loss(
            current_summary_prediction, current_summary
        ) + functional.mse_loss(next_summary_prediction, next_summary)
        world_loss = functional.mse_loss(
            transition.next_latents,
            next_latent.detach().unsqueeze(1).expand_as(transition.next_latents),
        )
        relevance_loss = functional.binary_cross_entropy_with_logits(
            proposal.relevance_logits, changed.float()
        )
        changed_float = changed.float()
        changed_count = changed_float.sum().clamp_min(1.0)
        delta_loss = (
            (proposal.delta_mean - delta).square() * changed_float
        ).sum() / changed_count
        scale = proposal.delta_log_scale.exp()
        nll_loss = (
            (
                (proposal.delta_mean - delta).square() / (scale.square() + 1e-6)
                + 2.0 * proposal.delta_log_scale
            )
            * changed_float
        ).sum() / changed_count
        true_distance = (next_summary - current_summary).square().mean(dim=-1, keepdim=True)
        distance_loss = functional.mse_loss(
            transition.distances, true_distance.expand_as(transition.distances)
        )
        latent_values = torch.cat((current_latent, next_latent), dim=0)
        latent_loss = latent_values.mean(dim=0).square().mean() + (
            latent_values.std(dim=0).mean() - 1.0
        ).square()
        total = (
            summary_loss
            + world_loss
            + 0.35 * delta_loss
            + 0.05 * nll_loss
            + 0.15 * relevance_loss
            + 0.1 * distance_loss
            + 0.02 * latent_loss
        )
        return {
            "total": total,
            "summary": summary_loss,
            "world": world_loss,
            "delta": delta_loss,
            "relevance": relevance_loss,
            "distance": distance_loss,
        }

    @torch.no_grad()
    def evaluate(step: int, edge_pool: Tensor, split: str, batches: int) -> dict[str, float]:
        model.eval()
        accumulated: dict[str, float] = {}
        for offset in range(batches):
            start = (offset * args.batch_size) % len(edge_pool)
            indexes = edge_pool[start : start + args.batch_size]
            if len(indexes) < args.batch_size:
                indexes = torch.cat((indexes, edge_pool[: args.batch_size - len(indexes)]))
            with torch.cuda.amp.autocast(dtype=torch.float16):
                values = losses(batch(indexes))
            for name, value in values.items():
                accumulated[name] = accumulated.get(name, 0.0) + float(value)
        return {
            "step": float(step),
            "split": split,
            **{name: value / batches for name, value in accumulated.items()},
        }

    started = time.monotonic()
    deadline = started + args.training_seconds
    metrics: list[dict[str, float]] = []
    best_test = math.inf
    best_state: dict[str, Tensor] | None = None
    step = 0
    order = train_edges[torch.randperm(len(train_edges), generator=generator)]
    cursor = 0
    while time.monotonic() < deadline and step < args.maximum_steps:
        if cursor + args.batch_size > len(order):
            order = train_edges[torch.randperm(len(train_edges), generator=generator)]
            cursor = 0
        indexes = order[cursor : cursor + args.batch_size]
        cursor += args.batch_size
        progress = min((time.monotonic() - started) / args.training_seconds, 1.0)
        learning_rate = 3e-4 * (0.05 + 0.95 * 0.5 * (1.0 + math.cos(math.pi * progress)))
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        model.train()
        with torch.cuda.amp.autocast(dtype=torch.float16):
            values = losses(batch(indexes))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(values["total"]).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        step += 1
        if step == 1 or step % args.evaluation_interval == 0:
            train_metric = evaluate(step, train_edges, "train", 8)
            test_metric = evaluate(step, test_edges, "test", 8)
            metrics.extend((train_metric, test_metric))
            elapsed = time.monotonic() - started
            print(
                f"step={step} elapsed={elapsed:.0f}s "
                f"train={train_metric['total']:.4f} test={test_metric['total']:.4f}",
                flush=True,
            )
            if test_metric["total"] < best_test:
                best_test = test_metric["total"]
                best_state = {
                    name: value.detach().cpu().contiguous()
                    for name, value in model.state_dict().items()
                }
    if best_state is None:
        raise RuntimeError("Training did not produce a checkpoint")
    save_file(best_state, output / "best_model.safetensors")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8"
    )
    plot_metrics(metrics, output / "loss_curves.png")
    report = {
        "schema_version": 1,
        "status": "complete",
        "generator": GENERATOR,
        "gpu": gpu.model_dump(),
        "interventions": str(repository_relative_output_path(repo_root, interventions_dir)),
        "reference_training": str(repository_relative_output_path(repo_root, reference_dir)),
        "parameter_count": parameter_count,
        "training_seconds": time.monotonic() - started,
        "steps": step,
        "train_transition_count": len(train_edges),
        "test_transition_count": len(test_edges),
        "best_test_total": best_test,
        "feature_mean": feature_mean,
        "feature_standard_deviation": feature_std,
        "plots": ["loss_curves.png"],
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
