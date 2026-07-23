#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from synth import RTX_2080_TI_NAME, STREAM_FEATURE_GPU_UUID, VitalPreset, pin_gpu


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", required=True, type=Path)
    parser.add_argument("--control-statistics", required=True, type=Path)
    parser.add_argument("--audio-projection", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--metadata-output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gpu = pin_gpu(STREAM_FEATURE_GPU_UUID, RTX_2080_TI_NAME)
    import torch
    from safetensors.torch import save_file

    from synth.audio_features import AudioFeatureKind, extract_feature, read_render
    from synth.preset_representation import ControlStatistics

    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible feature GPU, found {torch.cuda.device_count()}")
    if torch.cuda.get_device_name(0) != RTX_2080_TI_NAME:
        raise RuntimeError(f"Unexpected visible feature GPU {torch.cuda.get_device_name(0)}")

    tasks = json.loads(args.tasks.resolve(strict=True).read_text(encoding="utf-8"))
    statistics = ControlStatistics.model_validate_json(
        args.control_statistics.resolve(strict=True).read_text(encoding="utf-8")
    )
    projection = np.load(args.audio_projection.resolve(strict=True))
    features = []
    controls = []
    masks = []
    audio_targets = []
    template_indexes = []
    retained_metadata = []
    with torch.inference_mode():
        for task in tasks:
            audio = read_render(Path(task["audio"])).to("cuda")
            peak = float(audio.abs().max())
            if peak < 1e-5:
                continue
            feature = extract_feature(audio, AudioFeatureKind.LOG_MEL)
            preset = VitalPreset.from_file(Path(task["preset"]))
            control, mask = statistics.encode(preset)
            summary = np.concatenate(
                (
                    feature.float().mean(dim=-1).cpu().numpy().ravel(),
                    feature.float().std(dim=-1).cpu().numpy().ravel(),
                )
            )
            scaled_summary = (
                summary - projection["summary_mean"]
            ) / projection["summary_scale"]
            embedding = (
                scaled_summary - projection["pca_mean"]
            ) @ projection["pca_components"].T
            audio_target = (
                embedding - projection["audio_mean"]
            ) / projection["audio_scale"]
            features.append(feature.half().cpu().contiguous())
            controls.append(torch.from_numpy(control).half())
            masks.append(torch.from_numpy(mask))
            audio_targets.append(
                torch.from_numpy(np.asarray(audio_target, dtype=np.float32)).half()
            )
            template_indexes.append(int(task["template_index"]))
            retained_metadata.append(
                {
                    **task["metadata"],
                    "peak": peak,
                    "clipped_sample_fraction": float(
                        (audio.abs() >= 0.999).float().mean()
                    ),
                }
            )
    if not features:
        raise RuntimeError("Chunk contains no non-silent examples")

    tensors = {
        "feature": torch.stack(features),
        "controls": torch.stack(controls),
        "mask": torch.stack(masks),
        "audio_target": torch.stack(audio_targets),
        "template_index": torch.tensor(template_indexes, dtype=torch.int32),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = args.output.with_suffix(".partial.safetensors")
    save_file(tensors, temporary_output)
    temporary_output.replace(args.output)
    temporary_metadata = args.metadata_output.with_suffix(".partial.jsonl")
    temporary_metadata.write_text(
        "".join(json.dumps(item) + "\n" for item in retained_metadata),
        encoding="utf-8",
    )
    temporary_metadata.replace(args.metadata_output)
    print(
        json.dumps(
            {
                "gpu": gpu.model_dump(),
                "input_count": len(tasks),
                "retained_count": len(retained_metadata),
            }
        )
    )


if __name__ == "__main__":
    main()
