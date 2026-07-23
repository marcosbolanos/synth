#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from safetensors.torch import save_file

from synth import A6000_UUID, pin_a6000


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument(
        "--features",
        nargs="+",
        choices=("dac", "log_mel", "multi_stft", "waveform"),
        default=("dac", "log_mel", "multi_stft", "waveform"),
    )
    return parser.parse_args()


def main() -> None:
    gpu = pin_a6000()
    import torch

    from synth.audio_features import (
        AudioFeatureKind,
        extract_dac,
        extract_feature,
        read_render,
    )

    args = parse_args()
    dataset = args.dataset_dir.resolve(strict=True)
    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    feature_root = dataset / "features"
    feature_root.mkdir(exist_ok=True)
    timings: dict[str, float] = {}
    shapes: dict[str, list[int]] = {}
    kinds = tuple(AudioFeatureKind(name) for name in args.features)
    for kind in kinds:
        kind_root = feature_root / kind.value
        kind_root.mkdir(exist_ok=True)
        started = time.monotonic()
        dac_model = None
        if kind is AudioFeatureKind.DAC:
            import dac

            model_path = dac.utils.download(model_type="44khz")
            dac_model = dac.DAC.load(model_path).to("cuda").eval()
        for index, row in enumerate(rows, start=1):
            destination = kind_root / f"{row['preset_sha256']}.safetensors"
            if not destination.exists():
                audio = read_render(dataset / row["audio_file"]).to("cuda")
                with torch.inference_mode():
                    if kind is AudioFeatureKind.DAC:
                        feature, codes = extract_dac(audio, dac_model)
                        tensors = {
                            "feature": feature.to(dtype=torch.float16).cpu().contiguous(),
                            "codes": codes.to(dtype=torch.int16).cpu().contiguous(),
                        }
                    else:
                        feature = extract_feature(audio, kind)
                        tensors = {
                            "feature": feature.to(dtype=torch.float16).cpu().contiguous()
                        }
                save_file(tensors, destination)
                shapes[kind.value] = list(feature.shape)
            if index % 100 == 0:
                print(f"{kind.value}: {index}/{len(rows)}", flush=True)
        timings[kind.value] = time.monotonic() - started
    report = {
        "gpu_uuid": A6000_UUID,
        "gpu": gpu.model_dump(),
        "feature_shapes": shapes,
        "seconds": timings,
    }
    (feature_root / "feature_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
