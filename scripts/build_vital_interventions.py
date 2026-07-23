#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import save_file

from synth import create_output_directory, repository_relative_output_path
from synth.preset_representation import ControlStatistics, load_presets


GENERATOR = "build_vital_interventions_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", required=True, type=Path)
    parser.add_argument("--reference-training-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    dataset = args.dataset_dir.resolve(strict=True)
    reference = args.reference_training_dir.resolve(strict=True)
    rows = tuple(
        json.loads(line)
        for line in (dataset / "dataset.jsonl").read_text(encoding="utf-8").splitlines()
    )
    statistics = ControlStatistics.model_validate_json(
        (reference / "control_statistics.json").read_text(encoding="utf-8")
    )
    presets = load_presets(dataset, rows)
    controls_and_masks = tuple(statistics.encode(preset) for preset in presets)
    controls = np.stack(tuple(item[0] for item in controls_and_masks))
    masks = np.stack(tuple(item[1] for item in controls_and_masks))
    index_by_digest = {str(row["preset_sha256"]): index for index, row in enumerate(rows)}
    current: list[int] = []
    following: list[int] = []
    for child_index, row in enumerate(rows):
        parent_digest = row.get("parent_preset_sha256")
        if parent_digest is None:
            continue
        parent_index = index_by_digest[str(parent_digest)]
        current.extend((parent_index, child_index))
        following.extend((child_index, parent_index))
    current_array = np.asarray(current, dtype=np.int64)
    following_array = np.asarray(following, dtype=np.int64)
    action = controls[following_array] - controls[current_array]
    action_mask = masks[current_array] & masks[following_array] & (np.abs(action) > 1e-6)

    output = create_output_directory(repo_root, GENERATOR)
    save_file(
        {
            "controls": torch.from_numpy(controls),
            "masks": torch.from_numpy(masks),
            "current": torch.from_numpy(current_array),
            "next": torch.from_numpy(following_array),
            "action_delta": torch.from_numpy(action),
            "action_mask": torch.from_numpy(action_mask),
        },
        output / "interventions.safetensors",
    )
    state_lines = []
    for row in rows:
        digest = str(row["preset_sha256"])
        state_lines.append(
            json.dumps(
                {
                    "preset_sha256": digest,
                    "split": row["split"],
                    "pack": row["pack"],
                    "evaluation_family": row["evaluation_family"],
                    "is_augmented": row["is_augmented"],
                    "preset_file": str(
                        repository_relative_output_path(repo_root, dataset / str(row["preset_file"]))
                    ),
                    "audio_file": str(
                        repository_relative_output_path(repo_root, dataset / str(row["audio_file"]))
                    ),
                    "feature_file": str(
                        repository_relative_output_path(
                            repo_root,
                            dataset / "features" / "log_mel" / f"{digest}.safetensors",
                        )
                    ),
                }
            )
        )
    (output / "states.jsonl").write_text("\n".join(state_lines) + "\n", encoding="utf-8")
    report = {
        "schema_version": 1,
        "generator": GENERATOR,
        "dataset": str(repository_relative_output_path(repo_root, dataset)),
        "reference_training": str(repository_relative_output_path(repo_root, reference)),
        "state_count": len(rows),
        "transition_count": len(current),
        "forward_transition_count": len(current) // 2,
        "control_count": int(controls.shape[1]),
        "split_transition_counts": {
            split: sum(rows[index]["split"] == split for index in current)
            for split in ("train", "development", "test")
        },
    }
    (output / "intervention_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(output)


if __name__ == "__main__":
    main()
