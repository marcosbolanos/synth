#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt

from synth import create_output_directory
from synth.corpus import PresetCorpusEntry, build_corpus


GENERATOR_NAME = "investigate_vital_model_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset-root", required=True, type=Path)
    return parser.parse_args()


def distribution(values: list[int]) -> dict[str, float | int]:
    ordered = sorted(values)
    return {
        "minimum": ordered[0],
        "median": statistics.median(ordered),
        "p95": ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))],
        "maximum": ordered[-1],
        "mean": statistics.mean(ordered),
    }


def write_plot(entries: tuple[PresetCorpusEntry, ...], output: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].hist([entry.json_bytes for entry in entries], bins=50, label="Native JSON")
    axes[0].hist(
        [entry.externalized_json_bytes for entry in entries],
        bins=50,
        alpha=0.7,
        label="Asset references",
    )
    axes[0].set_title("Preset representation sizes")
    axes[0].set_xlabel("bytes")
    axes[0].legend()
    axes[1].scatter(
        [entry.control_count for entry in entries],
        [entry.wavetable_component_count for entry in entries],
        alpha=0.35,
        s=12,
    )
    axes[1].set_xlabel("controls")
    axes[1].set_ylabel("wavetable components")
    axes[1].set_title("Output structural complexity")
    figure.tight_layout()
    figure.savefig(output, dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    preset_root = args.preset_root.resolve(strict=True)
    entries = build_corpus(preset_root)
    repo_root = Path(__file__).resolve().parent.parent
    output = create_output_directory(repo_root, GENERATOR_NAME)
    with (output / "corpus.jsonl").open("w", encoding="utf-8") as destination:
        for entry in entries:
            destination.write(entry.model_dump_json() + "\n")
    fields = (
        "json_bytes",
        "externalized_json_bytes",
        "control_count",
        "active_modulation_count",
        "wavetable_component_count",
        "wavetable_keyframe_count",
        "lfo_point_count",
        "embedded_asset_count",
    )
    report = {
        "generator": GENERATOR_NAME,
        "preset_root": str(preset_root),
        "unique_presets": len(entries),
        "source_files": sum(len(entry.source_paths) for entry in entries),
        "splits": Counter(entry.split for entry in entries),
        "evaluation_families": Counter(
            entry.evaluation_family or "other" for entry in entries
        ),
        "packs": Counter(entry.pack for entry in entries),
        "distributions": {
            field: distribution([getattr(entry, field) for entry in entries])
            for field in fields
        },
    }
    (output / "investigation.json").write_text(
        json.dumps(report, indent=2, default=dict) + "\n", encoding="utf-8"
    )
    write_plot(entries, output / "preset_complexity.png")
    print(output)


if __name__ == "__main__":
    main()
