#!/usr/bin/env -S uv run

from __future__ import annotations

import argparse
from pathlib import Path

from synth import VitalRenderer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate VitalControlName from an installed Vital VST3."
    )
    parser.add_argument("--vital-vst3", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    renderer = VitalRenderer(args.vital_vst3)
    control_names = sorted(name.value for name in renderer.current_preset.settings.controls)
    lines = [
        "from enum import StrEnum",
        "",
        "",
        "class VitalControlName(StrEnum):",
    ]
    lines.extend(f'    {name.upper()} = "{name}"' for name in control_names)
    lines.append("")
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"generated {len(control_names)} controls in {args.output}")


if __name__ == "__main__":
    main()
