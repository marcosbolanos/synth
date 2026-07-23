from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from synth.models.vital_preset_model import VitalPreset


SplitName = Literal["train", "development", "test"]


class PresetCorpusEntry(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    preset_sha256: str
    source_paths: tuple[Path, ...]
    pack: str
    evaluation_family: Literal["dubstep", "midtempo"] | None
    split: SplitName
    preset_name: str
    preset_style: str
    json_bytes: int
    externalized_json_bytes: int
    control_count: int
    active_modulation_count: int
    wavetable_group_count: int
    wavetable_component_count: int
    wavetable_keyframe_count: int
    lfo_point_count: int
    embedded_asset_count: int


def evaluation_family(path: Path) -> Literal["dubstep", "midtempo"] | None:
    lowered = tuple(part.casefold() for part in path.parts)
    if any("dubstep" in part for part in lowered):
        return "dubstep"
    if any("midtempo" in part for part in lowered):
        return "midtempo"
    return None


def pack_name(path: Path, corpus_root: Path) -> str:
    relative = path.relative_to(corpus_root)
    family = evaluation_family(relative)
    if family is not None:
        for part in reversed(relative.parts[:-1]):
            if family in part.casefold():
                return part
    parents = relative.parts[:-1]
    return parents[-2] if len(parents) >= 2 else parents[-1]


def _externalize_assets(value: object, assets: dict[str, str]) -> object:
    if isinstance(value, dict):
        result: dict[str, object] = {}
        for key, child in value.items():
            if key in {"samples", "samples_stereo", "audio_file", "wave_data"} and isinstance(child, str) and child:
                digest = hashlib.sha256(child.encode("utf-8")).hexdigest()
                assets[digest] = child
                result[key] = {"asset_sha256": digest}
            else:
                result[key] = _externalize_assets(child, assets)
        return result
    if isinstance(value, list):
        return [_externalize_assets(child, assets) for child in value]
    return value


def _metrics(preset: VitalPreset) -> dict[str, int]:
    settings = preset.settings
    groups = tuple(group for wavetable in settings.wavetables for group in wavetable.groups)
    components = tuple(component for group in groups for component in group.components)
    keyframe_count = sum(len(component.keyframes or ()) for component in components)
    assets: dict[str, str] = {}
    externalized = _externalize_assets(preset.to_dict(), assets)
    externalized_bytes = len(
        json.dumps(externalized, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    )
    return {
        "externalized_json_bytes": externalized_bytes,
        "control_count": len(settings.controls),
        "active_modulation_count": sum(
            bool(route.source and route.destination) for route in settings.modulations
        ),
        "wavetable_group_count": len(groups),
        "wavetable_component_count": len(components),
        "wavetable_keyframe_count": keyframe_count,
        "lfo_point_count": sum(lfo.num_points for lfo in settings.lfos),
        "embedded_asset_count": len(assets),
    }


def _rank(seed: int, pack: str, digest: str) -> bytes:
    return hashlib.sha256(f"{seed}:{pack}:{digest}".encode()).digest()


def build_corpus(
    corpus_root: Path,
    *,
    seed: int = 20_260_722,
    test_fraction: float = 0.2,
    development_fraction: float = 0.1,
) -> tuple[PresetCorpusEntry, ...]:
    paths = tuple(sorted(corpus_root.rglob("*.vital")))
    if not paths:
        raise ValueError(f"No .vital files found beneath {corpus_root}")

    grouped: dict[str, list[tuple[Path, VitalPreset]]] = defaultdict(list)
    for path in paths:
        preset = VitalPreset.from_file(path)
        grouped[preset.sha256].append((path, preset))

    descriptions: dict[str, tuple[tuple[Path, ...], VitalPreset, str, str | None]] = {}
    evaluation_groups: dict[str, list[str]] = defaultdict(list)
    regular = []
    for digest, copies in grouped.items():
        source_paths = tuple(path for path, _ in copies)
        preset = copies[0][1]
        families = tuple(
            family for path in source_paths if (family := evaluation_family(path)) is not None
        )
        family = families[0] if families else None
        pack = next(
            (pack_name(path, corpus_root) for path in source_paths if evaluation_family(path)),
            pack_name(source_paths[0], corpus_root),
        )
        descriptions[digest] = (source_paths, preset, pack, family)
        if family is None:
            regular.append(digest)
        else:
            evaluation_groups[pack].append(digest)

    split_by_digest: dict[str, SplitName] = {}
    for pack, digests in evaluation_groups.items():
        ordered = sorted(digests, key=lambda digest: _rank(seed, pack, digest))
        test_count = max(1, round(len(ordered) * test_fraction)) if len(ordered) > 1 else 1
        for digest in ordered[:test_count]:
            split_by_digest[digest] = "test"
        for digest in ordered[test_count:]:
            split_by_digest[digest] = "train"

    development_count = round(len(regular) * development_fraction)
    development = set(
        sorted(regular, key=lambda digest: _rank(seed, "development", digest))[
            :development_count
        ]
    )
    for digest in regular:
        split_by_digest[digest] = "development" if digest in development else "train"

    entries = []
    for digest in sorted(descriptions):
        source_paths, preset, pack, family = descriptions[digest]
        entries.append(
            PresetCorpusEntry(
                preset_sha256=digest,
                source_paths=source_paths,
                pack=pack,
                evaluation_family=family,
                split=split_by_digest[digest],
                preset_name=preset.preset_name or source_paths[0].stem,
                preset_style=preset.preset_style,
                json_bytes=len(preset.to_json_bytes()),
                **_metrics(preset),
            )
        )
    return tuple(entries)
