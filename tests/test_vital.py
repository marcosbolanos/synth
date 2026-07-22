import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from synth import VitalControlName, VitalPreset
from synth.vital import (
    VitalNote,
    VitalRenderConfig,
    _component_state,
    _decode_juce_memory_block,
    _juce_memory_block_base64,
    _vst3_raw_state,
)


def line_generator() -> dict[str, object]:
    return {
        "num_points": 2,
        "points": [0.0, 0.0, 1.0, 1.0],
        "powers": [0.0, 0.0],
        "name": "Linear",
        "smooth": False,
    }


def preset_json() -> str:
    wavetable = {
        "groups": [],
        "name": "Init",
        "author": "",
        "version": "1.6.4",
        "remove_all_dc": True,
        "full_normalize": False,
    }
    settings = {
        "volume": 0.5,
        "sample": {"length": 0, "name": "", "sample_rate": 44_100, "samples": ""},
        "modulations": [{"source": "", "destination": ""} for _ in range(64)],
        "wavetables": [wavetable for _ in range(3)],
        "lfos": [line_generator() for _ in range(8)],
    }
    return json.dumps(
        {
            "synth_version": "1.6.4",
            "preset_name": "Typed Init",
            "author": "Synth",
            "comments": "Test preset",
            "preset_style": "Init",
            "macro1": "MACRO 1",
            "macro2": "MACRO 2",
            "macro3": "MACRO 3",
            "macro4": "MACRO 4",
            "settings": settings,
        }
    )


def test_vital_preset_from_json() -> None:
    preset = VitalPreset.from_json(preset_json())

    assert preset.preset_name == "Typed Init"
    assert preset.settings.controls["volume"] == 0.5
    assert len(preset.sha256) == 64


def test_vital_preset_semantic_json_round_trip() -> None:
    source = VitalPreset.from_json(preset_json())

    assert VitalPreset.from_json(source.to_json()) == source


def test_vital_preset_file_round_trip(tmp_path: Path) -> None:
    source = VitalPreset.from_json(preset_json())
    destination = tmp_path / "typed-init.vital"

    source.to_file(destination)

    assert VitalPreset.from_file(destination) == source


def test_with_control_returns_a_validated_copy() -> None:
    source = VitalPreset.from_json(preset_json())

    modified = source.with_control(VitalControlName.OSC_1_LEVEL, 0.75)

    assert modified.settings.controls[VitalControlName.OSC_1_LEVEL] == 0.75
    assert VitalControlName.OSC_1_LEVEL not in source.settings.controls


def test_unknown_control_is_rejected() -> None:
    data = json.loads(preset_json())
    data["settings"]["not_a_vital_control"] = 1.0

    with pytest.raises(ValidationError):
        VitalPreset.from_dict(data)


@pytest.mark.parametrize("data", [b"", b"a", b"Vital", bytes(range(256))])
def test_juce_memory_block_base64_round_trip(data: bytes) -> None:
    assert _decode_juce_memory_block(_juce_memory_block_base64(data)) == data


def test_vst3_state_contains_exact_native_preset() -> None:
    preset = VitalPreset.from_json(preset_json())
    preset_bytes = preset.to_json_bytes()

    assert _component_state(_vst3_raw_state(preset_bytes)) == preset_bytes


@pytest.mark.parametrize("note", [-1, 128])
def test_midi_note_contract(note: int) -> None:
    with pytest.raises(ValidationError):
        VitalNote(number=note, duration_seconds=1.0)


def test_render_config_is_stereo() -> None:
    with pytest.raises(ValidationError):
        VitalRenderConfig(channels=1)  # type: ignore[arg-type]
