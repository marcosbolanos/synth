from synth.augmentation import ControlVariationStatistics, variation_seed
from synth.models.vital_preset_model import VitalPreset
from test_vital import preset_json


def test_control_variation_is_deterministic_and_bounded() -> None:
    low = VitalPreset.from_json(preset_json())
    control_name = next(iter(low.settings.controls))
    high = low.with_control(control_name, 1.0)
    statistics = ControlVariationStatistics.fit((low, high))

    first, changed = statistics.vary(low, strength=0.32, seed=42)
    second, _ = statistics.vary(low, strength=0.32, seed=42)

    assert first.to_json_bytes() == second.to_json_bytes()
    assert changed == (control_name,)
    assert 0.0 <= first.settings.controls[control_name] <= 1.0


def test_variation_seed_changes_by_parent_and_index() -> None:
    assert variation_seed("a", 0) == variation_seed("a", 0)
    assert variation_seed("a", 0) != variation_seed("a", 1)
    assert variation_seed("a", 0) != variation_seed("b", 0)
