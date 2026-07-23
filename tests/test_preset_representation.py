from synth.models.vital_preset_model import VitalPreset
from synth.preset_representation import CONTROL_NAMES, ControlStatistics
from test_vital import preset_json


def test_control_representation_reconstructs_template() -> None:
    preset = VitalPreset.from_json(preset_json())
    statistics = ControlStatistics.fit((preset,))
    vector, mask = statistics.encode(preset)
    reconstructed = statistics.apply(preset, vector)

    assert vector.shape == (len(CONTROL_NAMES),)
    assert mask.sum() == 1
    assert reconstructed.settings.controls == preset.settings.controls


def test_control_representation_clips_to_observed_vital_domain() -> None:
    preset = VitalPreset.from_json(preset_json())
    statistics = ControlStatistics.fit((preset,))
    vector, _ = statistics.encode(preset)
    vector[:] = 1_000.0

    reconstructed = statistics.apply(preset, vector)

    assert reconstructed.settings.controls == preset.settings.controls
