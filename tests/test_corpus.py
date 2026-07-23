from pathlib import Path

from synth.corpus import evaluation_family


def test_evaluation_family_is_path_based() -> None:
    assert evaluation_family(Path("Vendor/BLA Midtempo/Presets/a.vital")) == "midtempo"
    assert evaluation_family(Path("Vendor/Dubstep Pack/a.vital")) == "dubstep"
    assert evaluation_family(Path("Vendor/Ambient/a.vital")) is None
