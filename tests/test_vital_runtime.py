from pathlib import Path

from synth import VitalRuntime


def test_runtime_command_pins_local_loader() -> None:
    repo_root = Path(__file__).resolve().parent.parent
    runtime = VitalRuntime.from_repo_root(repo_root)
    command = runtime.command(repo_root / "scripts" / "render_vital_vst3.py", "--help")

    assert command[0] == str(runtime.loader)
    assert command[1] == "--library-path"
    assert str(runtime.library_directory) in command[2]
    assert command[3] == str(runtime.python)
