# Pedalboard + latest Vital handoff

> **Temporary AI handoff:** Delete `doc/handoff.md` before merging this branch
> into `main`. Confirm that the deletion is included in the final merge diff.

## Goal

Replace the source-built Vital 1.0.6 headless renderer with an offline renderer
that hosts the latest official Vital Linux VST3 binary through Spotify
Pedalboard. Vital remains a black box: send it MIDI events, process audio blocks,
and write the returned stereo audio to a WAV file.

Do not remove the existing source-based renderer until the VST3 path has passed
the preset-loading and audio-equivalence checks below.

## Linux compatibility

Pedalboard supports VST3 instruments on Linux. Its published Linux wheels cover
x86_64 and aarch64 on manylinux and musllinux, and its documentation notes that
most Linux VSTs require a relatively modern system with glibc newer than 2.27.
Vital must be a native Linux VST3 build matching the machine architecture; a
Windows or macOS `.vst3` bundle cannot be loaded on Linux.

Plugin compatibility is not absolute. Run Vital from the Python main thread for
the initial implementation because Pedalboard warns that some plugins fail,
hang, or render incorrectly on background threads. Use separate worker processes
only after a single-process render is reliable.

References:

- <https://github.com/spotify/pedalboard#compatibility>
- <https://spotify.github.io/pedalboard/reference/pedalboard.html#pedalboard.VST3Plugin>
- <https://spotify.github.io/pedalboard/compatibility.html>

## Install Python dependencies

Use the repository's required package manager:

```bash
uv add pedalboard mido
```

Run Python commands and project scripts through `uv run`.

## Install and prepare Vital

1. Download the latest **native Linux VST3** installer from the official Vital
   account/download page: <https://account.vital.audio/>.
2. Install Vital using its official installer. Do not copy a Windows plugin or
   introduce Wine/yabridge into the first proof of concept.
3. Locate the installed `Vital.vst3` bundle. Common locations include
   `~/.vst3/Vital.vst3`, `/usr/local/lib/vst3/Vital.vst3`, and
   `/usr/lib/vst3/Vital.vst3`.
4. Open the official standalone or plugin editor once, authenticate it, install
   any required factory content, select the desired oversampling mode, and verify
   that it can play a note. Use Vital's offline mode afterward if appropriate.
5. Record the Vital version, plugin path, architecture, and SHA-256 digest of the
   installed plugin binary in test output. Do not commit account credentials,
   activation data, the proprietary Vital binary, or factory content.

Pass the plugin path explicitly to scripts, preferably with a required CLI
argument such as `--vital-vst3`. Do not silently search several locations or
fall back to the source-built renderer.

## Prove that Pedalboard can load and render Vital

Start with a minimal main-thread smoke test:

```python
from pathlib import Path

from mido import Message
from pedalboard import VST3Plugin, load_plugin
from pedalboard.io import AudioFile


SAMPLE_RATE: int = 44_100
CHANNELS: int = 2
BUFFER_SIZE: int = 512


def render_init_patch(plugin_path: Path, output_path: Path) -> None:
    plugin = load_plugin(str(plugin_path))
    if not isinstance(plugin, VST3Plugin):
        raise TypeError(f"Expected VST3Plugin, received {type(plugin).__name__}")
    if not plugin.is_instrument:
        raise TypeError(f"Expected an instrument plugin at {plugin_path}")

    audio = plugin(
        [
            Message("note_on", note=60, velocity=100, time=0.0),
            Message("note_off", note=60, velocity=0, time=4.0),
        ],
        duration=6.0,
        sample_rate=SAMPLE_RATE,
        num_channels=CHANNELS,
        buffer_size=BUFFER_SIZE,
        reset=True,
    )

    with AudioFile(str(output_path), "w", SAMPLE_RATE, CHANNELS) as output:
        output.write(audio)
```

The completed smoke test must assert that the output is stereo, finite, not
silent, and has the requested frame count. Generated WAVs and diagnostic files
must follow the repository convention:
`data/outputs/<generator_name_vX_timestamp>/files`.

## Establish how native `.vital` presets are loaded

This is the main unresolved integration point. Pedalboard's documented
`VST3Plugin.load_preset()` method loads Steinberg `.vstpreset` files, not native
`.vital` files.

The Vital 1.0.6 source stores JSON in the plugin's raw host state, closely
matching a `.vital` file and adding tuning state. The current closed-source Vital
may retain that behavior, but it is not a documented public contract. Test it;
do not assume it.

Perform this experiment in a disposable subprocess because Pedalboard warns that
invalid raw plugin state can crash the entire Python process:

1. Load the latest Vital VST3 and capture `plugin.raw_state` for its init patch.
2. Save a distinctive patch from the same Vital version as both `.vital` and,
   through a compatible host, `.vstpreset` when possible.
3. Compare the native preset JSON with `plugin.raw_state` and
   `plugin.preset_data` without modifying the installed originals.
4. Try assigning exact bytes only when their framing is understood.
5. Render a fixed MIDI phrase and compare it with a reference render made in the
   official Vital standalone or a DAW.
6. Restart the process, reload the state, and verify the same patch parameters
   and audio characteristics.

If direct native-state loading is not reliable, use `.vstpreset` snapshots as
the supported interchange format. Do not implement GUI automation or downgrade
the version strings inside `.vital` files.

## Reproducibility requirements

Pin or record all inputs that can change rendering:

- Pedalboard version and Python version
- Vital version and plugin binary digest
- Exact preset or VST3 state bytes and digest
- Sample rate, channel count, block size, BPM, and transport position
- MIDI event times, notes, velocities, and note-off events
- Vital oversampling and other non-preset configuration
- Warm-up duration, note duration, and effect/release tail duration

Reload the plugin for each independent render until tests prove that `reset=True`
fully clears Vital's oscillator, modulation, delay, and reverb state. For batch
work, prefer isolated worker processes so a plugin crash cannot destroy the
orchestrator.

## Acceptance criteria

- A native Linux Vital VST3 is loaded without opening its editor.
- Init-patch MIDI renders to a non-silent deterministic WAV.
- A preset made by the installed Vital version can be loaded without losing
  custom wavetables, LFO shapes, samples, modulation routes, or new spectral
  features.
- Repeated fresh-process renders match within an explicitly chosen tolerance.
- Failures identify the exact plugin path and operation; there is no dependency,
  plugin-path, preset-format, or renderer fallback.
- Existing source-renderer behavior remains available until migration is
  explicitly approved.
- **Before merge to `main`, delete this handoff file and verify that
  `doc/handoff.md` is absent from the merge diff.**
