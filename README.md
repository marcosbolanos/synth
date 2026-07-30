# Synth Experiments

## Dependencies

- `uv`
- The official native Linux Vital VST3

The `src/vital` submodule is retained only as reference material for inspecting
Vital's preset and state formats. It is not built or executed by this project.

The supplied Vital 1.6.4 binary requires a newer C runtime than this host. Install
the checksum-pinned user-local runtime and VST3 without sudo:

```bash
uv run scripts/setup_vital_runtime.py
```

## Render the official Vital VST3 with Pedalboard

Install the official native Linux Vital VST3, then render its init patch as a
black-box instrument from the Python main thread:

```bash
uv run scripts/render_vital_vst3.py --vital-vst3 /path/to/Vital.vst3
```

The plugin path is mandatory. Pedalboard plus the official VST3 is the only
supported rendering backend. The script writes the validated stereo WAV and
reproducibility metadata beneath `data/outputs/render_vital_vst3_v2_*/files`.

## Tailnet output gallery

The read-only gallery displays generated audio, images, HTML, and metadata. It
must be bound to an exact Tailscale IP so it is not exposed on LAN or public
interfaces:

```bash
uv run python -m webui.app --host 100.119.77.112 --port 8765
```

Pull completed visualization artifacts from the configured compute host:

```bash
uv run scripts/pull_remote_outputs.py
```

Edit `scripts/pull_remote_outputs.toml` to change the host, repository paths, or
number of recent manifests. Producers publish a run only after writing its
typed `gallery_manifest.json`. A successful verified pull restarts the configured
gallery service so newly synced routes become live immediately.

## External Vital presets

Download and safely extract the community-maintained Jek's Vital Presets
archive into `data/external/Vital Presets/Jeks Vital Presets`:

```bash
uv run scripts/download_jeks_vital_presets.py
```

The downloader validates the current Dropbox archive size and archive member
paths, stages extraction transactionally, and refuses to replace an existing
copy. The source archive is retained alongside the extracted files.

Render the 16 presets in the BLA Midtempo pack through the official Vital VST3:

```bash
uv run scripts/render_midtempo_presets.py \
  --vital-vst3 /home/marcos/.vst3/Vital.vst3 \
  --preset-dir "data/external/Vital Presets/Jeks Vital Presets/files/Jek's Vital Presets/BLA - Midtempo for Vital/Presets"
```

Each preset renders in a fresh subprocess. Native `.vital` JSON is wrapped in
the JUCE component-state envelope expected by Pedalboard's VST3 state API.

### Typed preset API

`VitalPreset` models the full native preset language: 903 Vital 1.6.4 control
names, modulation routes, samples, LFO and custom-warp curves, random state,
and discriminated wavetable source/modifier graphs.

```python
from pathlib import Path

from synth import VitalControlName, VitalPreset, VitalRenderer

preset = VitalPreset.from_file(Path("input.vital"))
modified = preset.with_control(VitalControlName.OSC_1_LEVEL, 0.75)
modified.to_file(Path("modified.vital"))

renderer = VitalRenderer(Path("/home/marcos/.vst3/Vital.vst3"))
audio = renderer.render_vital_audio(note=36, duration_seconds=4.0, preset=modified)
```

## Sound-to-preset experiment

The experiment uses C2 held for two seconds plus a two-second tail. Architecture
selection compares DAC, log-mel, multi-resolution STFT, and waveform encodings,
then trains only on the A6000 selected by its NVIDIA UUID.

```bash
uv run scripts/investigate_vital_corpus.py --preset-root "data/external/Vital Presets/Jeks Vital Presets/files"
uv run scripts/build_vital_dataset.py --corpus-jsonl /path/to/corpus.jsonl --vital-vst3 "$HOME/.vst3/Vital.vst3"
uv run scripts/extract_vital_audio_features.py --dataset-dir /path/to/dataset/files
uv run scripts/run_architecture_investigation.py --dataset-dir /path/to/dataset/files --corpus-investigation /path/to/investigation.json
uv run scripts/train_vital_transformer.py --dataset-dir /path/to/dataset/files --architecture-report /path/to/architecture_investigation.json --vital-vst3 "$HOME/.vst3/Vital.vst3"
uv run scripts/preference_finetune_vital.py --dataset-dir /path/to/dataset/files --training-dir /path/to/training/files --vital-vst3 "$HOME/.vst3/Vital.vst3"
uv run scripts/augment_vital_dataset.py --dataset-dir /path/to/dataset/files --vital-vst3 "$HOME/.vst3/Vital.vst3"
uv run scripts/extract_vital_audio_features.py --dataset-dir /path/to/augmented/files --features log_mel
uv run scripts/train_vital_transformer.py --dataset-dir /path/to/augmented/files --architecture-report /path/to/architecture_investigation.json --vital-vst3 "$HOME/.vst3/Vital.vst3"
uv run scripts/compare_vital_augmentation.py --baseline-training-dir /path/to/baseline/files --augmented-training-dir /path/to/augmented-training/files --augmented-dataset-dir /path/to/augmented/files
```

For the large diversified run, start the resumable producer on an unreserved
2080 Ti and the consumer on the A6000. The producer overlaps CPU rendering with
GPU feature extraction, writes compact tensor shards, and deletes transient
presets and audio. The consumer keeps a 25% replay fraction of original
examples, checkpoints each shard, and deletes consumed shards:

```bash
uv run scripts/stream_vital_augmentation.py \
  --dataset-dir /path/to/augmented/files \
  --reference-training-dir /path/to/augmented-training/files \
  --vital-vst3 "$HOME/.vst3/Vital.vst3" \
  --feature-gpu-uuid GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea

uv run scripts/train_vital_stream_transformer.py \
  --stream-dir /path/to/stream/files \
  --reference-dataset-dir /path/to/augmented/files \
  --reference-training-dir /path/to/augmented-training/files \
  --vital-vst3 "$HOME/.vst3/Vital.vst3"

uv run scripts/compare_vital_stream_scale.py \
  --baseline-training-dir /path/to/baseline/files \
  --four-variant-training-dir /path/to/augmented-training/files \
  --stream-training-dir /path/to/stream-training/files
```

The latest investigation and training report are available at
`/vital-investigation`, `/vital-transformer`, `/vital-augmentation`, and
`/vital-stream-scale` in the output gallery.
