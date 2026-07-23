from __future__ import annotations

from typing import Final

from synth.vital import VitalRenderConfig


MIDI_NOTE: Final[int] = 36
NOTE_DURATION_SECONDS: Final[float] = 2.0
RENDER_CONFIG: Final[VitalRenderConfig] = VitalRenderConfig(
    sample_rate=44_100,
    channels=2,
    buffer_size=512,
    velocity=110,
    tail_duration_seconds=2.0,
)
RENDER_FRAME_COUNT: Final[int] = 176_400
