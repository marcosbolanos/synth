from __future__ import annotations

import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Final, Literal

import numpy as np
from mido import Message
from numpy.typing import NDArray
from pedalboard import VST3Plugin, load_plugin
from pydantic import BaseModel, ConfigDict, Field

from synth.models.vital_preset_model import VitalPreset


JUCE_BASE64_TABLE: Final[bytes] = (
    b".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
)
JUCE_BASE64_VALUES: Final[dict[int, int]] = {
    character: index for index, character in enumerate(JUCE_BASE64_TABLE)
}


class VitalRenderConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    sample_rate: int = Field(default=44_100, ge=8_000, le=384_000)
    channels: Literal[2] = 2
    buffer_size: int = Field(default=512, ge=1, le=32_768)
    velocity: int = Field(default=110, ge=1, le=127)
    tail_duration_seconds: float = Field(default=2.0, ge=0.0, le=30.0)


class VitalNote(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    number: int = Field(ge=0, le=127)
    duration_seconds: float = Field(gt=0.0, le=120.0)


class VitalPluginIdentity(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    descriptive_name: str
    manufacturer_name: str
    version: str


def _juce_memory_block_base64(data: bytes) -> str:
    encoded = bytearray()
    accumulator = 0
    bit_count = 0
    for byte in data:
        accumulator |= byte << bit_count
        bit_count += 8
        while bit_count >= 6:
            encoded.append(JUCE_BASE64_TABLE[accumulator & 63])
            accumulator >>= 6
            bit_count -= 6
    if bit_count:
        encoded.append(JUCE_BASE64_TABLE[accumulator & 63])
    return f"{len(data)}.{encoded.decode('ascii')}"


def _decode_juce_memory_block(encoded: str) -> bytes:
    length_text, characters = encoded.split(".", maxsplit=1)
    expected_length = int(length_text)
    decoded = bytearray()
    accumulator = 0
    bit_count = 0
    for character in characters.encode("ascii"):
        accumulator |= JUCE_BASE64_VALUES[character] << bit_count
        bit_count += 6
        while bit_count >= 8 and len(decoded) < expected_length:
            decoded.append(accumulator & 0xFF)
            accumulator >>= 8
            bit_count -= 8
    if len(decoded) != expected_length:
        raise ValueError(
            f"Expected {expected_length} decoded JUCE bytes, received {len(decoded)}"
        )
    return bytes(decoded)


def _vst3_raw_state(native_preset: bytes) -> bytes:
    component_state = _juce_memory_block_base64(native_preset)
    xml = (
        '<?xml version="1.0" encoding="UTF-8"?> '
        f"<VST3PluginState><IComponent>{component_state}</IComponent>"
        "</VST3PluginState>"
    ).encode("utf-8")
    return b"VC2!" + struct.pack("<I", len(xml)) + xml + b"\0"


def _component_state(raw_state: bytes) -> bytes:
    if raw_state[:4] != b"VC2!":
        raise ValueError("Expected a JUCE VC2 VST3 state header")
    xml_length = struct.unpack("<I", raw_state[4:8])[0]
    root = ET.fromstring(raw_state[8 : 8 + xml_length])
    component = root.find("IComponent")
    if component is None or component.text is None:
        raise ValueError("VST3 state does not contain IComponent data")
    return _decode_juce_memory_block(component.text)


def _preset_from_component_state(component_state: bytes) -> VitalPreset:
    json_start = component_state.index(b"{")
    preset_data, _ = json.JSONDecoder().raw_decode(
        component_state[json_start:].decode("utf-8", errors="ignore")
    )
    preset_data.pop("tuning", None)
    return VitalPreset.from_dict(preset_data)


class VitalRenderer:
    """A main-thread Pedalboard host for one official Vital VST3 instance."""

    def __init__(
        self,
        plugin_path: Path,
        config: VitalRenderConfig = VitalRenderConfig(),
    ) -> None:
        self.plugin_path = plugin_path.resolve(strict=True)
        self.config = config
        plugin = load_plugin(str(self.plugin_path))
        if not isinstance(plugin, VST3Plugin) or not plugin.is_instrument:
            raise TypeError(f"Expected a VST3 instrument at {self.plugin_path}")
        self._plugin = plugin
        self.identity = VitalPluginIdentity(
            name=plugin.name,
            descriptive_name=plugin.descriptive_name,
            manufacturer_name=plugin.manufacturer_name,
            version=plugin.version,
        )

    @property
    def current_preset(self) -> VitalPreset:
        return _preset_from_component_state(_component_state(self._plugin.raw_state))

    def render_vital_audio(
        self,
        note: int,
        duration_seconds: float,
        preset: VitalPreset,
    ) -> NDArray[np.float32]:
        request = VitalNote(number=note, duration_seconds=duration_seconds)
        self._plugin.raw_state = _vst3_raw_state(preset.to_json_bytes())

        loaded_preset = self.current_preset
        if loaded_preset.preset_name != preset.preset_name:
            raise ValueError(
                f"Vital loaded {loaded_preset.preset_name!r}, "
                f"expected {preset.preset_name!r}"
            )

        render_duration = (
            request.duration_seconds + self.config.tail_duration_seconds
        )
        audio = self._plugin(
            [
                Message(
                    "note_on",
                    note=request.number,
                    velocity=self.config.velocity,
                    time=0.0,
                ),
                Message(
                    "note_off",
                    note=request.number,
                    velocity=0,
                    time=request.duration_seconds,
                ),
            ],
            duration=render_duration,
            sample_rate=self.config.sample_rate,
            num_channels=self.config.channels,
            buffer_size=self.config.buffer_size,
            reset=True,
        )
        expected_shape = (
            self.config.channels,
            int(render_duration * self.config.sample_rate),
        )
        if audio.shape != expected_shape:
            raise ValueError(f"Expected audio shape {expected_shape}, received {audio.shape}")
        if audio.dtype != np.float32:
            raise TypeError(f"Expected float32 audio, received {audio.dtype}")
        if not np.isfinite(audio).all():
            raise ValueError(f"Preset produced non-finite audio: {preset.preset_name}")
        if float(np.max(np.abs(audio))) <= 0.0:
            raise ValueError(f"Preset produced silent audio: {preset.preset_name}")
        return audio
