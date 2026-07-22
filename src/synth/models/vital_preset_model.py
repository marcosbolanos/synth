from __future__ import annotations

import hashlib
import json
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from synth.models.vital_control_name import VitalControlName


FiniteFloat = Annotated[float, Field(allow_inf_nan=False)]


class VitalModel(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)


class VitalLineGenerator(VitalModel):
    num_points: int = Field(ge=2)
    points: list[FiniteFloat]
    powers: list[FiniteFloat]
    name: str = ""
    smooth: bool = False

    @model_validator(mode="after")
    def validate_array_lengths(self) -> VitalLineGenerator:
        if len(self.points) != self.num_points * 2:
            raise ValueError("points must contain an x and y value for every point")
        if len(self.powers) != self.num_points:
            raise ValueError("powers must contain one value for every point")
        return self


class VitalModulationSource(StrEnum):
    AFTERTOUCH = "aftertouch"
    ENV_1 = "env_1"
    ENV_2 = "env_2"
    ENV_3 = "env_3"
    ENV_4 = "env_4"
    ENV_5 = "env_5"
    ENV_6 = "env_6"
    LFO_1 = "lfo_1"
    LFO_2 = "lfo_2"
    LFO_3 = "lfo_3"
    LFO_4 = "lfo_4"
    LFO_5 = "lfo_5"
    LFO_6 = "lfo_6"
    LFO_7 = "lfo_7"
    LFO_8 = "lfo_8"
    LIFT = "lift"
    MACRO_CONTROL_1 = "macro_control_1"
    MACRO_CONTROL_2 = "macro_control_2"
    MACRO_CONTROL_3 = "macro_control_3"
    MACRO_CONTROL_4 = "macro_control_4"
    MOD_WHEEL = "mod_wheel"
    NOTE = "note"
    NOTE_IN_OCTAVE = "note_in_octave"
    PITCH_WHEEL = "pitch_wheel"
    RANDOM = "random"
    RANDOM_1 = "random_1"
    RANDOM_2 = "random_2"
    RANDOM_3 = "random_3"
    RANDOM_4 = "random_4"
    STEREO = "stereo"
    VELOCITY = "velocity"


class VitalModulation(VitalModel):
    source: VitalModulationSource | Literal[""]
    destination: VitalControlName | Literal[""]
    line_mapping: VitalLineGenerator | None = None


class VitalSample(VitalModel):
    length: int = Field(ge=0)
    name: str
    sample_rate: int = Field(ge=0)
    samples: str
    samples_stereo: str | None = None


class VitalRandomValue(VitalModel):
    seed: int


class VitalPositionKeyframe(VitalModel):
    position: int = Field(ge=0)


class VitalWaveSourceKeyframe(VitalPositionKeyframe):
    wave_data: str


class VitalAudioFileSourceKeyframe(VitalPositionKeyframe):
    start_position: FiniteFloat
    window_fade: FiniteFloat
    window_size: FiniteFloat


class VitalLineSourceKeyframe(VitalPositionKeyframe):
    line: VitalLineGenerator
    pull_power: FiniteFloat


class VitalFrequencyFilterKeyframe(VitalPositionKeyframe):
    cutoff: FiniteFloat
    shape: FiniteFloat


class VitalWaveFolderKeyframe(VitalPositionKeyframe):
    fold_boost: FiniteFloat


class VitalWaveWindowKeyframe(VitalPositionKeyframe):
    left_position: FiniteFloat
    right_position: FiniteFloat


class VitalWaveWarpKeyframe(VitalPositionKeyframe):
    horizontal_power: FiniteFloat
    vertical_power: FiniteFloat


class VitalPhaseShiftKeyframe(VitalPositionKeyframe):
    mix: FiniteFloat
    phase: FiniteFloat


class VitalSlewLimiterKeyframe(VitalPositionKeyframe):
    down_run_rise: FiniteFloat
    up_run_rise: FiniteFloat


class VitalWaveSource(VitalModel):
    type: Literal["Wave Source"]
    interpolation: int
    interpolation_style: int
    keyframes: list[VitalWaveSourceKeyframe] | None = None


class VitalShepardToneSource(VitalModel):
    type: Literal["Shepard Tone Source"]
    interpolation: int
    interpolation_style: int
    keyframes: list[VitalWaveSourceKeyframe]


class VitalAudioFileSource(VitalModel):
    type: Literal["Audio File Source"]
    audio_file: str
    audio_sample_rate: int = Field(gt=0)
    fade_style: int
    interpolation_style: int
    keyframes: list[VitalAudioFileSourceKeyframe]
    normalize_gain: bool
    normalize_mult: bool
    phase_style: int
    random_seed: int
    window_size: FiniteFloat


class VitalLineSource(VitalModel):
    type: Literal["Line Source"]
    interpolation_style: int
    keyframes: list[VitalLineSourceKeyframe]
    num_points: int = Field(gt=0)


class VitalFrequencyFilter(VitalModel):
    type: Literal["Frequency Filter"]
    interpolation_style: int
    keyframes: list[VitalFrequencyFilterKeyframe]
    normalize: bool
    style: int


class VitalWaveFolder(VitalModel):
    type: Literal["Wave Folder"]
    interpolation_style: int
    keyframes: list[VitalWaveFolderKeyframe]


class VitalWaveWindow(VitalModel):
    type: Literal["Wave Window"]
    interpolation_style: int
    keyframes: list[VitalWaveWindowKeyframe]
    window_shape: int


class VitalWaveWarp(VitalModel):
    type: Literal["Wave Warp"]
    horizontal_asymmetric: bool
    interpolation_style: int
    keyframes: list[VitalWaveWarpKeyframe]
    vertical_asymmetric: bool


class VitalPhaseShift(VitalModel):
    type: Literal["Phase Shift"]
    interpolation_style: int
    keyframes: list[VitalPhaseShiftKeyframe]
    style: int


class VitalSlewLimiter(VitalModel):
    type: Literal["Slew Limiter"]
    interpolation_style: int
    keyframes: list[VitalSlewLimiterKeyframe]


VitalWavetableComponent = Annotated[
    VitalWaveSource
    | VitalShepardToneSource
    | VitalAudioFileSource
    | VitalLineSource
    | VitalFrequencyFilter
    | VitalWaveFolder
    | VitalWaveWindow
    | VitalWaveWarp
    | VitalPhaseShift
    | VitalSlewLimiter,
    Field(discriminator="type"),
]


class VitalWavetableGroup(VitalModel):
    components: list[VitalWavetableComponent]


class VitalWavetable(VitalModel):
    groups: list[VitalWavetableGroup]
    name: str
    author: str | None = None
    version: str
    remove_all_dc: bool
    full_normalize: bool


class VitalSettings(VitalModel):
    controls: dict[VitalControlName, FiniteFloat]
    sample: VitalSample
    modulations: list[VitalModulation] = Field(min_length=64, max_length=64)
    wavetables: list[VitalWavetable] = Field(min_length=3, max_length=3)
    lfos: list[VitalLineGenerator] = Field(min_length=8, max_length=8)
    custom_warps: list[VitalLineGenerator] | None = Field(
        default=None, min_length=3, max_length=3
    )
    random_values: list[VitalRandomValue] | None = Field(
        default=None, min_length=3, max_length=3
    )

    @classmethod
    def from_vital_dict(cls, settings: dict[str, object]) -> VitalSettings:
        data = dict(settings)
        structural = {
            name: data.pop(name)
            for name in (
                "sample",
                "modulations",
                "wavetables",
                "lfos",
                "custom_warps",
                "random_values",
            )
            if name in data
        }
        return cls.model_validate({"controls": data, **structural})

    def to_vital_dict(self) -> dict[str, object]:
        structures = self.model_dump(
            mode="json", exclude={"controls"}, exclude_unset=True
        )
        controls = {name.value: value for name, value in self.controls.items()}
        return {**controls, **structures}

    def with_control(
        self, name: VitalControlName, value: float
    ) -> VitalSettings:
        controls = {**self.controls, name: value}
        return VitalSettings.model_validate(
            {**self.model_dump(), "controls": controls}
        )


class VitalPreset(VitalModel):
    synth_version: str
    preset_name: str | None = None
    author: str
    comments: str
    preset_style: str
    macro1: str
    macro2: str
    macro3: str
    macro4: str
    settings: VitalSettings

    @classmethod
    def from_dict(cls, preset: dict[str, object]) -> VitalPreset:
        data = dict(preset)
        settings = data.pop("settings")
        if not isinstance(settings, dict):
            raise TypeError("Vital preset settings must be a JSON object")
        return cls.model_validate(
            {**data, "settings": VitalSettings.from_vital_dict(settings)}
        )

    @classmethod
    def from_json(cls, preset_json: str | bytes) -> VitalPreset:
        preset = json.loads(preset_json)
        if not isinstance(preset, dict):
            raise TypeError("A Vital preset must be a JSON object")
        return cls.from_dict(preset)

    @classmethod
    def from_file(cls, source_path: Path) -> VitalPreset:
        return cls.from_json(source_path.resolve(strict=True).read_bytes())

    def to_dict(self) -> dict[str, object]:
        metadata = self.model_dump(
            mode="json", exclude={"settings"}, exclude_unset=True
        )
        return {**metadata, "settings": self.settings.to_vital_dict()}

    def to_json(self, *, indent: int | None = None) -> str:
        separators = (",", ":") if indent is None else None
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            allow_nan=False,
            indent=indent,
            separators=separators,
        )

    def to_json_bytes(self, *, indent: int | None = None) -> bytes:
        return self.to_json(indent=indent).encode("utf-8")

    def to_file(self, destination: Path, *, overwrite: bool = False) -> None:
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Refusing to replace Vital preset: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(self.to_json_bytes())

    def with_control(
        self, name: VitalControlName, value: float
    ) -> VitalPreset:
        return self.model_copy(update={"settings": self.settings.with_control(name, value)})

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.to_json_bytes()).hexdigest()
