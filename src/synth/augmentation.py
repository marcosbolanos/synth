from __future__ import annotations

import hashlib
import re
from enum import StrEnum
from typing import Annotated

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from synth.models.vital_control_name import VitalControlName
from synth.models.vital_preset_model import VitalPreset


AUGMENTATION_STRENGTHS = (0.08, 0.14, 0.22, 0.32)


class VariationMode(StrEnum):
    LOCAL = "local"
    CORRELATED = "correlated"
    INTERPOLATED = "interpolated"
    MODULATION = "modulation"
    CATEGORICAL = "categorical"


class ControlVariationStatistics(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    minimum: dict[VitalControlName, float]
    maximum: dict[VitalControlName, float]
    robust_scale: dict[VitalControlName, float]
    continuous_controls: frozenset[VitalControlName]
    categorical_values: dict[VitalControlName, tuple[float, ...]]

    @classmethod
    def fit(
        cls,
        presets: tuple[VitalPreset, ...],
    ) -> ControlVariationStatistics:
        observed: dict[VitalControlName, list[float]] = {
            name: [] for name in VitalControlName
        }
        for preset in presets:
            for name, value in preset.settings.controls.items():
                observed[name].append(value)

        minimum: dict[VitalControlName, float] = {}
        maximum: dict[VitalControlName, float] = {}
        robust_scale: dict[VitalControlName, float] = {}
        continuous_controls: set[VitalControlName] = set()
        categorical_values: dict[VitalControlName, tuple[float, ...]] = {}
        for name, values in observed.items():
            if not values:
                continue
            array = np.asarray(values, dtype=np.float64)
            minimum[name] = float(array.min())
            maximum[name] = float(array.max())
            quantile_10, quantile_90 = np.quantile(array, (0.1, 0.9))
            robust_scale[name] = float((quantile_90 - quantile_10) / 2.563)
            unique = np.unique(array)
            integer_valued = bool(np.allclose(unique, np.round(unique), atol=1e-6))
            if (
                maximum[name] > minimum[name]
                and robust_scale[name] > 1e-8
                and not (integer_valued and len(unique) <= 32)
            ):
                continuous_controls.add(name)
            elif 1 < len(unique) <= 32:
                categorical_values[name] = tuple(float(value) for value in unique)
        return cls(
            minimum=minimum,
            maximum=maximum,
            robust_scale=robust_scale,
            continuous_controls=frozenset(continuous_controls),
            categorical_values=categorical_values,
        )

    @staticmethod
    def _preset(
        source: VitalPreset,
        controls: dict[VitalControlName, float],
        *,
        mode: VariationMode,
        strength: float,
        changed: tuple[VitalControlName, ...],
    ) -> VitalPreset:
        settings = source.settings.model_copy(update={"controls": controls})
        return source.model_copy(
            update={
                "preset_name": f"{source.preset_name or 'Preset'} · {mode.value} variation",
                "author": "synth augmentation",
                "comments": (
                    f"Controlled {mode.value} variation; strength={strength:.3f}; "
                    f"changed_controls={len(changed)}"
                ),
                "settings": settings,
            }
        )

    def vary(
        self,
        preset: VitalPreset,
        *,
        strength: Annotated[float, Field(gt=0.0, le=1.0)],
        seed: int,
    ) -> tuple[VitalPreset, tuple[VitalControlName, ...]]:
        if not 0.0 < strength <= 1.0:
            raise ValueError(f"Strength must be in (0, 1], received {strength}")
        rng = np.random.default_rng(seed)
        eligible = tuple(
            name
            for name in preset.settings.controls
            if name in self.continuous_controls
        )
        if not eligible:
            raise ValueError("Preset has no continuous controls eligible for augmentation")

        selection = rng.random(len(eligible)) < 0.08
        if not selection.any():
            selection[int(rng.integers(0, len(eligible)))] = True
        controls = dict(preset.settings.controls)
        changed: list[VitalControlName] = []
        for selected, name in zip(selection, eligible, strict=True):
            if not selected:
                continue
            original = controls[name]
            delta = float(rng.normal()) * strength * self.robust_scale[name]
            varied = float(
                np.clip(
                    original + delta,
                    self.minimum[name],
                    self.maximum[name],
                )
            )
            if varied != original:
                controls[name] = varied
                changed.append(name)
        if not changed:
            name = eligible[int(rng.integers(0, len(eligible)))]
            direction = 1.0 if controls[name] < self.maximum[name] else -1.0
            varied = controls[name] + direction * strength * self.robust_scale[name]
            controls[name] = float(
                np.clip(varied, self.minimum[name], self.maximum[name])
            )
            changed.append(name)

        result = self._preset(
            preset,
            controls,
            mode=VariationMode.LOCAL,
            strength=strength,
            changed=tuple(changed),
        )
        return result, tuple(changed)

    def vary_correlated(
        self,
        preset: VitalPreset,
        *,
        strength: float,
        seed: int,
    ) -> tuple[VitalPreset, tuple[VitalControlName, ...]]:
        rng = np.random.default_rng(seed)
        groups: dict[str, list[VitalControlName]] = {}
        for name in preset.settings.controls:
            if name not in self.continuous_controls:
                continue
            match = re.match(r"^(osc|filter|env|lfo|random)_(\d+)_", name.value)
            group = "_".join(match.groups()) if match else name.value.split("_", 1)[0]
            groups.setdefault(group, []).append(name)
        eligible_groups = tuple(
            names for names in groups.values() if len(names) >= 2
        )
        if not eligible_groups:
            raise ValueError("Preset has no correlated control group")
        names = eligible_groups[int(rng.integers(0, len(eligible_groups)))]
        shared_direction = float(rng.normal())
        controls = dict(preset.settings.controls)
        changed = []
        for name in names:
            if rng.random() >= 0.45:
                continue
            delta = (
                shared_direction + float(rng.normal(scale=0.25))
            ) * strength * self.robust_scale[name]
            varied = float(
                np.clip(
                    controls[name] + delta,
                    self.minimum[name],
                    self.maximum[name],
                )
            )
            if varied != controls[name]:
                controls[name] = varied
                changed.append(name)
        if not changed:
            name = names[int(rng.integers(0, len(names)))]
            direction = 1.0 if controls[name] < self.maximum[name] else -1.0
            controls[name] = float(
                np.clip(
                    controls[name] + direction * strength * self.robust_scale[name],
                    self.minimum[name],
                    self.maximum[name],
                )
            )
            changed.append(name)
        result = self._preset(
            preset,
            controls,
            mode=VariationMode.CORRELATED,
            strength=strength,
            changed=tuple(changed),
        )
        return result, tuple(changed)

    def interpolate(
        self,
        preset: VitalPreset,
        donor: VitalPreset,
        *,
        strength: float,
        seed: int,
    ) -> tuple[VitalPreset, tuple[VitalControlName, ...]]:
        rng = np.random.default_rng(seed)
        eligible = tuple(
            name
            for name, value in preset.settings.controls.items()
            if name in self.continuous_controls
            and name in donor.settings.controls
            and donor.settings.controls[name] != value
        )
        if not eligible:
            raise ValueError("Preset and donor share no differing continuous controls")
        controls = dict(preset.settings.controls)
        selection = rng.random(len(eligible)) < 0.12
        if not selection.any():
            selection[int(rng.integers(0, len(eligible)))] = True
        changed = []
        for selected, name in zip(selection, eligible, strict=True):
            if not selected:
                continue
            controls[name] = float(
                controls[name]
                + strength * (donor.settings.controls[name] - controls[name])
            )
            changed.append(name)
        result = self._preset(
            preset,
            controls,
            mode=VariationMode.INTERPOLATED,
            strength=strength,
            changed=tuple(changed),
        )
        return result, tuple(changed)

    def vary_modulation(
        self,
        preset: VitalPreset,
        *,
        strength: float,
        seed: int,
    ) -> tuple[VitalPreset, tuple[VitalControlName, ...]]:
        rng = np.random.default_rng(seed)
        eligible = []
        for name in preset.settings.controls:
            match = re.fullmatch(r"modulation_(\d+)_amount", name.value)
            if match is None or name not in self.continuous_controls:
                continue
            route = preset.settings.modulations[int(match.group(1)) - 1]
            if route.source and route.destination:
                eligible.append(name)
        if not eligible:
            return self.vary(preset, strength=strength, seed=seed)
        controls = dict(preset.settings.controls)
        count = min(len(eligible), max(1, int(rng.integers(1, 4))))
        selected = rng.choice(len(eligible), size=count, replace=False)
        changed = []
        for index in selected:
            name = eligible[int(index)]
            varied = float(
                np.clip(
                    controls[name]
                    + float(rng.normal()) * strength * self.robust_scale[name],
                    self.minimum[name],
                    self.maximum[name],
                )
            )
            if varied != controls[name]:
                controls[name] = varied
                changed.append(name)
        if not changed:
            return self.vary(preset, strength=strength, seed=seed)
        result = self._preset(
            preset,
            controls,
            mode=VariationMode.MODULATION,
            strength=strength,
            changed=tuple(changed),
        )
        return result, tuple(changed)

    def vary_categorical(
        self,
        preset: VitalPreset,
        *,
        seed: int,
    ) -> tuple[VitalPreset, tuple[VitalControlName, ...]]:
        rng = np.random.default_rng(seed)
        excluded = {
            VitalControlName.BYPASS,
            VitalControlName.BEATS_PER_MINUTE,
        }
        eligible = tuple(
            name
            for name, value in preset.settings.controls.items()
            if name in self.categorical_values
            and name not in excluded
            and any(candidate != value for candidate in self.categorical_values[name])
        )
        if not eligible:
            return self.vary(preset, strength=0.14, seed=seed)
        name = eligible[int(rng.integers(0, len(eligible)))]
        original = preset.settings.controls[name]
        alternatives = tuple(
            value for value in self.categorical_values[name] if value != original
        )
        controls = dict(preset.settings.controls)
        controls[name] = alternatives[int(rng.integers(0, len(alternatives)))]
        changed = (name,)
        result = self._preset(
            preset,
            controls,
            mode=VariationMode.CATEGORICAL,
            strength=1.0,
            changed=changed,
        )
        return result, changed


def variation_seed(parent_sha256: str, variation_index: int) -> int:
    digest = hashlib.sha256(
        f"vital-augmentation-v1:{parent_sha256}:{variation_index}".encode()
    ).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)
