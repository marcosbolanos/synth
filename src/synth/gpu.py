from __future__ import annotations

import csv
import os
import subprocess
from io import StringIO
from typing import Final

from pydantic import BaseModel, ConfigDict


A6000_UUID: Final[str] = "GPU-025e10e6-263f-d814-6dd5-added86fc8af"
A6000_NAME: Final[str] = "NVIDIA RTX A6000"
STREAM_FEATURE_GPU_UUID: Final[str] = "GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea"
RTX_2080_TI_NAME: Final[str] = "NVIDIA GeForce RTX 2080 Ti"


class NvidiaGpu(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    uuid: str
    name: str
    memory_total_mib: int


def nvidia_gpus() -> tuple[NvidiaGpu, ...]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=uuid,name,memory.total",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = csv.reader(StringIO(result.stdout), skipinitialspace=True)
    return tuple(
        NvidiaGpu(uuid=uuid, name=name, memory_total_mib=int(memory_mib))
        for uuid, name, memory_mib in rows
    )


def pin_gpu(uuid: str, expected_name: str) -> NvidiaGpu:
    matches = tuple(gpu for gpu in nvidia_gpus() if gpu.uuid == uuid)
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one GPU with UUID {uuid}")
    gpu = matches[0]
    if gpu.name != expected_name:
        raise RuntimeError(f"Expected {expected_name}, received {gpu.name}")
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu.uuid
    return gpu


def pin_a6000() -> NvidiaGpu:
    return pin_gpu(A6000_UUID, A6000_NAME)
