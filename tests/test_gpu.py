from synth.gpu import (
    A6000_NAME,
    A6000_UUID,
    STREAM_FEATURE_GPU_UUID,
    NvidiaGpu,
    pin_a6000,
)


def test_pin_a6000_uses_uuid(monkeypatch) -> None:
    gpu = NvidiaGpu(uuid=A6000_UUID, name=A6000_NAME, memory_total_mib=49_140)
    monkeypatch.setattr("synth.gpu.nvidia_gpus", lambda: (gpu,))

    assert pin_a6000() == gpu
    assert __import__("os").environ["CUDA_VISIBLE_DEVICES"] == A6000_UUID


def test_stream_feature_gpu_avoids_reserved_gpu() -> None:
    assert STREAM_FEATURE_GPU_UUID == "GPU-a3d54441-08da-ed66-fa0f-6aaaaf97baea"
    assert STREAM_FEATURE_GPU_UUID != "GPU-fcb13561-e5da-20e1-2ff7-a9bdbfa68c26"
