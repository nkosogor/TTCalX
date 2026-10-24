import pytest


def _gpu_ok():
    try:
        import cupy as cp
        import cufinufft  # noqa: F401
        return cp.cuda.runtime.getDeviceCount() > 0
    except Exception:
        return False


GPU = _gpu_ok()
needs_gpu = pytest.mark.skipif(not GPU, reason="needs CuPy + cuFINUFFT + a CUDA device")
