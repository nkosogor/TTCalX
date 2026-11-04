"""GPU CLEAN imager for OVRO-LWA all-sky data (CuPy + cuFINUFFT + python-casacore).

Products (see docs/src/gpu-clean-imager.md):
  * 10-s snapshot: CLEANed Stokes I + dirty Stokes V + QA JSON     (ttcalx_imager.snapshot)
  * 1-h and N x 10-min Stokes I (multiscale CLEAN) + V (dirty)      (ttcalx_imager.hour)

The modules that do not need a GPU (bands, msio, weights, rotate, beam, fits) can be
imported without CuPy installed.
"""
__version__ = "0.1.0"

import os as _os

# Speed settings of the 86afd89 imager patch that are NOT on by default (env, read at call time):
# the defaults keep 748884a's hour-job GPU / host memory (no chunk cache, cuFFT plans cleared
# after every interval) and its float operations where gating is cheap (LAPACK SVD, FFT local RMS,
# full-size scale FFTs, FFT residual updates).  The gsi v3 entry points (``ttcalx-image hour
# --archive``, ``cli.run_hour_snaps``, the gsi v3 engine) switch them on together, as validated
# (with the device-cache reserve and the GPU OOM fallback).  The rest of the patch (compact w-term
# factors with float32 interpolation, the Hogbom kernels) is always on: same images to float32
# rounding, validated by the 2026-10-27 regression set.  The gsi v4 speed switches (2026-11 speed rounds 1-4,
# each default off) are on in FAST_ENV too: the same images (bitwise, or the same transform / float operations
# to float32 rounding, at the GPU rerun floor).
FAST_ENV = {
    "TTCALX_CHUNK_CACHE": "store",
    "TTCALX_KEEP_FFT_PLANS": "1",
    "TTCALX_CACHE_RESERVE": "auto",
    "TTCALX_CACHE_OOM_FALLBACK": "1",
    "TTCALX_SVD_METHOD": "gram",
    "TTCALX_RMS_FILTER": "circulant",
    "TTCALX_SCALE_FFT": "small",
    "TTCALX_SPARSE_UPDATE_MAX": "64",
    "TTCALX_FAST_EXPAND": "1",
    "TTCALX_FAST_AK": "1",
    "TTCALX_HOUR_MULTI": "1",
    "TTCALX_PSF_MEMO": "1",
    "TTCALX_HOUR_SETUP_FAST": "1",
    "TTCALX_SVD_PREFETCH": "1",
    "TTCALX_FAST_COORDS": "1",
    "TTCALX_FIT_WINDOW": "512",
    "TTCALX_FAST_UPSAMPLE": "1",
    "TTCALX_NUFFT_PRUNED": "1",
    "TTCALX_FUSED_WACC": "1",
}


def use_fast_settings(env=None):
    """Set every FAST_ENV variable that is not already set (an explicit value wins); returns the
    values now in effect."""
    env = _os.environ if env is None else env
    for k, v in FAST_ENV.items():
        env.setdefault(k, v)
    return {k: env[k] for k in FAST_ENV}
