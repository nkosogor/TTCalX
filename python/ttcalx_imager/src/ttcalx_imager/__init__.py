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


# v5 speed switches (2026-11-24), default off: not in FAST_ENV.  TTCALX_V5=1 in the environment of a process that
# calls use_fast_settings (the gsi v3 engine; gsi: imaging.gpu_env: {TTCALX_V5: "1"}) switches them on together.
# Snapshot: zenith-frame gridding with the per-snapshot router (zenith.py; same products to the gridding accuracy,
# headers equal except the restoring beam).  TTCALX_PINNED_H2D (snapshot.h2d's page-locked staging buffer) is NOT
# in the set: no measured speedup (CuPy 13.6 already stages pageable host arrays through pinned memory); opt-in.
SNAP_V5_ENV = {
    "TTCALX_SNAP_ZEN": "1",
}


def v5_env():
    """The variables TTCALX_V5=1 sets (snapshot and hour-stage v5 switches)."""
    return dict(SNAP_V5_ENV, **HOUR_V5_ENV)


def use_fast_settings(env=None):
    """Set every FAST_ENV variable that is not already set (an explicit value wins); with TTCALX_V5=1 also the v5
    switches (v5_env()).  Returns the values now in effect."""
    env = _os.environ if env is None else env
    want = dict(FAST_ENV)
    if env.get("TTCALX_V5", "0") == "1":
        want.update(v5_env())
    for k, v in want.items():
        env.setdefault(k, v)
    return {k: env[k] for k in want}


# Hour-stage v5 speed switches (2026-11 "pexact" set; each default off, env read at call time), validated on top of
# FAST_ENV (TTCALX_FUSED_CAK needs TTCALX_FAST_AK + TTCALX_FUSED_WACC + TTCALX_NUFFT_PRUNED): the same hour images at
# the GPU rerun floor (55 / 73 MHz op_hour vs production, 2026-11-24; TTCALX_UVMAX_GPU and TTCALX_WMAX_MEMO
# bit-identical), ~1.2x faster at 55-73 MHz.  TTCALX_MS_INCR (needs TTCALX_SPARSE_UPDATE_MAX) is NOT in the set: at
# 73 MHz, with the other switches, it moves I-Deep-NoTaper-Robust-0 reproducibly (residual 0.067 sigma, model flux
# +0.6 %) for ~7 s; alone it is not faster.
# Not in FAST_ENV: use_hour_v5_settings() (or TTCALX_V5=1 with use_fast_settings) turns them on.
HOUR_V5_ENV = {
    "TTCALX_REPROJ_FAST": "1",
    "TTCALX_SVD_QUANTUM": "0.25",
    "TTCALX_FUSED_CAK": "1",
    "TTCALX_WMAX_MEMO": "1",
    "TTCALX_UVMAX_GPU": "1",
}


def use_hour_v5_settings(env=None):
    """Set every FAST_ENV and HOUR_V5_ENV variable that is not already set (an explicit value wins); returns the
    HOUR_V5_ENV values now in effect."""
    env = _os.environ if env is None else env
    use_fast_settings(env)
    for k, v in HOUR_V5_ENV.items():
        env.setdefault(k, v)
    return {k: env[k] for k in HOUR_V5_ENV}
