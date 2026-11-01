"""Run ``hour.image_hour_snaps`` on the CPU (no CuPy device, no cuFINUFFT) to compare inputs.

Everything image_hour_snaps does itself runs for real (sorting, QA drops, interval groups,
HourGrid weights with the real WeightGrid, the chgcentre rotation, the chunk cache in every
TTCALX_CHUNK_CACHE mode, the PSF visibilities, the major-cycle residual), with numpy standing in
for CuPy.  Only the GPU kernels are replaced: SampleSet by a deterministic CPU binning
"gridder" that records a digest of every (x, y, w, wt) and every gridded visibility vector, and
the multiscale CLEAN by one residual call.  Two runs whose call logs and images are equal give
the real gridder bit-identical inputs (the GPU products can then only differ by cuFINUFFT's
floating-point atomics).
"""
import contextlib
import hashlib
import importlib
import sys
import types
from functools import partial
from types import SimpleNamespace

import numpy as np


def _np_cupy(free_bytes):
    m = types.ModuleType("np_cupy")
    m.__getattr__ = lambda k: getattr(np, k)          # module-level __getattr__ (PEP 562)
    m.asarray = np.asarray
    m.asnumpy = lambda a, *x, **k: np.asarray(a)
    m.ndarray = np.ndarray
    m.cuda = SimpleNamespace(Device=lambda *a: SimpleNamespace(synchronize=lambda: None),
                             runtime=SimpleNamespace(memGetInfo=lambda: (free_bytes[0], 16e9)))
    m.get_default_memory_pool = lambda: SimpleNamespace(total_bytes=lambda: 0, used_bytes=lambda: 0,
                                                        free_all_blocks=lambda: None)
    m.fft = SimpleNamespace(config=SimpleNamespace(get_plan_cache=lambda: SimpleNamespace(
        set_memsize=lambda n: None, clear=lambda: None)), fftfreq=np.fft.fftfreq)
    return m


def import_hour():
    """ttcalx_imager.hour, importable without cuFINUFFT / a CUDA device (stub modules for what
    is missing; the stubs are never called by the CPU runs)."""
    try:
        return importlib.import_module("ttcalx_imager.hour")
    except Exception:
        pass
    for name in ("cufinufft",):
        sys.modules.setdefault(name, types.ModuleType(name))
    try:
        import cupy  # noqa: F401
        import cupyx.scipy.ndimage  # noqa: F401
    except Exception:
        fake = _np_cupy([0])
        sys.modules["cupy"] = fake
        for n in ("cupyx", "cupyx.scipy", "cupyx.scipy.ndimage"):
            sys.modules[n] = types.ModuleType(n)
        sys.modules["cupyx"].scipy = sys.modules["cupyx.scipy"]
        sys.modules["cupyx.scipy"].ndimage = sys.modules["cupyx.scipy.ndimage"]
    for k in [k for k in sys.modules if k.startswith("ttcalx_imager.") and k.split(".")[-1] in
              ("hour", "gridder", "clean", "reproject", "snapshot")]:
        del sys.modules[k]
    return importlib.import_module("ttcalx_imager.hour")


def _h(*arrays):
    d = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        d.update(a.dtype.str.encode()); d.update(str(a.shape).encode()); d.update(a.tobytes())
    return d.hexdigest()[:16]


class Recorder:
    def __init__(self):
        self.calls = []
        self.images = {}


def _fake_sampleset(rec, xp):
    class FakeSampleSet:
        """Folds w >= 0 exactly as gridder.SampleSet (device branch); 'grids' by binning."""

        def __init__(self, u, v, w, wt, N, scale_rad, svd=None, eps=1e-4, precompute=True, pre=None):
            flip = w < 0
            s = xp.where(flip, -1.0, 1.0).astype(xp.float32)
            u, v, w = u * s, v * s, w * s
            self.flip = flip
            self.x = (2 * np.pi * scale_rad) * u
            self.y = (2 * np.pi * scale_rad) * v
            self.w = w
            self.wt = xp.asarray(wt, dtype=xp.float32)
            self.N = N
            self.npts = int(self.w.size)
            ix = np.floor(np.asarray(self.x, np.float64) * (N / (2 * np.pi))).astype(np.int64) % N
            iy = np.floor(np.asarray(self.y, np.float64) * (N / (2 * np.pi))).astype(np.int64) % N
            self._idx = ix * N + iy
            rec.calls.append(("SampleSet", self.npts, _h(self.x, self.y, self.w, self.wt, self.flip)))

        def grid(self, vis):
            c = np.where(self.flip, np.conj(vis), vis) * self.wt
            rec.calls.append(("grid", _h(vis)))
            re = np.bincount(self._idx, c.real.astype(np.float64), minlength=self.N * self.N)
            im = np.bincount(self._idx, c.imag.astype(np.float64), minlength=self.N * self.N)
            return (re + 1e-3 * im + 1.0).astype(np.float32).reshape(self.N, self.N)   # + 1: PSF peak != 0

        def grid_psf(self):
            rec.calls.append(("grid_psf",))
            return (np.bincount(self._idx, self.wt.astype(np.float64), minlength=self.N * self.N) + 1.0
                    ).astype(np.float32).reshape(self.N, self.N)          # + 1: the PSF peak is never 0

        def degrid(self, model):
            rec.calls.append(("degrid", _h(model)))
            q = np.asarray(model, np.float32).ravel()[self._idx].astype(np.complex64)
            return np.where(self.flip, np.conj(q), q)
    return FakeSampleSet


@contextlib.contextmanager
def cpu_hour(cache="auto", free_gb=1000.0, N=None):
    """Context in which ttcalx_imager.hour (image_hour, image_hour_snaps, the CLI) runs on the
    CPU; yields the Recorder.  cache: the TTCALX_CHUNK_CACHE mode; free_gb: the 'free GPU memory'
    the chunk-cache decision sees; N: image size override (default: the band's)."""
    import os
    hour = import_hour()
    from ttcalx_imager import weights
    rec = Recorder()
    fcp = _np_cupy([free_gb * 1e9])
    saved = {k: getattr(hour, k) for k in ("cp", "SampleSet", "get_svd", "to_ws", "deconvolve_multiscale",
                                           "write_image", "WeightGrid")}
    geo = hour.bands.geometry
    old_env = os.environ.get("TTCALX_CHUNK_CACHE")

    def deconv(dirty, psf, sdeg, theo, params, residual_fn=None, beam=None, log=None):
        m = (dirty * np.float32(0.5)).astype(np.float32)
        r = residual_fn(m)
        return dirty, r, m, dict(beam=(0.1, 0.1, 0.0), nit=1, nmajor=1)

    def write_image(name, img, h, pol):
        rec.images[os.path.basename(name)] = np.array(img)
        return name

    try:
        hour.cp = fcp
        hour.SampleSet = _fake_sampleset(rec, fcp)
        hour.get_svd = lambda *a, **k: SimpleNamespace(K=1)
        hour.to_ws = lambda x: x.T[::-1]
        hour.deconvolve_multiscale = deconv
        hour.write_image = write_image
        hour.WeightGrid = partial(weights.WeightGrid, xp=fcp)
        if N is not None:
            hour.bands.geometry = lambda b: (N, geo(b)[1])
        os.environ["TTCALX_CHUNK_CACHE"] = cache
        yield rec
    finally:
        for k, v in saved.items():
            setattr(hour, k, v)
        hour.bands.geometry = geo
        if old_env is None:
            os.environ.pop("TTCALX_CHUNK_CACHE", None)
        else:
            os.environ["TTCALX_CHUNK_CACHE"] = old_env


def run(snaps, band, out_prefix, ra, dec, products, cache="auto", free_gb=1000.0, mode="vis", N=None, **kw):
    """image_hour_snaps(snaps, ...) on the CPU; returns (summary, Recorder)."""
    with cpu_hour(cache, free_gb, N) as rec:
        s = import_hour().image_hour_snaps(snaps, band, out_prefix, ra, dec, products=products, mode=mode, **kw)
    return s, rec


def comparable(summary):
    """The summary without timings, as JSON text (NaN == NaN)."""
    import json
    return json.dumps({k: v for k, v in summary.items() if k != "timing"}, sort_keys=True, default=float)
