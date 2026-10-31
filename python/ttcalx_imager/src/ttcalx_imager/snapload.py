"""Host-only (numpy + casacore, no CuPy) snapshot loading for the pipelined snapshot imager.

``load_snapshot`` = read_ms + sample selection + the per-sample host arrays of SampleSet, i.e.
everything ``SnapshotImager.image`` does before the GPU, with bit-identical values.
``SharedLoader`` runs it in a separate *process* (python-casacore holds the GIL while reading,
so a reader thread would stall the GPU-issuing thread) and hands the arrays back through one
POSIX shared-memory block per snapshot (no pickling of the ~100-200 MB of samples).

Segment ownership: the child creates a block named ``ttcalx_<parent pid>_...`` and leaves it
registered with the resource tracker (shared with the parent under spawn); the parent unlinks
it in ``release``.  A block the parent never consumes (early generator close, crash, kill) is
unlinked by the tracker once both processes are gone; ``sweep_stale_shm`` removes blocks of
dead processes left by older versions.  The helper exits with its parent (PR_SET_PDEATHSIG).
"""
import ctypes
import glob
import math
import multiprocessing as mp
import os
import signal
import time
import uuid
from concurrent.futures import ProcessPoolExecutor
from types import SimpleNamespace

import numpy as np

from .msio import C, is_empty, read_ms

META = ("path", "stem", "time", "tstart", "tmid", "flag_frac", "zero_frac", "phase_dir", "freqs", "chan_width",
        "telescope", "observer", "field_name")


def prepare_samples(u, v, w, scale_deg):
    """Per-sample host arrays of image_arrays / SampleSet (same expressions, same values)."""
    sc = math.radians(scale_deg)
    flip = w < 0
    s = np.where(flip, -1.0, 1.0)
    uf, vf, wf = u * s, v * s, w * s
    return dict(flip=flip, x=(2 * np.pi * uf * sc).astype(np.float32), y=(2 * np.pi * vf * sc).astype(np.float32),
                wfold=wf.astype(np.float32), u32=u.astype(np.float32), v32=v.astype(np.float32),
                w32=w.astype(np.float32), wabsmax=float(np.abs(w).max()),
                theo_pix=1.0 / float(np.sqrt(u * u + v * v + w * w).max()) / sc)


def load_snapshot(ms_path, N, scale_deg, pols="IV", prep=True):
    """read_ms + sample selection (+ prepare_samples).  'ms' is a light metadata namespace."""
    t0 = time.perf_counter()
    ms = read_ms(ms_path, want_v="V" in pols)
    T = {"read MS": time.perf_counter() - t0}
    meta = SimpleNamespace(**{k: getattr(ms, k) for k in META})
    L = dict(ms=meta, timing=T, empty=is_empty(ms))
    if not L["empty"]:
        t1 = time.perf_counter()
        wnat = ms.natw[:, None] * ms.ok
        sel = wnat > 0
        lam = C / ms.freqs
        L["u"] = (ms.uvw[:, 0:1] / lam[None])[sel]; L["v"] = (ms.uvw[:, 1:2] / lam[None])[sel]
        L["w"] = (ms.uvw[:, 2:3] / lam[None])[sel]
        L["wnat"] = wnat[sel]; L["visI"] = ms.visI[sel]
        L["visV"] = ms.visV[sel] if "V" in pols else None
        L["pre"] = prepare_samples(L["u"], L["v"], L["w"], scale_deg) if prep else None
        T["select+prep (host)"] = time.perf_counter() - t1
    return L


_ARR = ("u", "v", "w", "wnat", "visI", "visV")
_PRE = ("flip", "x", "y", "wfold", "u32", "v32", "w32")


def _to_shm(ms_path, N, scale_deg, pols):
    """Child side: load, copy the arrays into one shared-memory block, return its layout."""
    L = load_snapshot(ms_path, N, scale_deg, pols, prep=True)
    if L["empty"]:
        return L, None, None
    arrs = [(k, L.pop(k)) for k in _ARR if L.get(k) is not None]
    pre = L["pre"]; arrs += [("pre." + k, pre.pop(k)) for k in _PRE]
    t1 = time.perf_counter()
    off = 0; lay = []
    for k, a in arrs:
        lay.append((k, a.dtype.str, a.shape, off)); off += (a.nbytes + 63) // 64 * 64
    def fill(buf):
        for (k, dt, shp, o), (_, a) in zip(lay, arrs):
            np.ndarray(shp, dt, buffer=buf, offset=o)[...] = a
    name = _new_shm(max(off, 1), fill)
    L["timing"]["to shm"] = time.perf_counter() - t1
    return L, name, lay


SHM_PREFIX = "ttcalx_"


def _new_shm(size, fill):
    """Create a block named ttcalx_<parent pid>_<random>, fill(buf) it, close it and return its
    name; unlinked again if fill fails (e.g. casacore throws on a corrupt MS)."""
    from multiprocessing import shared_memory
    shm = shared_memory.SharedMemory(name=f"{SHM_PREFIX}{os.getppid()}_{uuid.uuid4().hex[:12]}",
                                     create=True, size=size)
    try:
        fill(shm.buf)
    except BaseException:
        shm.close(); shm.unlink()
        raise
    name = shm.name
    shm.close()
    return name


def _attach(name):
    from multiprocessing import shared_memory
    return shared_memory.SharedMemory(name=name)


def _unlink(shm):
    try:
        shm.close()
    except BufferError:        # a view is still alive somewhere; unlink anyway
        pass
    try:
        shm.unlink()
    except FileNotFoundError:
        pass


def sweep_stale_shm():
    """Remove /dev/shm/ttcalx_<pid>_* blocks whose owner process is gone (daemon start-up)."""
    n = 0
    for f in glob.glob(f"/dev/shm/{SHM_PREFIX}*"):
        try:
            pid = int(os.path.basename(f)[len(SHM_PREFIX):].split("_")[0])
            os.kill(pid, 0)
        except ProcessLookupError:
            try:
                os.unlink(f); n += 1
            except OSError:
                pass
        except (ValueError, PermissionError):
            pass
    return n


def _init_child(parent_pid):
    """Helper process: die with the parent (it would otherwise be orphaned on SIGKILL, holding
    its end of the call queue and never seeing EOF)."""
    try:
        ctypes.CDLL("libc.so.6", use_errno=True).prctl(1, signal.SIGTERM)       # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    if os.getppid() != parent_pid:          # parent died before prctl took effect
        os._exit(1)


class SharedLoader:
    """load_snapshot in one helper process; result(fut) -> (L, release) with L's arrays as views
    of a shared-memory block; call release() once the arrays are no longer needed (after H2D)."""

    def __init__(self, nproc=1):
        self.nproc = nproc
        self.ex = ProcessPoolExecutor(nproc, mp_context=mp.get_context("spawn"), initializer=_init_child,
                                      initargs=(os.getpid(),))

    def restart(self):
        """New helper pool after the old one broke (helper killed / crashed on a bad MS)."""
        self.ex.shutdown(wait=False, cancel_futures=True)
        self.__init__(self.nproc)

    def submit(self, ms_path, N, scale_deg, pols="IV"):
        return self.ex.submit(_to_shm, ms_path, N, scale_deg, pols)

    @staticmethod
    def result(fut):
        L, name, lay = fut.result()
        if name is None:
            return L, (lambda: None)
        shm = _attach(name)
        pre = L["pre"]
        for k, dt, shp, o in lay:
            a = np.ndarray(shp, dt, buffer=shm.buf, offset=o)
            if k.startswith("pre."):
                pre[k[4:]] = a
            else:
                L[k] = a
        L.setdefault("visV", None)

        def release():
            for k in _ARR:
                L.pop(k, None)
            for k in _PRE:
                pre.pop(k, None)
            _unlink(shm)
        return L, release

    @staticmethod
    def discard(fut):
        """Unlink the block of a finished (unconsumed) future; cancel it if not started."""
        if fut.cancel() or not fut.done() or fut.exception() is not None:
            return
        name = fut.result()[1]
        if name is not None:
            _unlink(_attach(name))

    def shutdown(self, pending=()):
        """pending: futures never passed to result(); their blocks are unlinked."""
        for f in pending:
            if not f.cancel():
                try:
                    f.exception()           # wait for a running read
                except BaseException:
                    pass
                try:
                    self.discard(f)
                except Exception:
                    pass
        self.ex.shutdown(wait=True)


# ----------------------------------------------------------------------------- raw columns -> GPU
def _raw_to_shm(ms_path, want_v=True, column="DATA"):
    """Child side: read the main-table columns of a snapshot MS straight into one shared-memory
    block (casacore getcolnp, no intermediate copies); small columns + metadata are pickled."""
    from casacore.tables import table
    from .msio import _subtable_col
    t0 = time.perf_counter()
    t = table(ms_path, ack=False)
    nrow = t.nrows()
    if nrow == 0:
        t.close()
        raise ValueError(f"{ms_path}: main table has no rows")
    a1 = t.getcol("ANTENNA1"); a2 = t.getcol("ANTENNA2"); uvw = t.getcol("UVW"); wcol = t.getcol("WEIGHT")
    tcol = t.getcol("TIME")
    shp = t.getcell(column, 0).shape                       # (nchan, ncorr)
    big = [(column, (nrow,) + shp, np.complex64), ("FLAG", (nrow,) + shp, np.bool_)]
    off = 0; lay = []
    for k, s, dt in big:
        nb = int(np.prod(s)) * np.dtype(dt).itemsize
        lay.append((k, np.dtype(dt).str, s, off)); off += (nb + 63) // 64 * 64
    def fill(buf):
        for k, dt, s, o in lay:
            t.getcolnp(k, np.ndarray(s, dt, buffer=buf, offset=o))
    try:
        name = _new_shm(off, fill)
    finally:
        t.close()
    small = dict(a1=a1, a2=a2, uvw=uvw, wcol=wcol, time=float(tcol[0]), tstart=float(tcol.min()), tmid=float(tcol.mean()),
                 freqs=np.asarray(_subtable_col(ms_path, "SPECTRAL_WINDOW", "CHAN_FREQ")[0], np.float64),
                 chan_width=np.asarray(_subtable_col(ms_path, "SPECTRAL_WINDOW", "CHAN_WIDTH")[0], np.float64))
    pdir = _subtable_col(ms_path, "FIELD", "PHASE_DIR")[0, 0]
    names = _subtable_col(ms_path, "FIELD", "NAME", [""])
    tel = _subtable_col(ms_path, "OBSERVATION", "TELESCOPE_NAME", ["OVRO-LWA"])
    obs = _subtable_col(ms_path, "OBSERVATION", "OBSERVER", [""])
    small.update(phase_dir=(float(pdir[0]), float(pdir[1])), telescope=str(tel[0]) if len(tel) else "OVRO-LWA",
                 observer=str(obs[0]) if len(obs) else "", field_name=str(names[0]) if len(names) else "",
                 path=ms_path, stem=os.path.basename(ms_path.rstrip("/")).rsplit(".ms", 1)[0], column=column)
    return small, name, lay, time.perf_counter() - t0


class RawLoader(SharedLoader):
    """Like SharedLoader, but the child only reads the raw columns (into shared memory); the
    selection / Stokes / per-sample arrays are computed on the GPU (snapshot.gpu_select)."""

    def submit(self, ms_path, N=None, scale_deg=None, pols="IV"):
        return self.ex.submit(_raw_to_shm, ms_path, "V" in pols)

    @staticmethod
    def result(fut):
        small, name, lay, dt = fut.result()
        shm = _attach(name)
        raw = {k: np.ndarray(s, d, buffer=shm.buf, offset=o) for k, d, s, o in lay}

        def release():
            raw.clear()
            _unlink(shm)
        return small, raw, dt, release
