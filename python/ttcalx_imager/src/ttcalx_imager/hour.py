"""1-h and N x 10-min Stokes I (multiscale CLEAN) + V (dirty) images from peeled snapshot MSes.

Equivalent of the orca main-pipeline imaging of one band-hour (concat -> chgcentre -> pilot V
QA -> wsclean I-Deep-Taper-Robust-0 / I-Taper-10min / V-Taper-Deep / V-Taper-10min), without
the concatenation and AOFlagger steps:

  1. load every snapshot (CPU worker processes) and compute its exact chgcentre rotation R
     to the hour phase centre (rotate.py);
  2. QA: per-snapshot dirty V rms (from snapshot-mode QA JSON or computed here), reject by the
     orca median + 3 MAD rule (qa.py);
  3. one global Briggs/Tukey density grid over all accepted samples of an interval, in the hour
     frame (= WSClean weighting of the concatenated, re-phased MS);
  4. w-snapshot gridding (default): blocks of ``block`` consecutive snapshots are gridded in the
     frame of their middle snapshot (small w range, K ~ 6-8 SVD terms instead of ~37 in the hour
     frame) with the global weights, reprojected to the hour SIN grid and summed with weight
     sum_w(block).  The PSF (unit point source at the phase centre, with the w-term, as WSClean)
     is imaged the same way;
  5. multiscale CLEAN with WSClean major cycles: the band-limited hour model is reprojected
     into each block frame (times the SIN flux Jacobian n_hour / n_block) and degridded there;
  6. restore with the WSClean beam fit of the hour PSF.

``mode='vis'`` grids everything directly in the hour frame instead (slower, same result).
"""
import gc
import json
import math
import multiprocessing as mp
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cupy as cp

from . import bands
from .clean import deconvolve_multiscale
from .fits import make_header, write_image
from .gridder import SampleSet, get_svd, to_ws
from .gsiv import fast_expand
from .msio import load_samples
from .qa import qa_reject
from .reproject import Reprojector, fft_upsample, prefilter
from .rotate import parse_phase_centre, rotate_samples  # noqa: F401 (parse_phase_centre: API)
from .weights import WeightGrid


# ----------------------------------------------------------------------------- loading
def middle_zenith(paths):
    """PHASE_DIR of the middle snapshot (the snapshots are phased to zenith)."""
    from casacore.tables import table
    p = sorted(paths)[len(paths) // 2]
    t = table(p + "/FIELD", ack=False); pd = t.getcol("PHASE_DIR")[0, 0]; t.close()
    return float(pd[0]), float(pd[1])


def load_all(paths, ra, dec, nproc=8, want_v=True):
    """Load snapshots in worker processes (spawned: CUDA must not be forked)."""
    n = len(paths)
    if nproc <= 1:
        return [load_samples(p, ra, dec, want_v) for p in paths]
    with ProcessPoolExecutor(nproc, mp_context=mp.get_context("spawn")) as ex:
        return list(ex.map(load_samples, paths, [ra] * n, [dec] * n, [want_v] * n, chunksize=2))


def _nsamp(s):
    """Kept samples of a snapshot without expanding a LazySnap (gsiv.py)."""
    n = getattr(s, "n", None)
    return int(n) if n is not None else len(s["u"])


def _with(s, **kw):
    """Copy of a snapshot mapping with some keys replaced (a LazySnap stays lazy)."""
    if hasattr(s, "derive"):
        return s.derive(**kw)
    d = dict(s); d.update(kw)
    return d


def _uvmax(s):
    """max |(u, v, w)| [lambda] of a snapshot, float32 as computed here before (a LazySnap
    computes it once and keeps the scalar instead of re-expanding u, v, w every time)."""
    if hasattr(s, "uvmax"):
        return s.uvmax()
    return float(np.sqrt(s["u"] ** 2 + s["v"] ** 2 + s["w"] ** 2).max())


_DENSITY_MEMO = {}      # TTCALX_HOUR_SETUP_FAST: (snapshot set, geometry) -> WeightGrid density grid (one entry)


# ----------------------------------------------------------------------------- chunk cache control
class CacheCtl:
    """Device chunk-cache policy of one hour interval, shared by its HourGrid and the w-snapshot
    block grids.  device=False: no chunk is kept on the GPU (the mode's host / store behaviour as
    when a chunk does not fit); reserve: bytes to leave free besides TTCALX_CACHE_MARGIN_GB (the
    larger of the two counts); dev_bytes: what this interval has cached on the GPU."""

    def __init__(self, device=True, reserve=0.0):
        self.device = device; self.reserve = float(reserve); self.dev_bytes = 0


def deconv_reserve_bytes(N, theo_pix, params):
    """Upper estimate [bytes] of the GPU memory the multiscale CLEAN of an N x N interval needs on
    top of the gridding (clean.MultiScaleClean + the major cycles): the P x P real-FFT kernels of
    every scale (S_s and S_s * PSF, P = fast_len(1.5 N + 2)) and of the scale images, the per-scale
    N x N images and masks of the minor loop, the 2x band-limited model of the major cycle, the
    padded FFT temporaries and the cuFFT plan cache (TTCALX_FFT_PLAN_CACHE_GB), times 1.25 for what
    the CuPy pool does not see (cuFINUFFT plans) and pool fragmentation: at 64 MHz the cache left
    the unscaled 7.1 GB free and the first major cycle still ran out of memory (a 312 MB model
    upsampling), while an 8 GB margin was enough for every product.  At 3122 px (64-82 MHz, 7-8
    scales): 8.8-9.5 GB."""
    from .clean import fast_len
    scales = [0]
    if params.multiscale:
        s = 4 * theo_pix
        while s < N * 0.5:
            scales.append(s); s *= 2
    if params.multiscale_max_scales:
        scales = scales[:params.multiscale_max_scales]
    nsc = len(scales)
    P = fast_len(N + N // 2 + 2); F = P * (P // 2 + 1) * 8
    hmax = max(int(math.ceil(0.5 * s)) for s in scales)
    Ps = fast_len(N + (2 * hmax + 1) // 2 + 2); Fs = Ps * (Ps // 2 + 1) * 8
    I = N * N * 4
    kernels = (2 * nsc + 1) * F + max(nsc - 1, 0) * Fs                  # KS, KP, Kpsf; KSs
    minor = nsc * (3 * I + 3 * N * N) + nsc * int(2.25 * I) + 10 * I     # Pss, R_s, |R_s|, masks; full_ps; R, M, ...
    fft_tmp = 2 * P * P * 4 + 2 * F
    major = 40 * I                                                       # fft_upsample (2N)^2 complex, prefilter, coords
    plans = float(os.environ.get("TTCALX_FFT_PLAN_CACHE_GB", "1.5")) * 1e9
    return 1.25 * (kernels + minor + fft_tmp + major + plans)


def _reserve(N, sn, sc, params):
    """TTCALX_CACHE_RESERVE: unset / 0 (default: the margin only), 'auto' (deconv_reserve_bytes)
    or GB."""
    v = os.environ.get("TTCALX_CACHE_RESERVE", "0").strip().lower()
    if v in ("", "0", "none", "off"):
        return 0.0
    if v == "auto":
        return float(deconv_reserve_bytes(N, theoretical_beam_pix(sn, sc), params))
    return float(v) * 1e9


def _is_gpu_oom(e):
    """A GPU allocation failure (CuPy pool, cuFFT plan / workspace, cuFINUFFT plan) -- not a CUDA
    fatal error, which a retry in the same process cannot help."""
    oom = getattr(getattr(getattr(cp, "cuda", None), "memory", None), "OutOfMemoryError", None)
    if oom is not None and isinstance(e, oom):
        return True
    m = str(e)
    return ("CUFFT_ALLOC_FAILED" in m or "cudaErrorMemoryAllocation" in m or "out of memory" in m.lower()) and \
        "illegal" not in m.lower()


# ----------------------------------------------------------------------------- gridding
class HourGrid:
    """Samples of many snapshots (kept on the host) gridded in chunks into one image.

    rotate=True: each snapshot's stored-frame (u, v, w, vis) are rotated by its R (snapshot
    dict 'R') on the fly.  weights=None: one global Briggs grid over all snapshots."""

    def __init__(self, snaps, N, scale_deg, robust=0.0, tukey_inner=None, eps=1e-4, svd_tol=1e-4,
                 chunk_pts=40_000_000, rotate=True, weights=None, need_svd=True, log=None, ctl=None):
        self.snaps = snaps; self.N = N; self.sdeg = scale_deg; self.sc = math.radians(scale_deg)
        self.eps = eps; self.chunk_pts = chunk_pts; self.rotate = rotate
        self._cache = {}
        self.ctl = CacheCtl() if ctl is None else ctl
        log = log or (lambda *a: None)
        t0 = time.perf_counter()

        def rot_uvw(i):
            u, v, w, _ = self._rot([i], *self._dev3(i))
            return u, v, w

        # TTCALX_HOUR_SETUP_FAST=1 (default 0): the density grid of a snapshot set is computed once and re-used by
        # every product with the same snapshots (it depends on u, v and the natural weights only; robust and the
        # taper enter in finalize / weights), and the weights, their sum and wmax come from one rotation pass
        fast = os.environ.get("TTCALX_HOUR_SETUP_FAST", "0") == "1"
        sw = 0.0; wmax = 0.0; done = False
        if weights is None:
            WG = WeightGrid(N, self.sc, robust, tukey_inner)
            key = None
            if fast:
                key = (tuple(s["stem"] for s in snaps), N, round(self.sc, 15), WG.npix,
                       b"".join(np.asarray(s["R"], np.float64).tobytes() for s in snaps))
            if key is not None and key in _DENSITY_MEMO:
                WG.grid = _DENSITY_MEMO[key]
            else:
                for i, s in enumerate(snaps):
                    u, v, _ = rot_uvw(i); WG.add(u, v, self._dev(i, "nw"))
                if key is not None:
                    _DENSITY_MEMO.clear(); _DENSITY_MEMO[key] = WG.grid
            WG.finalize()
            self.iw = []
            for i, s in enumerate(snaps):
                u, v, w = rot_uvw(i); self.iw.append(cp.asnumpy(WG.weights(u, v, self._dev(i, "nw"))))
                if fast:
                    sw += float(self.iw[i].sum()); wmax = max(wmax, float(cp.abs(w).max()))
            done = fast
        else:
            self.iw = weights
        if not done:
            for i in range(len(snaps)):
                _, _, w = rot_uvw(i)
                sw += float(self.iw[i].sum()); wmax = max(wmax, float(cp.abs(w).max()))
        self.sumw = sw
        self.svd = get_svd(N, self.sc, wmax, svd_tol) if need_svd else None
        self.chunks = self._subchunks(range(len(snaps)))
        self.npts = sum(self._n(i) for i in range(len(snaps)))
        log(f"HourGrid: {len(snaps)} snaps, {self.npts / 1e6:.1f} M samples, {len(self.chunks)} chunks, "
            f"K={self.svd.K if self.svd else 0} (wmax {wmax:.0f}), {time.perf_counter() - t0:.1f} s")

    def _n(self, i):
        """Kept samples of snapshot i (record n for a LazySnap: no expansion)."""
        return _nsamp(self.snaps[i])

    def _dev(self, i, k):
        """Sample array k of snapshot i on the GPU.  A LazySnap (gsiv.py) expands u, v, w, nw on
        the GPU from its per-row arrays (msio.expand_coord / expand_nw: one float64 multiply, a
        mask and a cast, so the values equal the host expansion bit for bit) instead of expanding
        on the host and copying 4 x 4 B/sample."""
        s = self.snaps[i]
        if k in ("u", "v", "w", "nw") and hasattr(s, "expand"):
            return s.expand(k, cp)
        return cp.asarray(s[k])

    def _cat_dev(self, ids, k):
        return cp.concatenate([self._dev(i, k) for i in ids])

    def _dev3(self, i):
        """(u, v, w) of snapshot i on the GPU (= _dev(i, 'u'), ...; TTCALX_FAST_EXPAND: one transfer)."""
        s = self.snaps[i]
        if hasattr(s, "expand_uvw") and fast_expand():
            return s.expand_uvw(cp)
        return tuple(self._dev(i, k) for k in ("u", "v", "w"))

    def _cat_dev3(self, ids):
        parts = [self._dev3(i) for i in ids]
        return tuple(cp.concatenate([p[k] for p in parts]) for k in range(3))

    def _subchunks(self, ids):
        out = []; cur = []; n = 0
        for i in ids:
            if cur and n + self._n(i) > self.chunk_pts:
                out.append(cur); cur = []; n = 0
            cur.append(i); n += self._n(i)
        if cur:
            out.append(cur)
        return out

    def _rot(self, ids, u, v, w, vis=None):
        """Stored-frame coordinates of snapshots ``ids`` (concatenated) -> common frame (chgcentre)."""
        if not self.rotate:
            return u, v, w, vis
        out = [[], [], [], []]; o = 0
        for i in ids:
            n = self._n(i); sl = slice(o, o + n)
            r = rotate_samples(u[sl], v[sl], w[sl], self.snaps[i]["R"], None if vis is None else vis[sl], xp=cp)
            for lst, a in zip(out, r):
                lst.append(a)
            o += n
        u, v, w = (cp.concatenate(x) for x in out[:3])
        return u, v, w, (cp.concatenate(out[3]) if vis is not None else None)

    def _psf_vis(self, i):
        """Stored-frame visibilities of a unit point source at the hour phase centre (column 'PSF'):
        exp(2 pi i (w_hour - w_stored)), so that the chgcentre re-phasing in _rot turns them into
        1 in the hour frame (exp(2 pi i (w_hour - w_block)) in a w-snapshot block frame)."""
        s = self.snaps[i]
        Rh = s.get("R_hour", s["R"])
        memo = getattr(s, "_memo", None) if os.environ.get("TTCALX_PSF_MEMO", "0") == "1" else None
        key = ("psf_vis", np.asarray(Rh, np.float64).tobytes())
        if memo is not None and key in memo:
            # TTCALX_PSF_MEMO=1 (default 0): the snapshot's PSF visibilities depend only on its samples and R_hour,
            # so every product / interval re-uses the first computation (kept on the host, ~8 B per sample)
            return cp.asarray(memo[key])
        u, v, w = (x.astype(cp.float64) for x in self._dev3(i))
        wh = Rh[2, 0] * u + Rh[2, 1] * v + Rh[2, 2] * w
        out = cp.exp(2j * np.pi * (wh - w)).astype(cp.complex64)
        if memo is not None:
            memo[key] = cp.asnumpy(out)
        return out

    def _chunk(self, ids, col):
        """(SampleSet, rotated visibilities of ``col``) of the snapshots ``ids``.

        Chunk cache (env TTCALX_CHUNK_CACHE = off (default, as 748884a) | auto | device | host | store): the
        dirty image, the PSF, every major cycle and V of one interval re-use the same samples, so
        the first call keeps them -- on the GPU (rotated, folded SampleSet + rotated I + unrotated
        w, ~29 B/sample) if that leaves TTCALX_CACHE_MARGIN_GB (default 5) free, else the
        host-side concatenations (no repeated np.concatenate, ~1.3 s per call for 60 snapshots at
        55 MHz).  store (gsi v3 archive / hour store snapshots): the device cache as auto, but no
        host cache -- a chunk that is not cached on the GPU is rebuilt per call from the store,
        u, v, w expanded on the GPU from the per-row uvw (no ~24 B/sample host duplicate).
        The arrays and operations are the same as without the cache (bit-identical images).
        self.ctl (CacheCtl): device=False keeps nothing on the GPU (OOM fallback of
        image_hour_snaps); the free memory to leave is max(margin, ctl.reserve)."""
        mode = os.environ.get("TTCALX_CHUNK_CACHE", "off")
        store = mode == "store"
        key = tuple(ids)
        c = self._cache.get(key) if mode != "off" else None
        if c is not None and "S" in c:                       # device cache
            S = c["S"]
            if col in c["vis"]:
                return S, c["vis"][col]
            if not col:
                return S, None
            if col == "PSF":
                vis = cp.concatenate([self._psf_vis(i) for i in ids])
            elif store:
                vis = self._cat_dev(ids, col)
            else:
                vis = cp.asarray(np.concatenate([self.snaps[i][col] for i in ids]))
            if self.rotate:      # rotate_samples: vis * exp(2 pi i (w - w')), w' = unfolded S.w
                wn = cp.where(S.flip, -S.w, S.w)
                vis = vis * cp.exp(2j * np.pi * (c["w0"] - wn)).astype(vis.dtype)
            return S, vis
        if c is not None:                                    # host cache
            hc = c["host"]
            cat = lambda k: hc[k] if k in hc else np.concatenate([self.snaps[i][k] for i in ids])
        else:
            hc = {}
            cat = lambda k: hc.setdefault(k, np.concatenate([self.snaps[i][k] for i in ids])) \
                if k in ("u", "v", "w", "I") else np.concatenate([self.snaps[i][k] for i in ids])
        if store:
            vis = cp.concatenate([self._psf_vis(i) for i in ids]) if col == "PSF" else (
                self._cat_dev(ids, col) if col else None)
            u0, v0, w0 = self._cat_dev3(ids)
            u, v, w, vis = self._rot(ids, u0, v0, w0, vis)
            wt = cp.concatenate([cp.asarray(self.iw[i]) for i in ids])
        else:
            if col == "PSF":
                vis = cp.concatenate([self._psf_vis(i) for i in ids])
            else:
                vis = cp.asarray(cat(col)) if col else None
            w0 = cp.asarray(cat("w"))
            u, v, w, vis = self._rot(ids, cp.asarray(cat("u")), cp.asarray(cat("v")), w0, vis)
            if "wt" not in hc:
                hc["wt"] = np.concatenate([self.iw[i] for i in ids])
            wt = cp.asarray(hc["wt"])
        S = SampleSet(u, v, w, wt, self.N, self.sc, svd=self.svd, eps=self.eps, precompute=False, scope="hour")
        if mode != "off" and c is None:
            npts = int(S.npts)
            need = npts * (S.x.itemsize + S.y.itemsize + S.w.itemsize + S.wt.itemsize + 1 + w0.itemsize + 8)
            pool = cp.get_default_memory_pool()
            free = cp.cuda.runtime.memGetInfo()[0] + pool.total_bytes() - pool.used_bytes()
            margin = max(float(os.environ.get("TTCALX_CACHE_MARGIN_GB", "5")) * 1e9, self.ctl.reserve)
            if self.ctl.device and (mode == "device" or (mode in ("auto", "store") and free - need > margin)):
                self.ctl.dev_bytes += need
                d = dict(S=S, w0=w0, vis={})
                if col == "I":
                    d["vis"]["I"] = vis
                elif "I" in self.snaps[ids[0]]:
                    vi = self._cat_dev(ids, "I") if store else cp.asarray(cat("I"))
                    if self.rotate:
                        wn = cp.where(S.flip, -S.w, S.w)
                        vi = vi * cp.exp(2j * np.pi * (w0 - wn)).astype(vi.dtype)
                    d["vis"]["I"] = vi
                self._cache[key] = d
            elif not store:
                self._cache[key] = dict(host=hc)
        return S, vis

    def _vis_for(self, ids, col, S, w0):
        """Rotated visibilities of column ``col`` for the SampleSet S of snapshots ``ids`` (w0: their
        stored-frame w): the expressions of _chunk's device-cache branch, so the values are the ones
        _chunk(ids, col) returns."""
        if col == "PSF":
            vis = cp.concatenate([self._psf_vis(i) for i in ids])
        elif os.environ.get("TTCALX_CHUNK_CACHE", "off") == "store":
            vis = self._cat_dev(ids, col)
        else:
            vis = cp.asarray(np.concatenate([self.snaps[i][col] for i in ids]))
        if self.rotate:      # rotate_samples: vis * exp(2 pi i (w - w')), w' = unfolded S.w
            wn = cp.where(S.flip, -S.w, S.w)
            vis = vis * cp.exp(2j * np.pi * (w0 - wn)).astype(vis.dtype)
        return vis

    def image_multi(self, cols):
        """[image(c) for c in cols] (dirty images, no model) with one SampleSet per chunk: the chunk is
        built (or taken from the cache) and its points set once for all columns."""
        accs = [cp.zeros((self.N, self.N), cp.float32) for _ in cols]; sumw = 0.0
        for chunk in self.chunks:
            S, vis = self._chunk(chunk, cols[0])
            sumw += float(S.wt.sum())
            accs[0] += S.grid(vis); del vis
            if len(cols) > 1:
                c = self._cache.get(tuple(chunk))
                if c is not None and "S" in c:
                    w0 = c["w0"]
                elif os.environ.get("TTCALX_CHUNK_CACHE", "off") == "store":
                    w0 = self._cat_dev(chunk, "w")
                else:
                    w0 = cp.asarray(np.concatenate([self.snaps[i]["w"] for i in chunk]))
                for j, col in enumerate(cols[1:], 1):
                    vis = self._vis_for(chunk, col, S, w0)
                    accs[j] += S.grid(vis); del vis
            del S
        return [to_ws(a) / sumw for a in accs]

    def image(self, col="I", model=None, psf=False, ids=None):
        """Weighted dirty (model=None) or residual image / sum of weights, WSClean orientation.
        col='PSF': image of a unit point source at the phase centre WITH the w-term (WSClean's PSF).
        psf=True: 2-D PSF without the w-term (not what WSClean writes; kept for comparisons)."""
        acc = cp.zeros((self.N, self.N), cp.float32); sumw = 0.0
        for chunk in (self.chunks if ids is None else self._subchunks(ids)):
            S, vis = self._chunk(chunk, None if psf else col)
            sumw += float(S.wt.sum())
            if psf:
                acc += S.grid_psf(); continue
            if model is not None:
                vis = vis - S.degrid(model)
            acc += S.grid(vis)
            del S, vis
        return to_ws(acc) / sumw


class WSnap:
    """w-snapshot imaging with the global weights of an HourGrid (see module docstring)."""

    def __init__(self, G, block=6, order=5, up=2, log=None):
        self.G = G; self.N = G.N; self.sc = G.sc; self.order = order; self.up = up
        self.P = Reprojector(self.N, self.sc)
        n = len(G.snaps); self.sub = []; self.Rb = []
        for b0 in range(0, n, block):
            ids = list(range(b0, min(n, b0 + block)))
            Rb = G.snaps[ids[len(ids) // 2]]["R"]
            sn = []
            for i in ids:     # snapshot frame -> block frame: R_b^T R_i (both chgcentre-exact)
                sn.append(_with(G.snaps[i], R=Rb.T @ G.snaps[i]["R"], R_hour=G.snaps[i]["R"]))
            self.sub.append(HourGrid(sn, self.N, G.sdeg, rotate=block > 1, weights=[G.iw[i] for i in ids],
                                     eps=G.eps, svd_tol=1e-4, ctl=G.ctl))
            self.Rb.append(Rb)
        self._T = None
        (log or (lambda *a: None))(f"WSnap: {len(self.sub)} blocks of {block}, K = {sorted(set(g.svd.K for g in self.sub))}")

    def _bandlimit(self, M):
        """2x band-limited upsampling of the hour model, keeping |q| < umax (data support) and
        tapering to 0 at the original Nyquist so that spline reprojection is accurate."""
        if self._T is None:
            f = cp.fft.fftfreq(self.N).astype(cp.float32)
            q = cp.sqrt(f[:, None] ** 2 + f[None, :] ** 2) / self.sc
            umax = max(_uvmax(s) for s in self.G.snaps) * 1.02
            qn = 0.5 / self.sc
            t = cp.clip((q - umax) / max(qn - umax, 1), 0, 1)
            self._T = cp.where(q <= umax, 1.0, 0.5 * (1 + cp.cos(cp.pi * t))).astype(cp.float32)
        return fft_upsample(M, self.up, self._T)

    def image_multi(self, cols):
        """[image(c) for c in cols] (dirty images) in one pass over the blocks (HourGrid.image_multi),
        each block image reprojected with the same coordinates as image() computes for it."""
        accs = [cp.zeros((self.N, self.N), cp.float32) for _ in cols]; wsum = 0.0
        for g, R in zip(self.sub, self.Rb):
            ds = g.image_multi(cols)
            co = self.P.coords(R)
            for j, d in enumerate(ds):
                r, _ = self.P(d, R, self.order, coords=co)
                accs[j] += g.sumw * r
            wsum += g.sumw
            del ds
        return [a / wsum for a in accs]

    def image(self, col="I", model=None):
        acc = cp.zeros((self.N, self.N), cp.float32); wsum = 0.0
        Mb = prefilter(self._bandlimit(model), self.order) if model is not None else None
        for g, R in zip(self.sub, self.Rb):
            ms = None
            if Mb is not None:
                co = self.P.coords(R.T)                       # block pixel -> hour-frame position
                ms, _ = self.P(Mb, None, self.order, coords=co, up=self.up, prefiltered=True)
                ms = ms * (co[2] / cp.maximum(self.P.n, 1e-6))  # model Jacobian n_hour / n_block
            d = g.image(col, model=ms)
            r, _ = self.P(d, R, self.order)
            acc += g.sumw * r; wsum += g.sumw
        return acc / wsum


# ----------------------------------------------------------------------------- driver
def theoretical_beam_pix(snaps, sc):
    return 1.0 / max(_uvmax(s) for s in snaps) / sc


def snapshot_qa(snaps, N, sdeg, params, qa="auto", qa_dir=None, log=None):
    """Per-snapshot V rms: from <qa_dir>/<stem>-qa.json (snapshot mode) if present, else computed."""
    from .snapshot import v_rms_from_samples
    out = []
    for s in snaps:
        r = None
        if qa in ("auto", "json") and qa_dir:
            p = os.path.join(qa_dir, f"{s['stem']}-qa.json")
            if os.path.exists(p):
                r = json.load(open(p)).get("V_rms_box")
        if r is None and qa in ("auto", "compute"):
            r = v_rms_from_samples(s, N, sdeg, params.robust, params.taper_inner_tukey, scope="hour")
        out.append(float("nan") if r is None else float(r))
    return out


def _as_product(p, pols):
    """HourProduct, or a legacy (tag, n_intervals, CleanParams) tuple."""
    if isinstance(p, bands.HourProduct):
        return p
    tag, nint, params = p
    return bands.HourProduct(tag, nint, params, v_tag="V" if "V" in pols else None, naming="legacy")


def _names(prod, name):
    """File names of one image of a product (orca = WSClean single-pol names)."""
    if prod.naming == "orca":
        return dict(image=f"{name}-image.fits", residual=f"{name}-residual.fits", model=f"{name}-model.fits",
                    dirty=f"{name}-dirty.fits", psf=f"{name}-psf.fits")
    return dict(image=f"{name}-I-image.fits", residual=f"{name}-I-residual.fits", model=f"{name}-I-model.fits",
                dirty=f"{name}-I-dirty.fits", psf=f"{name}-psf.fits", V=f"{name}-V-image.fits")


def _want_v(products, pols, qa):
    return any(p.v_tag for p in products) or qa in ("auto", "compute")


def image_hour(paths, band, out_prefix, phase_centre=None, products=(("1h", 1, bands.DEEP_I),),
               mode="wsnap", block=6, pols="IV", qa="auto", qa_dir=None, exclude=(), nproc=8,
               beam=None, write_aux=True, verbose=False, exclude_stems=()):
    """Image one band-hour from its snapshot MSes = load_all + :func:`image_hour_snaps`.
    products: HourProduct (bands.hour_products) or legacy (tag, n_intervals, CleanParams)
    tuples, all made from one load, e.g. bands.hour_products(bands.ORCA_ALL) or
    (("1h", 1, DEEP_I), ("10min", 6, TENMIN_I)).  Returns the summary dict (also written to
    <out_prefix>-summary.json); summary["timing"] has seconds per step, keyed "<tag>: <step>"
    and accumulated over intervals."""
    T = {}; t0 = time.perf_counter()
    products = [_as_product(p, pols) for p in products]
    ra, dec = parse_phase_centre(phase_centre) if phase_centre else middle_zenith(paths)
    snaps = load_all(list(paths), ra, dec, nproc, want_v=_want_v(products, pols, qa))
    T["load + rotation (CPU)"] = time.perf_counter() - t0
    return image_hour_snaps(snaps, band, out_prefix, ra, dec, products=products, mode=mode, block=block, pols=pols,
                            qa=qa, qa_dir=qa_dir, exclude=exclude, exclude_stems=exclude_stems, beam=beam,
                            write_aux=write_aux, verbose=verbose, timing=T, t0=t0)


def image_hour_snaps(snaps, band, out_prefix, ra, dec, products=(("1h", 1, bands.DEEP_I),), mode="wsnap",
                     block=6, pols="IV", qa="auto", qa_dir=None, exclude=(), exclude_stems=(), beam=None,
                     write_aux=True, verbose=False, timing=None, t0=None):
    """Image one band-hour from loaded snapshots: msio.load_samples dicts (:func:`image_hour`)
    or gsiv.LazySnap mappings (an archive / the gsi v3 hour store), any order.

    (ra, dec): the hour phase centre the snapshots' R rotate to.  exclude: snapshot indices in
    time order; exclude_stems: snapshot stems (both dropped as QA rejections, the interval
    boundaries still count them).  timing / t0: the caller's timing dict and start time (the
    load is timed there).  The snapshot list is sorted in place and gets 'tidx'."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    T = {} if timing is None else timing
    t0 = time.perf_counter() if t0 is None else t0
    last = [time.perf_counter()]

    def lap(k):
        cp.cuda.Device().synchronize(); now = time.perf_counter(); T[k] = T.get(k, 0) + now - last[0]; last[0] = now

    products = [_as_product(p, pols) for p in products]
    # cuFFT plans: every interval of every product uses the same few sizes (deconvolution P x P
    # r2c/c2r, the scale-image size, the 2x model upsampling), and creating one costs ~0.25 s
    # (~2-3 s per interval).  TTCALX_KEEP_FFT_PLANS=1 keeps them across intervals, in an LRU plan cache
    # capped at TTCALX_FFT_PLAN_CACHE_GB (default 1.5); default 0: cleared after every interval (748884a).
    keep_plans = os.environ.get("TTCALX_KEEP_FFT_PLANS", "0") == "1"
    # TTCALX_CACHE_OOM_FALLBACK=1: a GPU OOM in an interval whose chunks were partly kept on the GPU is
    # imaged once more without the device chunk cache (same inputs, bit-identical products).  Default 0
    # (raise, as 86afd89) for MS hour jobs; the archive / in-memory entry points (cli --archive,
    # run_hour_snaps: gsi v3) turn it on
    oom_fallback = os.environ.get("TTCALX_CACHE_OOM_FALLBACK", "0") == "1"
    # TTCALX_HOUR_MULTI=1 (default 0): an I product's dirty I, PSF and (if made) dirty V are gridded in one
    # pass over the chunks (one SampleSet / setpts per chunk instead of three), the same images
    multi = os.environ.get("TTCALX_HOUR_MULTI", "0") == "1"
    if keep_plans:
        cp.fft.config.get_plan_cache().set_memsize(int(float(os.environ.get("TTCALX_FFT_PLAN_CACHE_GB", "1.5")) * 1e9))
    N, sdeg = bands.geometry(band)
    snaps.sort(key=lambda s: s["time"])
    nfull = len(snaps)
    for i, s in enumerate(snaps):
        s["tidx"] = i
    # QA always with the orca pilot-snapshot weighting (briggs 0, inner Tukey 30)
    vr = snapshot_qa(snaps, N, sdeg, bands.SNAPSHOT_CLEAN_I, qa, qa_dir, log) if qa != "none" else [float("nan")] * nfull
    bad, med, sd = qa_reject(vr) if qa != "none" else ([], None, None)
    xs = set(exclude_stems)
    drop = set(bad) | set(int(x) for x in exclude) | {i for i, s in enumerate(snaps) if s["stem"] in xs} | \
        {i for i, s in enumerate(snaps) if _nsamp(s) == 0 or s["zero_frac"] >= 0.999}
    lap("snapshot QA")
    log(f"QA: {nfull} snapshots, median V rms {med}, mad {sd}, rejected {sorted(drop)}")
    summary = dict(band=band, N=N, scale_deg=sdeg, phase_centre=[math.degrees(ra), math.degrees(dec)], mode=mode,
                   block=block, n_loaded=nfull, rejected=sorted(drop),
                   qa=[dict(stem=s["stem"], time=s["time"], V_rms_box=r, flag_frac=s["flagfrac"], zero_frac=s["zero_frac"])
                       for s, r in zip(snaps, vr)], qa_median=med, qa_mad_std=sd, timing=T, products={})
    # interval start times include rejected snapshots: orca flags them in the concatenated MS, so
    # WSClean's DATE-OBS is the first time step of the interval whether or not it is flagged
    tstart_all = [s["tstart"] for s in snaps]
    snaps = [s for i, s in enumerate(snaps) if i not in drop]
    for prod in products:
        tag, nint, params = prod.tag, prod.n_intervals, prod.params
        key = tag if prod.image_i else prod.v_tag
        bnd = [int(math.floor(k * nfull / nint)) for k in range(nint + 1)]   # wsclean -intervals-out
        groups = [[s for s in snaps if bnd[k] <= s["tidx"] < bnd[k + 1]] for k in range(nint)]
        t0s = [min(tstart_all[bnd[k]:bnd[k + 1]], default=None) for k in range(nint)]
        outs = []
        dev_ok = True        # device chunk cache allowed (off for the rest of the product after an OOM fallback)

        def interval(gi, sn, name, fn, tsfx, ctl):
            G = HourGrid(sn, N, sdeg, robust=params.robust, tukey_inner=params.taper_inner_tukey or None,
                         need_svd=(mode != "wsnap"), log=log, ctl=ctl)
            if mode == "wsnap":
                W = WSnap(G, block=block, log=log)
                imgfn = W.image
            else:
                imgfn = G.image
            multi_fn = W.image_multi if mode == "wsnap" else G.image_multi
            lap(f"{key}: weights + gridder setup")
            vpre = None
            s0 = sn[0]
            hist = (f"ttcalx-image hour ({mode}, block {block}): {key} multiscale={params.multiscale} "
                    f"bias {params.multiscale_scale_bias} niter={params.niter} mgain={params.mgain} "
                    f"auto-mask={params.auto_mask} auto-threshold={params.auto_threshold} local-rms "
                    f"horizon={params.horizon_deg}deg briggs {params.robust} tukey {params.taper_inner_tukey}")
            hdr = lambda b: make_header(N, sdeg, ra, dec, s0["freqs"], s0["chan_width"], t0s[gi],
                                        beam=b, telescope=s0["telescope"], observer=s0["observer"],
                                        obj=s0["field_name"], extra=dict(NSNAP=(len(sn), "snapshots imaged")),
                                        history=[hist])
            files = []; di = {}; theo = theoretical_beam_pix(sn, G.sc)
            if prod.image_i:
                if multi:
                    # TTCALX_HOUR_MULTI: dirty I, the PSF and dirty V in one pass over the chunks (the same images)
                    out3 = multi_fn(["I", "PSF"] + (["V"] if prod.v_tag else []))
                    dirty, psf = out3[0], out3[1]; vpre = out3[2] if prod.v_tag else None
                    del out3
                    psf = psf / psf[N // 2, N // 2]; lap(f"{tag}: dirty I + psf" + (" + dirty V" if prod.v_tag else ""))
                else:
                    dirty = imgfn("I"); lap(f"{tag}: dirty I")
                    # PSF with the w-term in the hour frame, as WSClean (w-snapshot gridded like the data)
                    psf = imgfn("PSF"); psf = psf / psf[N // 2, N // 2]; lap(f"{tag}: psf")
                restored, R, M, di = deconvolve_multiscale(dirty, psf, sdeg, theo, params,
                                                           residual_fn=lambda m: imgfn("I", model=m), beam=beam, log=log)
                lap(f"{tag}: deconvolution (incl. major cycles)")
                h = hdr(di["beam"])
                files.append(write_image(fn["image"], restored, h, "I"))
                if write_aux:
                    files += [write_image(fn["residual"], R, h, "I"), write_image(fn["model"], M, h, "I"),
                              write_image(fn["dirty"], dirty, h, "I"), write_image(fn["psf"], psf, h, "I")]
                lap(f"{tag}: write")
                del dirty, psf, restored, R, M
            if prod.v_tag:
                vkey = prod.v_tag if prod.naming == "orca" else tag
                if vpre is not None:
                    v = vpre; vpre = None
                else:
                    v = imgfn("V"); lap(f"{vkey}: dirty V")
                h = hdr(di.get("beam"))
                vname = fn["V"] if prod.naming == "legacy" else f"{out_prefix}-{prod.v_tag}{tsfx}-image.fits"
                files.append(write_image(vname, v, h, "V"))
                lap(f"{vkey}: write")
                del v
            out = dict(name=name, n_snap=len(sn), npts=G.npts, theo_beam_pix=theo, files=files,
                       K=sorted(set(g.svd.K for g in W.sub)) if mode == "wsnap" else G.svd.K, **di)
            # imgfn (a bound method of W / G) and the residual closure keep the gridders alive:
            # drop every reference before the next interval allocates its own (82 MHz 10-min OOM)
            del G, imgfn, multi_fn, vpre
            if mode == "wsnap":
                del W
            gc.collect()
            if not keep_plans:
                cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
            return out

        for gi, sn in enumerate(groups):
            tsfx = "" if nint == 1 else f"-t{gi:04d}"
            name = f"{out_prefix}-{tag}{tsfx}"
            fn = _names(prod, name)
            if not sn:
                outs.append(dict(name=name, n_snap=0)); continue
            reserve = _reserve(N, sn, math.radians(sdeg), params) if prod.image_i else 0.0
            while True:
                ctl = CacheCtl(device=dev_ok, reserve=reserve)
                T0 = dict(T); err = None
                try:
                    outs.append(interval(gi, sn, name, fn, tsfx, ctl))
                except Exception as e:      # noqa: BLE001 (re-raised unless a GPU OOM with a device cache)
                    if not (oom_fallback and ctl.device and ctl.dev_bytes and _is_gpu_oom(e)):
                        raise
                    err = f"{type(e).__name__}: {e}"
                if err is None:
                    break
                # outside the except clause: the failed attempt's frames (gridders, device cache) are gone.
                # Again with no chunk on the GPU (host cache, or rebuilt per use from the store): the gridder
                # gets the same arrays (bit-identical inputs), only slower; the failed attempt's time is
                # booked under "<key>: OOM fallback" instead of its steps
                dev_ok = False
                gc.collect()
                cp.fft.config.get_plan_cache().clear()
                cp.get_default_memory_pool().free_all_blocks()
                spent = sum(T.get(k, 0) - T0.get(k, 0) for k in T)
                T.clear(); T.update(T0); lap(f"{key}: OOM fallback")
                T[f"{key}: OOM fallback"] += spent
                msg = (f"hour {os.path.basename(name)}: GPU out of memory with {ctl.dev_bytes / 1e9:.2f} GB of device "
                       f"chunk cache ({err}); again without the device chunk cache")
                print(f"WARNING: {msg}", file=sys.stderr, flush=True); log(msg)
                summary.setdefault("cache_fallback", []).append(dict(name=name, error=err,
                                                                     device_cache_GB=ctl.dev_bytes / 1e9))
        summary["products"][key] = outs
        if prod.image_i and prod.v_tag and prod.naming == "orca":
            summary["products"][prod.v_tag] = [dict(name=o["name"], n_snap=o["n_snap"],
                                                    files=[f for f in o.get("files", []) if prod.v_tag in os.path.basename(f)])
                                               for o in outs]
    _DENSITY_MEMO.clear()
    T["TOTAL"] = time.perf_counter() - t0
    json.dump(summary, open(f"{out_prefix}-summary.json", "w"), indent=1, default=float)
    return summary
