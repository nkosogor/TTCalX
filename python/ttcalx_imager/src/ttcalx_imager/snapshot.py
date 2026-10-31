"""10-s snapshot product: CLEANed Stokes I + dirty Stokes V + QA JSON, all on one GPU.

    read MS (casacore, CPU) -> Briggs/Tukey weights (GPU) -> dirty I, PSF (cuFINUFFT + SVD w-term)
    -> Clark/Hogbom clean with auto-mask / local-rms -> 2 major cycles (DFT predict of the sparse
    model, or degridding above ``dft_max`` components) -> restore with the WSClean beam fit
    -> dirty V with the same weights -> FITS + <prefix>-qa.json

Emulates orca SNAPSHOT_CLEAN_I_PARAMS (wsclean -pol I -niter 50000 -mgain 0.9999 -auto-mask 5
-auto-threshold 1 -local-rms -horizon-mask 10deg -weight briggs 0 -taper-inner-tukey 30) and the
dirty V pilot snapshot (-pol V -niter 0, same weighting).

Keep one ``SnapshotImager`` per process: cuFINUFFT plans and SVD w-term tables are cached per
geometry (the first snapshot of a geometry costs ~1-2 s extra).
"""
import json
import math
import time
import traceback
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures.process import BrokenProcessPool

import numpy as np
import cupy as cp

from . import bands, gridder
from .clean import clean_snapshot, mad_std
from .fits import make_header, write_image, write_images
from .gridder import SampleSet, dft_predict, get_svd, to_ws
from .msio import C, is_empty, read_ms
from .snapload import RawLoader, SharedLoader, load_snapshot, prepare_samples  # noqa: F401  (re-exported)
from .weights import WeightGrid

QA_BOX = 1024      # orca analyze_snapshot_quality: std of the central 1024 x 1024 pixels


def box_std(img, box=QA_BOX):
    N = img.shape[0]; h = min(box, N) // 2
    return float(img[N // 2 - h:N // 2 + h, N // 2 - h:N // 2 + h].std())


def snapshot_svd(N, sc, wf, tol):
    """w-term factorisation for a snapshot (|w| <~ 50 lambda): wmax rounded to 0.25 lambda.

    Keeps ONE snapshot table (nw=600 entries of gridder._SVD_CACHE): a table of another key is
    dropped before a new one is built, so the GPU never holds two, and consecutive snapshots of
    the same geometry (after fix_uvw the snapshot w range is constant over an hour) reuse it.
    The table for a given key is exactly the one the stock code builds, so images are unchanged."""
    wq = math.ceil((float(wf) * 1.0001 + 1e-6) / 0.25) * 0.25
    key = (N, round(sc, 12), round(wq, 3), tol, 600)
    if key not in gridder._SVD_CACHE:
        for k in [k for k in gridder._SVD_CACHE if k[4] == 600]:
            del gridder._SVD_CACHE[k]
    return get_svd(N, sc, float(wf) * 1.0001 + 1e-6, tol, quantum=0.25, nw=600, pad=0.0)



def gpu_select(small, raw, scale_deg, pols="IV", zero_limit=0.999):
    """GPU version of snapload.load_snapshot from the raw MS columns (RawLoader): the same
    float64/complex64 element operations as msio.read_ms + load_snapshot + prepare_samples, in
    the same sample order (row-major boolean selection), hence the same values; returns the
    load_snapshot dict with CuPy arrays."""
    from types import SimpleNamespace
    t1 = time.perf_counter()
    column = small["column"]
    keep = cp.asarray(small["a1"] != small["a2"])
    data = cp.asarray(raw[column]); flag = cp.asarray(raw["FLAG"])
    wcol = cp.asarray(small["wcol"]); uvw = cp.asarray(small["uvw"].astype(np.float64))
    ok = ~(flag[:, :, 0] | flag[:, :, 3]) & keep[:, None]          # = read_ms ok on the kept rows
    nkeep = int(keep.sum()); nchan = data.shape[1]
    nok = int(ok.sum())
    zero = int(((data[:, :, 0] == 0) & (data[:, :, 3] == 0) & ok).sum())
    meta = SimpleNamespace(path=small["path"], stem=small["stem"], time=small["time"], tstart=small["tstart"],
                           tmid=small["tmid"], flag_frac=float(1.0 - nok / (nkeep * nchan)) if nkeep * nchan else 1.0,
                           zero_frac=float(zero / nok) if nok else 1.0, phase_dir=small["phase_dir"],
                           freqs=small["freqs"], chan_width=small["chan_width"], telescope=small["telescope"],
                           observer=small["observer"], field_name=small["field_name"])
    L = dict(ms=meta, timing={}, empty=(nok == 0) or meta.zero_frac >= zero_limit)
    if not L["empty"]:
        natw = (0.5 * (wcol[:, 0] + wcol[:, 3])).astype(cp.float32)
        wnat = natw[:, None] * ok
        sel = (wnat > 0) & keep[:, None]
        lam = cp.asarray(C / small["freqs"])
        u = (uvw[:, 0:1] / lam[None])[sel]; v = (uvw[:, 1:2] / lam[None])[sel]; w = (uvw[:, 2:3] / lam[None])[sel]
        L.update(u=u, v=v, w=w, wnat=wnat[sel])
        L["visI"] = (0.5 * (data[:, :, 0][sel] + data[:, :, 3][sel])).astype(cp.complex64)
        L["visV"] = ((0.5j * (data[:, :, 2][sel] - data[:, :, 1][sel])).astype(cp.complex64)
                     if "V" in pols else None)
        del data, flag
        sc = math.radians(scale_deg)
        flip = w < 0
        sg = cp.where(flip, -1.0, 1.0)
        uf, vf, wf = u * sg, v * sg, w * sg
        L["pre"] = dict(flip=flip, x=(2 * np.pi * uf * sc).astype(cp.float32), y=(2 * np.pi * vf * sc).astype(cp.float32),
                        wfold=wf.astype(cp.float32), u32=u.astype(cp.float32), v32=v.astype(cp.float32),
                        w32=w.astype(cp.float32), wabsmax=float(cp.abs(w).max()),
                        theo_pix=1.0 / float(cp.sqrt(u * u + v * v + w * w).max()) / sc)
    cp.cuda.Device().synchronize()
    L["timing"]["select (GPU)"] = time.perf_counter() - t1
    return L


class SnapshotImager:
    def __init__(self, eps=1e-4, svd_tol=1e-4, dft_max=8000, verbose=False):
        self.eps = eps; self.svd_tol = svd_tol; self.dft_max = dft_max
        self.log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    def image_arrays(self, u, v, w, wnat, visI, visV, N, scale_deg, params=bands.SNAPSHOT_CLEAN_I, beam=None,
                     timing=None, pre=None):
        """Core imager on per-sample arrays (host numpy): u, v, w [lambda] float64, wnat natural
        weights (> 0), visI / visV (or None) complex64.  Returns a dict with the cupy images
        restored, residual, model, dirty, psf, V (or None) and an 'info' dict."""
        T = {} if timing is None else timing
        last = [time.perf_counter()]

        def lap(k):
            cp.cuda.Device().synchronize()
            now = time.perf_counter(); T[k] = T.get(k, 0) + now - last[0]; last[0] = now

        sc = math.radians(scale_deg)
        ud, vd, nwd = cp.asarray(u), cp.asarray(v), cp.asarray(wnat)
        WG = WeightGrid(N, sc, params.robust, params.taper_inner_tukey or None)
        WG.add(ud, vd, nwd); WG.finalize()
        iw = WG.weights(ud, vd, nwd)
        sw = float(iw.sum())
        del ud, vd, nwd
        lap("weights")
        if pre is None:
            svd = snapshot_svd(N, sc, np.abs(w).max(), self.svd_tol)
            lap("svd table")
            S = SampleSet(u, v, w, iw, N, sc, svd=svd, eps=self.eps)
            u32 = cp.asarray(u, dtype=cp.float32); v32 = cp.asarray(v, dtype=cp.float32); w32 = cp.asarray(w, dtype=cp.float32)
            theo_pix = 1.0 / float(np.sqrt(u * u + v * v + w * w).max()) / sc     # WSClean theoretical beam
        else:
            svd = snapshot_svd(N, sc, pre["wabsmax"], self.svd_tol)
            lap("svd table")
            S = SampleSet(None, None, None, iw, N, sc, svd=svd, eps=self.eps, pre=pre)
            u32 = cp.asarray(pre["u32"]); v32 = cp.asarray(pre["v32"]); w32 = cp.asarray(pre["w32"])
            theo_pix = pre["theo_pix"]
        dI = cp.asarray(visI)
        lap("gridder setup")
        dirty = to_ws(S.grid(dI)) / sw
        lap("dirty I")
        # PSF = image of unit visibilities WITH the w-term: WSClean's PSF agrees with a direct DFT
        # of that to ~1e-5, the w-less PSF differs by up to ~1e-3 of the peak at el < 60 deg.
        # Normalised to 1 at the centre, as WSClean writes it.
        psf = to_ws(S.grid(cp.ones(S.npts, cp.complex64)))
        psf = psf / psf[N // 2, N // 2]
        lap("psf")
        sky = to_ws(svd.inside)

        def residual(model):
            if int(cp.count_nonzero(model)) <= self.dft_max:
                pred = dft_predict(u32, v32, w32, model, sc)
            else:
                pred = S.degrid(model)
            return to_ws(S.grid(dI - pred)) / sw

        restored, res, model, ci = clean_snapshot(dirty, psf, residual, scale_deg, theo_pix, sky, params,
                                                  beam=beam, log=self.log)
        lap("clean + major cycles + restore")
        imgV = None
        if visV is not None:
            imgV = to_ws(S.grid(cp.asarray(visV))) / sw
            lap("dirty V")
        info = dict(ci, K=svd.K, npts=S.npts, theo_beam_pix=theo_pix, sumw=sw, I_peak=float(restored.max()),
                    I_res_rms_box=box_std(res), I_res_mad=mad_std(res[sky]))
        if imgV is not None:
            info.update(V_rms_box=box_std(imgV), V_maxabs=float(cp.abs(imgV).max()))
        return dict(restored=restored, residual=res, model=model, dirty=dirty, psf=psf, V=imgV, info=info)

    def image(self, ms_path, band=None, N=None, scale_deg=None, out_prefix=None, pols="IV",
              params=bands.SNAPSHOT_CLEAN_I, beam=None, write_aux=False, write_dirty=False, write_psf=False):
        """Image one snapshot MS.  Geometry from ``band`` (orca tiers) or explicit N / scale_deg.
        Returns an info dict (also written to <out_prefix>-qa.json if out_prefix is given)."""
        t0 = time.perf_counter()
        if N is None:
            N, scale_deg = bands.geometry(band if band is not None else ms_path)
        L = load_snapshot(ms_path, N, scale_deg, pols, prep=True)
        info, prods = self._compute(L, N, scale_deg, out_prefix, pols, params, beam, write_aux, write_dirty, write_psf)
        if prods:
            prods = [(p, a.get(), st) for p, a, st in prods]
        return self._finish(info, prods, t0)

    def _compute(self, L, N, scale_deg, out_prefix, pols, params, beam, write_aux, write_dirty, write_psf):
        ms = L["ms"]; T = dict(L["timing"])
        info = dict(ms=ms.path, stem=ms.stem, N=N, scale_deg=scale_deg, time=ms.time, flag_frac=ms.flag_frac,
                    zero_frac=ms.zero_frac, status="ok")
        prods = None
        if L["empty"]:
            info["status"] = "empty"
        else:
            r = self.image_arrays(L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"],
                                  N, scale_deg, params, beam, timing=T, pre=L.get("pre"))
            info.update(r["info"])
            if out_prefix:
                h = make_header(N, scale_deg, *ms.phase_dir, ms.freqs, ms.chan_width, ms.tstart, beam=info["beam"],
                                telescope=ms.telescope, observer=ms.observer, obj=ms.field_name,
                                history=[f"ttcalx-image snapshot: clean I niter={params.niter} auto-mask={params.auto_mask} "
                                         f"auto-threshold={params.auto_threshold} local-rms horizon={params.horizon_deg}deg "
                                         f"briggs {params.robust} tukey {params.taper_inner_tukey} nmajor={params.nmajor}"])
                prods = [("I-image", r["restored"], "I")]
                if write_aux:
                    prods += [("I-residual", r["residual"], "I"), ("I-model", r["model"], "I")]
                if write_dirty:
                    prods.append(("I-dirty", r["dirty"], "I"))
                if write_psf:
                    prods.append(("psf", r["psf"], "I"))
                if r["V"] is not None:
                    prods.append(("V-image", r["V"], "V"))
                prods = [(f"{out_prefix}-{name}.fits", arr, st) for name, arr, st in prods]
                info["_header"] = h
            info["gpu_pool_GB"] = cp.get_default_memory_pool().total_bytes() / 1e9
        info["timing"] = T
        info["_prefix"] = out_prefix
        return info, prods

    @staticmethod
    def _finish(info, prods, t0):
        """Write FITS (host arrays) + qa.json; TOTAL = wall time since t0 (thread-safe, host only)."""
        T = info["timing"]; h = info.pop("_header", None); out_prefix = info.pop("_prefix", None)
        if prods:
            t1 = time.perf_counter()
            info["files"] = write_images(prods, h)
            T["write FITS"] = time.perf_counter() - t1
        T["TOTAL"] = time.perf_counter() - t0
        if out_prefix:
            json.dump(info, open(f"{out_prefix}-qa.json", "w"), indent=1, default=float)
        return info

    def image_many(self, requests, prefetch=1, free_pool=True, reader="raw", on_done=None):
        """Pipelined ``image`` over a sequence of request dicts (keyword arguments of ``image``):
        the MS of snapshot i+1..i+prefetch is read + prepared in a host thread while the GPU
        images snapshot i, and the FITS / qa.json of snapshot i are written in another host
        thread.  Yields the info dicts in request order (each after its files are complete).
        GPU work is issued from the calling thread only and in the same order as one-at-a-time
        ``image`` calls, so the images are the same.  free_pool: hand cached CuPy blocks back
        after each snapshot (as the gsi-v2 daemon does).
        reader: "raw" (default: a helper process reads the raw columns into shared memory, the
        selection runs on the GPU), "process" (helper process does read_ms + host selection) or
        "thread" (slowest: python-casacore holds the GIL while reading).
        requests may be a queue.Queue (None ends it): then a request is started as soon as it
        arrives and finished snapshots are delivered without waiting for the next one.
        on_done(info): called from the writer thread when a snapshot's files are complete
        (instead of yielding).  A failing snapshot (read, imaging or file writing) gives info
        status "error" with its traceback, and the loop goes on; a dead helper is replaced.
        The process readers use multiprocessing "spawn": a calling script needs an
        ``if __name__ == "__main__":`` guard (or use reader="thread")."""
        def geom(rq):
            N, sc = rq.get("N"), rq.get("scale_deg")
            if N is None:
                b = rq.get("band"); N, sc = bands.geometry(b if b is not None else rq["ms_path"])
            return N, sc

        def load(rq):
            N, sc = geom(rq)
            return load_snapshot(rq["ms_path"], N, sc, rq.get("pols", "IV"), prep=True)

        import queue as _queue
        if isinstance(requests, _queue.Queue):          # live request stream (daemon); None ends it
            q = requests

            def take(block):
                try:
                    return q.get(block)
                except _queue.Empty:
                    return False
        else:
            it = iter(requests)

            def take(block):
                return next(it, None)
        proc = {"process": SharedLoader, "raw": RawLoader}.get(reader)
        proc = proc(1) if proc else None
        rd = None if proc else ThreadPoolExecutor(1, thread_name_prefix="snap-read")
        wr = ThreadPoolExecutor(1, thread_name_prefix="snap-write")
        pend = deque(); ended = [False]

        def submit(block):
            """queue the next request's read; False if none available now / ever."""
            if ended[0]:
                return False
            rq = take(block)
            if rq is False:
                return False
            if rq is None:
                ended[0] = True; return False
            try:
                if proc:
                    N, sc = geom(rq)
                    f = proc.submit(rq["ms_path"], N, sc, rq.get("pols", "IV"))
                else:
                    f = rd.submit(load, rq)
            except Exception as e:      # e.g. a bad request: reported in order below
                f = e
            pend.append((rq, f))
            return True

        def deliver(info):
            if on_done is not None:
                try:
                    on_done(info)
                except Exception:           # a failing callback must not end the loop
                    traceback.print_exc()
            return info

        def finish(info, prods, t0, rq):
            try:
                info = self._finish(info, prods, t0)
            except Exception as e:          # e.g. FITS / qa.json write failure
                return failed(rq, e)
            if rq.get("_id") is not None:
                info["_id"] = rq["_id"]      # request tag (not written to the qa.json)
            return deliver(info)

        def failed(rq, e):
            tb = "".join(traceback.format_exception(type(e), e, e.__traceback__))
            info = dict(ms=rq.get("ms_path"), status="error", error=repr(e), traceback=tb, timing={}, request=rq)
            if rq.get("_id") is not None:
                info["_id"] = rq["_id"]
            return deliver(info)

        prev = None
        try:
            while True:
                if not pend:
                    if prev is not None:                    # nothing queued: deliver what is done
                        r = prev.result(); prev = None
                        if on_done is None:
                            yield r
                    if not submit(True):
                        break
                while len(pend) < 1 + max(0, prefetch) and submit(False):
                    pass
                rq, fut = pend.popleft()
                t_wait = time.perf_counter()
                try:
                    if isinstance(fut, Exception):
                        raise fut
                    if reader == "raw":
                        small, raw, dt, release = proc.result(fut)
                        try:
                            L = gpu_select(small, raw, geom(rq)[1], rq.get("pols", "IV"))
                        finally:
                            release()
                        release = (lambda: None)
                        L["timing"]["read MS (raw)"] = dt
                    elif proc:
                        L, release = proc.result(fut)
                    else:
                        L, release = fut.result(), (lambda: None)
                    while len(pend) < max(0, prefetch) and submit(False):
                        pass
                    L["timing"]["wait read"] = time.perf_counter() - t_wait
                    N, sc = geom(rq)
                    kw = {k: rq.get(k, d) for k, d in (("out_prefix", None), ("pols", "IV"),
                                                       ("params", bands.SNAPSHOT_CLEAN_I), ("beam", None),
                                                       ("write_aux", False), ("write_dirty", False), ("write_psf", False))}
                    t0 = time.perf_counter()
                    try:
                        info, prods = self._compute(L, N, sc, **kw)
                        if prods:
                            t1 = time.perf_counter()
                            prods = [(p, a.get(), st) for p, a, st in prods]      # D2H on this thread
                            info["timing"]["D2H"] = time.perf_counter() - t1
                    finally:
                        release()       # host sample arrays are on the GPU (or done with) by now
                        del L
                    if free_pool:
                        cp.get_default_memory_pool().free_all_blocks()
                    nxt = wr.submit(finish, info, prods, t0, rq)
                except Exception as e:
                    if free_pool:
                        cp.get_default_memory_pool().free_all_blocks()
                    if isinstance(e, BrokenProcessPool):
                        # the helper died (killed, or crashed reading this MS): new helper, re-read the rest
                        proc.restart()
                        for i, (rq2, f2) in enumerate(pend):
                            if not isinstance(f2, Exception):
                                pend[i] = (rq2, proc.submit(rq2["ms_path"], *geom(rq2), rq2.get("pols", "IV")))
                    nxt = wr.submit(failed, rq, e)
                if prev is not None:
                    r = prev.result()
                    if on_done is None:
                        yield r
                prev = nxt
            if prev is not None:
                r = prev.result()
                if on_done is None:
                    yield r
        finally:
            # early close / exception: unconsumed prefetched reads must not leave shared memory behind
            futs = [f for _, f in pend if not isinstance(f, Exception)]
            if proc:
                proc.shutdown(futs)
            else:
                rd.shutdown(wait=True, cancel_futures=True)
            wr.shutdown(wait=True)


def v_rms_from_samples(s, N, scale_deg, robust=0.0, tukey_inner=30.0, svd_tol=1e-4, eps=1e-4):
    """QA statistic of one loaded snapshot (msio.load_samples dict): std of the central box of the
    dirty Stokes V image made with the snapshot's own Briggs weights (= orca pilot snapshot)."""
    if len(s["u"]) == 0:
        return float("nan")                # no unflagged data: rejected by qa_reject
    sc = math.radians(scale_deg)
    u = cp.asarray(s["u"]); v = cp.asarray(s["v"]); w = cp.asarray(s["w"]); nw = cp.asarray(s["nw"])
    WG = WeightGrid(N, sc, robust, tukey_inner or None)
    WG.add(u, v, nw); WG.finalize()
    iw = WG.weights(u, v, nw)
    svd = snapshot_svd(N, sc, float(cp.abs(w).max()), svd_tol)
    S = SampleSet(u, v, w, iw, N, sc, svd=svd, eps=eps, precompute=False)
    img = to_ws(S.grid(cp.asarray(s["V"]))) / float(iw.sum())
    return box_std(img)
