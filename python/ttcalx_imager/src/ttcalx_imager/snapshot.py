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

import numpy as np
import cupy as cp

from . import bands
from .clean import clean_snapshot, mad_std
from .fits import make_header, write_image
from .gridder import SampleSet, dft_predict, get_svd, to_ws
from .msio import C, is_empty, read_ms
from .weights import WeightGrid

QA_BOX = 1024      # orca analyze_snapshot_quality: std of the central 1024 x 1024 pixels


def box_std(img, box=QA_BOX):
    N = img.shape[0]; h = min(box, N) // 2
    return float(img[N // 2 - h:N // 2 + h, N // 2 - h:N // 2 + h].std())


def snapshot_svd(N, sc, wf, tol):
    """w-term factorisation for a snapshot (|w| <~ 50 lambda): wmax rounded to 0.25 lambda."""
    return get_svd(N, sc, float(wf) * 1.0001 + 1e-6, tol, quantum=0.25, nw=600, pad=0.0)


class SnapshotImager:
    def __init__(self, eps=1e-4, svd_tol=1e-4, dft_max=8000, verbose=False):
        self.eps = eps; self.svd_tol = svd_tol; self.dft_max = dft_max
        self.log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)

    def image_arrays(self, u, v, w, wnat, visI, visV, N, scale_deg, params=bands.SNAPSHOT_CLEAN_I, beam=None,
                     timing=None):
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
        svd = snapshot_svd(N, sc, np.abs(w).max(), self.svd_tol)
        S = SampleSet(u, v, w, iw, N, sc, svd=svd, eps=self.eps)
        u32 = cp.asarray(u, dtype=cp.float32); v32 = cp.asarray(v, dtype=cp.float32); w32 = cp.asarray(w, dtype=cp.float32)
        dI = cp.asarray(visI)
        theo_pix = 1.0 / float(np.sqrt(u * u + v * v + w * w).max()) / sc     # WSClean theoretical beam
        lap("gridder setup")
        dirty = to_ws(S.grid(dI)) / sw
        lap("dirty I")
        psf = to_ws(cp.where(svd.inside, S.grid_psf(), 0)) / sw
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
        T = {}; t0 = time.perf_counter()
        if N is None:
            N, scale_deg = bands.geometry(band if band is not None else ms_path)
        ms = read_ms(ms_path, want_v="V" in pols)
        T["read MS"] = time.perf_counter() - t0
        info = dict(ms=ms_path, stem=ms.stem, N=N, scale_deg=scale_deg, time=ms.time, flag_frac=ms.flag_frac,
                    zero_frac=ms.zero_frac, status="ok")
        if is_empty(ms):
            info["status"] = "empty"
        else:
            wnat = ms.natw[:, None] * ms.ok
            sel = wnat > 0
            lam = C / ms.freqs
            # per-sample coordinates in wavelengths, float64 on the host
            u = (ms.uvw[:, 0:1] / lam[None])[sel]; v = (ms.uvw[:, 1:2] / lam[None])[sel]; w = (ms.uvw[:, 2:3] / lam[None])[sel]
            r = self.image_arrays(u, v, w, wnat[sel], ms.visI[sel], ms.visV[sel] if "V" in pols else None,
                                  N, scale_deg, params, beam, timing=T)
            info.update(r["info"])
            if out_prefix:
                t1 = time.perf_counter()
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
                info["files"] = [write_image(f"{out_prefix}-{name}.fits", arr, h, stokes=st) for name, arr, st in prods]
                T["write FITS"] = time.perf_counter() - t1
            info["gpu_pool_GB"] = cp.get_default_memory_pool().total_bytes() / 1e9
        T["TOTAL"] = time.perf_counter() - t0
        info["timing"] = T
        if out_prefix:
            json.dump(info, open(f"{out_prefix}-qa.json", "w"), indent=1, default=float)
        return info


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
