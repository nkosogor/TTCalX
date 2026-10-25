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
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import cupy as cp

from . import bands
from .clean import deconvolve_multiscale
from .fits import make_header, write_image
from .gridder import SampleSet, get_svd, to_ws
from .msio import load_samples
from .qa import qa_reject
from .reproject import Reprojector, fft_upsample
from .rotate import rotate_samples
from .weights import WeightGrid


# ----------------------------------------------------------------------------- loading
def parse_phase_centre(pc):
    """'02h30m00s 37d12m57.057s' (or 'ra_deg dec_deg') -> (ra, dec) rad."""
    from astropy.coordinates import Angle
    a, b = pc.split()
    ra = Angle(a, unit="hourangle") if any(c in a for c in "hm:") else Angle(float(a), unit="deg")
    return ra.rad, Angle(b, unit="deg").rad


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


# ----------------------------------------------------------------------------- gridding
class HourGrid:
    """Samples of many snapshots (kept on the host) gridded in chunks into one image.

    rotate=True: each snapshot's stored-frame (u, v, w, vis) are rotated by its R (snapshot
    dict 'R') on the fly.  weights=None: one global Briggs grid over all snapshots."""

    def __init__(self, snaps, N, scale_deg, robust=0.0, tukey_inner=None, eps=1e-4, svd_tol=1e-4,
                 chunk_pts=40_000_000, rotate=True, weights=None, need_svd=True, log=None):
        self.snaps = snaps; self.N = N; self.sdeg = scale_deg; self.sc = math.radians(scale_deg)
        self.eps = eps; self.chunk_pts = chunk_pts; self.rotate = rotate
        log = log or (lambda *a: None)
        t0 = time.perf_counter()

        def rot_uvw(i):
            s = snaps[i]
            u, v, w, _ = self._rot([i], cp.asarray(s["u"]), cp.asarray(s["v"]), cp.asarray(s["w"]))
            return u, v, w

        if weights is None:
            WG = WeightGrid(N, self.sc, robust, tukey_inner)
            for i, s in enumerate(snaps):
                u, v, _ = rot_uvw(i); WG.add(u, v, cp.asarray(s["nw"]))
            WG.finalize()
            self.iw = []
            for i, s in enumerate(snaps):
                u, v, _ = rot_uvw(i); self.iw.append(cp.asnumpy(WG.weights(u, v, cp.asarray(s["nw"]))))
        else:
            self.iw = weights
        sw = 0.0; wmax = 0.0
        for i in range(len(snaps)):
            _, _, w = rot_uvw(i)
            sw += float(self.iw[i].sum()); wmax = max(wmax, float(cp.abs(w).max()))
        self.sumw = sw
        self.svd = get_svd(N, self.sc, wmax, svd_tol) if need_svd else None
        self.chunks = self._subchunks(range(len(snaps)))
        self.npts = sum(len(s["u"]) for s in snaps)
        log(f"HourGrid: {len(snaps)} snaps, {self.npts / 1e6:.1f} M samples, {len(self.chunks)} chunks, "
            f"K={self.svd.K if self.svd else 0} (wmax {wmax:.0f}), {time.perf_counter() - t0:.1f} s")

    def _subchunks(self, ids):
        out = []; cur = []; n = 0
        for i in ids:
            if cur and n + len(self.snaps[i]["u"]) > self.chunk_pts:
                out.append(cur); cur = []; n = 0
            cur.append(i); n += len(self.snaps[i]["u"])
        if cur:
            out.append(cur)
        return out

    def _rot(self, ids, u, v, w, vis=None):
        """Stored-frame coordinates of snapshots ``ids`` (concatenated) -> common frame (chgcentre)."""
        if not self.rotate:
            return u, v, w, vis
        out = [[], [], [], []]; o = 0
        for i in ids:
            n = len(self.snaps[i]["u"]); sl = slice(o, o + n)
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
        u, v, w = (cp.asarray(s[k], dtype=cp.float64) for k in ("u", "v", "w"))
        wh = Rh[2, 0] * u + Rh[2, 1] * v + Rh[2, 2] * w
        return cp.exp(2j * np.pi * (wh - w)).astype(cp.complex64)

    def _chunk(self, ids, col):
        cat = lambda k: np.concatenate([self.snaps[i][k] for i in ids])
        if col == "PSF":
            vis = cp.concatenate([self._psf_vis(i) for i in ids])
        else:
            vis = cp.asarray(cat(col)) if col else None
        u, v, w, vis = self._rot(ids, cp.asarray(cat("u")), cp.asarray(cat("v")), cp.asarray(cat("w")), vis)
        wt = cp.asarray(np.concatenate([self.iw[i] for i in ids]))
        return SampleSet(u, v, w, wt, self.N, self.sc, svd=self.svd, eps=self.eps, precompute=False), vis

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
                d = dict(G.snaps[i]); d["R"] = Rb.T @ G.snaps[i]["R"]; d["R_hour"] = G.snaps[i]["R"]; sn.append(d)
            self.sub.append(HourGrid(sn, self.N, G.sdeg, rotate=block > 1, weights=[G.iw[i] for i in ids],
                                     eps=G.eps, svd_tol=1e-4))
            self.Rb.append(Rb)
        self._T = None
        (log or (lambda *a: None))(f"WSnap: {len(self.sub)} blocks of {block}, K = {sorted(set(g.svd.K for g in self.sub))}")

    def _bandlimit(self, M):
        """2x band-limited upsampling of the hour model, keeping |q| < umax (data support) and
        tapering to 0 at the original Nyquist so that spline reprojection is accurate."""
        if self._T is None:
            f = cp.fft.fftfreq(self.N).astype(cp.float32)
            q = cp.sqrt(f[:, None] ** 2 + f[None, :] ** 2) / self.sc
            umax = max(float(np.sqrt(s["u"] ** 2 + s["v"] ** 2 + s["w"] ** 2).max()) for s in self.G.snaps) * 1.02
            qn = 0.5 / self.sc
            t = cp.clip((q - umax) / max(qn - umax, 1), 0, 1)
            self._T = cp.where(q <= umax, 1.0, 0.5 * (1 + cp.cos(cp.pi * t))).astype(cp.float32)
        return fft_upsample(M, self.up, self._T)

    def image(self, col="I", model=None):
        acc = cp.zeros((self.N, self.N), cp.float32); wsum = 0.0
        Mb = self._bandlimit(model) if model is not None else None
        for g, R in zip(self.sub, self.Rb):
            ms = None
            if Mb is not None:
                co = self.P.coords(R.T)                       # block pixel -> hour-frame position
                ms, _ = self.P(Mb, None, self.order, coords=co, up=self.up)
                ms = ms * (co[2] / cp.maximum(self.P.n, 1e-6))  # model Jacobian n_hour / n_block
            d = g.image(col, model=ms)
            r, _ = self.P(d, R, self.order)
            acc += g.sumw * r; wsum += g.sumw
        return acc / wsum


# ----------------------------------------------------------------------------- driver
def theoretical_beam_pix(snaps, sc):
    return 1.0 / max(float(np.sqrt(s["u"] ** 2 + s["v"] ** 2 + s["w"] ** 2).max()) for s in snaps) / sc


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
            r = v_rms_from_samples(s, N, sdeg, params.robust, params.taper_inner_tukey)
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


def image_hour(paths, band, out_prefix, phase_centre=None, products=(("1h", 1, bands.DEEP_I),),
               mode="wsnap", block=6, pols="IV", qa="auto", qa_dir=None, exclude=(), nproc=8,
               beam=None, write_aux=True, verbose=False):
    """Image one band-hour.  products: HourProduct (bands.hour_products) or legacy
    (tag, n_intervals, CleanParams) tuples, all made from one load, e.g.
    bands.hour_products(bands.ORCA_ALL) or (("1h", 1, DEEP_I), ("10min", 6, TENMIN_I)).
    Returns the summary dict (also written to <out_prefix>-summary.json); summary["timing"]
    has seconds per step, keyed "<tag>: <step>" and accumulated over intervals."""
    log = (lambda *a: print(*a, flush=True)) if verbose else (lambda *a: None)
    T = {}; t0 = time.perf_counter(); last = [t0]

    def lap(k):
        cp.cuda.Device().synchronize(); now = time.perf_counter(); T[k] = T.get(k, 0) + now - last[0]; last[0] = now

    products = [_as_product(p, pols) for p in products]
    want_v = any(p.v_tag for p in products)
    N, sdeg = bands.geometry(band)
    ra, dec = parse_phase_centre(phase_centre) if phase_centre else middle_zenith(paths)
    snaps = load_all(list(paths), ra, dec, nproc, want_v=want_v or qa in ("auto", "compute"))
    snaps.sort(key=lambda s: s["time"])
    lap("load + rotation (CPU)")
    nfull = len(snaps)
    for i, s in enumerate(snaps):
        s["tidx"] = i
    # QA always with the orca pilot-snapshot weighting (briggs 0, inner Tukey 30)
    vr = snapshot_qa(snaps, N, sdeg, bands.SNAPSHOT_CLEAN_I, qa, qa_dir, log) if qa != "none" else [float("nan")] * nfull
    bad, med, sd = qa_reject(vr) if qa != "none" else ([], None, None)
    drop = set(bad) | set(int(x) for x in exclude) | {i for i, s in enumerate(snaps) if len(s["u"]) == 0 or s["zero_frac"] >= 0.999}
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
        for gi, sn in enumerate(groups):
            tsfx = "" if nint == 1 else f"-t{gi:04d}"
            name = f"{out_prefix}-{tag}{tsfx}"
            fn = _names(prod, name)
            if not sn:
                outs.append(dict(name=name, n_snap=0)); continue
            G = HourGrid(sn, N, sdeg, robust=params.robust, tukey_inner=params.taper_inner_tukey or None,
                         need_svd=(mode != "wsnap"), log=log)
            if mode == "wsnap":
                W = WSnap(G, block=block, log=log)
                imgfn = W.image
            else:
                imgfn = G.image
            lap(f"{key}: weights + gridder setup")
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
                v = imgfn("V"); lap(f"{vkey}: dirty V")
                h = hdr(di.get("beam"))
                vname = fn["V"] if prod.naming == "legacy" else f"{out_prefix}-{prod.v_tag}{tsfx}-image.fits"
                files.append(write_image(vname, v, h, "V"))
                lap(f"{vkey}: write")
                del v
            outs.append(dict(name=name, n_snap=len(sn), npts=G.npts, theo_beam_pix=theo, files=files,
                             K=sorted(set(g.svd.K for g in W.sub)) if mode == "wsnap" else G.svd.K, **di))
            # imgfn (a bound method of W / G) and the residual closure keep the gridders alive:
            # drop every reference before the next interval allocates its own (82 MHz 10-min OOM)
            del G, imgfn
            if mode == "wsnap":
                del W
            gc.collect()
            cp.fft.config.get_plan_cache().clear()
            cp.get_default_memory_pool().free_all_blocks()
        summary["products"][key] = outs
        if prod.image_i and prod.v_tag and prod.naming == "orca":
            summary["products"][prod.v_tag] = [dict(name=o["name"], n_snap=o["n_snap"],
                                                    files=[f for f in o.get("files", []) if prod.v_tag in os.path.basename(f)])
                                               for o in outs]
    T["TOTAL"] = time.perf_counter() - t0
    json.dump(summary, open(f"{out_prefix}-summary.json", "w"), indent=1, default=float)
    return summary
