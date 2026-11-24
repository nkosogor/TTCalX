"""Snapshot gridding in the snapshot's stored (zenith) frame, everything else in the LST-hour frame
(``TTCALX_SNAP_ZEN``, default off).

The gsi v3 engine images each snapshot at the LST-hour phase centre (chgcentre UVW + re-phased DATA).
Late and early in the hour the hour-frame |w| is large, so the w-term needs K ~ 40-60 NUFFT layers per
gridding pass; in the snapshot's own stored (zenith) frame the array is nearly coplanar and K ~ 7-8.
This module replaces ONLY the gridding of the snapshot imager (dirty I, PSF, the two major-cycle
residuals, dirty V): each pass is

    vis_hour * b            b = exp(+2 pi i (w_hour - w_stored)) undoes chgcentre's re-phasing
    -> zenith-frame gridding (the production NUFFT + SVD w-term, K_z layers, x' table extended to
       n in [-NLOW, 1] so that directions below the zenith-frame horizon are covered)
    -> order-5 B-spline (FIR prefilter) reprojection to the production hour grid (N, scale, LST-hour
       centre: (l, m, -n)_hour = R (l, m, -n)_stored with uvw_hour = R uvw_stored)
    -> x production's a_k interpolation gain (see ``gain_map``) / sum of weights, zero beyond the horizon.

Hour pixels whose direction is less than ``ring_deg`` above the zenith-frame horizon or within ``ring_px``
zenith-grid pixels of it (and every pixel below it) are not interpolated from the zenith image (its w-term factor varies too fast there for a
spline: the "horizon ring" error of the 2026-11-23 prototype, 1-2 sigma near bright sources at el ~ 0):
they are evaluated per w-term layer: the layer's band-limited NUFFT output F_k interpolated at the exact
zenith-frame position (the same quintic B-spline + FIR prefilter, evaluated only where the ring needs it)
times b_k at the exact x' = n - 1/2 of the direction.

Everything else is the production code: Briggs / Tukey weights on the hour-frame uv (WeightGrid), the
theoretical beam, the PSF normalisation, the restoring-beam fit, clean_snapshot (Clark / Hogbom, local RMS,
auto-mask, MAD), the prediction (float32 DFT up to ``dft_max`` components, else the hour-frame
SampleSet.degrid), restore, QA statistics, FITS header and writing.

Production's hour-frame gridder interpolates the a_k(w) (600 points) and b_k(x') (2001 points) tables
linearly, which loses up to ~0.3 % (55 MHz) / ~0.5 % (73 MHz, K 58-59) of the flux at the hour ends.
``gain_map`` reproduces that loss per pixel (to second order) so that the products stay those of production
(TTCALX_SNAP_ZEN_AK_LOSS=1, the default).  TTCALX_SNAP_ZEN_AK_LOSS=0 gives the loss-free image instead
(differs from production by that loss; not validated as a product change -- see README "Known issues").

Per-snapshot router (``use_zen``): the zenith path is used only where it is faster, i.e. when the hour
table's rank K_h is at least K_z + TTCALX_SNAP_ZEN_DK (mid-hour at 41-82 MHz the hour frame is as cheap).

Switches (environment, read per snapshot):
  TTCALX_SNAP_ZEN          0 (default: production) | 1 / auto (per-snapshot router) | force (always, tests)
  TTCALX_SNAP_ZEN_DK       router margin in w-term layers (default DK_DEFAULT)
  TTCALX_SNAP_ZEN_GZ       zenith grid oversampling of the I / V / residual passes (default 1)
  TTCALX_SNAP_ZEN_PSFGZ    zenith grid oversampling of the PSF pass (default 1.5: beam within ~2e-5)
  TTCALX_SNAP_ZEN_RING_DEG horizon ring: zenith-frame elevation below which pixels are evaluated per layer (6)
  TTCALX_SNAP_ZEN_RING_PX  ... and distance to the zenith-frame horizon in zenith-grid pixels (8)
  TTCALX_SNAP_ZEN_AK_LOSS  1 (default): emulate production's a_k / b_k interpolation loss; 0: exact
The engine hands the stored-frame UVW over as small["uvw_stored"] (snapshot.gpu_select builds the
zenith-frame samples from it); without it the production path runs.
"""
import math
import os

import numpy as np
import cupy as cp

from . import gridder
from .gridder import SvdWTerm, fast_ak, pruned_on, to_ws

NW = 600                 # a_k(w) table points (as the production snapshot table)
DX = 1.0 / 2000.0        # b_k(x') table step (as production: 2001 points on [-1/2, 1/2])
NLOW = 0.15              # x' table extended to n = -NLOW (directions below the zenith-frame horizon)
RMAX = 1.006             # zenith image continued (n = 0) up to this radius before the spline prefilter
FIR_H = 20               # quintic B-spline prefilter: FIR half length (taps decay as 0.43^k)
EDGE = 16                # zenith grid margin beyond the horizon [pixels] (ring spline support + FIR tail)
DK_DEFAULT = 4           # router margin (see README; measured 2026-11-24)
RING_DEG_DEFAULT = 6.0   # horizon ring: at least this zenith-frame elevation ...
RING_PX_DEFAULT = 8.0    # ... and at least this many zenith-grid pixels (radially) from the horizon
QUANTUM = 0.25           # snapshot table key quantum (snapshot.snapshot_svd)


_OOM_PREFIXES = ("Error setting non-uniform points", "Error creating plan")


def is_oom(e):
    """True for an out-of-GPU-memory error of the zenith path (CuPy OutOfMemoryError, a CUDA "out of memory", or a
    cuFINUFFT plan / setpts allocation failure); SnapshotImager.image_arrays then images the snapshot on the
    production path.  Anything else is a bug and is raised."""
    if isinstance(e, cp.cuda.memory.OutOfMemoryError):
        return True
    msg = str(e)
    return "out of memory" in msg.lower() or msg.startswith(_OOM_PREFIXES)


def mode():
    """TTCALX_SNAP_ZEN: 'off' (default), 'auto' (1 / auto: per-snapshot router) or 'force'."""
    v = os.environ.get("TTCALX_SNAP_ZEN", "0").strip().lower()
    if v in ("", "0", "off", "no", "false"):
        return "off"
    return "force" if v == "force" else "auto"


def _envf(name, default):
    v = os.environ.get(name, "")
    return float(v) if v.strip() else float(default)


def settings():
    return dict(gz=_envf("TTCALX_SNAP_ZEN_GZ", 1.0), psfgz=_envf("TTCALX_SNAP_ZEN_PSFGZ", 1.5),
                ring_deg=_envf("TTCALX_SNAP_ZEN_RING_DEG", RING_DEG_DEFAULT),
                ring_px=_envf("TTCALX_SNAP_ZEN_RING_PX", RING_PX_DEFAULT), dk=_envf("TTCALX_SNAP_ZEN_DK", DK_DEFAULT),
                ak_loss=os.environ.get("TTCALX_SNAP_ZEN_AK_LOSS", "1") != "0")


def table_wq(wabs):
    """The snapshot table key of a w range (snapshot.snapshot_svd: wmax * 1.0001 + 1e-6 up to 0.25 lambda)."""
    return math.ceil((float(wabs) * 1.0001 + 1e-6) / QUANTUM) * QUANTUM


def _zen_nx(nlow=NLOW):
    return int(round((0.5 - (-0.5 - nlow)) / DX)) + 1


def hour_rank(wabs_h, tol):
    """K of the production hour-frame snapshot table for this w range (the factors production would use: cached)."""
    return gridder._svd_factors(table_wq(wabs_h), tol, NW, 2001)[0]


def zen_rank(wabs_z, tol, nlow=NLOW):
    return gridder._svd_factors(table_wq(wabs_z), tol, NW, _zen_nx(nlow), xmin=-0.5 - nlow)[0]


def use_zen(wabs_h, wabs_z, tol):
    """Router: (use the zenith path?, K_h, K_z)."""
    m = mode()
    if m == "off":
        return False, None, None
    if float(os.environ.get("GSI_SVD_QUANTUM", "0") or 0) > 0:
        # the gsi engine then replaces snapshot.snapshot_svd with another table quantum: the a_k-loss emulation and
        # the router assume production's 0.25 lambda key, so keep the production path
        return False, None, None
    kh, kz = hour_rank(wabs_h, tol), zen_rank(wabs_z, tol)
    return (m == "force" or kh >= kz + settings()["dk"]), kh, kz


class ZenSamples:
    """The stored-frame (zenith) samples of a snapshot for snapshot.gpu_select's selection: max |w| [lambda] at
    once, for the router; w, u, v (float64, the selection's order) and R (uvw = R uvw_stored, rotate.rotation_from_uvw
    over the cross-correlations, as the v3 record's rotation) on first use.  Item access: zen["u"] etc."""

    def __init__(self, small, sel, lam):
        self._small = small; self._sel = sel; self._lam = lam
        self._ud = cp.asarray(np.asarray(small["uvw_stored"], np.float64))
        # max |w| over the selection without the per-sample array (the router alone needs it; zen["w"] is built only
        # when the zenith path runs): per row |w_row| / (smallest selected lambda) -- the same float64 value, since
        # |x / l| = |x| / l and correctly rounded division is monotonic in l > 0 (rows with no sample give 0)
        lmin = cp.where(sel, lam[None], cp.inf).min(axis=1)
        self.wabs = float((cp.abs(self._ud[:, 2]) / lmin).max())
        del lmin
        self._c = {}

    @property
    def w(self):
        return self["w"]

    def drop_samples(self):
        """Free the per-sample stored-frame u, v, w (float64; rebuilt on the next access).  R and wabs are kept."""
        for k in ("u", "v", "w"):
            self._c.pop(k, None)

    def _col(self, i):
        return (self._ud[:, i:i + 1] / self._lam[None])[self._sel]

    def __getitem__(self, k):
        if k not in self._c:
            if k == "w":
                self._c[k] = self._col(2)
            elif k == "u":
                self._c[k] = self._col(0)
            elif k == "v":
                self._c[k] = self._col(1)
            elif k == "R":
                from .rotate import rotation_from_uvw
                sm = self._small
                keep = np.asarray(sm["a1"]) != np.asarray(sm["a2"])
                us = np.asarray(sm["uvw_stored"], np.float64)[keep]; uh = np.asarray(sm["uvw"], np.float64)[keep]
                R = rotation_from_uvw(us, uh)
                # the fit's residual (ZenSnapshot falls back when uvw is not R uvw_stored to ~1e-6 of max |uvw|)
                self.R_res = float(np.abs(us @ R.T - uh).max()) if uh.size else 0.0
                self.R_scale = float(np.abs(uh).max()) if uh.size else 0.0
                self._c[k] = R
            else:
                raise KeyError(k)
        return self._c[k]


# ----------------------------------------------------------------------------- caches
_TABLES = {}             # (Nz, scz, wq, tol, nlow, rmax) -> ZenTable (one per grid size kept)
_GEOM = {}               # (Nz, scz, rmax, xmin, nx) -> (inside, x', j, t)
_PLANS = {}              # (Nz, eps, pruned) -> [type-1 plan, token of the points set]
_GAIN = {}               # (N, sc) -> (x'^2, s (1 - s)) float64 WSClean orientation
_TAPS = {}


def clear():
    _TABLES.clear(); _GEOM.clear(); _PLANS.clear(); _GAIN.clear()


def _geometry(Nz, scz, rmax, xmin, nx):
    """Zenith grid (raw orientation): inside r < rmax, x' = n - 1/2 (n = sqrt(1 - r^2), 0 for 1 <= r < rmax) and
    its cell index / fraction on the x' table grid (as gridder._geometry, on [xmin, 1/2])."""
    key = (int(Nz), round(float(scz), 15), float(rmax), float(xmin), int(nx))
    hit = _GEOM.get(key)
    if hit is None:
        j = (cp.arange(Nz, dtype=cp.float64) - Nz // 2) * scz
        r2 = j[:, None] ** 2 + j[None, :] ** 2
        inside = r2 < rmax ** 2
        xp = cp.where(inside, cp.sqrt(cp.maximum(1 - r2, 0)) - 0.5, 0.5).astype(cp.float32).ravel()
        del j, r2
        xgd = cp.asarray(np.linspace(xmin, 0.5, nx), dtype=cp.float32)
        jj = cp.clip(cp.searchsorted(xgd, xp, side="right") - 1, 0, nx - 2)
        t = ((xp.astype(cp.float64) - xgd[jj].astype(cp.float64))
             / (xgd[jj + 1].astype(cp.float64) - xgd[jj].astype(cp.float64))).astype(cp.float32)
        if len(_GEOM) >= 2:
            _GEOM.pop(next(iter(_GEOM)))
        hit = _GEOM[key] = (inside, xp, jj.astype(cp.int16), t)
    return hit


def _min_ns(N, sc, rt):
    """The lowest stored-frame n of the hour sky (n_hour = 0 circle, the extreme of a rotated hemisphere)."""
    r = cp.asnumpy(rt).reshape(3, 3)
    # n_s = -(a l + b m) + c n_h with (a, b, c) = row 2 of R^T: the minimum over the hemisphere n_h >= 0 is on the
    # horizon (-hypot(a, b)) unless c < 0 (then -|(a, b, c)|)
    a, b, c = r[2]
    return -math.sqrt(a * a + b * b + c * c) if c < 0 else -math.hypot(a, b)


class ZenTable(SvdWTerm):
    """SvdWTerm on x' in [-1/2 - nlow, 1/2] (directions down to n = -nlow) for the zenith grid (Nz, scz)."""

    def __init__(self, Nz, scz, wq, tol, nlow=NLOW, rmax=RMAX, nw=NW):
        xmin = -0.5 - nlow; nx = _zen_nx(nlow)
        K, wg, A, Vh = gridder._svd_factors(wq, tol, nw, nx, xmin=xmin)
        self.K = K; self.wmax = wq; self.N = Nz; self.nx = nx; self.xmin = xmin
        self.wg = cp.asarray(wg, dtype=cp.float32)
        self.Ar = [cp.asarray(A[:, k].real, dtype=cp.float32) for k in range(K)]
        self.Ai = [cp.asarray(A[:, k].imag, dtype=cp.float32) for k in range(K)]
        self.inside, self.xp, self.jx, self.tx = _geometry(Nz, scz, rmax, xmin, nx)
        self.xgd = cp.asarray(np.linspace(xmin, 0.5, nx), dtype=cp.float32)
        self.Br = [cp.asarray(Vh[k].real, dtype=cp.float32) for k in range(K)]
        self.Bi = [cp.asarray(Vh[k].imag, dtype=cp.float32) for k in range(K)]


def _table(Nz, scz, wq, tol):
    key = (int(Nz), round(float(scz), 12), round(wq, 3), tol, NLOW, RMAX)
    t = _TABLES.get(key)
    if t is None:
        for k in [k for k in _TABLES if k[0] == Nz]:       # one table per grid size (w range moves slowly)
            del _TABLES[k]
        t = _TABLES[key] = ZenTable(Nz, scz, wq, tol)
    return t


def _plan(Nz, eps):
    """Type-1 plan of a zenith grid (no type-2 plan: the zenith path never degrids)."""
    pr = pruned_on("snap")
    key = (int(Nz), eps, pr)
    p = _PLANS.get(key)
    if p is None:
        if pr:
            from .pruned import PrunedPlan
            p = [PrunedPlan(1, Nz, eps, 1), None]
        else:
            import cufinufft
            p = [cufinufft.Plan(1, (Nz, Nz), n_trans=1, eps=eps, isign=1, dtype="complex64"), None]
        _PLANS[key] = p
    return p


def _quintic_taps(H=FIR_H, M=8192):
    """FIR approximation (2H + 1 taps) of the quintic B-spline prefilter 1 / B5(omega)."""
    if H not in _TAPS:
        w = 2 * np.pi * np.arange(M) / M
        B = (66 + 52 * np.cos(w) + 2 * np.cos(2 * w)) / 120.0
        h = np.real(np.fft.ifft(1.0 / B))
        _TAPS[H] = np.array([h[k % M] for k in range(-H, H + 1)], np.float64)
    return _TAPS[H]


def grid_size(N, sc, g):
    """(Nz, scz) of a zenith grid with oversampling g: N g pixels on the hour image's field (scz = sc N / Nz),
    enlarged (same scz) until the horizon plus the ring gather's support fits."""
    Nz = int(round(N * g)); Nz += Nz % 2
    scz = sc * N / Nz
    need = 2 * (int(math.ceil(1.0 / scz)) + EDGE)
    return max(Nz, need + need % 2), scz


# ----------------------------------------------------------------------------- kernels
_B5 = r"""
__device__ __forceinline__ float ttcalx_b5(float x) {
    x = fabsf(x);
    if (x < 1.f) { const float x2 = x * x; return (66.f - 60.f * x2 + 30.f * x2 * x2 - 10.f * x2 * x2 * x) / 120.f; }
    if (x < 2.f) { const float x2 = x * x;
        return (51.f + 75.f * x - 210.f * x2 + 150.f * x2 * x - 45.f * x2 * x2 + 5.f * x2 * x2 * x) / 120.f; }
    if (x < 3.f) { const float u = 3.f - x; return u * u * u * u * u / 120.f; }
    return 0.f;
}
"""

_FIR = cp.RawKernel(r'''
extern "C" __global__ void ttcalx_zen_fir(const float* __restrict__ in, float* __restrict__ out, const int N,
                                          const int axis, const float* __restrict__ h, const int H) {
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)N * N) return;
    const int y = (int)(t / N), x = (int)(t % N);
    float acc = 0.f;
    if (axis == 1) {
        for (int k = -H; k <= H; ++k) { const int xx = x + k; if (xx >= 0 && xx < N) acc += h[k + H] * in[(long long)y * N + xx]; }
    } else {
        for (int k = -H; k <= H; ++k) { const int yy = y + k; if (yy >= 0 && yy < N) acc += h[k + H] * in[(long long)yy * N + x]; }
    }
    out[t] = acc;
}''', "ttcalx_zen_fir")

_COORD = r"""
// hour pixel t of the N x N WSClean-orientation grid -> raw zenith-grid position (fx = ix, fy = iy) and n_stored
__device__ __forceinline__ bool ttcalx_zen_pos(const long long t, const int N, const double sc, const int Nz,
    const double scz, const double* __restrict__ rt, double* fx, double* fy, double* ns) {
    const int y = (int)(t / N), x = (int)(t % N);
    const double l = (double)(x - N / 2) * sc, m = (double)(((N - y) % N) - N / 2) * sc;
    const double r2 = l * l + m * m;
    if (!(r2 < 1.0)) return false;
    const double c = -sqrt(1.0 - r2);
    const double ls = rt[0] * l + rt[1] * m + rt[2] * c;
    const double ms = rt[3] * l + rt[4] * m + rt[5] * c;
    *ns = -(rt[6] * l + rt[7] * m + rt[8] * c);
    *fx = ls / scz + Nz / 2; *fy = ms / scz + Nz / 2;
    return true;
}
"""

_GATHER = cp.RawKernel(_B5 + _COORD + r'''
extern "C" __global__ void ttcalx_zen_gather(const float* __restrict__ pf, const int Nz, const double scz, const int N,
    const double sc, const double* __restrict__ rt, const float* __restrict__ gain, const float scale,
    float* __restrict__ out)
{
    // out[y, x] = scale * gain * quintic spline of the prefiltered raw zenith image at the pixel's direction
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)N * N) return;
    double fx, fy, ns;
    if (!ttcalx_zen_pos(t, N, sc, Nz, scz, rt, &fx, &fy, &ns)) { out[t] = 0.f; return; }
    const int ix0 = (int)floor(fx), iy0 = (int)floor(fy);
    const float tx = (float)(fx - ix0), ty = (float)(fy - iy0);
    float wx[6], wy[6];
    for (int k = 0; k < 6; ++k) { wx[k] = ttcalx_b5(tx - (float)(k - 2)); wy[k] = ttcalx_b5(ty - (float)(k - 2)); }
    float acc = 0.f;
    for (int a = 0; a < 6; ++a) {
        const int ix = ix0 - 2 + a;
        if (ix < 0 || ix >= Nz) continue;
        float s = 0.f;
        const float* row = pf + (long long)ix * Nz;
        for (int b = 0; b < 6; ++b) { const int iy = iy0 - 2 + b; if (iy >= 0 && iy < Nz) s += wy[b] * row[iy]; }
        acc += wx[a] * s;
    }
    out[t] = acc * scale * (gain ? gain[t] : 1.f);
}''', "ttcalx_zen_gather")

_RINGSEL = cp.RawKernel(_COORD + r'''
extern "C" __global__ void ttcalx_zen_ringsel(const int N, const double sc, const int Nz, const double scz,
    const double* __restrict__ rt, const double nring, bool* __restrict__ ring)
{
    const long long t = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= (long long)N * N) return;
    double fx, fy, ns;
    ring[t] = ttcalx_zen_pos(t, N, sc, Nz, scz, rt, &fx, &fy, &ns) && ns < nring;
}''', "ttcalx_zen_ringsel")

_RINGPREP = cp.RawKernel(_COORD + r"""
extern "C" __global__ void ttcalx_zen_ringprep(const long long* __restrict__ idx, const long long n, const int N,
    const double sc, const int Nz, const double scz, const double* __restrict__ rt, const float* __restrict__ xg,
    const int nx, int* __restrict__ i0, float* __restrict__ txy, short* __restrict__ jb, float* __restrict__ tb,
    bool* __restrict__ supp)
{
    // ring target r: spline cell (ix0, iy0) and fractions (as ttcalx_zen_gather), the b_k table cell / fraction of
    // x' = n_stored - 1/2, and its 6 x 6 spline support marked in supp (the zenith pixels whose coefficients it needs)
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double fx, fy, ns;
    ttcalx_zen_pos(idx[r], N, sc, Nz, scz, rt, &fx, &fy, &ns);
    const int ix0 = (int)floor(fx), iy0 = (int)floor(fy);
    i0[2 * r] = ix0; i0[2 * r + 1] = iy0;
    txy[2 * r] = (float)(fx - ix0); txy[2 * r + 1] = (float)(fy - iy0);
    for (int a = 0; a < 6; ++a) {
        const int ix = ix0 - 2 + a;
        if (ix < 0 || ix >= Nz) continue;
        for (int b = 0; b < 6; ++b) { const int iy = iy0 - 2 + b; if (iy >= 0 && iy < Nz) supp[(long long)ix * Nz + iy] = true; }
    }
    const float x = (float)(ns - 0.5);
    int lo = 0, hi = nx;                         // searchsorted(xg, x, 'right') - 1, clipped to [0, nx - 2]
    while (lo < hi) { const int mid = (lo + hi) >> 1; if (xg[mid] <= x) lo = mid + 1; else hi = mid; }
    int j = lo - 1; j = j < 0 ? 0 : (j > nx - 2 ? nx - 2 : j);
    jb[r] = (short)j;
    tb[r] = (float)(((double)x - (double)xg[j]) / ((double)xg[j + 1] - (double)xg[j]));
}""", "ttcalx_zen_ringprep")

_DILATE = cp.RawKernel(r"""
extern "C" __global__ void ttcalx_zen_dilate(const long long* __restrict__ q, const long long n, const int Nz,
                                             const int H, bool* __restrict__ out)
{
    // out: the pixels within H along axis 0 of the listed ones (the column FIR's support)
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int ix = (int)(q[i] / Nz), iy = (int)(q[i] % Nz);
    for (int a = -H; a <= H; ++a) { const int x = ix + a; if (x >= 0 && x < Nz) out[(long long)x * Nz + iy] = true; }
}""", "ttcalx_zen_dilate")

_RFIR = cp.RawKernel(r"""
#include <cupy/complex.cuh>
extern "C" __global__ void ttcalx_zen_rfir(const complex<float>* __restrict__ in, complex<float>* __restrict__ out,
    const long long* __restrict__ q, const long long n, const int Nz, const int axis, const float* __restrict__ h,
    const int H)
{
    // the prefilter's FIR (zero outside the grid) of a complex raw image, at the listed pixels only
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const long long p = q[i];
    const int ix = (int)(p / Nz), iy = (int)(p % Nz);
    float re = 0.f, im = 0.f;
    if (axis == 1) {
        for (int k = -H; k <= H; ++k) { const int y = iy + k;
            if (y >= 0 && y < Nz) { const complex<float> v = in[(long long)ix * Nz + y]; re += h[k + H] * v.real(); im += h[k + H] * v.imag(); } }
    } else {
        for (int k = -H; k <= H; ++k) { const int x = ix + k;
            if (x >= 0 && x < Nz) { const complex<float> v = in[(long long)x * Nz + iy]; re += h[k + H] * v.real(); im += h[k + H] * v.imag(); } }
    }
    out[p] = complex<float>(re, im);
}""", "ttcalx_zen_rfir")

_RINGACC = cp.RawKernel(_B5 + r"""
#include <cupy/complex.cuh>
extern "C" __global__ void ttcalx_zen_ringacc(const complex<float>* __restrict__ c, const int Nz, const long long n,
    const int* __restrict__ i0, const float* __restrict__ txy, const short* __restrict__ jb,
    const float* __restrict__ tb, const float* __restrict__ br, const float* __restrict__ bi, float* __restrict__ acc)
{
    // acc[r] += Re(F_k(position_r) b_k(x'_r)): F_k = quintic spline of the layer's raw NUFFT output (prefiltered
    // coefficients c at the target's support, as the zenith image's own reprojection)
    const long long r = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    const int ix0 = i0[2 * r], iy0 = i0[2 * r + 1];
    const float tx = txy[2 * r], ty = txy[2 * r + 1];
    float wx[6], wy[6];
    for (int k = 0; k < 6; ++k) { wx[k] = ttcalx_b5(tx - (float)(k - 2)); wy[k] = ttcalx_b5(ty - (float)(k - 2)); }
    float re = 0.f, im = 0.f;
    for (int a = 0; a < 6; ++a) {
        const int ix = ix0 - 2 + a;
        if (ix < 0 || ix >= Nz) continue;
        float sre = 0.f, sim = 0.f;
        for (int b = 0; b < 6; ++b) {
            const int iy = iy0 - 2 + b;
            if (iy >= 0 && iy < Nz) { const complex<float> v = c[(long long)ix * Nz + iy]; sre += wy[b] * v.real(); sim += wy[b] * v.imag(); }
        }
        re += wx[a] * sre; im += wx[a] * sim;
    }
    const int j = jb[r]; const float t = tb[r];
    const float b_re = br[j] + t * (br[j + 1] - br[j]), b_im = bi[j] + t * (bi[j + 1] - bi[j]);
    acc[r] = __fadd_rn(acc[r], re * b_re - im * b_im);
}""", "ttcalx_zen_ringacc")


def _launch(kernel, n, args, bs=256):
    kernel(((int(n) + bs - 1) // bs,), (bs,), args)


def prefilter(img):
    """Quintic B-spline prefilter (FIR, zero outside the grid) of a square float32 image."""
    N = img.shape[0]
    h = cp.asarray(_quintic_taps(), dtype=cp.float32)
    src = cp.ascontiguousarray(img, dtype=cp.float32); tmp = cp.empty_like(src)
    _launch(_FIR, src.size, (src, tmp, np.int32(N), np.int32(1), h, np.int32(FIR_H)))
    _launch(_FIR, src.size, (tmp, src, np.int32(N), np.int32(0), h, np.int32(FIR_H)))
    return src


def gain_map(wfold_h, wabs_h, iw, sw, N, sc):
    """Per-pixel mean gain (WSClean orientation) of production's hour-frame snapshot gridder relative to the
    exact w-term: linear interpolation of a_k(w) (NW points on [0, wq]) and b_k(x') (2001 points) multiplies a
    sample's term at pixel p by 1 - 2 pi^2 [t (1 - t) (dw x'_p)^2 + s_p (1 - s_p) (dx w)^2] to second order
    (t, s_p: fractional table positions); averaged over the samples with the imaging weights.  Production's
    sky-dependent flux loss (up to ~0.5 % at the hour ends at 73 MHz) is thus kept (see the module doc)."""
    wq = table_wq(wabs_h)
    dw = wq / (NW - 1.0)
    wg = cp.asarray(np.linspace(0, wq, NW), dtype=cp.float32).astype(cp.float64)
    w = wfold_h.astype(cp.float64)
    j = cp.clip(cp.searchsorted(wg, w, side="right") - 1, 0, NW - 2)
    t = (w - wg[j]) / (wg[j + 1] - wg[j])
    iw64 = iw.astype(cp.float64)
    g1 = float(cp.dot(iw64, t * (1 - t))) / sw
    g2 = float(cp.dot(iw64, w * w)) / sw
    del wg, w, j, t, iw64
    key = (int(N), round(float(sc), 15))
    if key not in _GAIN:
        _, xp, _, tx = gridder._geometry(N, sc, 1.0, 2001)
        x2 = to_ws(xp.reshape(N, N)).astype(cp.float64) ** 2
        s = to_ws(tx.reshape(N, N)).astype(cp.float64)
        _GAIN.clear()
        _GAIN[key] = (x2, s * (1 - s))
    x2, s1s = _GAIN[key]
    return (1.0 - 2 * np.pi ** 2 * (g1 * dw * dw * x2 + g2 * DX * DX * s1s)).astype(cp.float32), dict(wq=wq, g1=g1, g2=g2)


# ----------------------------------------------------------------------------- per snapshot
class _Grid:
    """One zenith grid of a snapshot: table, plan, folded sample coordinates, ring targets."""

    def __init__(self, Z, g, tol, eps):
        self.Nz, self.scz = grid_size(Z.N, Z.sc, g)
        self.table = _table(self.Nz, self.scz, table_wq(Z.wabs_z), tol)
        self.plan = _plan(self.Nz, eps)
        self.x = (2 * np.pi * Z.uf * self.scz).astype(cp.float32)
        self.y = (2 * np.pi * Z.vf * self.scz).astype(cp.float32)
        self.token = object()
        # ring targets: hour pixels (inside the hour horizon) less than ring_deg above the zenith-frame horizon or
        # within ring_px zenith-grid pixels of it (the spline's error grows as the horizon gets closer in pixels)
        N = Z.N
        r_px = max(0.0, 1.0 - Z.opts["ring_px"] * self.scz)
        self.n_ring = max(math.sin(math.radians(Z.opts["ring_deg"])), math.sqrt(max(0.0, 1.0 - r_px * r_px)))
        ring = cp.empty(N * N, cp.bool_)
        _launch(_RINGSEL, N * N, (np.int32(N), np.float64(Z.sc), np.int32(self.Nz), np.float64(self.scz), Z.rt,
                                  np.float64(self.n_ring), ring))
        self.ridx = cp.flatnonzero(ring)
        del ring
        n = int(self.ridx.size)
        self.i0 = cp.empty(2 * n, cp.int32); self.txy = cp.empty(2 * n, cp.float32)
        self.jb = cp.empty(n, cp.int16); self.tb = cp.empty(n, cp.float32)
        self.q1 = self.q0 = None
        if n:
            supp = cp.zeros(self.Nz * self.Nz, cp.bool_)
            _launch(_RINGPREP, n, (self.ridx, np.int64(n), np.int32(N), np.float64(Z.sc), np.int32(self.Nz),
                                   np.float64(self.scz), Z.rt, self.table.xgd, np.int32(self.table.nx), self.i0, self.txy,
                                   self.jb, self.tb, supp))
            self.q0 = cp.flatnonzero(supp)                      # spline coefficients needed by the ring
            supp.fill(False)
            _launch(_DILATE, self.q0.size, (self.q0, np.int64(self.q0.size), np.int32(self.Nz), np.int32(FIR_H), supp))
            self.q1 = cp.flatnonzero(supp)                      # row-FIR outputs needed by their column FIR
            del supp

    def ring_layer(self, f, k, racc, bufs):
        """racc += Re(F_k b_k) at the ring targets: F_k = f (raw layer image) prefiltered at the needed pixels only
        (the FIR of ``prefilter``) and B-spline interpolated (as the reprojection of the zenith image)."""
        n = int(self.ridx.size); Nz = self.Nz
        h = cp.asarray(_quintic_taps(), dtype=cp.float32)
        fx, cc = bufs
        _launch(_RFIR, self.q1.size, (f, fx, self.q1, np.int64(self.q1.size), np.int32(Nz), np.int32(1), h, np.int32(FIR_H)))
        _launch(_RFIR, self.q0.size, (fx, cc, self.q0, np.int64(self.q0.size), np.int32(Nz), np.int32(0), h, np.int32(FIR_H)))
        _launch(_RINGACC, n, (cc, np.int32(Nz), np.int64(n), self.i0, self.txy, self.jb, self.tb,
                              self.table.Br[k], self.table.Bi[k], racc))

    def setpts(self):
        if self.plan[1] is not self.token:
            gridder._room_for_setpts(int(self.x.size))
            self.plan[0].setpts(self.x, self.y)
            self.plan[1] = self.token


class ZenFar(Exception):
    """The hour sky reaches n_stored < -NLOW (snapshot imaged more than asin(NLOW) ~ 8.6 deg from its zenith): the
    extended b_k(x') table would be extrapolated there, so the snapshot is not imaged on the zenith path
    (SnapshotImager.image_arrays then runs the production path)."""

    def __init__(self, n_min):
        self.n_min = n_min
        super().__init__(f"zenith: hour pixels reach n_stored = {n_min:.3f} < -{NLOW} (snapshot imaged > "
                         f"{math.degrees(math.asin(NLOW)):.1f} deg from its zenith): production path")


class ZenBadR(Exception):
    """The hour-frame uvw is not a rotation of the stored-frame uvw (least-squares R residual > R_TOL max |uvw|):
    the zenith path's b = exp(2 pi i (w_hour - w_stored)) and reprojection would be wrong, so the snapshot is
    imaged on the production path (as ZenFar)."""

    def __init__(self, res, scale):
        self.res, self.scale = res, scale
        super().__init__(f"zenith: uvw is not R uvw_stored (fit residual {res:.3g} m, max |uvw| {scale:.3g} m): "
                         "production path")


R_TOL = 1e-6             # ZenBadR: relative residual of the R fit


class ZenSnapshot:
    """Zenith-frame gridding of one snapshot; ``image(vis_hour)`` stands in for to_ws(S.grid(vis_hour)) / sw.

    zen: dict(u, v, w: stored-frame per-sample coordinates [lambda] (float64 CuPy, the samples of the hour-frame
    selection, unfolded), R: uvw_hour = R uvw_stored); w_hour: the hour-frame w [lambda] of the same samples
    (float64); pre: gpu_select's hour-frame 'pre' dict; iw / sw: production imaging weights and their sum."""

    def __init__(self, zen, w_hour, pre, iw, sw, N, sc, tol=1e-4, eps=1e-4, opts=None):
        self.N = N; self.sc = sc; self.sw = sw
        R = zen["R"]
        res = getattr(zen, "R_res", None)
        if res is not None and res > R_TOL * zen.R_scale:
            raise ZenBadR(res, zen.R_scale)
        self.rt = cp.asarray(np.ascontiguousarray(np.asarray(R, np.float64).T).ravel())
        lo = float(_min_ns(N, sc, self.rt))
        if lo < -NLOW:      # before any GPU work: the caller falls back to the production path
            raise ZenFar(lo)
        self.opts = opts or settings()
        u, v, w = zen["u"], zen["v"], zen["w"]
        self.b = cp.exp(2j * np.pi * (w_hour - w)).astype(cp.complex64)       # zenith vis = hour vis * b
        self.flip = w < 0
        sg = cp.where(self.flip, -1.0, 1.0)
        self.uf, self.vf = u * sg, v * sg
        self.wf = (w * sg).astype(cp.float32)
        self.wabs_z = float(cp.abs(w).max()) if not hasattr(zen, "wabs") else zen.wabs
        self.wt = cp.asarray(iw, dtype=cp.float32)
        self.eph = cp.exp(1j * cp.pi * self.wf).astype(cp.complex64)
        self.gain, self.gain_info = (gain_map(pre["wfold"], pre["wabsmax"], iw, sw, N, sc) if self.opts["ak_loss"]
                                     else (None, {}))
        self.grids = {"main": _Grid(self, self.opts["gz"], tol, eps)}
        pg = self.opts["psfgz"]
        self.grids["psf"] = self.grids["main"] if pg == self.opts["gz"] else _Grid(self, pg, tol, eps)
        del self.uf, self.vf, u, v, w
        if hasattr(zen, "drop_samples"):
            zen.drop_samples()          # the float64 stored-frame u, v, w are not needed through CLEAN

    @property
    def K(self):
        return self.grids["main"].table.K

    def info(self):
        g, p = self.grids["main"], self.grids["psf"]
        return dict(K=g.table.K, Nz=g.Nz, Nz_psf=p.Nz, ring_pix=int(g.ridx.size), ring_el=math.degrees(math.asin(g.n_ring)),
                    wabs=self.wabs_z,
                    **{f"gain_{k}": v for k, v in self.gain_info.items()})

    def image(self, vis_hour, grid="main", norm=True):
        """Hour-frame WSClean-orientation image of hour-frame samples vis_hour (complex64): the production
        to_ws(S.grid(vis_hour)) / sw (norm=False: without / sw), zero beyond the horizon."""
        G = self.grids[grid]; T = G.table; N = self.N; Nz = G.Nz
        G.setpts()
        c = vis_hour * self.b
        c = cp.where(self.flip, cp.conj(c), c) * self.wt
        c = c * self.eph
        acc = cp.zeros((Nz, Nz), cp.float32)
        n = int(G.ridx.size)
        racc = cp.zeros(n, cp.float32)
        bufs = (cp.empty((Nz, Nz), cp.complex64), cp.empty((Nz, Nz), cp.complex64)) if n else None
        idx = T.a_index(self.wf) if fast_ak() else None
        for k in range(T.K):
            f = G.plan[0].execute((c * T.a(k, self.wf, idx)).astype(cp.complex64, copy=idx is None))
            T.acc_re(acc, f, k)
            if n:
                G.ring_layer(f, k, racc, bufs)
            del f
        del c, idx, bufs
        pf = prefilter(cp.where(T.inside, acc, 0))
        del acc
        scale = 1.0 / self.sw if norm else 1.0
        out = cp.empty((N, N), cp.float32)
        _launch(_GATHER, N * N, (pf, np.int32(Nz), np.float64(G.scz), np.int32(N), np.float64(self.sc), self.rt,
                                 self.gain if self.gain is not None else np.uint64(0), np.float32(scale), out))
        del pf
        if n:
            rv = racc * np.float32(scale)
            if self.gain is not None:
                rv *= self.gain.ravel()[G.ridx]
            out.ravel()[G.ridx] = rv
        return out
