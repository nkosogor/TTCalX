"""cuFINUFFT gridding / degridding with the full w-term, for all-sky images.

Conventions
-----------
* Imaging kernel (= WSClean / ducc0 ``vis2dirty`` with divide_by_n=False):
      I(l, m) = Re sum_j c_j exp(+2 pi i (u_j l + v_j m - w_j (n - 1))),  n = sqrt(1 - l^2 - m^2)
  and the prediction (degridding) kernel is its adjoint
      V_j = sum_pix M(l, m) exp(-2 pi i (u_j l + v_j m - w_j (n - 1))).
* Samples are Hermitian-folded to w >= 0 ((u, v, w, V) -> (-u, -v, -w, V*)); predicted
  visibilities are unfolded again.
* w-term: exp(-2 pi i w (n-1)) = exp(+i pi w) exp(-2 pi i w x'), x' = (n-1) + 1/2 in [-1/2, 1/2],
  and exp(-2 pi i w x') ~= sum_k a_k(w) b_k(x') from an SVD of that kernel on a (w, x') grid
  (truncated at max error ``tol``).  Each term is one 2-D type-1 (grid) or type-2 (degrid)
  NUFFT: the image is sum_k Re(F_k(c a_k e^{i pi w}) * b_k(x'(l, m))).  K ~ 6-10 for a
  10-s snapshot (w <~ 50 lambda), ~37 for the hour frame at 41 MHz.
* Raw NUFFT output is indexed [ix(l), iy(m)] with l = (ix - N/2) d, m = (iy - N/2) d.
  ``to_ws`` turns it into WSClean/FITS memory order [y, x]: x = ix, y = (N - iy) % N, i.e.
  RA increases to the left (CDELT1 < 0), Dec upwards.
* Pixels with l^2 + m^2 >= 1 (beyond the horizon) are set to 0 in dirty/residual images.
"""
import itertools
import math

import numpy as np
import cupy as cp
import cufinufft

C = 299792458.0


def to_ws(x):
    """raw [ix(l), iy(m)] -> WSClean [y, x]"""
    n = x.shape[0]
    return x.T[(n - cp.arange(n)) % n, :]


def from_ws(img):
    """WSClean [y, x] -> raw [ix(l), iy(m)]"""
    n = img.shape[0]
    return img[(n - cp.arange(n)) % n, :].T


# ----------------------------------------------------------------------------- w-term factorisation
# The factorisation is kept compact: per w-range only the 1-D tables a_k(w) (nw points) and
# b_k(x') (nx points), ~0.2 MB on the host; per image geometry one float32 x'(l, m) image and the
# horizon mask.  b_k(l, m) = interp(x'(l, m), b_k) is evaluated on the fly inside the gridding
# kernels (equal to the former cp.interp(...).astype(complex64) tables to float32 rounding), so no K x N x N
# complex64 table is held (0.6 GB at 82 MHz, K = 8; 37 x N^2 for the hour frame).
# Both caches survive _SVD_CACHE.clear() (the snapshot daemon clears that every snapshot).

_FACTOR_CACHE = {}      # (wmax, tol, nw, nx) -> (K, wg, A [nw, K] complex128, Vh [K, nx] complex128)
_FACTOR_CACHE_MAX = 512
_GEOM_CACHE = {}        # (N, scale_rad, rmax, nx) -> (inside, x', cell j int16, fraction t float32)
_GEOM_CACHE_MAX = 4
SVD_METHOD = "gram"     # "gram" (top eigenpairs of M M^H, 2-4x faster, same K) or "lapack" (np.linalg.svd)


def _svd_factors(wmax, tol, nw, nx, method=None):
    """Truncated factorisation exp(-2 pi i w x') ~= sum_{k<K} A[:, k](w) Vh[k](x') on the (w, x') grid,
    K = smallest rank with max |M - A_K Vh_K| < tol (as before).  Cached (compact, host)."""
    method = method or SVD_METHOD
    key = (float(wmax), float(tol), int(nw), int(nx), method)
    hit = _FACTOR_CACHE.get(key)
    if hit is not None:
        return hit
    wg = np.linspace(0, wmax, nw); xg = np.linspace(-0.5, 0.5, nx)
    M = np.exp(-2j * np.pi * np.outer(wg, xg))
    res = None
    if method == "gram":
        # M M^H = U S^2 U^H; the leading singular triplets from its top eigenpairs.  The singular
        # values agree with LAPACK's SVD to ~1e-11 S_0 (the w-term needs S_K / S_0 ~ 1e-5).
        import scipy.linalg as sla
        kmax = min(nw, 96)
        G = M @ M.conj().T
        ev, U = sla.eigh(G, subset_by_index=[nw - kmax, nw - 1], driver="evr")
        ev = ev[::-1]; U = U[:, ::-1]
        S = np.sqrt(np.maximum(ev, 0.0))
        P = U.conj().T @ M                         # = S Vh
        R = M.copy()
        for K in range(1, kmax + 1):
            R -= np.outer(U[:, K - 1], P[K - 1])
            if np.abs(R).max() < tol:
                A = U[:, :K] * S[:K]
                Vh = P[:K] / S[:K, None]
                res = (K, wg, A, Vh)
                break
    if res is None:                                # lapack, or rank above kmax
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        K = 1
        while K < len(S) and np.abs(M - (U[:, :K] * S[:K]) @ Vh[:K]).max() >= tol:
            K += 1
        res = (K, wg, U[:, :K] * S[:K], Vh[:K].copy())
    if len(_FACTOR_CACHE) >= _FACTOR_CACHE_MAX:
        _FACTOR_CACHE.pop(next(iter(_FACTOR_CACHE)))
    _FACTOR_CACHE[key] = res
    return res


def _geometry(N, scale_rad, rmax=1.0, nx=2001):
    """Per image geometry (raw orientation, cached): inside [N, N] bool, x' = n - 1/2 float32 [N*N], and the
    linear-interpolation cell j (int16) and fraction t (float32) of x' on the nx-point x' grid of b_k."""
    key = (int(N), round(float(scale_rad), 15), float(rmax), int(nx))
    hit = _GEOM_CACHE.get(key)
    if hit is None:
        j = (cp.arange(N, dtype=cp.float64) - N // 2) * scale_rad
        r2 = j[:, None] ** 2 + j[None, :] ** 2
        inside = r2 < rmax ** 2                    # raw orientation (symmetric anyway)
        xp = (cp.where(inside, cp.sqrt(cp.maximum(1 - r2, 0)) - 1.0, 0.0) + 0.5).astype(cp.float32).ravel()
        del j, r2
        xgd = cp.asarray(np.linspace(-0.5, 0.5, nx), dtype=cp.float32)
        jj = cp.clip(cp.searchsorted(xgd, xp, side="right") - 1, 0, nx - 2)
        t = ((xp.astype(cp.float64) - xgd[jj].astype(cp.float64))
             / (xgd[jj + 1].astype(cp.float64) - xgd[jj].astype(cp.float64))).astype(cp.float32)
        if len(_GEOM_CACHE) >= _GEOM_CACHE_MAX:
            _GEOM_CACHE.pop(next(iter(_GEOM_CACHE)))
        hit = _GEOM_CACHE[key] = (inside, xp, jj.astype(cp.int16), t)
    return hit


# b_k(l, m) = linear interpolation of the 1-D table b_k(x') at x'(l, m) (as the former cp.interp table,
# to float32 rounding), evaluated inside the kernels from the cached cell index / fraction.
_wgrid_acc = cp.ElementwiseKernel(
    "complex64 f, int16 j, float32 t, raw float32 br, raw float32 bi", "float32 acc",
    """const float b_re = br[j] + t * (br[j + 1] - br[j]), b_im = bi[j] + t * (bi[j + 1] - bi[j]);
       const float o = f.real() * b_re - f.imag() * b_im;
       acc = __fadd_rn(acc, o);""", "wgrid_acc")
_wdegrid_in = cp.ElementwiseKernel(
    "float32 m, int16 j, float32 t, raw float32 br, raw float32 bi", "complex64 out",
    """const complex<float> b(br[j] + t * (br[j + 1] - br[j]), bi[j] + t * (bi[j + 1] - bi[j]));
       out = complex<float>(m, 0.f) * conj(b);""", "wdegrid_in")


class SvdWTerm:
    """exp(-2 pi i w x') ~= sum_k a_k(w) b_k(x'),  w in [0, wmax], x' in [-1/2, 1/2]."""

    def __init__(self, N, scale_rad, wmax, tol=1e-4, nw=800, nx=2001, rmax=1.0):
        K, wg, A, Vh = _svd_factors(wmax, tol, nw, nx)
        self.K = K; self.wmax = wmax; self.N = N; self.nx = nx
        self.wg = cp.asarray(wg, dtype=cp.float32)
        self.Ar = [cp.asarray(A[:, k].real, dtype=cp.float32) for k in range(K)]
        self.Ai = [cp.asarray(A[:, k].imag, dtype=cp.float32) for k in range(K)]
        self.inside, self.xp, self.jx, self.tx = _geometry(N, scale_rad, rmax, nx)
        self.xgd = cp.asarray(np.linspace(-0.5, 0.5, nx), dtype=cp.float32)
        self.Br = [cp.asarray(Vh[k].real, dtype=cp.float32) for k in range(K)]
        self.Bi = [cp.asarray(Vh[k].imag, dtype=cp.float32) for k in range(K)]

    @property
    def b(self):
        """Materialised b_k(l, m) images (compatibility / tests only: K x N x N complex64)."""
        return [(cp.interp(self.xp, self.xgd, self.Br[k]) + 1j * cp.interp(self.xp, self.xgd, self.Bi[k]))
                .reshape(self.N, self.N).astype(cp.complex64) for k in range(self.K)]

    def acc_re(self, acc, f, k):
        """acc += Re(f * b_k) (raw orientation, f complex64 [N, N])."""
        _wgrid_acc(f.ravel(), self.jx, self.tx, self.Br[k], self.Bi[k], acc.ravel())

    def model_conj_b(self, m, k):
        """m * conj(b_k) as complex64 [N, N] (m float32 raw orientation)."""
        out = cp.empty((self.N, self.N), cp.complex64)
        _wdegrid_in(m.ravel(), self.jx, self.tx, self.Br[k], self.Bi[k], out.ravel())
        return out

    def a(self, k, w):
        return (cp.interp(w, self.wg, self.Ar[k]) + 1j * cp.interp(w, self.wg, self.Ai[k])).astype(cp.complex64)


_SVD_CACHE = {}


def get_svd(N, scale_rad, wmax, tol=1e-4, quantum=2.0, nw=800, pad=1e-3):
    """Cached factorisation; wmax is rounded up to a multiple of ``quantum`` so that consecutive
    snapshots / blocks reuse it (hour: quantum 2, nw 800; snapshot: quantum 0.25, nw 600)."""
    wq = math.ceil(wmax / quantum) * quantum + pad
    key = (N, round(scale_rad, 12), round(wq, 3), tol, nw)
    if key not in _SVD_CACHE:
        _SVD_CACHE[key] = SvdWTerm(N, scale_rad, wq, tol, nw=nw)
    return _SVD_CACHE[key]


_PLAN_CACHE = {}
_TOKENS = itertools.count()


def get_plans(N, eps=1e-4):
    """(type-1, type-2) single-transform complex64 cuFINUFFT plans for an N x N image."""
    if (N, eps) not in _PLAN_CACHE:
        _PLAN_CACHE[(N, eps)] = [cufinufft.Plan(1, (N, N), n_trans=1, eps=eps, isign=1, dtype="complex64"),
                                 cufinufft.Plan(2, (N, N), n_trans=1, eps=eps, isign=-1, dtype="complex64"),
                                 None, None]    # tokens of the SampleSets whose points are set
    return _PLAN_CACHE[(N, eps)]


def clear_caches():
    _SVD_CACHE.clear(); _PLAN_CACHE.clear(); _FACTOR_CACHE.clear(); _GEOM_CACHE.clear()


# ----------------------------------------------------------------------------- sample sets
class SampleSet:
    """Visibility samples resident on the GPU, folded to w >= 0, ready for (de)gridding.

    u, v, w: per-sample coordinates [lambda], either host float64 (numpy) or device float32
    (cupy); wt: imaging weights (float32, 0 = not used).  ``svd`` = None disables the w-term."""

    def __init__(self, u, v, w, wt, N, scale_rad, svd=None, eps=1e-4, precompute=True, pre=None):
        self.N = N; self.sc = scale_rad; self.svd = svd
        if pre is not None:             # host arrays of snapshot.prepare_samples (same values as below)
            self.flip = cp.asarray(pre["flip"])
            self.x = cp.asarray(pre["x"]); self.y = cp.asarray(pre["y"]); self.w = cp.asarray(pre["wfold"])
        elif isinstance(u, np.ndarray):
            flip = w < 0
            s = np.where(flip, -1.0, 1.0)
            u, v, w = u * s, v * s, w * s
            self.flip = cp.asarray(flip)
            self.x = cp.asarray(2 * np.pi * u * scale_rad, dtype=cp.float32)
            self.y = cp.asarray(2 * np.pi * v * scale_rad, dtype=cp.float32)
            self.w = cp.asarray(w, dtype=cp.float32)
        else:
            flip = w < 0
            s = cp.where(flip, -1.0, 1.0).astype(cp.float32)
            u, v, w = u * s, v * s, w * s
            self.flip = flip
            self.x = (2 * np.pi * scale_rad) * u
            self.y = (2 * np.pi * scale_rad) * v
            self.w = w
        self.wt = cp.asarray(wt, dtype=cp.float32)
        self.npts = int(self.w.size)
        self.plans = get_plans(N, eps)
        self.token = next(_TOKENS)
        self.a = None
        if svd is not None and precompute:
            self.a = [svd.a(k, self.w) for k in range(svd.K)]
            self.ephase = cp.exp(1j * cp.pi * self.w).astype(cp.complex64)

    def _setpts(self, t):
        """Point the type-t plan (1 grid, 2 degrid) at this sample set (skipped if already)."""
        if self.plans[t + 1] != self.token:
            self.plans[t - 1].setpts(self.x, self.y)
            self.plans[t + 1] = self.token

    def _ak(self, k):
        return self.a[k] if self.a is not None else self.svd.a(k, self.w)

    def _eph(self):
        return self.ephase if self.a is not None else cp.exp(1j * cp.pi * self.w).astype(cp.complex64)

    def grid(self, vis):
        """Weighted sum (raw orientation, NOT normalised) of vis (complex64, unfolded)."""
        self._setpts(1)
        c = cp.where(self.flip, cp.conj(vis), vis) * self.wt
        if self.svd is None:
            return self.plans[0].execute(c.astype(cp.complex64)).real
        c = c * self._eph()
        acc = cp.zeros((self.N, self.N), cp.float32)
        for k in range(self.svd.K):
            f = self.plans[0].execute((c * self._ak(k)).astype(cp.complex64))
            self.svd.acc_re(acc, f, k)
        return cp.where(self.svd.inside, acc, 0)

    def grid_psf(self):
        """Sum of weights gridded without w-term (PSF at the phase centre), raw orientation."""
        self._setpts(1)
        return self.plans[0].execute(self.wt.astype(cp.complex64)).real

    def degrid(self, model_ws):
        """Predict (unfolded) visibilities of a WSClean-orientation model image [Jy/pixel]."""
        self._setpts(2)
        m = cp.ascontiguousarray(from_ws(model_ws).astype(cp.float32))
        q = cp.zeros(self.npts, cp.complex64)
        for k in range(self.svd.K):
            q += cp.conj(self._ak(k)) * self.plans[1].execute(self.svd.model_conj_b(m, k))
        q *= cp.conj(self._eph())
        return cp.where(self.flip, cp.conj(q), q)


# ----------------------------------------------------------------------------- direct predict
_DFT_SRC = r"""
extern "C" __global__ void dft_predict(const float* __restrict__ uu, const float* __restrict__ vv,
    const float* __restrict__ ww, const long npts,
    const float* __restrict__ cl, const float* __restrict__ cm, const float* __restrict__ cn1,
    const float* __restrict__ cs, const int ncomp, float2* __restrict__ vis)
{
    // V = sum_k S_k exp(2 pi i (-(u l + v m) + w (n-1)))   (adjoint of the imaging kernel)
    extern __shared__ float sh[];
    float* sl = sh; float* sm = sh + blockDim.x; float* sn = sh + 2 * blockDim.x; float* ss = sh + 3 * blockDim.x;
    const long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    float u = 0.f, v = 0.f, w = 0.f;
    if (idx < npts) { u = uu[idx]; v = vv[idx]; w = ww[idx]; }
    float re = 0.f, im = 0.f;
    for (int k0 = 0; k0 < ncomp; k0 += blockDim.x) {
        const int k = k0 + threadIdx.x;
        __syncthreads();
        if (k < ncomp) { sl[threadIdx.x] = cl[k]; sm[threadIdx.x] = cm[k]; sn[threadIdx.x] = cn1[k]; ss[threadIdx.x] = cs[k]; }
        __syncthreads();
        const int kk = min((int)blockDim.x, ncomp - k0);
        for (int j = 0; j < kk; ++j) {
            float ph = -u * sl[j] - v * sm[j] + w * sn[j];
            ph = ph - rintf(ph);              // reduce to [-1/2, 1/2] turns before sincospi
            float s, co;
            sincospif(2.f * ph, &s, &co);
            re += ss[j] * co; im += ss[j] * s;
        }
    }
    if (idx < npts) vis[idx] = make_float2(re, im);
}
"""
_dft = cp.RawKernel(_DFT_SRC, "dft_predict")


def dft_predict(u, v, w, model_ws, scale_rad):
    """Exact (DFT) prediction of a sparse model, faster than degridding below ~8000 components.
    u, v, w: cupy float32 per-sample coordinates [lambda] (unfolded)."""
    N = model_ws.shape[0]
    ys, xs = cp.nonzero(model_ws)
    s = model_ws[ys, xs].astype(cp.float32)
    l = (xs.astype(cp.float64) - N // 2) * scale_rad
    m = (((N - ys) % N).astype(cp.float64) - N // 2) * scale_rad
    n1 = cp.sqrt(cp.maximum(1.0 - l * l - m * m, 0.0)) - 1.0
    npts = u.size
    vis = cp.empty(npts, cp.complex64)
    bs = 256
    _dft(((npts + bs - 1) // bs,), (bs,), (u, v, w, np.int64(npts), l.astype(cp.float32), m.astype(cp.float32),
                                        n1.astype(cp.float32), s, np.int32(s.size), vis), shared_mem=4 * 4 * bs)
    return vis
