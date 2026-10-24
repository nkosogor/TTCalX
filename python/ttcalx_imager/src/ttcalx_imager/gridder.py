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


_re_mul = cp.ElementwiseKernel("complex64 f, complex64 b", "float32 o",
                               "o = f.real() * b.real() - f.imag() * b.imag()", "re_mul")


# ----------------------------------------------------------------------------- w-term factorisation
class SvdWTerm:
    """exp(-2 pi i w x') ~= sum_k a_k(w) b_k(x'),  w in [0, wmax], x' in [-1/2, 1/2]."""

    def __init__(self, N, scale_rad, wmax, tol=1e-4, nw=800, nx=2001, rmax=1.0):
        wg = np.linspace(0, wmax, nw); xg = np.linspace(-0.5, 0.5, nx)
        M = np.exp(-2j * np.pi * np.outer(wg, xg))
        U, S, Vh = np.linalg.svd(M, full_matrices=False)
        K = 1
        while K < len(S) and np.abs(M - (U[:, :K] * S[:K]) @ Vh[:K]).max() >= tol:
            K += 1
        self.K = K; self.wmax = wmax
        self.wg = cp.asarray(wg, dtype=cp.float32)
        A = U[:, :K] * S[:K]
        self.Ar = [cp.asarray(A[:, k].real, dtype=cp.float32) for k in range(K)]
        self.Ai = [cp.asarray(A[:, k].imag, dtype=cp.float32) for k in range(K)]
        j = (cp.arange(N, dtype=cp.float64) - N // 2) * scale_rad
        r2 = j[:, None] ** 2 + j[None, :] ** 2
        self.inside = r2 < rmax ** 2                       # raw orientation (symmetric anyway)
        xp = (cp.where(self.inside, cp.sqrt(cp.maximum(1 - r2, 0)) - 1.0, 0.0) + 0.5).astype(cp.float32).ravel()
        xgd = cp.asarray(xg, dtype=cp.float32)
        self.b = [(cp.interp(xp, xgd, cp.asarray(Vh[k].real, dtype=cp.float32))
                   + 1j * cp.interp(xp, xgd, cp.asarray(Vh[k].imag, dtype=cp.float32))).reshape(N, N).astype(cp.complex64)
                  for k in range(K)]

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
    _SVD_CACHE.clear(); _PLAN_CACHE.clear()


# ----------------------------------------------------------------------------- sample sets
class SampleSet:
    """Visibility samples resident on the GPU, folded to w >= 0, ready for (de)gridding.

    u, v, w: per-sample coordinates [lambda], either host float64 (numpy) or device float32
    (cupy); wt: imaging weights (float32, 0 = not used).  ``svd`` = None disables the w-term."""

    def __init__(self, u, v, w, wt, N, scale_rad, svd=None, eps=1e-4, precompute=True):
        self.N = N; self.sc = scale_rad; self.svd = svd
        if isinstance(u, np.ndarray):
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
            acc += _re_mul(f, self.svd.b[k])
        return cp.where(self.svd.inside, acc, 0)

    def grid_psf(self):
        """Sum of weights gridded without w-term (PSF at the phase centre), raw orientation."""
        self._setpts(1)
        return self.plans[0].execute(self.wt.astype(cp.complex64)).real

    def degrid(self, model_ws):
        """Predict (unfolded) visibilities of a WSClean-orientation model image [Jy/pixel]."""
        self._setpts(2)
        m = cp.ascontiguousarray(from_ws(model_ws).astype(cp.complex64))
        q = cp.zeros(self.npts, cp.complex64)
        for k in range(self.svd.K):
            q += cp.conj(self._ak(k)) * self.plans[1].execute(cp.ascontiguousarray(m * cp.conj(self.svd.b[k])))
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
