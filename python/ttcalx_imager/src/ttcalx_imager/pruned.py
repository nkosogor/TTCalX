"""Pruned-FFT 2-D NUFFT (TTCALX_NUFFT_PRUNED, default off): cuFINUFFT spread / interpolation only
(``gpu_spreadinterponly``) on the same fine grid and kernel as its own type-1 / type-2 transforms, the
fine-grid FFT done as contiguous batched 1-D cuFFTs that skip what is not needed, and the kernel
deconvolution applied while the N x N modes are gathered.

Why: cuFINUFFT's 2-D cuFFT of the nf x nf fine grid (nf ~ 2N, 6250 at 82 MHz) is most of a transform
(6.4 of 8.2 ms at 82 MHz), and ~3/4 of that is the strided column pass.  At the all-sky geometries the
samples fill only |x| <~ 0.6 pi of the grid (synthesised beam ~3-5 pixels) and type 1 needs only N of the
nf modes per axis.  Type 1 here: (1) row FFTs of the fine-grid rows the spreading touched (a band of
~0.3 nf rows), (2) a tiled transpose of the N needed mode columns of that band, (3) contiguous row FFTs of
those N rows (the column transform), (4) a tiled transposing gather of the N x N modes times the
deconvolution.  Type 2 is the mirror image.

The same NUFFT up to float32 FFT rounding: the spread grid is cuFINUFFT's (same kernel, nf, points; its fine
grid is centred, x = 0 at cell nf / 2), the deconvolution factors are calibrated against cuFINUFFT's own type-1
plan (a unit point at the origin), and only the FFT is done differently.
"""
import math

import numpy as np
import cupy as cp
import cufinufft
from cupy.cuda import cufft as _cf

_Q = 256                # band edges rounded to multiples of this (stable bands, few cuFFT plan sizes)
_CHUNK = 512            # rows per batched 1-D cuFFT call (bounds the cuFFT work areas)
_BUF = {}               # (nf, N) -> shared buffers and their zero-invariant state
_CAL = {}               # (N, nf, eps, upsampfac) -> deconvolution d [N] complex64
_FFTP = {}              # (nf, batch, inplace) -> cufft PlanNd

_TSRC = r"""
// out[orow(q) * old + ocol(p)] = in[irow(p) * ild + icol(q)] (* d[p] d[q]): a tiled transpose with index maps
// map(i; mode, base) = mode ? (i - N/2 + nf) % nf : i + base  (mode: the modeord-0 mode k = i - N/2 at its FFT bin)
__device__ __forceinline__ long long mp(int i, int mode, int base, int N, int nf) {
    return mode ? (long long)((i - N / 2 + nf) % nf) : (long long)(i + base);
}
extern "C" __global__ void ttcalx_tgather(const float2* __restrict__ in, float2* __restrict__ out,
    const int P, const int Q, const long long ild, const long long old, const int N, const int nf,
    const int ir_m, const int ir_b, const int ic_m, const int ic_b, const int or_m, const int or_b,
    const int oc_m, const int oc_b, const float2* __restrict__ d)
{
    __shared__ float2 tile[32][33];
    const int q0 = blockIdx.x * 32, p0 = blockIdx.y * 32;
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        const int p = p0 + k, q = q0 + threadIdx.x;
        if (p < P && q < Q) {
            float2 v = in[mp(p, ir_m, ir_b, N, nf) * ild + mp(q, ic_m, ic_b, N, nf)];
            if (d) {
                const float2 a = d[p], b = d[q];
                const float2 s = make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
                v = make_float2(v.x * s.x - v.y * s.y, v.x * s.y + v.y * s.x);
            }
            tile[k][threadIdx.x] = v;
        }
    }
    __syncthreads();
    for (int k = threadIdx.y; k < 32; k += blockDim.y) {
        const int q = q0 + k, p = p0 + threadIdx.x;
        if (p < P && q < Q)
            out[mp(q, or_m, or_b, N, nf) * old + mp(p, oc_m, oc_b, N, nf)] = tile[threadIdx.x][k];
    }
}
"""
_tg = cp.RawKernel(_TSRC, "ttcalx_tgather")


def _tgather(inp, out, P, Q, N, nf, ir, ic, orr, oc, d=None):
    """out[orow(q), ocol(p)] = inp[irow(p), icol(q)] (* d[p] d[q]); maps (mode, base) as in the kernel."""
    grid = ((Q + 31) // 32, (P + 31) // 32)
    dptr = d if d is not None else np.uint64(0)
    _tg(grid, (32, 8), (inp, out, np.int32(P), np.int32(Q), np.int64(inp.shape[1]), np.int64(out.shape[1]),
                        np.int32(N), np.int32(nf), np.int32(ir[0]), np.int32(ir[1]), np.int32(ic[0]), np.int32(ic[1]),
                        np.int32(orr[0]), np.int32(orr[1]), np.int32(oc[0]), np.int32(oc[1]), dptr))


def _bufs(nf, N):
    b = _BUF.get((nf, N))
    if b is None:
        b = _BUF[(nf, N)] = dict(G=cp.zeros((nf, nf), cp.complex64), T=cp.zeros((N, nf), cp.complex64),
                                 U=cp.empty((N, nf), cp.complex64), Gs=None, Ts=None)
    return b


def _plan1d(nf, batch, inplace):
    key = (nf, batch, inplace)
    p = _FFTP.get(key)
    if p is None:
        p = _FFTP[key] = _cf.PlanNd([nf], [nf], 1, nf, [nf], 1, nf, _cf.CUFFT_C2C, batch, "C", -1, None)
    return p


def _fft_rows(A, r0, r1, direction, out=None):
    """1-D FFT along the last axis of rows r0..r1-1 of A (C order; in place, or into the same rows of out),
    in chunks of _CHUNK rows."""
    nf = A.shape[1]
    r = r0
    while r < r1:
        n = min(_CHUNK, r1 - r)
        v = A[r:r + n]
        _plan1d(nf, n, out is None).fft(v, v if out is None else out[r:r + n], direction)
        r += n


class PrunedPlan:
    """Drop-in for cufinufft.Plan(t, (N, N), n_trans=1, eps, isign, dtype='complex64', **opts) as used by
    gridder.SampleSet: setpts(x, y) and execute(data) (type 1: N x N modes from the strengths at the points,
    type 2: the values at the points from N x N modes), modeord 0, x along the first mode axis."""

    def __init__(self, nufft_type, N, eps, isign, upsampfac=2.0):
        self.t = nufft_type; self.N = N; self.eps = eps; self.isign = isign
        self.up = float(upsampfac)
        self.nf = _nf_of(N, self.up)
        self.sp = cufinufft.Plan(nufft_type, (self.nf, self.nf), n_trans=1, eps=eps, isign=isign, dtype="complex64",
                                 upsampfac=self.up, gpu_spreadinterponly=1)
        self.h = 16                    # >= half the spreading kernel width (5 cells at eps 1e-4, upsampfac 2) + margin
        self.d = None; self.band = None

    def _dir(self):
        """exp(+i k x) (isign > 0) is cuFFT's INVERSE (unnormalised) transform."""
        return _cf.CUFFT_INVERSE if self.isign > 0 else _cf.CUFFT_FORWARD

    def _set_band(self, lo, hi):
        nf = self.nf
        r0 = max(0, (nf // 2 + lo - self.h) // _Q * _Q)
        r1 = min(nf, -(-(nf // 2 + hi + self.h) // _Q) * _Q)
        self.band = (r0, r1)

    def setpts(self, x, y):
        if self.d is None:
            self.d = _calibrate(self.N, self.nf, self.eps, self.up)
        self.sp.setpts(x, y)
        s = self.nf / (2 * math.pi)
        xmin, xmax = (float(v) for v in cp.stack([x.min(), x.max()]).get())
        self._set_band(int(math.floor(xmin * s)), int(math.ceil(xmax * s)))

    def execute(self, data):
        N, nf = self.N, self.nf
        B = _bufs(nf, N); G, T, U = B["G"], B["T"], B["U"]
        r0, r1 = self.band; R = r1 - r0; dr = self._dir()
        if self.t == 1:
            self.sp.execute(data, out=G); B["Gs"] = "spread"   # spread (cuFINUFFT clears the grid first)
            _fft_rows(G, r0, r1, dr)                           # y -> ky along the band rows
            if B["Ts"] != ("t1", self.band):                   # T: zero outside the band columns
                T.fill(0); B["Ts"] = ("t1", self.band)
            # T[j, r0 + a] = G[r0 + a, bin(ky_j)]
            _tgather(G, T, R, N, N, nf, (0, r0), (1, 0), (0, 0), (0, r0))
            _fft_rows(T, 0, N, dr, out=U)                      # x -> kx (rows of T = columns of G)
            f = cp.empty((N, N), cp.complex64)
            # f[ix, iy] = U[iy, bin(kx_ix)] d[ix] d[iy]
            _tgather(U, f, N, N, N, nf, (0, 0), (1, 0), (0, 0), (0, 0), self.d)
            return f
        f = cp.ascontiguousarray(data, dtype=cp.complex64)
        if B["Ts"] != "t2":                                    # T: zero outside the mode columns
            T.fill(0); B["Ts"] = "t2"
        # T[iy, bin(kx_ix)] = f[ix, iy] d[ix] d[iy]
        _tgather(f, T, N, N, N, nf, (0, 0), (0, 0), (0, 0), (1, 0), self.d)
        _fft_rows(T, 0, N, dr, out=U)                          # kx -> x
        if B["Gs"] != ("t2", self.band):                       # G: zero outside band rows x mode columns
            G.fill(0); B["Gs"] = ("t2", self.band)
        else:
            G[r0:r1].fill(0)
        # G[r0 + a, bin(ky_j)] = U[j, r0 + a]
        _tgather(U, G, N, R, N, nf, (0, 0), (0, r0), (0, r0), (1, 0))
        _fft_rows(G, r0, r1, dr)                               # ky -> y along the band rows
        return self.sp.execute(G)


def _nf_of(N, up):
    """cuFINUFFT's fine grid size: the smallest even 2,3,5-smooth integer >= up * N (and >= 2 nspread)."""
    n = max(int(up * N), 2 * 16)
    while True:
        if n % 2 == 0:
            m = n
            for p in (2, 3, 5):
                while m % p == 0:
                    m //= p
            if m == 1:
                return n
        n += 1


def _calibrate(N, nf, eps, up):
    """Deconvolution factors d [N] (complex64: the kernel's 1 / phi_hat(k) times the (-1)^k of the centred fine
    grid) with modes = d[ix] d[iy] * (pruned FFT of the spread grid): from a unit point at the origin through
    cuFINUFFT's own type-1 plan and through the spread-only path with d = 1.  Type 2 deconvolves with the same
    factors (the same kernel and grid)."""
    key = (N, nf, eps, up)
    hit = _CAL.get(key)
    if hit is not None:
        return hit
    z = cp.zeros(1, cp.float32); one = cp.ones(1, cp.complex64)
    ref = cufinufft.Plan(1, (N, N), n_trans=1, eps=eps, isign=1, dtype="complex64",
                         **({} if up == 2.0 else {"upsampfac": up}))
    ref.setpts(z, z); fr = ref.execute(one).astype(cp.complex128)
    del ref
    p = PrunedPlan(1, N, eps, 1, up)
    p.d = cp.ones(N, cp.complex64)
    p.setpts(z, z); fp = p.execute(one).astype(cp.complex128)
    del p
    D = fr / fp
    c = N // 2
    d = (D[:, c] / cp.sqrt(D[c, c])).astype(cp.complex64)
    sep = float(cp.abs(cp.outer(d.astype(cp.complex128), d.astype(cp.complex128)) - D).max() / cp.abs(D).max())
    if not sep < 1e-5:
        raise RuntimeError(f"pruned NUFFT calibration: deconvolution not separable ({sep:.2e})")
    _CAL[key] = d
    return d


def clear():
    _BUF.clear(); _CAL.clear(); _FFTP.clear()
