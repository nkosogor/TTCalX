"""Reprojection between SIN images of different phase centres (w-snapshot imaging).

With the gridder convention exp(+2 pi i (u l + v m - w (n-1))), the direction cosines
(l, m, -n) transform like (u, v, w): if uvw_target = R uvw_frame then
(l, m, -n)_target = R (l, m, -n)_frame.  Images are resampled with cubic/quintic splines.

Flux convention: dirty/residual images are surface brightness (Jy/beam) and are resampled
as-is; a *model* (Jy/pixel) moved from frame A to frame B is multiplied by the SIN pixel
solid-angle Jacobian n_A / n_B (a pixel covers dl dm / n steradian).
"""
import os

import numpy as np
import cupy as cp
from cupyx.scipy import ndimage as cnd


def target_dirs(N, sc):
    """(l, m, n, inside) of every pixel of an N x N WSClean-orientation [y, x] image."""
    j = (cp.arange(N, dtype=cp.float64) - N // 2) * sc
    y = cp.arange(N)
    l = j[None, :] * cp.ones((N, 1))
    m = ((((N - y) % N) - N // 2) * sc)[:, None] * cp.ones((1, N))
    r2 = l * l + m * m
    inside = r2 < 1
    n = cp.sqrt(cp.maximum(1 - r2, 0))
    return l, m, n, inside


# TTCALX_FAST_COORDS=1 (default 0): Reprojector.coords in one kernel with the float64 operations of the CuPy
# expression below, each rounded on its own (__dmul_rn / __dadd_rn / __ddiv_rn: no fused multiply-add), so the
# coordinates are the same bits; ~10x less memory traffic than the per-operation temporaries
_coords_kernel = cp.ElementwiseKernel(
    "float64 l, float64 m, float64 n, bool inside, raw float64 r, float64 sc, float64 half",
    "float32 ys32, float32 xs32, bool ok, float64 ns",
    """const double c = -n;
       const double ls = __dadd_rn(__dadd_rn(__dmul_rn(r[0], l), __dmul_rn(r[1], m)), __dmul_rn(r[2], c));
       const double ms = __dadd_rn(__dadd_rn(__dmul_rn(r[3], l), __dmul_rn(r[4], m)), __dmul_rn(r[5], c));
       ns = -__dadd_rn(__dadd_rn(__dmul_rn(r[6], l), __dmul_rn(r[7], m)), __dmul_rn(r[8], c));
       xs32 = (float)__dadd_rn(__ddiv_rn(ls, sc), half);
       ys32 = (float)__dadd_rn(half, -__ddiv_rn(ms, sc));
       ok = inside && (ns > 0);""", "ttcalx_reproject_coords")


class Reprojector:
    def __init__(self, N, sc):
        self.N = N; self.sc = sc
        self.l, self.m, self.n, self.inside = target_dirs(N, sc)

    def coords(self, R):
        """For each pixel of the target grid: (y, x) pixel position in the source frame, validity
        and n in the source frame, where uvw_target = R uvw_source."""
        if os.environ.get("TTCALX_FAST_COORDS", "0") == "1":
            N = self.N
            yx = cp.empty((2, N, N), cp.float32); ok = cp.empty((N, N), cp.bool_); ns = cp.empty((N, N), cp.float64)
            r = cp.asarray(np.ascontiguousarray(np.asarray(R, np.float64).T).ravel())
            _coords_kernel(self.l, self.m, self.n, self.inside, r, float(self.sc), float(N // 2), yx[0], yx[1], ok, ns)
            return yx, ok, ns
        Rt = cp.asarray(np.asarray(R).T)
        a, b, c = self.l, self.m, -self.n
        ls = Rt[0, 0] * a + Rt[0, 1] * b + Rt[0, 2] * c
        ms = Rt[1, 0] * a + Rt[1, 1] * b + Rt[1, 2] * c
        ns = -(Rt[2, 0] * a + Rt[2, 1] * b + Rt[2, 2] * c)
        N, sc = self.N, self.sc
        xs = ls / sc + N // 2
        ys = N // 2 - ms / sc
        ok = self.inside & (ns > 0)
        return cp.stack([ys, xs]).astype(cp.float32), ok, ns

    def __call__(self, img, R, order=3, coords=None, up=1, prefiltered=False):
        """Resample source-frame img onto the target grid (zero outside the horizon).
        prefiltered=True: img is already prefilter(img, order) (the same source resampled
        to several targets is spline-filtered once; identical result)."""
        if coords is None:
            coords = self.coords(R)
        yx, ok, _ = coords
        if order in (3, 5) and reproj_fast():
            # TTCALX_REPROJ_FAST: FFT prefilter + one float32 B-spline kernel (zero outside ok, as below)
            return interp_prefiltered(img if prefiltered else prefilter_fft(img, order), yx, ok, order, up), ok
        if up != 1:
            yx = yx * up
        out = cnd.map_coordinates(img.astype(cp.float32), yx, order=order, mode="constant", cval=0.0,
                                  prefilter=not prefiltered)
        return cp.where(ok, out, 0), ok


def prefilter(img, order):
    """The spline prefilter map_coordinates(mode='constant', prefilter=True) applies to img."""
    img = img.astype(cp.float32)
    if order < 2:
        return cp.ascontiguousarray(img)
    if order in (3, 5) and reproj_fast():
        return prefilter_fft(img, order)
    return cp.ascontiguousarray(cnd.spline_filter(img, order, output=cp.float32, mode="constant"))


def fft_upsample(img, up=2, taper=None):
    """Band-limited (zero-padded FFT) upsampling, sample i -> up*i; taper: fftfreq-ordered filter."""
    N = img.shape[0]
    F = cp.fft.fft2(img.astype(cp.complex64))
    if taper is not None:
        F *= taper
    if up == 2 and N % 2 == 0 and img.shape[1] == N and os.environ.get("TTCALX_FAST_UPSAMPLE", "0") == "1":
        return _upsample2_polyphase(F)
    F = cp.fft.fftshift(F)
    M = N * up; o = (M - N) // 2
    G = cp.zeros((M, M), cp.complex64); G[o:o + N, o:o + N] = F
    return (cp.fft.ifft2(cp.fft.ifftshift(G)).real * (up * up)).astype(cp.float32)


def _upsample2_polyphase(F):
    """fft_upsample(up=2) from the (tapered) N x N spectrum F (fftfreq order, N even) with four N x N inverse FFTs
    instead of one 2N x 2N: the zero-padded 2N-point inverse DFT at the even / odd output samples is the N-point
    inverse DFT of F times exp(i pi f / N) ** parity along each axis (f = the fftfreq bin in [-N/2, N/2), as the
    zero padding places it).  The same values to FFT rounding (TTCALX_FAST_UPSAMPLE=1, default 0): at the hour sizes
    (2N = 2 x 3122, 2 x 2358, 2 x 1508) the 2N transform is a slow Bluestein one."""
    N = F.shape[0]
    f = cp.fft.fftfreq(N, d=1.0 / N)                          # bins -N/2 .. N/2 - 1 (float64)
    ph = cp.exp(1j * cp.pi * f / N).astype(cp.complex64)
    out = cp.empty((2 * N, 2 * N), cp.float32)
    for sy in (0, 1):
        Fy = F * ph[:, None] if sy else F
        for sx in (0, 1):
            out[sy::2, sx::2] = cp.fft.ifft2(Fy * ph[None, :] if sx else Fy).real
        del Fy
    return out


# ----------------------------------------------------------------------------- TTCALX_REPROJ_FAST
# TTCALX_REPROJ_FAST=1 (default 0; hour stage, v5): the order-3 / order-5 spline reprojection of Reprojector and
# prefilter() as (a) the B-spline prefilter by one zero-padded real-FFT division by the sampled B-spline transfer
# function instead of cupyx's recursive (IIR) filter with mirror boundaries -- the two differ only within ~30 px
# of the image edge, where the w-snapshot block images and the band-limited model are zero (beyond the horizon)
# -- and (b) the interpolation of map_coordinates(mode="constant", prefilter=False) in one float32 kernel
# (B-spline weights in float32, taps outside the image mirrored as cupyx does, points outside [0, n - 1] or
# not ok -> 0) instead of cupyx's float64 kernel.  The same reprojection to float32 rounding; validated on
# band-hours at 55 / 64 / 73 MHz (2026-11-23/24: images at the GPU rerun floor).
_INTERP_SRC = r"""
__device__ __forceinline__ float ttcalx_b5(float d) {
    d = fabsf(d);
    if (d < 1.f) { const float d2 = d * d; return 11.f / 20.f - 0.5f * d2 + 0.25f * d2 * d2 - d2 * d2 * d / 12.f; }
    if (d < 2.f) { const float d2 = d * d;
        return 17.f / 40.f + 0.625f * d - 1.75f * d2 + 1.25f * d2 * d - 0.375f * d2 * d2 + d2 * d2 * d / 24.f; }
    if (d < 3.f) { const float e = 3.f - d, e2 = e * e; return e2 * e2 * e / 120.f; }
    return 0.f;
}
__device__ __forceinline__ float ttcalx_b3(float d) {
    d = fabsf(d);
    if (d < 1.f) return 2.f / 3.f - d * d + 0.5f * d * d * d;
    if (d < 2.f) { const float e = 2.f - d; return e * e * e / 6.f; }
    return 0.f;
}
__device__ __forceinline__ int ttcalx_mirror(int i, const int n) {
    // cupyx mode="mirror" reflection about the edge samples (period 2n - 2)
    if (n == 1) return 0;
    const int p = 2 * n - 2;
    i = i % p; if (i < 0) i += p;
    return i < n ? i : p - i;
}
template <int ORD>
__device__ __forceinline__ float ttcalx_interp_at(const float* __restrict__ img, const int ny, const int nx,
                                                  const float y, const float x) {
    constexpr int T = ORD + 1;
    float wy[T], wx[T];
    const float fy = floorf(y), fx = floorf(x);
    const int y0 = (int)fy - ORD / 2, x0 = (int)fx - ORD / 2;
    #pragma unroll
    for (int j = 0; j < T; ++j) {
        if (ORD == 5) { wy[j] = ttcalx_b5(y - fy - (float)(j - 2)); wx[j] = ttcalx_b5(x - fx - (float)(j - 2)); }
        else { wy[j] = ttcalx_b3(y - fy - (float)(j - 1)); wx[j] = ttcalx_b3(x - fx - (float)(j - 1)); }
    }
    int xx[T];
    #pragma unroll
    for (int b = 0; b < T; ++b) xx[b] = ttcalx_mirror(x0 + b, nx);
    float acc = 0.f;
    #pragma unroll
    for (int a = 0; a < T; ++a) {
        const float* row = img + (long long)ttcalx_mirror(y0 + a, ny) * nx;
        float r = 0.f;
        #pragma unroll
        for (int b = 0; b < T; ++b) r += wx[b] * __ldg(row + xx[b]);
        acc += wy[a] * r;
    }
    return acc;
}
extern "C" __global__ void ttcalx_interp(const float* __restrict__ img, const int ny, const int nx,
    const float* __restrict__ yx, const bool* __restrict__ ok, const long long n_out, const float up,
    const int order, float* __restrict__ out)
{
    const long long o = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= n_out) return;
    const float y = yx[o] * up, x = yx[o + n_out] * up;
    if (!ok[o] || y < 0.f || x < 0.f || y > (float)(ny - 1) || x > (float)(nx - 1)) { out[o] = 0.f; return; }
    out[o] = order == 5 ? ttcalx_interp_at<5>(img, ny, nx, y, x) : ttcalx_interp_at<3>(img, ny, nx, y, x);
}
"""
_INTERP_MOD = []

# the prefilter's division by the B-spline transfer function a(f_y) b(f_x), formed per element in float64 and
# rounded to float32 (no P x P table is kept)
_pf_divide = cp.ElementwiseKernel(
    "raw float64 a, raw float64 b, int64 q", "complex64 F",
    "const float g = (float)(1.0 / (a[i / q] * b[i % q])); F = complex<float>(F.real() * g, F.imag() * g);",
    "ttcalx_bspline_prefilter_divide")


def reproj_fast() -> bool:
    """TTCALX_REPROJ_FAST=1 (default 0): FFT spline prefilter + float32 interpolation kernel (see above)."""
    return os.environ.get("TTCALX_REPROJ_FAST", "0") == "1"


def _interp_kernel():
    if not _INTERP_MOD:
        _INTERP_MOD.append(cp.RawModule(code=_INTERP_SRC, options=("-std=c++14",)))
    return _INTERP_MOD[0].get_function("ttcalx_interp")


def interp_prefiltered(img, yx, ok, order, up=1):
    """map_coordinates(img, yx * up, order, mode="constant", prefilter=False) where ok, else 0, with img already
    spline-prefiltered (float32 kernel; order 3 or 5).  yx: float32 [2, ...] (y, x) source positions."""
    assert order in (3, 5)
    img = cp.ascontiguousarray(img, dtype=cp.float32)
    ny, nx = img.shape
    yx = cp.ascontiguousarray(yx, dtype=cp.float32)
    okc = cp.ascontiguousarray(ok, dtype=cp.bool_)
    n_out = int(okc.size)
    out = cp.empty(okc.shape, cp.float32)
    _interp_kernel()(((n_out + 255) // 256,), (256,),
                     (img, np.int32(ny), np.int32(nx), yx, okc, np.int64(n_out), np.float32(up), np.int32(order), out))
    return out


def _fast_len2357(n):
    """Smallest 2^a 3^b 5^c 7^d >= n (a fast cuFFT size)."""
    m = n
    while True:
        k = m
        for p in (2, 3, 5, 7):
            while k % p == 0:
                k //= p
        if k == 1:
            return m
        m += 1


def prefilter_fft(img, order, pad=48):
    """B-spline prefilter (order 3 / 5) of img by a zero-padded real-FFT division by the B-spline transfer
    function (TTCALX_REPROJ_FAST); the padded input and the spectrum are freed as soon as they are used."""
    img = cp.asarray(img, cp.float32)
    n0, n1 = img.shape
    P0, P1 = _fast_len2357(n0 + pad), _fast_len2357(n1 + pad)

    def tf(P):
        w = 2 * np.pi * np.arange(P) / P
        return (66 + 52 * np.cos(w) + 2 * np.cos(2 * w)) / 120.0 if order == 5 else (4 + 2 * np.cos(w)) / 6.0
    x = cp.zeros((P0, P1), cp.float32); x[:n0, :n1] = img
    F = cp.fft.rfft2(x); del x
    _pf_divide(cp.asarray(tf(P0)), cp.asarray(tf(P1)[: P1 // 2 + 1]), F.shape[1], F)
    y = cp.fft.irfft2(F, s=(P0, P1)); del F
    return cp.ascontiguousarray(y[:n0, :n1])
