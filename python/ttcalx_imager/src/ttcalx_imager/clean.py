"""Image-plane deconvolution on the GPU, emulating WSClean/radler for our settings.

* ``SnapshotClean`` + ``clean_snapshot``: Hogbom/Clark CLEAN, as
  ``wsclean -niter 50000 -gain 0.1 -mgain 0.9999 -auto-mask 5 -auto-threshold 1 -local-rms
  -horizon-mask 10deg`` with exactly 2 major cycles (what -mgain 0.9999 gives in practice).
* ``MultiScaleClean`` + ``deconvolve_multiscale``: radler-like multiscale CLEAN with the
  WSClean major-cycle logic for ``-multiscale -multiscale-scale-bias 0.8 -mgain 0.95
  -auto-mask 3 -auto-threshold 0.5 -local-rms -horizon-mask 10deg``.

Images are cupy float32 arrays in WSClean orientation [y, x]; the PSF is N x N with its peak at
(N/2, N/2).  The minor loops run as one CUDA block doing exact Hogbom iterations on a candidate
pixel set (Clark-style: pixels above max(threshold, sidelobe_fraction * peak)); after every
minor loop the full residual is updated exactly by FFT convolution.

Auto-masking (radler): phase 1 cleans to auto_mask * sigma, the pixels (per scale) that received
components become the mask, phase 2 cleans inside the mask to auto_threshold * sigma.
Local RMS (radler RmsImage::Make): sigma map = sqrt(Gaussian-smoothed residual^2), Gaussian FWHM
= 25 theoretical beams; peaks are compared after scaling by min(rms) / rms(x, y).
"""
import math
import os
import warnings

import numpy as np
import cupy as cp

from .beam import DegeneratePSFError, check_psf_support, fit_beam

_HOGBOM_SRC = r"""
extern "C" __global__ void hogbom_set(const float* __restrict__ psf, const int pn, const int pc,
    const int* __restrict__ px, const int* __restrict__ py, float* val, const float* __restrict__ fac,
    float* comp, const int K, const float gain, const float thresh, const int maxiter,
    int* out_niter, float* out_peak)
{
    // Hogbom CLEAN restricted to K candidate pixels (px, py); val = residual at those pixels,
    // fac = local-RMS weighting of the peak search; PSF pn x pn with centre pc.
    __shared__ float sv[1024];
    __shared__ int si[1024];
    const int tid = threadIdx.x;
    const int nt = blockDim.x;
    int it = 0;
    float peak = 0.f;
    for (; it < maxiter; ++it) {
        float best = -1.f; int bi = -1;
        for (int k = tid; k < K; k += nt) {
            const float a = fabsf(val[k]) * fac[k];
            if (a > best) { best = a; bi = k; }
        }
        sv[tid] = best; si[tid] = bi;
        __syncthreads();
        for (int s = nt >> 1; s > 0; s >>= 1) {
            if (tid < s && sv[tid + s] > sv[tid]) { sv[tid] = sv[tid + s]; si[tid] = si[tid + s]; }
            __syncthreads();
        }
        peak = sv[0];
        const int ip = si[0];
        if (peak < thresh || ip < 0) break;
        const float amp = gain * val[ip];
        const int x0 = px[ip], y0 = py[ip];
        __syncthreads();
        if (tid == 0) comp[ip] += amp;
        for (int k = tid; k < K; k += nt) {
            const int dx = px[k] - x0 + pc, dy = py[k] - y0 + pc;
            if (dx >= 0 && dx < pn && dy >= 0 && dy < pn) val[k] -= amp * psf[(size_t)dy * pn + dx];
        }
        __syncthreads();
    }
    if (tid == 0) { *out_niter = it; *out_peak = peak; }
}
"""
_hogbom = cp.RawKernel(_HOGBOM_SRC, "hogbom_set")

# Same algorithm and the same float operations as hogbom_set (bit-identical result, including
# the order in which exactly equal peaks are resolved), but: warp-shuffle argmax (2 __syncthreads per iteration instead of 12), and the PSF
# subtraction is fused with the next iteration's per-thread peak search (one pass over the set).
_HOGBOM2_SRC = r"""
extern "C" __global__ void hogbom_set2(const float* __restrict__ psf, const int pn, const int pc,
    const int* __restrict__ px, const int* __restrict__ py, float* val, const float* __restrict__ fac,
    float* comp, const int K, const float gain, const float thresh, const int maxiter,
    float* out, const float* __restrict__ G)
{
    // G (optional, K x K): G[i K + k] = the PSF value hogbom_set reads for candidate k when the
    // component is at candidate i (a coalesced row read instead of K scattered PSF gathers)
    __shared__ float wv[32];
    __shared__ int wi[32];
    __shared__ float s_peak, s_amp;
    __shared__ int s_ip;
    // launched with exactly 1024 threads on candidate arrays permuted by hogbom_on_set: thread
    // p = 32 w + l (warp w, lane l) holds, at positions p + 1024 j, the candidates j 1024 + t of
    // hogbom_set's thread t = 32 l + w (NaN-padded).  The intra-warp shuffles then combine bits
    // 5-9 of t and the cross-warp step bits 0-4, so exact ties are resolved in the same order as
    // hogbom_set's tree, whose last (deciding) comparison is on bit 0 of t.
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nt = 1024, nw = 32;
    const int tid = threadIdx.x;
    float best = -1.f; int bi = -1;
    for (int k = tid; k < K; k += nt) {
        const float a = fabsf(val[k]) * fac[k];
        if (a > best) { best = a; bi = k; }
    }
    int it = 0;
    float peak = 0.f;
    for (; it < maxiter; ++it) {
        // argmax over the block; on ties the lower thread wins (as the shared-memory tree)
        float v = best; int i = bi;
        for (int o = 16; o > 0; o >>= 1) {
            const float v2 = __shfl_down_sync(0xffffffffu, v, o);
            const int i2 = __shfl_down_sync(0xffffffffu, i, o);
            if (v2 > v) { v = v2; i = i2; }
        }
        if (lane == 0) { wv[wid] = v; wi[wid] = i; }
        __syncthreads();
        if (wid == 0) {
            v = lane < nw ? wv[lane] : -2.f; i = lane < nw ? wi[lane] : -1;
            for (int o = 16; o > 0; o >>= 1) {
                const float v2 = __shfl_down_sync(0xffffffffu, v, o);
                const int i2 = __shfl_down_sync(0xffffffffu, i, o);
                if (v2 > v) { v = v2; i = i2; }
            }
            if (lane == 0) {
                s_peak = v; s_ip = i;
                if (i >= 0 && !(v < thresh)) { s_amp = gain * val[i]; comp[i] += s_amp; }
            }
        }
        __syncthreads();
        peak = s_peak;
        const int ip = s_ip;
        if (peak < thresh || ip < 0) break;
        const float amp = s_amp;
        const int x0 = px[ip], y0 = py[ip];
        best = -1.f; bi = -1;
        for (int k = tid; k < K; k += nt) {
            const int dx = px[k] - x0 + pc, dy = py[k] - y0 + pc;
            float vk = val[k];
            if (dx >= 0 && dx < pn && dy >= 0 && dy < pn) {
                const float p = G ? G[(size_t)ip * K + k] : psf[(size_t)dy * pn + dx];
                vk -= amp * p; val[k] = vk;
            }
            const float a = fabsf(vk) * fac[k];
            if (a > best) { best = a; bi = k; }
        }
    }
    if (tid == 0) { out[0] = (float)it; out[1] = peak; }
}
"""
_hogbom2 = cp.RawKernel(_HOGBOM2_SRC, "hogbom_set2")


# Multi-block (cooperative, one grid barrier per iteration) version for large candidate sets,
# where the single-block kernels are bound by the latency of the PSF gathers of one SM.  The
# argmax is a 64-bit atomicMax of (float bits of |val| * fac, ~rank) with rank = the position of
# the candidate in hogbom_set's tie order (bit-reversed thread index t = k mod 1024, then
# j = k div 1024), so it selects exactly the component hogbom_set selects; the float operations
# on val and comp are the same (bit-identical result).
_HOGBOM_MB_SRC = r"""
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
__device__ __forceinline__ unsigned long long hkey(const float a, const int k, const int J) {
    if (!(a == a)) return 0ull;                                   // NaN: never selected
    const unsigned int t = k & 1023, j = k >> 10;
    const unsigned int rank = (__brev(t) >> 22) * (unsigned int)J + j;
    return ((unsigned long long)__float_as_uint(a) << 32) | (unsigned long long)(0xffffffffu - rank);
}
extern "C" __global__ void hogbom_mb(const float* __restrict__ psf, const int pn, const int pc,
    const int* __restrict__ px, const int* __restrict__ py, float* val, const float* __restrict__ fac,
    float* comp, const int K, const int J, const float gain, const float thresh, const int maxiter,
    unsigned long long* keys, float* cval, float* out)
{
    cg::grid_group grid = cg::this_grid();
    __shared__ unsigned long long sk[32];
    __shared__ float sv[32];
    const int gtid = blockIdx.x * blockDim.x + threadIdx.x, gn = gridDim.x * blockDim.x;
    const int lane = threadIdx.x & 31, wid = threadIdx.x >> 5, nw = blockDim.x >> 5, B = gridDim.x;
    unsigned long long best = 0ull; float bestv = 0.f;
    for (int k = gtid; k < K; k += gn) {
        const float vk = val[k];
        const unsigned long long key = hkey(fabsf(vk) * fac[k], k, J);
        if (key > best) { best = key; bestv = vk; }
    }
    int it = 0;
    float peak = 0.f;
    for (; it < maxiter; ++it) {
        const int slot = it % 3;
        unsigned long long b = best; float bv = bestv;
        for (int o = 16; o > 0; o >>= 1) {
            const unsigned long long b2 = __shfl_down_sync(0xffffffffu, b, o);
            const float v2 = __shfl_down_sync(0xffffffffu, bv, o);
            if (b2 > b) { b = b2; bv = v2; }
        }
        if (lane == 0) { sk[wid] = b; sv[wid] = bv; }
        __syncthreads();
        if (wid == 0) {
            b = lane < nw ? sk[lane] : 0ull; bv = lane < nw ? sv[lane] : 0.f;
            for (int o = 16; o > 0; o >>= 1) {
                const unsigned long long b2 = __shfl_down_sync(0xffffffffu, b, o);
                const float v2 = __shfl_down_sync(0xffffffffu, bv, o);
                if (b2 > b) { b = b2; bv = v2; }
            }
            if (lane == 0) {
                cval[slot * B + blockIdx.x] = bv;
                if (b) atomicMax(&keys[slot], b);
            }
        }
        grid.sync();
        const unsigned long long g = *((volatile unsigned long long*)&keys[slot]);
        if (gtid == 0) keys[(it + 2) % 3] = 0ull;
        if (g == 0ull) { peak = -1.f; break; }
        peak = __uint_as_float((unsigned int)(g >> 32));
        if (peak < thresh) break;
        const unsigned int rank = 0xffffffffu - (unsigned int)(g & 0xffffffffull);
        const unsigned int t = __brev(rank / (unsigned int)J) >> 22, j = rank % (unsigned int)J;
        const int ip = (int)(j * 1024u + t);
        const float amp = gain * *((volatile float*)&cval[slot * B + (ip % gn) / blockDim.x]);
        const int x0 = px[ip], y0 = py[ip];
        best = 0ull; bestv = 0.f;
        for (int k = gtid; k < K; k += gn) {
            const int dx = px[k] - x0 + pc, dy = py[k] - y0 + pc;
            float vk = val[k];
            if (k == ip) comp[k] += amp;
            if (dx >= 0 && dx < pn && dy >= 0 && dy < pn) { vk -= amp * psf[(size_t)dy * pn + dx]; val[k] = vk; }
            const unsigned long long key = hkey(fabsf(vk) * fac[k], k, J);
            if (key > best) { best = key; bestv = vk; }
        }
        __syncthreads();
    }
    if (gtid == 0) { out[0] = (float)it; out[1] = peak; }
}
"""
_hogbom_mb = None
HOGBOM_MB_MIN_K = 6144
HOGBOM_SMALL_K = 512


def _hogbom_mb_kernel():
    global _hogbom_mb
    if _hogbom_mb is None:
        _hogbom_mb = cp.RawKernel(_HOGBOM_MB_SRC, "hogbom_mb", enable_cooperative_groups=True)
    return _hogbom_mb


def _hogbom_on_set_mb(psf, ys, xs, vals, fac, gain, thresh, maxiter):
    K = int(ys.size)
    ker = _hogbom_mb_kernel()
    nsm = cp.cuda.Device().attributes["MultiProcessorCount"]
    nth = 256
    B = nsm * int(os.environ.get("TTCALX_HOGBOM_MB_BPS", "2"))
    B = min(B, max(1, -(-K // nth)))
    comp = cp.zeros(K, cp.float32)
    keys = cp.zeros(3, cp.uint64); cval = cp.zeros(3 * B, cp.float32); out = cp.zeros(2, cp.float32)
    vals = cp.ascontiguousarray(vals, dtype=cp.float32)
    ker((B,), (nth,), (psf, np.int32(psf.shape[0]), np.int32(psf.shape[0] // 2), xs.astype(cp.int32),
                       ys.astype(cp.int32), vals, cp.ascontiguousarray(fac, dtype=cp.float32), comp, np.int32(K),
                       np.int32(-(-K // 1024)), np.float32(gain), np.float32(thresh), np.int32(maxiter),
                       keys, cval, out))
    o = out.get()
    return comp, int(o[0]), float(o[1])


_MB_BROKEN = [False]


def _mb_errors():
    """What makes the multi-block Hogbom unusable: a refused cooperative launch (driver / runtime
    error) or a kernel that does not compile (cupy.cuda.compiler.CompileException)."""
    c = cp.cuda
    errs = (getattr(getattr(c, "driver", None), "CUDADriverError", None),
            getattr(getattr(c, "runtime", None), "CUDARuntimeError", None),
            getattr(getattr(c, "compiler", None), "CompileException", None))
    return tuple(e for e in errs if isinstance(e, type))


def hogbom_on_set(psf, ys, xs, vals, fac, gain, thresh, maxiter):
    """Run the Hogbom kernel; returns (component amplitudes per candidate, n_iter, last peak)."""
    K = int(ys.size)
    if K >= HOGBOM_MB_MIN_K and not _MB_BROKEN[0]:
        try:
            return _hogbom_on_set_mb(psf, ys, xs, vals, fac, gain, thresh, maxiter)
        except _mb_errors() as e:
            # cooperative launch refused (grid not co-resident, e.g. MPS SM limit or a too
            # large TTCALX_HOGBOM_MB_BPS), or the cooperative-groups kernel does not compile
            # (NVRTC without cooperative_groups.h): the single-block kernel gives the same components
            _MB_BROKEN[0] = True
            warnings.warn(f"multi-block Hogbom unavailable ({e}); using the single-block kernel")
    if K <= HOGBOM_SMALL_K:
        # tiny sets (most multiscale sub-minor cycles): the original kernel, one host sync
        comp = cp.zeros(K, cp.float32)
        o = cp.zeros(2, cp.int32)
        pn = psf.shape[0]
        _hogbom((1,), (1024,), (psf, np.int32(pn), np.int32(pn // 2), xs.astype(cp.int32), ys.astype(cp.int32),
                                vals, fac, comp, np.int32(K), np.float32(gain), np.float32(thresh), np.int32(maxiter),
                                o[0:1], o[1:2]))
        o = o.get()
        return comp, int(o[0]), float(o[1:2].view(np.float32)[0])
    Kp = max(1024, -(-K // 1024) * 1024)
    pos = _hog_perm(cp.arange(K, dtype=cp.int32))           # candidate k -> permuted position
    px = cp.full(Kp, -(1 << 28), cp.int32); py = cp.full(Kp, -(1 << 28), cp.int32)
    pv = cp.full(Kp, np.nan, cp.float32); pf = cp.ones(Kp, cp.float32)
    px[pos] = xs; py[pos] = ys; pv[pos] = vals; pf[pos] = fac
    comp = cp.zeros(Kp, cp.float32)
    out = cp.zeros(2, cp.float32)
    pn = psf.shape[0]
    G = None
    if 1024 < Kp <= HOGBOM_MATRIX_MAX_K:
        G = cp.empty((Kp, Kp), cp.float32)
        _hog_matrix(psf, np.int32(pn), np.int32(pn // 2), px[:, None], py[:, None], px[None, :], py[None, :], G)
    _hogbom2((1,), (1024,), (psf, np.int32(pn), np.int32(pn // 2), px, py, pv, pf, comp, np.int32(Kp),
                             np.float32(gain), np.float32(thresh), np.int32(maxiter), out,
                             G if G is not None else np.uintp(0)))
    o = out.get()
    return comp[pos], int(o[0]), float(o[1])


HOGBOM_MATRIX_MAX_K = int(os.environ.get("TTCALX_HOGBOM_MATRIX_MAX_K", "4096"))
_hog_matrix = cp.ElementwiseKernel(
    "raw float32 psf, int32 pn, int32 pc, int32 xi, int32 yi, int32 xk, int32 yk", "float32 g",
    "const int dx = xk - xi + pc, dy = yk - yi + pc;"
    "g = (dx >= 0 && dx < pn && dy >= 0 && dy < pn) ? psf[(size_t)dy * pn + dx] : 0.f;", "hog_matrix")
_hog_perm = cp.ElementwiseKernel("int32 k", "int32 p",
                                 "const int t = k & 1023; p = (k - t) + ((t & 31) << 5) + (t >> 5);", "hog_perm")


_SPARSE_SRC = r"""
extern "C" __global__ void sparse_conv_sub(const float* __restrict__ R, float* out, const int N,
    const float* __restrict__ ker, const int kn, const int kc,
    const int* __restrict__ cy, const int* __restrict__ cx, const float* __restrict__ ca, const int nc)
{
    // out[y, x] = R[y, x] - sum_j ca[j] ker[y - cy[j] + kc, x - cx[j] + kc]  (kernel kn x kn, zero outside)
    __shared__ int sy[64], sx[64];
    __shared__ float sa[64];
    const int t = threadIdx.y * blockDim.x + threadIdx.x;
    if (t < nc) { sy[t] = cy[t]; sx[t] = cx[t]; sa[t] = ca[t]; }
    __syncthreads();
    const int x = blockIdx.x * blockDim.x + threadIdx.x, y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;
    float acc = 0.f;
    for (int j = 0; j < nc; ++j) {
        const int dy = y - sy[j] + kc, dx = x - sx[j] + kc;
        if (dy >= 0 && dy < kn && dx >= 0 && dx < kn) acc += sa[j] * ker[(size_t)dy * kn + dx];
    }
    out[(size_t)y * N + x] = R[(size_t)y * N + x] - acc;
}
"""
_sparse = cp.RawKernel(_SPARSE_SRC, "sparse_conv_sub")


def sparse_conv_sub(R, ker, kc, cy, cx, ca):
    """R - sum_j ca_j ker(. - (cy_j, cx_j)) for <= 64 components (new array)."""
    N = R.shape[0]; kn = ker.shape[0]; nc = int(ca.size)
    assert nc <= 64
    R = cp.ascontiguousarray(R, dtype=cp.float32); ker = cp.ascontiguousarray(ker, dtype=cp.float32)
    out = cp.empty_like(R)
    _sparse(((N + 31) // 32, (N + 7) // 8), (32, 8), (R, out, np.int32(N), ker, np.int32(kn), np.int32(kc),
                                                     cy, cx, ca.astype(cp.float32), np.int32(nc)))
    return out


# ----------------------------------------------------------------------------- helpers
def fast_len(n):
    """smallest 2^a 3^b 5^c 7^d >= n"""
    best = 1 << int(math.ceil(math.log2(n)))
    for a in (1, 3, 9, 27, 81, 243):
        for b in (1, 5, 25, 125, 625, 3125):
            for c in (1, 7, 49, 343):
                m = a * b * c
                if m > best:
                    continue
                p = m
                while p < n:
                    p *= 2
                best = min(best, p)
    return best


class Convolver:
    """Linear (non-cyclic) convolution of N x N images with M x M kernels centred at M//2,
    by zero-padded real FFTs of size P = fast_len(N + M//2 + extra)."""

    def __init__(self, N, M=None, extra=2):
        self.N = N
        self.M = N if M is None else M
        self.P = fast_len(N + self.M // 2 + extra)

    def kfft(self, k):
        P, M = self.P, k.shape[0]; c = M // 2
        kp = cp.zeros((P, P), cp.float32)
        k = cp.asarray(k, dtype=cp.float32)
        kp[:M - c, :M - c] = k[c:, c:]; kp[:M - c, P - c:] = k[c:, :c]
        kp[P - c:, :M - c] = k[:c, c:]; kp[P - c:, P - c:] = k[:c, :c]
        return cp.fft.rfft2(kp)

    def img_fft(self, img):
        x = cp.zeros((self.P, self.P), cp.float32); x[:self.N, :self.N] = img
        return cp.fft.rfft2(x)

    def back(self, F):
        return cp.fft.irfft2(F, s=(self.P, self.P))[:self.N, :self.N]

    def __call__(self, img, K):
        return self.back(self.img_fft(img) * K)


def gaussian_kernel(N, bmaj_deg, bmin_deg, bpa_deg, scale_deg):
    """Peak-1 elliptical Gaussian (2h+1)^2 in [y, x] orientation; BPA from +y (north) towards -x
    (east for CDELT1 < 0), as BPA in the FITS header."""
    s2f = 2.0 * math.sqrt(2.0 * math.log(2.0))
    smaj = bmaj_deg / scale_deg / s2f
    smin = bmin_deg / scale_deg / s2f
    half = int(min(N // 2 - 1, math.ceil(5 * smaj)))
    y, x = cp.mgrid[-half:half + 1, -half:half + 1].astype(cp.float32)
    pa = math.radians(bpa_deg)
    a = -x * math.sin(pa) + y * math.cos(pa)   # along the major axis, unit vector (-sin pa, cos pa)
    b = x * math.cos(pa) + y * math.sin(pa)    # along the minor axis
    k = cp.exp(-0.5 * (a * a / smaj ** 2 + b * b / smin ** 2)).astype(cp.float32)
    return k, half


def horizon_mask(N, scale_deg, horizon_deg=10.0):
    """True where elevation > horizon_deg (r = sqrt(l^2 + m^2) <= cos(horizon))."""
    j = cp.arange(N, dtype=cp.float32) - N // 2
    s = math.radians(scale_deg)
    r2 = (j[None, :] * s) ** 2 + (j[:, None] * s) ** 2
    return r2 <= math.cos(math.radians(horizon_deg)) ** 2


def rms_image(img, beam_fwhm_pix, window=25.0):
    """radler RmsImage::Make: sqrt(Gaussian-weighted mean of img^2), Gaussian FWHM = window*beam
    (cyclic FFT smoothing with a unit-sum Gaussian)."""
    N = img.shape[0]
    sig = window * beam_fwhm_pix / (2.0 * math.sqrt(2.0 * math.log(2.0)))
    if os.environ.get("TTCALX_RMS_FILTER", "fft") != "circulant":      # 748884a (default)
        f = cp.fft.fftfreq(N).astype(cp.float32)
        g = cp.exp(-2.0 * (math.pi * sig) ** 2 * (f[:, None] ** 2 + f[None, :] ** 2)).astype(cp.float32)
        sm = cp.fft.ifft2(cp.fft.fft2(img.astype(cp.float32) ** 2) * g).real
        return cp.sqrt(cp.maximum(sm, 0)).astype(cp.float32)
    # TTCALX_RMS_FILTER=circulant: the Gaussian filter is separable, so the 2-D cyclic convolution is A X A^T with A the N x N
    # circulant matrix of the 1-D kernel ifft(g1): two SGEMMs instead of a complex N x N FFT pair
    # (N = 2358 = 2 3^2 131 needs a Bluestein cuFFT plan, ~0.5-0.9 s to create after every
    # plan-cache clear in the hour driver).  Same filter; differences ~1e-6 of the max.
    A = _circulant_gauss(N, round(sig, 9))
    x2 = img.astype(cp.float32) ** 2
    sm = A @ x2 @ A.T
    return cp.sqrt(cp.maximum(sm, 0)).astype(cp.float32)


_CIRC_CACHE = {}


def _circulant_gauss(N, sig):
    key = (N, sig)
    if key not in _CIRC_CACHE:
        if len(_CIRC_CACHE) >= 4:
            _CIRC_CACHE.pop(next(iter(_CIRC_CACHE)))       # oldest
        f = np.fft.fftfreq(N).astype(np.float32).astype(np.float64)
        a = np.fft.ifft(np.exp(-2.0 * (math.pi * sig) ** 2 * f ** 2)).real
        idx = (np.arange(N)[:, None] - np.arange(N)[None, :]) % N
        _CIRC_CACHE[key] = cp.asarray(a[idx], dtype=cp.float32)
    return _CIRC_CACHE[key]


def local_rms_factor(res, theo_pix, scale_deg, horizon, window=25.0):
    """(fac, min_rms): fac = min_rms / rms(x, y).  The minimum is taken over the horizon mask,
    excluding pixels whose RMS window reaches the zeroed region beyond the horizon (r >= 1)."""
    N = res.shape[0]
    rms = rms_image(res, theo_pix, window)
    jj = cp.arange(N, dtype=cp.float32) - N // 2
    far = cp.sqrt(jj[None, :] ** 2 + jj[:, None] ** 2) < (1.0 / math.radians(scale_deg) - 3 * window * theo_pix / 2.3548)
    sel = rms[horizon & far]
    if sel.size == 0:
        check_psf_support(scale_deg, theo_pix, window)       # raises with the numbers
        raise DegeneratePSFError(f"degenerate PSF: no pixel for the local-RMS minimum (theoretical beam "
                                 f"{theo_pix:.4g} px, window {window:g})")
    minrms = float(sel.min())
    return (minrms / cp.maximum(rms, 1e-30)).astype(cp.float32), minrms


# the same float32 operations as cp.where(m, cp.abs(x) * f, 0), in one kernel
_masked_absf = cp.ElementwiseKernel("T x, T f, bool m", "T y", "y = m ? abs(x) * f : (T)0", "masked_absf")


def mad_std(x):
    """radler StdDevFromMAD (median of |x|, i.e. about zero)."""
    return float(1.4826 * cp.median(cp.abs(x)))


# ----------------------------------------------------------------------------- snapshot (Clark)
class SnapshotClean:
    def __init__(self, psf, N, scale_deg, theo_pix, beam=None, gain=0.1, horizon_deg=10.0, kmax=60000,
                 log=None):
        self.N = N; self.scale = scale_deg
        self.psf = cp.asarray(psf, dtype=cp.float32)
        self.cv = Convolver(N, self.psf.shape[0], extra=1)
        self.Kpsf = self.cv.kfft(self.psf)
        self.gain = gain
        self.mask = horizon_mask(N, scale_deg, horizon_deg)
        self.beam = tuple(beam) if beam is not None else fit_beam(self.psf, scale_deg, theo_pix)
        self.kmax = kmax
        self.log = log or (lambda *a: None)
        # Clark candidate fraction: peak |PSF| outside ~2.5 restoring-beam FWHM of the centre
        p = self.psf; c = p.shape[0] // 2
        hw = max(3, int(2.5 * self.beam[0] / scale_deg))
        q = cp.abs(p).copy(); q[c - hw:c + hw + 1, c - hw:c + hw + 1] = 0
        self.clark_frac = float(q.max())

    def conv(self, img):
        return self.cv(img, self.Kpsf)

    def clean_full(self, res, model, thresh, fac=None, maxiter=50000, niter_done=0):
        """Clark cycles over the whole (horizon-masked) image until the scaled peak < thresh."""
        N = self.N
        facimg = cp.ones((N, N), cp.float32) if fac is None else fac
        nit = niter_done; ncyc = 0
        while nit < maxiter:
            a = cp.where(self.mask, cp.abs(res) * facimg, 0)
            peak = float(a.max())
            if peak < thresh:
                break
            t = max(thresh, peak * self.clark_frac)
            sel = a >= t
            if int(sel.sum()) > self.kmax:
                t = float(cp.sort(a[sel])[-self.kmax]); sel = a >= t
            ys, xs = cp.nonzero(sel)
            comp, n, _ = hogbom_on_set(self.psf, ys, xs, res[ys, xs].astype(cp.float32),
                                       facimg[ys, xs].astype(cp.float32), self.gain, t, maxiter - nit)
            nit += n; ncyc += 1
            dm = cp.zeros((N, N), cp.float32); dm[ys, xs] = comp
            model += dm
            res = res - self.conv(dm)
            if n == 0:
                break
        self.log(f"  clean_full: {nit - niter_done} iters, {ncyc} Clark cycles, thresh {thresh:.4g}, frac {self.clark_frac:.3f}")
        return model, res, nit

    def clean_masked(self, res, model, thresh, maskpix, maxiter=50000, niter_done=0):
        """Hogbom restricted to a fixed pixel set (auto-mask phase 2), exact, one kernel launch."""
        ys, xs = cp.nonzero(maskpix & self.mask)
        comp, n, _ = hogbom_on_set(self.psf, ys, xs, res[ys, xs].astype(cp.float32), cp.ones(ys.size, cp.float32),
                                   self.gain, thresh, maxiter - niter_done)
        dm = cp.zeros_like(model); dm[ys, xs] = comp
        model += dm
        res = res - self.conv(dm)
        self.log(f"  clean_masked: {n} iters on {ys.size} pixels, thresh {thresh:.4g}")
        return model, res, niter_done + n

    def restore(self, model, residual):
        k, h = gaussian_kernel(self.N, *self.beam, self.scale)
        cv = Convolver(self.N, 2 * h + 1, extra=1)
        return residual + cv(model, cv.kfft(k))


def clean_snapshot(dirty, psf, residual_fn, scale_deg, theo_pix, sky, params, beam=None, log=None):
    """Snapshot CLEAN driver (radler auto-mask logic with a fixed number of major cycles).

    phase 1: Clark clean to auto_mask * min(local rms) (local rms of the dirty image)
    major 1: exact residual from the visibilities
    phase 2: Hogbom inside the phase-1 component mask to auto_threshold * MAD-sigma
    major 2: exact residual; restore with the fitted beam.
    residual_fn(model) -> residual image.  sky: pixels inside the horizon (r < 1) for the MAD.
    Returns (restored, residual, model, info)."""
    N = dirty.shape[0]
    check_psf_support(scale_deg, theo_pix, params.local_rms_window)    # before the beam fit
    D = SnapshotClean(psf, N, scale_deg, theo_pix, beam=beam, gain=params.gain, horizon_deg=params.horizon_deg, log=log)
    model = cp.zeros((N, N), cp.float32)
    res = dirty.copy()
    sig0 = mad_std(res[sky])
    fac, minrms = local_rms_factor(res, theo_pix, scale_deg, D.mask, params.local_rms_window)
    t1 = params.auto_mask * minrms
    model, res, nit = D.clean_full(res, model, t1, fac=fac, maxiter=params.niter)
    nmaj = 0
    if params.nmajor >= 1:
        res = residual_fn(model); nmaj += 1
    sig1 = mad_std(res[sky])
    t2 = params.auto_threshold * sig1
    model, res, nit = D.clean_masked(res, model, t2, model != 0, maxiter=params.niter, niter_done=nit)
    if params.nmajor >= 2:
        res = residual_fn(model); nmaj += 1
    restored = D.restore(model, res)
    info = dict(niter=nit, nmajor=nmaj, thr1=t1, thr2=t2, minrms=minrms, sig0=sig0, sig1=sig1, beam=D.beam,
                ncomp=int(cp.count_nonzero(model)), clark_frac=D.clark_frac)
    return restored, res, model, info


# ----------------------------------------------------------------------------- multiscale (hour)
def shape_kernel(N, scale):
    """radler tapered-quadratic scale kernel of width `scale` px (unit sum), N x N centred."""
    c = N // 2
    k = cp.zeros((N, N), cp.float32)
    if scale == 0:
        k[c, c] = 1.0; return k
    R = 0.5 * scale; h = int(math.ceil(R))
    y, x = cp.mgrid[-h:h + 1, -h:h + 1].astype(cp.float32)
    rn = cp.sqrt(x * x + y * y) / R
    f = cp.where(rn < 1, (1 - rn * rn) * 0.5 * (1 + cp.cos(cp.pi * rn)), 0).astype(cp.float32)
    k[c - h:c + h + 1, c - h:c + h + 1] = f / f.sum()
    return k


class MultiScaleClean:
    """Scales 0 and 4 x theoretical beam, doubling up to N/2 (radler defaults).  Scale choice:
    argmax_s max|S_s * R| * fac * bias_s with bias_s = scale_bias^-log2(s / s_1) (radler: 1 for
    scale 0 and the first non-zero scale s_1, favours large scales); components of scale s stay
    ceil(s / 2) px from the image edge (radler's scale border); sub-minor loop = Hogbom on the
    candidate set of that scale with the twice-convolved PSF S_s * S_s * PSF, then exact FFT update
    of the full residual with S_s * PSF.  A cycle that raises the peak above divergence_limit x the
    loop's first peak is undone and ends the loop (radler -divergence-limit, default 4)."""

    def __init__(self, psf, N, scale_deg, theo_pix, beam=None, scale_bias=0.8, gain=0.1, horizon_deg=10.0,
                 kmax=60000, max_scale_frac=0.5, multiscale=True, max_scales=0, divergence_limit=4.0, log=None):
        self.N = N; self.sdeg = scale_deg; self.gain = gain; self.kmax = kmax
        self.divergence_limit = divergence_limit; self.diverged = False
        self.sparse_max = min(64, int(os.environ.get("TTCALX_SPARSE_UPDATE_MAX", "0")))    # 0: FFT updates (748884a); 64 = kernel limit
        self._small, self._fullps = {}, {}
        self.log = log or (lambda *a: None)
        self.psf = cp.asarray(psf, dtype=cp.float32)
        self.beam = tuple(beam) if beam is not None else fit_beam(self.psf, scale_deg, theo_pix)
        self.mask = horizon_mask(N, scale_deg, horizon_deg)
        b = self.beam[0] / scale_deg
        scales = [0]
        if multiscale:
            s = 4 * theo_pix
            while s < N * max_scale_frac:
                scales.append(s); s *= 2
        if max_scales:
            scales = scales[:max_scales]
        self.scales = scales
        self.C = Convolver(N, N, extra=2)
        Kpsf = self.C.kfft(self.psf)
        hmax = max(int(math.ceil(0.5 * s)) for s in scales)
        if os.environ.get("TTCALX_SCALE_FFT", "full") == "small":
            self.Cs = Convolver(N, 2 * hmax + 1, extra=2)
            self.KSs = [None if s == 0 else self.Cs.kfft(shape_kernel(N, s)) for s in scales]
        else:
            self.Cs = self.C; self.KSs = None
        self.KS, self.KP, self.Pss, self.pss0, self.bias, self.edge = [], [], [], [], [], []
        c = N // 2
        for k, s in enumerate(scales):
            e = int(math.ceil(0.5 * s)); m = cp.zeros((N, N), bool); m[e:N - e, e:N - e] = True
            self.edge.append(m)
            KS = self.C.kfft(shape_kernel(N, s))
            KP = KS * Kpsf
            self.KS.append(KS); self.KP.append(KP)
        if self.KSs is None:
            self.KSs = self.KS
        for k, s in enumerate(scales):
            KS, KP = self.KS[k], self.KP[k]
            delta = cp.zeros((N, N), cp.float32); delta[c, c] = 1
            Ps_img = self.C(delta, KP)
            Pss_img = self.C(Ps_img, KS)
            self.pss0.append(float(Pss_img[c, c]))
            self.Pss.append((Pss_img / float(Pss_img[c, c])).astype(cp.float32))
            self.bias.append(1.0 if s == 0 else (1.0 / scale_bias) ** math.log2(s / scales[1]))
        self.frac = []          # Clark fraction per scale: peak sidelobe of the normalised P_ss
        for k, s in enumerate(scales):
            q = cp.abs(self.Pss[k]).copy()
            hw = max(3, int(2.5 * b + s))
            q[c - hw:c + hw + 1, c - hw:c + hw + 1] = 0
            self.frac.append(float(q.max()))
        self.log(f"MultiScaleClean: beam {self.beam[0]*60:.2f}'x{self.beam[1]*60:.2f}' ({b:.1f} px), "
                 f"scales {[round(s, 1) for s in scales]}, clark frac {[round(f, 3) for f in self.frac]}")

    def scale_images(self, R, active=None):
        # the scale kernels have support +-ceil(s/2) px only: linear convolution needs a padded
        # size of N + hmax (self.Cs), not the N + N/2 of the PSF convolutions (self.C)
        if active is None:
            active = [True] * len(self.scales)
        F = self.Cs.img_fft(R) if any(a and s != 0 for a, s in zip(active, self.scales)) else None
        return [None if not active[k] else R if s == 0 else self.Cs.back(F * self.KSs[k])
                for k, s in enumerate(self.scales)]

    def small_shape(self, k):
        """(kernel, centre) of scale k's shape function cut to its support (2h+1)^2."""
        if k not in self._small:
            s = self.scales[k]; h = int(math.ceil(0.5 * s)); c = self.N // 2
            self._small[k] = (cp.ascontiguousarray(shape_kernel(self.N, s)[c - h:c + h + 1, c - h:c + h + 1]), h)
        return self._small[k]

    def full_ps(self, k):
        """(kernel, centre) of S_k * PSF as the FFT update applies it (offsets -N/2-h .. N/2-1+h)."""
        if k not in self._fullps:
            h = int(math.ceil(0.5 * self.scales[k])); c = self.N // 2; P = self.C.P
            img = cp.fft.irfft2(self.KP[k], s=(P, P))
            n = self.N + 2 * h
            self._fullps[k] = (cp.ascontiguousarray(cp.roll(img, (c + h, c + h), axis=(0, 1))[:n, :n]
                                                    .astype(cp.float32)), c + h)
        return self._fullps[k]

    def minor(self, R, M, stop, fac, scale_masks=None, maxiter=500000, niter_done=0, comp_masks=None,
              max_cycles=100000):
        """Clean until all scaled peaks < stop.  scale_masks: per-scale allowed pixels (phase 2)."""
        N = self.N; nit = niter_done; ncyc = 0
        if comp_masks is None:
            comp_masks = [cp.zeros((N, N), bool) for _ in self.scales]
        allow = [(self.mask if scale_masks is None else (self.mask & scale_masks[k])) & self.edge[k]
                 for k in range(len(self.scales))]
        per_scale = [0] * len(self.scales)
        first = None
        # a scale whose allowed region is empty (phase 2: no phase-1 components of that scale) has
        # peak 0 and can never be selected: its scale image is not computed (same result)
        active = [bool(v) for v in cp.stack([a.any() for a in allow]).get()]
        nsc = len(self.scales)
        while nit < maxiter and ncyc < max_cycles:
            Rs = self.scale_images(R, active)
            # fused masked |R_s| * fac (one kernel per scale), one device->host copy for all maxima
            absf = [_masked_absf(Rs[k], fac, allow[k]) if active[k] else None for k in range(nsc)]
            act = [k for k in range(nsc) if active[k]]
            pk = [0.0] * nsc
            if act:
                for k, p in zip(act, cp.stack([absf[k].max() for k in act]).get()):
                    pk[k] = float(p)
            score = [pk[k] * self.bias[k] for k in range(len(self.scales))]
            k = int(np.argmax(score))
            if first is None:
                first = score[k]
            elif self.divergence_limit and score[k] > self.divergence_limit * first:
                # the last cycle made things worse: undo it and stop (radler "divergence")
                M[...] = M_last; R = R_last; self.diverged = True
                nit -= n_last; per_scale[k_last] -= n_last
                self.log(f"   minor: diverging (peak {score[k]:.4g} > {self.divergence_limit:g} x {first:.4g}); "
                         f"last cycle undone")
                break
            if score[k] < stop:
                break
            t = max(stop / self.bias[k], pk[k] * self.frac[k])
            absk = absf[k] if absf[k] is not None else cp.zeros((N, N), cp.float32)
            if Rs[k] is None:
                Rs[k] = cp.zeros((N, N), cp.float32)
            sel = absk >= t
            if int(sel.sum()) > self.kmax:
                t = float(cp.sort(absk[sel])[-self.kmax]); sel = absk >= t
            ys, xs = cp.nonzero(sel)
            vals = (Rs[k][ys, xs] / self.pss0[k]).astype(cp.float32)
            comp, n, _ = hogbom_on_set(self.Pss[k], ys, xs, vals, fac[ys, xs].astype(cp.float32), self.gain,
                                       t / self.pss0[k], maxiter - nit)
            if n == 0:
                break
            if self.divergence_limit:
                M_last, R_last, n_last, k_last = M.copy(), R, n, k
            nit += n; ncyc += 1; per_scale[k] += n
            dm = cp.zeros((N, N), cp.float32); dm[ys, xs] = comp
            comp_masks[k] |= dm != 0
            nz = cp.nonzero(comp)[0] if self.sparse_max else None
            if nz is not None and int(nz.size) <= self.sparse_max:
                # few components (most cycles of the deep phase-2 loops): direct sums of the
                # shifted kernels instead of a P x P FFT pair (the same linear convolution)
                cy, cx, ca = ys[nz].astype(cp.int32), xs[nz].astype(cp.int32), comp[nz]
                if self.scales[k] == 0:
                    M += dm
                    R = sparse_conv_sub(R, self.psf, N // 2, cy, cx, ca)
                else:
                    kS, hS = self.small_shape(k)
                    M -= sparse_conv_sub(cp.zeros_like(M), kS, hS, cy, cx, ca)
                    kP, cP = self.full_ps(k)
                    R = sparse_conv_sub(R, kP, cP, cy, cx, ca)
            else:
                Fd = self.C.img_fft(dm)
                M += dm if self.scales[k] == 0 else self.C.back(Fd * self.KS[k])
                R = R - self.C.back(Fd * self.KP[k])
        self.log(f"   minor: {nit - niter_done} iters, {ncyc} cycles, stop {stop:.4g}, per scale {per_scale}")
        return R, M, nit, comp_masks

    def restore(self, M, R):
        k, h = gaussian_kernel(self.N, *self.beam, self.sdeg)
        big = cp.zeros((self.N, self.N), cp.float32); c = self.N // 2
        big[c - h:c + h + 1, c - h:c + h + 1] = k
        return R + self.C(M, self.C.kfft(big))


def deconvolve_multiscale(dirty, psf, scale_deg, theo_pix, params, residual_fn=None, beam=None, max_major=20,
                          log=None):
    """WSClean-like major-cycle driver for multiscale CLEAN with auto-mask/auto-threshold.

    Each minor loop stops at max(threshold, (1 - mgain) * peak); the residual is then recomputed
    from the visibilities by residual_fn(model).  When a minor loop reaches the auto-mask
    threshold, the component positions become the mask and the threshold drops to
    auto_threshold * sigma (phase 2); stops when the peak inside the mask is below it.
    Returns (restored, residual, model, info)."""
    log = log or (lambda *a: None)
    N = dirty.shape[0]
    D = MultiScaleClean(psf, N, scale_deg, theo_pix, beam=beam, scale_bias=params.multiscale_scale_bias,
                        gain=params.gain, horizon_deg=params.horizon_deg, multiscale=params.multiscale,
                        max_scales=params.multiscale_max_scales, log=log)
    sky = D.mask
    # pixels above the horizon (r < 1), the region of the MAD noise estimate (as clean_snapshot)
    above = horizon_mask(N, scale_deg, 0.0)
    R = dirty.copy(); M = cp.zeros((N, N), cp.float32)
    div = getattr(params, "divergence_limit", 4.0); D.divergence_limit = div; raw_prev = None
    # auto_mask = 0: no auto-mask phase (phase 2 from the start, unmasked); the stopping threshold
    # is max(threshold, auto_threshold * sigma) as WSClean's -threshold / -auto-threshold
    nit = 0; phase = 1 if params.auto_mask else 2; masks = None; comp_masks = None; nmajor = 0
    user_mask = None
    if params.fits_mask:      # WSClean -fits-mask: components (of every scale) only inside the mask
        from astropy.io import fits as _fits
        user_mask = cp.asarray(_fits.getdata(params.fits_mask).squeeze() > 0.5)
        if user_mask.shape != (N, N):
            raise ValueError(f"fits_mask {params.fits_mask}: shape {user_mask.shape} != image {(N, N)}")
        sky = sky & user_mask
        D.mask = sky
    info = dict(majors=[])
    while True:
        # radler applies -local-rms only while auto-masking (phase 1): its logs print "Lowest RMS
        # in image" for the first deconvolution only, and in the auto-threshold phase (phase 2)
        # thresholds use the plain MAD noise ("Estimated standard deviation") with unweighted peaks.
        if params.local_rms and phase == 1:
            fac, sigma = local_rms_factor(R, theo_pix, scale_deg, sky, params.local_rms_window)
        else:
            fac = cp.ones((N, N), cp.float32); sigma = mad_std(R[above])
        region = sky if (phase == 1 or not masks) else (sky & cp.stack(masks).any(axis=0))
        peak = float(cp.where(region, cp.abs(R) * fac, 0).max())
        thr = max(params.threshold, (params.auto_mask if phase == 1 else params.auto_threshold) * sigma)
        stop = max(thr, (1 - params.mgain) * peak) if residual_fn is not None else thr
        log(f" major {nmajor}: peak {peak:.4g} sigma {sigma:.4g} phase {phase} stop {stop:.4g}")
        raw = float(cp.where(sky, cp.abs(R), 0).max())
        if div and raw_prev is not None and raw > div * raw_prev:
            # the last major cycle raised the (unweighted) peak: keep the model before it and stop
            log(f"  diverged in major {nmajor}: peak |R| {raw:.4g} > {div:g} x {raw_prev:.4g}; "
                f"reverting to the previous model and stopping")
            M, R = M_prev, R_prev; info["diverged"] = True
            break
        raw_prev = raw if raw_prev is None else min(raw_prev, raw)
        info["majors"].append(dict(peak=peak, sigma=sigma, phase=phase, stop=stop, nit=nit))
        if peak < thr and phase == 2:
            break
        if peak < thr and phase == 1:
            phase = 2; masks = [m.copy() for m in comp_masks] if comp_masks else None
            continue
        final = stop <= thr * 1.0000001
        M_prev, R_prev = M.copy(), R.copy()
        R, M, nit, comp_masks = D.minor(R, M, stop, fac, scale_masks=masks if phase == 2 else None,
                                        maxiter=params.niter, niter_done=nit, comp_masks=comp_masks)
        did = nit - info["majors"][-1]["nit"]
        if residual_fn is not None and did > 0:
            R = residual_fn(M); nmajor += 1
        if not (bool(cp.isfinite(M).all()) and bool(cp.isfinite(R).all())):
            # divergence (e.g. deep cleaning of large scales whose PSF response is ~1e-4): keep the
            # last finite model and stop instead of writing NaN images
            log(f"  diverged in major {nmajor}: reverting to the previous model and stopping")
            M, R = M_prev, R_prev; info["diverged"] = True
            break
        if D.diverged:
            info["diverged"] = True
            break
        if did == 0 and phase == 2:
            break
        if nit >= params.niter or nmajor >= max_major:
            break
        if final:     # this minor loop went down to the phase threshold (WSClean "(final)")
            if phase == 1:
                phase = 2; masks = [m.copy() for m in comp_masks]
                continue
            break
    info.setdefault("diverged", False)
    info.update(nit=nit, nmajor=nmajor, beam=D.beam, scales=D.scales, niter_reached=nit >= params.niter,
                max_major_reached=nmajor >= max_major)
    return D.restore(M, R), R, M, info
