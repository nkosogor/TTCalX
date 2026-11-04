"""Reprojection between SIN images of different phase centres (w-snapshot imaging).

With the gridder convention exp(+2 pi i (u l + v m - w (n-1))), the direction cosines
(l, m, -n) transform like (u, v, w): if uvw_target = R uvw_frame then
(l, m, -n)_target = R (l, m, -n)_frame.  Images are resampled with cubic/quintic splines.

Flux convention: dirty/residual images are surface brightness (Jy/beam) and are resampled
as-is; a *model* (Jy/pixel) moved from frame A to frame B is multiplied by the SIN pixel
solid-angle Jacobian n_A / n_B (a pixel covers dl dm / n steradian).
"""
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
        import os
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
    return cp.ascontiguousarray(cnd.spline_filter(img, order, output=cp.float32, mode="constant"))


def fft_upsample(img, up=2, taper=None):
    """Band-limited (zero-padded FFT) upsampling, sample i -> up*i; taper: fftfreq-ordered filter."""
    N = img.shape[0]
    F = cp.fft.fft2(img.astype(cp.complex64))
    if taper is not None:
        F *= taper
    F = cp.fft.fftshift(F)
    M = N * up; o = (M - N) // 2
    G = cp.zeros((M, M), cp.complex64); G[o:o + N, o:o + N] = F
    return (cp.fft.ifft2(cp.fft.ifftshift(G)).real * (up * up)).astype(cp.float32)
