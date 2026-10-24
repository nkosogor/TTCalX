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


class Reprojector:
    def __init__(self, N, sc):
        self.N = N; self.sc = sc
        self.l, self.m, self.n, self.inside = target_dirs(N, sc)

    def coords(self, R):
        """For each pixel of the target grid: (y, x) pixel position in the source frame, validity
        and n in the source frame, where uvw_target = R uvw_source."""
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

    def __call__(self, img, R, order=3, coords=None, up=1):
        """Resample source-frame img onto the target grid (zero outside the horizon)."""
        if coords is None:
            coords = self.coords(R)
        yx, ok, _ = coords
        if up != 1:
            yx = yx * up
        out = cnd.map_coordinates(img.astype(cp.float32), yx, order=order, mode="constant", cval=0.0)
        return cp.where(ok, out, 0), ok


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
