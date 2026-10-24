"""Briggs weighting + inner Tukey taper with WSClean 3.4 semantics.

Matches WSClean's PSF to ~1e-3 of the peak.  The non-obvious details:

* The weight (density) grid is the *padded* image: Np = ceil(1.2 N) rounded up to a multiple of 4
  (WSClean 3.4: 2358 -> 2832, 3122 -> 3748), cell du = 1 / (Np * scale_rad).
* Samples are folded onto the half plane v >= 0 ((u, v) -> (-u, -v) if v < 0) and binned with
  floor(u / du) + Np/2 (no rounding to the nearest cell).
* Briggs: f2 = (5 * 10^-R)^2 / (sum_cells W^2 / sum_cells W);   w_img = w_nat / (1 + W_cell f2),
  with W the summed natural weight per cell (WSClean's "robustness" definition).
* -taper-inner-tukey L multiplies by 0.5 (1 - cos(pi r / L)) for r < L, with r the uv distance
  [lambda] of the *cell corner* (iu - Np/2, iv - Np/2) * du, not of the sample.

The grid can be accumulated over any number of chunks (``add``) before ``finalize``, which
gives one global density over e.g. a whole hour (= WSClean on the concatenated MS).
All functions take an array module ``xp`` (numpy or cupy).
"""
import math

import numpy as np


def padded_size(n, padding=1.2):
    p = int(math.ceil(n * padding))
    return (p + 3) // 4 * 4


class WeightGrid:
    def __init__(self, N, scale_rad, robust=0.0, tukey_inner=None, padding=1.2, xp=None):
        if xp is None:
            import cupy as xp
        self.xp = xp
        self.npix = padded_size(N, padding)
        self.du = 1.0 / (self.npix * scale_rad)
        self.robust = robust
        self.tukey = tukey_inner
        self.grid = xp.zeros(self.npix * self.npix, xp.float64)
        self.f2 = None

    def _idx(self, u, v, nw):
        xp = self.xp
        neg = v < 0
        uu = xp.where(neg, -u, u); vv = xp.where(neg, -v, v)
        n = self.npix
        iu = xp.floor(uu / self.du).astype(xp.int64) + n // 2
        iv = xp.floor(vv / self.du).astype(xp.int64) + n // 2
        good = (nw > 0) & (iu >= 0) & (iu < n) & (iv >= 0) & (iv < n)
        return xp.where(good, iv * n + iu, 0), good, iu, iv

    def add(self, u, v, nw):
        """Accumulate natural weights nw of samples (u, v) [lambda] into the density grid."""
        xp = self.xp
        idx, good, _, _ = self._idx(u, v, nw)
        self.grid += xp.bincount(idx, xp.where(good, nw, 0).astype(xp.float64), minlength=self.grid.size)

    def finalize(self):
        g = self.grid
        self.f2 = (5.0 * 10 ** (-self.robust)) ** 2 / (float((g ** 2).sum()) / float(g.sum()))
        return self

    def weights(self, u, v, nw):
        """Imaging weights (float32) of samples (u, v, nw); call after finalize()."""
        xp = self.xp
        idx, good, iu, iv = self._idx(u, v, nw)
        iw = xp.where(good, nw / (1.0 + self.grid[idx] * self.f2), 0.0)
        if self.tukey:
            n = self.npix
            uvd = xp.hypot((iu - n // 2) * self.du, (iv - n // 2) * self.du)
            iw = iw * xp.where(uvd < self.tukey, 0.5 * (1 - xp.cos(np.pi * uvd / self.tukey)), 1.0)
        return iw.astype(xp.float32)


def briggs_weights(u, v, nw, N, scale_rad, robust=0.0, tukey_inner=None, xp=None):
    """One-shot imaging weights for a single set of samples (e.g. one snapshot)."""
    g = WeightGrid(N, scale_rad, robust, tukey_inner, xp=xp)
    g.add(u, v, nw)
    g.finalize()
    return g.weights(u, v, nw)
