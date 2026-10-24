"""GPU tests (skipped without CuPy/cuFINUFFT/CUDA): gridder vs direct DFT, degrid vs DFT
predict, and a point source recovered by the snapshot CLEAN at the right pixel and flux."""
import math

import numpy as np
import pytest

from conftest import GPU, needs_gpu

pytestmark = [pytest.mark.gpu, needs_gpu]

if GPU:
    import cupy as cp
    from ttcalx_imager import bands
    from ttcalx_imager.gridder import SampleSet, SvdWTerm, dft_predict, from_ws, to_ws
    from ttcalx_imager.snapshot import SnapshotImager


def _samples(n, umax, wmax, seed=0):
    rng = np.random.default_rng(seed)
    r = umax * np.sqrt(rng.uniform(0.01, 1, n)); t = rng.uniform(0, 2 * np.pi, n)
    return r * np.cos(t), r * np.sin(t), rng.uniform(-wmax, wmax, n)


def test_dirty_matches_direct_dft():
    N, sc = 64, 0.025                     # l up to 0.8: strong w-term
    u, v, w = _samples(1500, 12.0, 15.0)
    rng = np.random.default_rng(1)
    vis = (rng.normal(size=u.size) + 1j * rng.normal(size=u.size)).astype(np.complex64)
    wt = rng.uniform(0.5, 1.5, u.size).astype(np.float32)
    svd = SvdWTerm(N, sc, np.abs(w).max() + 1e-3, tol=1e-5)
    S = SampleSet(u, v, w, wt, N, sc, svd=svd, eps=1e-6)
    got = cp.asnumpy(S.grid(cp.asarray(vis)))
    j = (np.arange(N) - N // 2) * sc
    L, M = np.meshgrid(j, j, indexing="ij")              # raw orientation [ix(l), iy(m)]
    inside = L ** 2 + M ** 2 < 1
    n1 = np.sqrt(np.maximum(1 - L ** 2 - M ** 2, 0)) - 1
    ph = np.exp(2j * np.pi * (u[:, None, None] * L + v[:, None, None] * M - w[:, None, None] * n1))
    ref = np.where(inside, np.real(np.tensordot(vis * wt, ph, axes=1)), 0)
    assert np.abs(got - ref).max() < 1e-3 * np.abs(ref).max()     # SVD/interp tables ~1e-4


def test_degrid_matches_dft_predict():
    N, sc = 96, 0.015
    u, v, w = _samples(2000, 20.0, 10.0, seed=4)
    svd = SvdWTerm(N, sc, np.abs(w).max() + 1e-3, tol=1e-5)
    S = SampleSet(u, v, w, np.ones(u.size, np.float32), N, sc, svd=svd, eps=1e-6)
    model = cp.zeros((N, N), cp.float32)
    model[30, 70] = 2.0; model[60, 20] = -0.7; model[48, 48] = 1.0
    q1 = cp.asnumpy(S.degrid(model))
    q2 = cp.asnumpy(dft_predict(*(cp.asarray(a, dtype=cp.float32) for a in (u, v, w)), model, sc))
    assert np.abs(q1 - q2).max() < 1e-3 * np.abs(q2).max()


def test_orientation_roundtrip():
    x = cp.arange(36, dtype=cp.float32).reshape(6, 6)
    assert bool((from_ws(to_ws(x)) == x).all())


def test_point_source_snapshot_clean():
    """10 Jy point source + noise: CLEAN puts it at the right pixel with the right flux."""
    N, sdeg = 256, 0.4
    sc = math.radians(sdeg)
    rng = np.random.default_rng(7)
    ant = rng.normal(0, 60, (80, 3)); ant[:, 2] *= 0.1
    b = (ant[:, None] - ant[None])[np.triu_indices(80, 1)]
    lam = 299792458.0 / np.array([40e6, 42e6, 44e6])
    u = (b[:, 0:1] / lam).ravel(); v = (b[:, 1:2] / lam).ravel(); w = (b[:, 2:3] / lam).ravel()
    y0, x0 = N // 2 + 23, N // 2 - 17                       # WSClean orientation [y, x]
    model = cp.zeros((N, N), cp.float32); model[y0, x0] = 10.0
    vis = cp.asnumpy(dft_predict(*(cp.asarray(a, dtype=cp.float32) for a in (u, v, w)), model, sc))
    vis = (vis + 0.3 * (rng.normal(size=vis.size) + 1j * rng.normal(size=vis.size))).astype(np.complex64)
    im = SnapshotImager()
    r = im.image_arrays(u, v, w, np.ones(u.size, np.float32), vis, None, N, sdeg, bands.SNAPSHOT_CLEAN_I)
    rest = cp.asnumpy(r["restored"]); mdl = cp.asnumpy(r["model"])
    assert np.unravel_index(np.argmax(rest), rest.shape) == (y0, x0)
    assert rest[y0, x0] == pytest.approx(10.0, rel=0.03)
    assert mdl[y0 - 2:y0 + 3, x0 - 2:x0 + 3].sum() == pytest.approx(10.0, rel=0.03)
    bmaj, bmin, bpa = r["info"]["beam"]
    assert 0 < bmin <= bmaj < 20 * sdeg
