"""Zenith-frame snapshot gridding (ttcalx_imager.zenith, TTCALX_SNAP_ZEN; GPU tests, skipped without CuPy/CUDA).

A synthetic snapshot: a nearly coplanar array observed at its zenith (stored frame) and re-phased (rotate.rephase,
as the gsi v3 engine) to a phase centre ~8 deg away (the LST-hour frame late in the hour), with point sources
up to the zenith-frame horizon."""
import math

import numpy as np
import pytest

from conftest import GPU, needs_gpu

pytestmark = [pytest.mark.gpu, needs_gpu]

if GPU:
    import cupy as cp
    from ttcalx_imager import bands, gridder, rotate, snapshot, zenith
    from ttcalx_imager.gridder import SampleSet, to_ws
    from ttcalx_imager.snapshot import SnapshotImager
    from ttcalx_imager.weights import WeightGrid

C = 299792458.0
N, SDEG = 256, 0.5


def _rot(ax, deg):
    a = math.radians(deg); c, s = math.cos(a), math.sin(a)
    if ax == 0:
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _snapshot(nant=90, freqs=(46e6, 47e6, 48e6), tilt=8.0, seed=3):
    """Rows (a1 < a2 plus autos), stored uvw [m] (zenith frame, nearly coplanar), hour uvw = R uvw_stored, and
    DATA: point sources (stored-frame l, m) predicted exactly in the stored frame, re-phased to the hour frame."""
    rng = np.random.default_rng(seed)
    ant = rng.normal(0, 45, (nant, 3)); ant[:, 2] *= 0.02     # max baseline ~45 lambda: ~2.5 px beam
    a1, a2 = (np.array(x, np.int32) for x in zip(*[(i, j) for i in range(nant) for j in range(i, nant)]))
    uvw_s = ant[a1] - ant[a2]
    R = _rot(0, tilt) @ _rot(1, 0.6 * tilt)
    uvw_h = uvw_s @ R.T
    # sources: (l, m) in the stored frame, flux; one 2 deg and one 5 deg above the zenith-frame horizon
    src = [(0.1, 0.2, 10.0), (-0.45, 0.3, 4.0)]
    for el, az in ((2.0, 200.0), (5.0, 20.0)):
        r = math.cos(math.radians(el))
        src.append((r * math.sin(math.radians(az)), r * math.cos(math.radians(az)), 6.0))
    f = np.asarray(freqs)
    lam = C / f
    us, vs, ws = (uvw_s[:, i:i + 1] / lam[None] for i in range(3))
    vis = np.zeros(us.shape, np.complex128)
    for l, m, s in src:
        n = math.sqrt(1 - l * l - m * m)
        vis += s * np.exp(-2j * np.pi * (us * l + vs * m + ws * (n - 1)))
    vis += 0.05 * (rng.normal(size=vis.shape) + 1j * rng.normal(size=vis.shape))
    data = np.zeros(vis.shape + (4,), np.complex64)
    data[:, :, 0] = vis; data[:, :, 3] = vis
    data[:, :, 1] = 0.02 * vis; data[:, :, 2] = -0.02 * vis          # some V
    data_h = rotate.rephase(data, uvw_s[:, 2], uvw_h[:, 2], f, xp=np)
    flag = np.zeros(data.shape, bool)
    flag[rng.random(len(a1)) < 0.05] = True
    small = dict(a1=a1, a2=a2, uvw=uvw_h, uvw_stored=uvw_s, wcol=np.ones((len(a1), 4), np.float32), time=5.3e9,
                 tstart=5.3e9, tmid=5.3e9, freqs=f, chan_width=np.full(f.size, 1e6), phase_dir=(1.0, 0.65),
                 telescope="OVRO-LWA", observer="", field_name="", path="x.ms", stem="x", column="DATA")
    return small, {"DATA": data_h, "FLAG": flag}, R, src


@pytest.fixture
def zen_env(monkeypatch):
    monkeypatch.setenv("TTCALX_SNAP_ZEN", "force")
    for k in ("DK", "GZ", "PSFGZ", "RING_DEG", "RING_PX", "AK_LOSS"):
        monkeypatch.delenv("TTCALX_SNAP_ZEN_" + k, raising=False)
    yield monkeypatch
    gridder.clear_caches()


def _weights(L, sc):
    p = bands.SNAPSHOT_CLEAN_I
    WG = WeightGrid(N, sc, p.robust, p.taper_inner_tukey)
    WG.add(L["u"], L["v"], L["wnat"]); WG.finalize()
    iw = WG.weights(L["u"], L["v"], L["wnat"])
    return iw, float(iw.sum())


def test_mode_parsing(monkeypatch):
    for v, m in (("", "off"), ("0", "off"), ("off", "off"), ("1", "auto"), ("auto", "auto"), ("force", "force")):
        monkeypatch.setenv("TTCALX_SNAP_ZEN", v)
        assert zenith.mode() == m
    monkeypatch.delenv("TTCALX_SNAP_ZEN")
    assert zenith.mode() == "off"


def test_extended_factor_table():
    """_svd_factors on x' in [xmin, 1/2] reproduces exp(-2 pi i w x') to the tolerance; the default (xmin = -1/2)
    key and factors are the production ones."""
    wq, tol = 6.25, 1e-4
    a = gridder._svd_factors(wq, tol, 600, 2001)
    assert gridder._svd_factors(wq, tol, 600, 2001, xmin=-0.5) is a
    nx = zenith._zen_nx()
    K, wg, A, Vh = gridder._svd_factors(wq, tol, 600, nx, xmin=-0.65)
    xg = np.linspace(-0.65, 0.5, nx)
    M = np.exp(-2j * np.pi * np.outer(wg, xg))
    assert np.abs(M - A @ Vh).max() < tol and K >= a[0]


def test_gpu_select_zen_samples(zen_env):
    """gpu_select: the zenith-frame samples (stored uvw / lambda, same selection) and R; none with the switch off."""
    small, raw, R, _ = _snapshot()
    sc = math.radians(SDEG)
    L = snapshot.gpu_select(small, raw, SDEG)
    z = L["zen"]
    assert np.allclose(z["R"], R, atol=1e-9)
    # the hour-frame samples are R times the stored ones
    uh = cp.asnumpy(L["u"]); wz = cp.asnumpy(z["w"]); uz, vz = cp.asnumpy(z["u"]), cp.asnumpy(z["v"])
    assert np.allclose(uh, R[0, 0] * uz + R[0, 1] * vz + R[0, 2] * wz, atol=1e-6)
    zen_env.setenv("TTCALX_SNAP_ZEN", "0")
    L0 = snapshot.gpu_select(small, raw, SDEG)
    assert "zen" not in L0
    for k in ("u", "v", "w", "wnat", "visI", "visV"):
        assert bool((L0[k] == L[k]).all())
    del sc


def test_zen_image_matches_hour_frame_gridding(zen_env):
    """ZenSnapshot.image = production to_ws(S.grid(vis)) / sw (hour frame, K_h layers) to the reprojection's
    accuracy, everywhere inside the hour horizon incl. the horizon ring; with the a_k-loss emulation on it is
    closer to production than without (production's linear table interpolation)."""
    small, raw, R, _ = _snapshot()
    sc = math.radians(SDEG)
    L = snapshot.gpu_select(small, raw, SDEG)
    iw, sw = _weights(L, sc)
    pre = L["pre"]
    svd = snapshot.snapshot_svd(N, sc, pre["wabsmax"], 1e-4)
    S = SampleSet(None, None, None, iw, N, sc, svd=svd, pre=pre)
    ref = to_ws(S.grid(L["visI"])) / sw
    Z = zenith.ZenSnapshot(L["zen"], L["w"], pre, iw, sw, N, sc)
    assert Z.K < svd.K and int(Z.grids["main"].ridx.size) > 0
    got = Z.image(L["visI"])
    peak = float(cp.abs(ref).max())
    err = float(cp.abs(got - ref).max()) / peak
    ring = cp.zeros(N * N, bool); ring[Z.grids["main"].ridx] = True
    ring = ring.reshape(N, N)
    err_ring = float(cp.abs(got - ref)[ring].max()) / peak
    rms = float(cp.sqrt(cp.mean((got - ref) ** 2))) / peak
    assert err < 5e-3 and err_ring < 2e-3 and rms < 2e-4, (err, err_ring, rms)
    assert bool((got[~to_ws(svd.inside)] == 0).all())
    # the a_k-loss emulation multiplies the hour image by production's per-pixel gain (<= 1, ~1e-3 here)
    zen_env.setenv("TTCALX_SNAP_ZEN_AK_LOSS", "0")
    exact = zenith.ZenSnapshot(L["zen"], L["w"], pre, iw, sw, N, sc).image(L["visI"])
    g = Z.gain
    assert float(g.max()) <= 1.0 and float(g[to_ws(svd.inside)].min()) > 0.99
    assert float(cp.abs(got - exact * g).max()) < 1e-6 * peak


def test_ring_layer_equals_dense_prefilter(zen_env):
    """The ring's sparse prefilter + spline (only the pixels the ring needs) = the dense FIR prefilter of the whole
    layer + the same B-spline weights, bitwise (the same float operations)."""
    small, raw, R, _ = _snapshot()
    sc = math.radians(SDEG)
    L = snapshot.gpu_select(small, raw, SDEG)
    iw, sw = _weights(L, sc)
    Z = zenith.ZenSnapshot(L["zen"], L["w"], L["pre"], iw, sw, N, sc)
    G = Z.grids["main"]; Nz = G.Nz; n = int(G.ridx.size)
    rng = np.random.default_rng(0)
    f = cp.asarray(rng.normal(size=(Nz, Nz)) + 1j * rng.normal(size=(Nz, Nz)), cp.complex64)
    racc = cp.zeros(n, cp.float32)
    bufs = (cp.empty((Nz, Nz), cp.complex64), cp.empty((Nz, Nz), cp.complex64))
    G.ring_layer(f, 0, racc, bufs)
    # dense: prefilter real and imaginary parts, spline at the targets, times b_0
    pr, pi = zenith.prefilter(f.real.copy()), zenith.prefilter(f.imag.copy())
    i0 = cp.asnumpy(G.i0).reshape(-1, 2); txy = cp.asnumpy(G.txy).reshape(-1, 2)
    b5 = np.vectorize(lambda x: _b5(np.float32(x)))
    pr, pi = cp.asnumpy(pr), cp.asnumpy(pi)
    br, bi = cp.asnumpy(G.table.Br[0]), cp.asnumpy(G.table.Bi[0]); jb = cp.asnumpy(G.jb); tb = cp.asnumpy(G.tb)
    got = cp.asnumpy(racc)
    for r in rng.choice(n, min(n, 200), replace=False):
        wx = b5(txy[r, 0] - np.arange(-2, 4)); wy = b5(txy[r, 1] - np.arange(-2, 4))
        ix, iy = i0[r]
        re = wx @ pr[ix - 2:ix + 4, iy - 2:iy + 4] @ wy; im = wx @ pi[ix - 2:ix + 4, iy - 2:iy + 4] @ wy
        j, t = jb[r], tb[r]
        bre = br[j] + t * (br[j + 1] - br[j]); bim = bi[j] + t * (bi[j + 1] - bi[j])
        assert abs(got[r] - (re * bre - im * bim)) <= 1e-5 * (abs(re) + abs(im) + 1e-12)


def _b5(x):
    x = abs(x)
    if x < 1:
        return (66 - 60 * x * x + 30 * x ** 4 - 10 * x ** 5) / 120
    if x < 2:
        return (51 + 75 * x - 210 * x * x + 150 * x ** 3 - 45 * x ** 4 + 5 * x ** 5) / 120
    if x < 3:
        return (3 - x) ** 5 / 120
    return 0.0


def test_image_arrays_zen_path(zen_env):
    """SnapshotImager.image_arrays with the zenith path (production weights, CLEAN, prediction, restore): the same
    components / beam / restored image as the production path to the gridding accuracy; the router keeps the
    production path when the hour frame is not more expensive (TTCALX_SNAP_ZEN=1, large margin) and the switch
    off ignores the zenith samples."""
    small, raw, R, src = _snapshot()
    im = SnapshotImager()
    L = snapshot.gpu_select(small, raw, SDEG)
    args = (L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"], N, SDEG)
    ref = im.image_arrays(*args, pre=L["pre"])
    z = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" in z["info"] and "zen" not in ref["info"] and z["info"]["K"] == ref["info"]["K"]
    peak = float(ref["restored"].max())
    assert float(cp.abs(z["restored"] - ref["restored"]).max()) < 3e-3 * peak
    assert float(cp.abs(z["V"] - ref["V"]).max()) < 1e-2 * float(cp.abs(ref["V"]).max())      # noise-like V
    for a, b in zip(z["info"]["beam"][:2], ref["info"]["beam"][:2]):
        assert a == pytest.approx(b, rel=1e-3)
    assert abs(z["info"]["ncomp"] - ref["info"]["ncomp"]) <= max(2, 0.02 * ref["info"]["ncomp"])
    # router: margin larger than any rank difference -> the production path (equal to the GPU rerun floor: the
    # cuFINUFFT spreading uses atomics, so two production runs differ at ~1e-7 of the peak)
    def same(x, y):
        return float(cp.abs(x["restored"] - y["restored"]).max()) < 1e-5 * peak and \
            float(cp.abs(x["V"] - y["V"]).max()) < 1e-5 * float(cp.abs(y["V"]).max())
    zen_env.setenv("TTCALX_SNAP_ZEN", "1"); zen_env.setenv("TTCALX_SNAP_ZEN_DK", "1000")
    a = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" not in a["info"] and same(a, ref)
    zen_env.setenv("TTCALX_SNAP_ZEN_DK", "0")
    assert "zen" in im.image_arrays(*args, pre=L["pre"], zen=L["zen"])["info"]
    zen_env.setenv("TTCALX_SNAP_ZEN", "0")
    b = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" not in b["info"] and same(b, ref)


def test_degrid_switch_uses_the_hour_frame_sampleset(zen_env):
    """Above dft_max components the zenith path predicts with the production hour-frame SampleSet.degrid (built on
    demand, the production table), as the production path."""
    small, raw, R, _ = _snapshot()
    im = SnapshotImager(dft_max=0)                    # every major cycle degrids
    L = snapshot.gpu_select(small, raw, SDEG)
    args = (L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"], N, SDEG)
    calls = []
    orig = SampleSet.degrid

    def spy(self, model):
        calls.append(self.svd.K)
        return orig(self, model)
    zen_env.setattr(SampleSet, "degrid", spy)
    ref = im.image_arrays(*args, pre=L["pre"])
    n_ref = len(calls)
    z = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert n_ref == 2 and len(calls) == 4 and calls[2] == calls[0]
    assert float(cp.abs(z["restored"] - ref["restored"]).max()) < 3e-3 * float(ref["restored"].max())


def test_pinned_h2d_same_bytes(monkeypatch):
    """TTCALX_PINNED_H2D: host arrays reach the GPU through the page-locked staging buffer unchanged (the buffer
    grows for a larger request and is reused); CuPy inputs pass through; off = cp.asarray."""
    rng = np.random.default_rng(1)
    a = (rng.normal(size=(500, 7, 4)) + 1j * rng.normal(size=(500, 7, 4))).astype(np.complex64)
    f = rng.random((500, 7, 4)) < 0.3
    g = cp.arange(5)
    for on in ("0", "1"):
        monkeypatch.setenv("TTCALX_PINNED_H2D", on)
        da, df, dg = snapshot.h2d(a, f, g)
        assert dg is g and np.array_equal(cp.asnumpy(da), a) and np.array_equal(cp.asnumpy(df), f)
        assert da.dtype == a.dtype and df.dtype == f.dtype and da.shape == a.shape
    n0 = snapshot._PIN["nbytes"]
    big = rng.normal(size=(1000, 9, 4)).astype(np.float32)
    (db,) = snapshot.h2d(big)
    assert np.array_equal(cp.asnumpy(db), big) and snapshot._PIN["nbytes"] >= max(n0, big.nbytes)
    da2, df2, _ = snapshot.h2d(a, f, g)
    assert np.array_equal(cp.asnumpy(da2), a) and np.array_equal(cp.asnumpy(df2), f) and np.array_equal(cp.asnumpy(db), big)
