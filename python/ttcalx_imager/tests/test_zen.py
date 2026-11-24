"""Zenith-frame snapshot gridding (ttcalx_imager.zenith, TTCALX_SNAP_ZEN; GPU tests, skipped without CuPy/CUDA).

A synthetic snapshot: a nearly coplanar array observed at its zenith (stored frame) and re-phased (rotate.rephase,
as the gsi v3 engine) to a phase centre ~8 deg away (the LST-hour frame late in the hour), with point sources
up to the zenith-frame horizon."""
import math
import os

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


def _snapshot(nant=90, freqs=(46e6, 47e6, 48e6), tilt=6.0, seed=3):
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


@pytest.fixture(params=["default", "fast"])
def zen_env(request, monkeypatch):
    """TTCALX_SNAP_ZEN=force on the default operations and on FAST_ENV (the production gsi setting)."""
    from ttcalx_imager import FAST_ENV
    for k, v in FAST_ENV.items():
        if request.param == "fast":
            monkeypatch.setenv(k, v)
        else:
            monkeypatch.delenv(k, raising=False)
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
    zi = z["info"]["zen_router"]                                      # the router's decision (force: routed)
    assert zi["routed"] is True and zi["used"] is True and zi["K_z"] == z["info"]["zen"]["K"]
    assert zi["K_h"] == zenith.hour_rank(L["pre"]["wabsmax"], im.svd_tol)
    assert "zen_router" not in ref["info"]                             # no zenith samples: the router never ran
    zen_env.setenv("TTCALX_SNAP_ZEN", "1"); zen_env.setenv("TTCALX_SNAP_ZEN_DK", "1000")
    a = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" not in a["info"] and same(a, ref)
    assert a["info"]["zen_router"] == dict(zi, routed=False, used=False)
    zen_env.setenv("TTCALX_SNAP_ZEN_DK", "0")
    assert "zen" in im.image_arrays(*args, pre=L["pre"], zen=L["zen"])["info"]
    zen_env.setenv("TTCALX_SNAP_ZEN", "0")
    b = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" not in b["info"] and same(b, ref) and "zen_router" not in b["info"]


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


@pytest.mark.parametrize("where", ["construct", "dirty", "psf", "cufinufft"])
def test_zen_out_of_memory_falls_back_to_production(zen_env, where):
    """An out-of-GPU-memory error in ZenSnapshot or in its first passes (dirty I, PSF; also cuFINUFFT's plan / setpts
    allocation errors): the zenith caches are freed and the snapshot is imaged on the production path (the same
    products as without the zenith samples, to the GPU rerun floor); any other error is raised."""
    small, raw, R, _ = _snapshot()
    im = SnapshotImager()
    L = snapshot.gpu_select(small, raw, SDEG)
    args = (L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"], N, SDEG)
    ref = im.image_arrays(*args, pre=L["pre"])
    oom = (RuntimeError("Error setting non-uniform points: 2") if where == "cufinufft" else
           cp.cuda.memory.OutOfMemoryError(1 << 34, 1 << 34))
    real_init, real_image = zenith.ZenSnapshot.__init__, zenith.ZenSnapshot.image

    def init(self, *a, **k):
        real_init(self, *a, **k)
        if where in ("construct", "cufinufft"):
            raise oom

    def image(self, vis, grid="main", norm=True):
        if (where == "dirty" and grid == "main") or (where == "psf" and grid == "psf"):
            raise oom
        return real_image(self, vis, grid, norm)
    zen_env.setattr(zenith.ZenSnapshot, "__init__", init)
    zen_env.setattr(zenith.ZenSnapshot, "image", image)
    z = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" not in z["info"] and "zen_oom" in z["info"] and z["info"]["K"] == ref["info"]["K"]
    assert not zenith._TABLES and not zenith._PLANS and "w" not in L["zen"]._c
    peak = float(ref["restored"].max())
    assert float(cp.abs(z["restored"] - ref["restored"]).max()) < 1e-5 * peak     # the GPU rerun floor
    # not an out-of-memory error: raised
    oom = ValueError("a bug")
    with pytest.raises(ValueError):
        im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert zenith.is_oom(RuntimeError("Error creating plan: 3")) and zenith.is_oom(RuntimeError("cudaErrorMemoryAllocation: out of memory"))
    assert not zenith.is_oom(RuntimeError("Error executing plan"))


def test_hour_set_degrid_not_precomputed(zen_env):
    """_HourSet builds its hour-frame SampleSet without the precomputed a_k arrays (K_h x npts, while the zenith
    buffers are alive); its degrid is bitwise that of the precomputed production SampleSet."""
    small, raw, R, _ = _snapshot()
    sc = math.radians(SDEG)
    L = snapshot.gpu_select(small, raw, SDEG)
    iw, sw = _weights(L, sc)
    pre = L["pre"]
    hs = snapshot._HourSet(N, sc, pre, iw, 1e-4, 1e-4)
    rng = np.random.default_rng(5)
    model = cp.zeros((N, N), cp.float32)
    model[rng.integers(60, 200, 40), rng.integers(60, 200, 40)] = cp.asarray(rng.uniform(0.1, 3.0, 40), cp.float32)
    got = hs.degrid(model)
    assert hs._S.a is None
    svd = snapshot.snapshot_svd(N, sc, pre["wabsmax"], 1e-4)
    ref = SampleSet(None, None, None, iw, N, sc, svd=svd, pre=pre).degrid(model)
    assert ref.dtype == got.dtype and bool((ref == got).all())


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


def test_far_from_zenith_falls_back_and_quantum_keeps_production(zen_env):
    """A snapshot imaged > asin(NLOW) from its zenith raises ZenFar in ZenSnapshot (b_k would be extrapolated) and
    image_arrays images it on the production path; with the engine's GSI_SVD_QUANTUM table override the router keeps
    the production path (the loss emulation assumes production's table key)."""
    small, raw, R, _ = _snapshot(tilt=8.0)
    sc = math.radians(SDEG)
    L = snapshot.gpu_select(small, raw, SDEG)
    iw, sw = _weights(L, sc)
    with pytest.raises(zenith.ZenFar) as ei:
        zenith.ZenSnapshot(L["zen"], L["w"], L["pre"], iw, sw, N, sc)
    assert ei.value.n_min < -zenith.NLOW
    im = SnapshotImager()
    args = (L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"], N, SDEG)
    ref = im.image_arrays(*args, pre=L["pre"])
    z = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])            # TTCALX_SNAP_ZEN=force
    assert "zen" not in z["info"] and z["info"]["zen_far_n_min"] == ei.value.n_min
    assert z["info"]["zen_router"]["routed"] is True and z["info"]["zen_router"]["used"] is False
    assert z["info"]["K"] == ref["info"]["K"]
    peak = float(ref["restored"].max())
    assert float(cp.abs(z["restored"] - ref["restored"]).max()) < 1e-5 * peak     # the GPU rerun floor
    # closer to its zenith: the zenith path
    small6, raw6, _, _ = _snapshot(tilt=6.0)
    L6 = snapshot.gpu_select(small6, raw6, SDEG)
    a6 = (L6["u"], L6["v"], L6["w"], L6["wnat"], L6["visI"], L6["visV"], N, SDEG)
    z6 = im.image_arrays(*a6, pre=L6["pre"], zen=L6["zen"])
    assert "zen" in z6["info"] and "zen_far_n_min" not in z6["info"]
    zen_env.setenv("GSI_SVD_QUANTUM", "1.0")
    assert zenith.use_zen(5.0, 1.0, 1e-4)[0] is False
    zen_env.setenv("GSI_SVD_QUANTUM", "0")
    assert zenith.use_zen(5.0, 1.0, 1e-4)[0] is True


def test_bad_rotation_or_shape_keeps_production(zen_env):
    """uvw that is not R uvw_stored (R fit residual > R_TOL max |uvw|) raises ZenBadR in ZenSnapshot and image_arrays
    images the snapshot on the production path; a uvw_stored of another shape gives no zenith samples (warning)."""
    small, raw, R, _ = _snapshot()
    rng = np.random.default_rng(9)
    bad = dict(small); bad["uvw"] = small["uvw"] + rng.normal(scale=1e-3, size=small["uvw"].shape)   # ~2e-5 relative
    im = SnapshotImager()
    L = snapshot.gpu_select(bad, raw, SDEG)
    args = (L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"], N, SDEG)
    ref = im.image_arrays(*args, pre=L["pre"])
    z = im.image_arrays(*args, pre=L["pre"], zen=L["zen"])
    assert "zen" not in z["info"] and z["info"]["zen_bad_R_rel"] > zenith.R_TOL
    assert float(cp.abs(z["restored"] - ref["restored"]).max()) < 1e-5 * float(ref["restored"].max())
    ok = snapshot.gpu_select(small, raw, SDEG)["zen"]
    ok["R"]
    assert ok.R_res <= zenith.R_TOL * ok.R_scale
    short = dict(small); short["uvw_stored"] = small["uvw_stored"][:-1]
    with pytest.warns(UserWarning, match="uvw_stored"):
        assert "zen" not in snapshot.gpu_select(short, raw, SDEG)


def test_zen_samples_lazy(zen_env):
    """ZenSamples: the router's max |w| equals max |zen["w"]| exactly (no per-sample array built for it); w, u, v are
    built on first use; the production pick of the router builds none of them."""
    small, raw, R, _ = _snapshot()
    # a few rows with only some channels selected, so the per-row smallest selected lambda matters
    raw = dict(raw); fl = raw["FLAG"].copy(); fl[::7, 2, :] = True; fl[::11, 0, :] = True; raw["FLAG"] = fl
    L = snapshot.gpu_select(small, raw, SDEG)
    z = L["zen"]
    assert z._c == {}
    lam = C / small["freqs"]
    full = (np.asarray(small["uvw_stored"])[:, 2:3] / lam[None])[cp.asnumpy(z._sel)]
    assert z.wabs == float(np.abs(full).max())
    assert z.wabs == float(cp.abs(z["w"]).max()) and np.array_equal(cp.asnumpy(z.w), full)
    zen_env.setenv("TTCALX_SNAP_ZEN", "1"); zen_env.setenv("TTCALX_SNAP_ZEN_DK", "1000")      # router -> production
    L2 = snapshot.gpu_select(small, raw, SDEG)
    a = SnapshotImager().image_arrays(L2["u"], L2["v"], L2["w"], L2["wnat"], L2["visI"], L2["visV"], N, SDEG,
                                      pre=L2["pre"], zen=L2["zen"])
    assert "zen" not in a["info"] and L2["zen"]._c == {}


def test_zen_snapshot_drops_stored_samples(zen_env):
    """ZenSnapshot frees the float64 stored-frame u, v, w of its ZenSamples once it has used them (they are not
    needed through CLEAN; R and the router's wabs stay); a later access rebuilds them, the same values."""
    small, raw, R, _ = _snapshot()
    sc = math.radians(SDEG)
    L = snapshot.gpu_select(small, raw, SDEG)
    iw, sw = _weights(L, sc)
    z = L["zen"]
    w0 = cp.asnumpy(z["w"])
    Z = zenith.ZenSnapshot(z, L["w"], L["pre"], iw, sw, N, sc)
    assert not {"u", "v", "w"} & set(z._c) and "R" in z._c
    assert np.array_equal(cp.asnumpy(z["w"]), w0) and z.wabs == float(np.abs(w0).max())
    assert float(cp.abs(Z.image(L["visI"])).max()) > 0


def test_clear_caches_clears_zenith(zen_env):
    """gridder.clear_caches (the gsi engine's before_hour / after_hour / band change) also drops the zenith tables and
    plans."""
    small, raw, R, _ = _snapshot()
    L = snapshot.gpu_select(small, raw, SDEG)
    SnapshotImager().image_arrays(L["u"], L["v"], L["w"], L["wnat"], L["visI"], L["visV"], N, SDEG, pre=L["pre"],
                                  zen=L["zen"])
    assert zenith._TABLES and zenith._PLANS
    gridder.clear_caches()
    assert not zenith._TABLES and not zenith._PLANS and not zenith._GEOM and not zenith._GAIN


def test_router_rank_prefetched_as_production(zen_env, monkeypatch):
    """zenith.hour_rank (the router's K_h) with TTCALX_SVD_PREFETCH=1: the neighbouring hour-table keys are
    factorised in the prefetch thread as production's snapshot_svd does, so a snapshot whose w range moves to the
    next key finds its factors cached (no synchronous SVD on the zenith path: the 2026-11-24 e2e alternation), and a
    key still being prefetched is waited for, not computed twice.  The same K as without the prefetch."""
    import threading
    monkeypatch.setenv("TTCALX_SVD_PREFETCH", "1")
    tol = 1e-4
    w0 = 7.3                                       # key 7.5; neighbours 7.25 / 7.75, two steps 7.0 / 8.0
    keys = [zenith.table_wq(w) for w in (w0, w0 - 0.25, w0 + 0.25, w0 - 0.5, w0 + 0.5)]
    for k in [k for k in gridder._FACTOR_CACHE if k[0] in keys and k[2:4] == (600, 2001)]:
        del gridder._FACTOR_CACHE[k]
    for k in [k for k in snapshot._PREFETCH if k != "ex" and k[0] in keys]:
        del snapshot._PREFETCH[k]
    main = threading.get_ident()
    computed = []
    orig = gridder._svd_factors

    def spy(wmax, tol_, nw, nx, *a, **kw):
        method = kw.get("method") or gridder.SVD_METHOD or os.environ.get("TTCALX_SVD_METHOD", "lapack")
        if (float(wmax), float(tol_), int(nw), int(nx), method) not in gridder._FACTOR_CACHE:
            computed.append((float(wmax), threading.get_ident() == main))
        return orig(wmax, tol_, nw, nx, *a, **kw)
    monkeypatch.setattr(gridder, "_svd_factors", spy)
    k0 = zenith.hour_rank(w0, tol)
    assert (keys[0], True) in computed                       # the first key: computed at once
    for w in keys[1:]:
        snapshot._PREFETCH[(float(w), float(tol), 600, 2001)].result()
    computed.clear()
    k1 = zenith.hour_rank(w0 - 0.25, tol)                    # the next snapshot's key: prefetched
    assert not [c for c in computed if c[1]], computed       # nothing factorised in the calling thread
    computed.clear()
    zenith.hour_rank(w0 + 0.5, tol)                          # a jump of two keys (73-82 MHz): prefetched too
    assert not [c for c in computed if c[1]], computed
    assert k0 == orig(keys[0], tol, 600, 2001)[0] and k1 == orig(keys[1], tol, 600, 2001)[0]
    # without the prefetch: the same ranks
    monkeypatch.setenv("TTCALX_SVD_PREFETCH", "0")
    assert zenith.hour_rank(w0, tol) == k0 and zenith.hour_rank(w0 - 0.25, tol) == k1

