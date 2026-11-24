"""Hour-stage v5 speed switches (ttcalx_imager.HOUR_V5_ENV, each default off): every switch against its off path
(bitwise where the operations are the same, else to the documented float32 tolerance), the memory bounds, and
that with the switches off nothing changes (no memo, no summary key)."""
import math

import numpy as np
import pytest

from ttcalx_imager import FAST_ENV, HOUR_V5_ENV, bands, gsiv, use_hour_v5_settings

from conftest import GPU, needs_gpu
from test_gsiv import _hour_snaps, same

if GPU:
    import cupy as cp


def _clear_v5(monkeypatch):
    for k in HOUR_V5_ENV:
        monkeypatch.delenv(k, raising=False)


# ----------------------------------------------------------------------------- settings (CPU)
def test_hour_v5_settings(monkeypatch):
    env = {"TTCALX_MS_INCR": "8"}
    got = use_hour_v5_settings(env)
    assert set(got) == set(HOUR_V5_ENV) and got["TTCALX_MS_INCR"] == "8"          # an explicit value wins
    assert all(env[k] == v for k, v in HOUR_V5_ENV.items() if k != "TTCALX_MS_INCR")
    assert all(env[k] == v for k, v in FAST_ENV.items())                           # on top of FAST_ENV
    assert not set(HOUR_V5_ENV) & set(FAST_ENV)


def test_hour_v5_off_and_wmax_memo_same_gridder_inputs(tmp_path, monkeypatch):
    """CPU hour run: TTCALX_WMAX_MEMO=1 gives the gridder the same inputs and get_svd the same wmax as the default
    (also with TTCALX_HOUR_SETUP_FAST=0, where the main grid uses the memo too); the memo is dropped after the
    call; with every switch off the summary has no hour_v5 key, with one on it records it."""
    hour_cpu = pytest.importorskip("hour_cpu")
    prods = bands.hour_products(["I-Deep-Taper-Robust-0", "V-Taper-Deep", "I-Taper-10min",
                                 "I-Deep-NoTaper-Robust-0.75"])
    dicts, p = _hour_snaps(tmp_path)
    _clear_v5(monkeypatch)
    for fast in ("0", "1"):
        monkeypatch.setenv("TTCALX_HOUR_SETUP_FAST", fast)
        runs = []
        for v in ("0", "1"):
            monkeypatch.setenv("TTCALX_WMAX_MEMO", v)
            svd_args = []
            with hour_cpu.cpu_hour("store", 1000, 64) as rec:
                hour = hour_cpu.import_hour()
                hour.get_svd = lambda N, sc, wmax, tol=1e-4, **k: svd_args.append((N, sc, wmax, tol, k)) or \
                    type("S", (), dict(K=1))()
                s = hour.image_hour_snaps(gsiv.load_archive(p, 1.3, 0.65), "41MHz", str(tmp_path / "x"), 1.3, 0.65,
                                          products=prods, mode="vis", qa="none", exclude_stems=(dicts[3]["stem"],))
                assert hour._WMAX_MEMO == {}
            runs.append((s, rec, svd_args))
        (s0, r0, a0), (s1, r1, a1) = runs
        assert "hour_v5" not in s0 and s1["hour_v5"] == {"TTCALX_WMAX_MEMO": "1"}
        s1 = dict(s1); s1.pop("hour_v5")
        assert hour_cpu.comparable(s0) == hour_cpu.comparable(s1)
        assert r1.calls == r0.calls and len(r0.images) > 0
        for k in r0.images:
            assert same(r1.images[k], r0.images[k]), k
        assert [(N, sc, np.float64(w).tobytes(), t, k) for N, sc, w, t, k in a1] == \
            [(N, sc, np.float64(w).tobytes(), t, k) for N, sc, w, t, k in a0] and a0
        assert all(k == dict(quantum=2.0) for *_, k in a0)                 # the default quantum


def test_svd_quantum_env(monkeypatch):
    from hour_cpu import import_hour
    hour = import_hour()
    monkeypatch.delenv("TTCALX_SVD_QUANTUM", raising=False)
    assert hour.svd_quantum() == 2.0
    monkeypatch.setenv("TTCALX_SVD_QUANTUM", "0.25")
    assert hour.svd_quantum() == 0.25


def test_uvmax_prefill_cpu(tmp_path):
    """prefill_uvmax sets the LazySnap memo with the uvmax expression (numpy standing in for CuPy here) and
    leaves snapshot dicts (no memo) alone."""
    from hour_cpu import import_hour
    hour = import_hour()
    dicts, p = _hour_snaps(tmp_path, n=3)
    a, b = gsiv.load_archive(p, 1.3, 0.65), gsiv.load_archive(p, 1.3, 0.65)
    ref = [s.uvmax() for s in a]
    assert hour.prefill_uvmax(b) == 3 and hour.prefill_uvmax(b) == 0
    assert [np.float64(x).tobytes() for x in ref] == [np.float64(s._memo["uvmax"]).tobytes() for s in b]
    assert hour.prefill_uvmax([dict(d) for d in dicts]) == 0


# ----------------------------------------------------------------------------- GPU
@needs_gpu
def test_uvmax_gpu_bitwise(monkeypatch):
    """TTCALX_UVMAX_GPU: the GPU expansion gives LazySnap.uvmax's float32 value bit for bit."""
    from ttcalx_imager import hour
    rng = np.random.default_rng(4)
    for nrow, nchan in ((500, 48), (30000, 192)):
        ok = rng.uniform(size=(nrow, nchan)) < 0.7
        meta = dict(path="x", stem="s", time=1.0, tstart=1.0, nrow=nrow, nchan=nchan, flagfrac=0.3, zero_frac=0.0,
                    telescope="OVRO_LWA", observer="", field_name="", pd=(0.0, 0.0), n=int(ok.sum()),
                    ok_shape=[nrow, nchan])
        arrays = dict(uvw=rng.normal(scale=900, size=(nrow, 3)), freqs=np.linspace(41e6, 64e6, nchan),
                      chan_width=np.full(nchan, 2.4e4), natw=rng.uniform(size=nrow).astype(np.float32),
                      ok_bits=gsiv.pack_mask(ok))
        for fe in ("0", "1"):
            monkeypatch.setenv("TTCALX_FAST_EXPAND", fe)
            a = gsiv.LazySnap(meta, arrays, np.eye(3), None)
            b = gsiv.LazySnap(meta, arrays, np.eye(3), None)
            assert hour.prefill_uvmax([b]) == 1
            assert np.float64(a.uvmax()).tobytes() == np.float64(b.uvmax()).tobytes()


def _band_limited(N, rng, edge):
    """Smooth random image, zero within `edge` px of the border (as the hour block / model images beyond the
    horizon)."""
    x = rng.normal(size=(N, N)).astype(np.float32)
    F = np.fft.rfft2(x); f0 = np.fft.fftfreq(N)[:, None]; f1 = np.fft.rfftfreq(N)[None, :]
    F *= np.exp(-(f0 ** 2 + f1 ** 2) / (2 * 0.12 ** 2))
    img = np.fft.irfft2(F, s=(N, N)).astype(np.float32)
    m = np.zeros((N, N), bool); m[edge:N - edge, edge:N - edge] = True
    return cp.asarray(np.where(m, img, 0).astype(np.float32))


@needs_gpu
@pytest.mark.parametrize("order", [3, 5])
def test_reproj_fast_equals_map_coordinates(monkeypatch, order):
    """TTCALX_REPROJ_FAST: FFT prefilter + float32 kernel = cupyx spline_filter + map_coordinates to float32
    rounding for images that are zero near their edge, including the 2x upsampled (up=2, prefiltered) model
    path of WSnap.image; points outside the source grid or not ok are 0 in both."""
    from ttcalx_imager.reproject import Reprojector, prefilter
    from ttcalx_imager.rotate import analytic_rotation
    rng = np.random.default_rng(order)
    N, sc = 512, math.radians(0.22)
    P = Reprojector(N, sc)
    img = _band_limited(N, rng, 48)
    big = _band_limited(2 * N, rng, 96)
    for d0, d1 in ((0.003, 0.002), (0.05, -0.03), (-0.2, 0.1)):
        R = analytic_rotation(1.0, 0.65, 1.0 + d0, 0.65 + d1)
        co = P.coords(R)
        out = {}
        for v in ("0", "1"):
            monkeypatch.setenv("TTCALX_REPROJ_FAST", v)
            a, ok = P(img, R, order)
            b, _ = P(prefilter(big, order), None, order, coords=co, up=2, prefiltered=True)
            out[v] = (a, b, ok)
        (a0, b0, ok0), (a1, b1, ok1) = out["0"], out["1"]
        assert a1.dtype == a0.dtype == cp.float32 and bool((ok0 == ok1).all())
        for x0, x1 in ((a0, a1), (b0, b1)):
            assert float(cp.abs(x1 - x0).max()) <= 2e-5 * float(cp.abs(x0).max())
            assert bool((x1[~ok0] == 0).all())


@needs_gpu
def test_prefilter_fft_inverts_bspline(monkeypatch):
    """prefilter_fft: sampling the B-spline expansion of the prefiltered image at the integer pixels gives the
    image back (interior; float32 rounding)."""
    from ttcalx_imager.reproject import interp_prefiltered, prefilter_fft
    rng = np.random.default_rng(2)
    N = 300
    img = _band_limited(N, rng, 30)
    yy, xx = cp.mgrid[:N, :N].astype(cp.float32)
    for order in (3, 5):
        c = prefilter_fft(img, order)
        back = interp_prefiltered(c, cp.stack([yy, xx]), cp.ones((N, N), bool), order)
        assert float(cp.abs(back - img).max()) <= 1e-5 * float(cp.abs(img).max())


@needs_gpu
def test_fused_cak_grid_degrid(monkeypatch):
    """TTCALX_FUSED_CAK: degrid bitwise equal to the FAST_AK path (deterministic interpolation, same float
    operations); grid equal to float32 rounding (cuFINUFFT spreading atomics)."""
    from ttcalx_imager import gridder
    rng = np.random.default_rng(9)
    N, sc = 512, math.radians(0.25)
    M = 30000
    u = rng.uniform(-60, 60, M); v = rng.uniform(-60, 60, M); w = rng.uniform(-30, 30, M)
    vis = cp.asarray(rng.normal(size=M) + 1j * rng.normal(size=M), cp.complex64)
    model = cp.zeros((N, N), cp.float32); model[100, 300] = 2.0; model[260, 200] = -0.7; model[256, 256] = 1.0
    for k, v_ in dict(TTCALX_NUFFT_PRUNED="1", TTCALX_FAST_AK="1", TTCALX_FUSED_WACC="1").items():
        monkeypatch.setenv(k, v_)
    gridder.clear_caches()
    svd = gridder.get_svd(N, sc, float(np.abs(w).max()), 1e-4, quantum=0.25)
    S = gridder.SampleSet(u, v, w, rng.uniform(0.5, 1.5, M).astype(np.float32), N, sc, svd=svd, precompute=False,
                          scope="hour")
    out = {}
    for f in ("0", "1"):
        monkeypatch.setenv("TTCALX_FUSED_CAK", f)
        out[f] = (S.grid(vis), S.degrid(model))
    assert cp.asnumpy(out["1"][1]).tobytes() == cp.asnumpy(out["0"][1]).tobytes()
    g0, g1 = out["0"][0], out["1"][0]
    assert float(cp.abs(g1 - g0).max()) <= 1e-6 * float(cp.abs(g0).max())
    gridder.clear_caches()


def _msclean_problem(N=384, theo=3.0, seed=5):
    from ttcalx_imager.clean import Convolver
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:N, :N] - N // 2
    psf = np.exp(-0.5 * (xx ** 2 + yy ** 2) / (theo / 2.3548) ** 2)
    psf += 0.05 * np.cos(2 * np.pi * xx / 23.0) * np.exp(-(xx ** 2 + yy ** 2) / (2 * 60.0 ** 2))   # sidelobes
    psf = cp.asarray(psf / psf[N // 2, N // 2], dtype=cp.float32)
    C = Convolver(N, N); K = C.kfft(psf)
    sky = cp.zeros((N, N), cp.float32)
    for _ in range(25):
        sky[int(rng.integers(60, N - 60)), int(rng.integers(60, N - 60))] = float(rng.uniform(1, 30))
    g = cp.asarray(np.exp(-0.5 * ((xx - 30) ** 2 + (yy + 20) ** 2) / 8.0 ** 2), cp.float32)
    sky += 0.4 * g                                                      # an extended source (larger scales)
    dirty = (C(sky, K) + cp.asarray(rng.normal(0, 0.05, (N, N)), dtype=cp.float32)).astype(cp.float32)
    return dirty, psf, (lambda M: (dirty - C(M, K)).astype(cp.float32)), theo


@needs_gpu
def test_ms_incr_equals_fft_scale_images(monkeypatch):
    """TTCALX_MS_INCR: multiscale CLEAN with incremental scale-image updates gives the default's model and
    residual to float32 rounding; the full-size kernels held stay within INCR_CACHE_GB (with 0 no S_s * S_k * PSF
    kernel is made, only the residual-update kernels already held are re-used); with no scale pair allowed
    (INCR_HMAX < 0) every cycle uses the FFT scale images: bitwise the default."""
    from ttcalx_imager import clean
    dirty, psf, residual, theo = _msclean_problem()
    params = bands.DEEP_I.with_(niter=4000)
    monkeypatch.setenv("TTCALX_SPARSE_UPDATE_MAX", "64")
    monkeypatch.setenv("TTCALX_SCALE_FFT", "small")
    orig = clean.MultiScaleClean._incr_scale_images
    st = dict(used=0, held=0, ngsk=0)

    def spy(self, *a):
        r = orig(self, *a)
        st["held"] = max(st["held"], sum(int(g.nbytes) for g, _ in list(self._gsk.values()) +
                                         list(self._fullps.values())))
        st["ngsk"] = max(st["ngsk"], len(self._gsk)); st["used"] += r is not None
        return r
    monkeypatch.setattr(clean.MultiScaleClean, "_incr_scale_images", spy)
    res = {}
    for name, incr, cap, hmax in (("off", "0", 1.0, 64), ("on", "16", 1.0, 64), ("cap0", "16", 0.0, 64),
                                  ("nopair", "16", 1.0, -1)):
        monkeypatch.setenv("TTCALX_MS_INCR", incr)
        monkeypatch.setattr(clean.MultiScaleClean, "INCR_CACHE_GB", cap)
        monkeypatch.setattr(clean.MultiScaleClean, "INCR_HMAX", hmax)
        st.update(used=0, held=0, ngsk=0)
        _, R, M, info = clean.deconvolve_multiscale(dirty, psf, 0.3, theo, params, residual_fn=residual,
                                                    beam=(0.9, 0.9, 0.0))
        res[name] = (cp.asnumpy(R), cp.asnumpy(M), info["nit"], dict(st))
    assert res["off"][3]["used"] == 0 and res["nopair"][3]["used"] == 0
    assert res["on"][3]["used"] > 10 and res["on"][3]["ngsk"] > 0 and res["on"][3]["held"] <= 1.0e9
    assert res["cap0"][3]["ngsk"] == 0
    R0, M0, n0, _ = res["off"]
    assert res["nopair"][0].tobytes() == R0.tobytes() and res["nopair"][1].tobytes() == M0.tobytes()
    for name in ("on", "cap0"):
        R1, M1, n1, _ = res[name]
        assert abs(n1 - n0) <= 0.01 * n0, name
        assert abs(M1.sum() / M0.sum() - 1) < 1e-3, name
        assert np.sqrt(np.mean((R1 - R0) ** 2)) < 0.02 * R0.std(), name


@needs_gpu
def test_hour_v5_switches_same_images(tmp_path, monkeypatch):
    """End to end (small w-snapshot hour, FAST_ENV on): every HOUR_V5_ENV switch on gives the same products as
    all off -- dirty, PSF and V to float32 / w-term-factor rounding, the CLEANed products to the small differences
    of a CLEAN that picks a few components differently (TTCALX_SVD_QUANTUM: other w-term factors); the
    summary lists the switches.  The image edge is ~11 px beyond the horizon, as at the production geometries
    (7-12 px): the FFT spline prefilter equals cupyx's mirror-boundary one only where the image is zero near
    its edge."""
    from astropy.io import fits
    from ttcalx_imager import hour
    from ttcalx_imager.rotate import analytic_rotation
    rng = np.random.default_rng(5)
    ant = rng.normal(0, 70, (64, 3)); ant[:, 2] *= 0.05
    b = (ant[:, None] - ant[None])[np.triu_indices(64, 1)]
    freqs = np.array([30.0e6]); lam = 299792458.0 / freqs[0]
    dec = math.radians(37.2); ra_h = 1.0
    nsnap = 12; ras = ra_h + (np.arange(nsnap) - nsnap // 2) * math.radians(1.25)
    N, sdeg = 256, 0.49                                  # image edge at l = 1.09
    l0, m0 = 0.1, -0.05

    def snap(i):
        u, v, w = (b[:, k] / lam for k in range(3))
        R = analytic_rotation(ras[i], dec, ra_h, dec)
        uh, vh, wh = R @ np.stack([u, v, w])
        n0 = math.sqrt(1 - l0 * l0 - m0 * m0)
        vis = 10 * np.exp(2j * np.pi * (uh * l0 + vh * m0 + wh * (n0 - 1))) * np.exp(-2j * np.pi * (wh - w))
        vis = vis + 0.05 * (rng.normal(size=u.size) + 1j * rng.normal(size=u.size))
        return dict(path=f"s{i}", stem=f"20261024_10{i:02d}03_30MHz", time=100.0 + 300 * i, tstart=95.0 + 300 * i,
                    freqs=freqs, chan_width=1e5, R=R, u=u.astype(np.float32), v=v.astype(np.float32),
                    w=w.astype(np.float32), nw=np.ones(u.size, np.float32), I=vis.astype(np.complex64),
                    V=(0.1 * vis).astype(np.complex64), flagfrac=0.0, zero_frac=0.0, telescope="OVRO-LWA",
                    observer="", field_name="")

    snaps = [snap(i) for i in range(nsnap)]
    monkeypatch.setattr(hour.bands, "geometry", lambda band: (N, sdeg))
    prods = [bands.HourProduct("t", 1, bands.DEEP_I.with_(niter=3000), v_tag="V"),
             bands.HourProduct("q", 2, bands.TENMIN_I.with_(niter=1000))]
    _clear_v5(monkeypatch)
    for k, v in FAST_ENV.items():
        monkeypatch.setenv(k, v)
    sums = {}
    for name in ("off", "on"):
        if name == "on":
            for k, v in HOUR_V5_ENV.items():
                monkeypatch.setenv(k, v)
        sums[name] = hour.image_hour_snaps([dict(s) for s in snaps], 30, str(tmp_path / name), ra_h, dec,
                                           products=prods, qa="none")
    assert "hour_v5" not in sums["off"] and sums["on"]["hour_v5"] == HOUR_V5_ENV
    get = lambda n, f: fits.getdata(f"{tmp_path}/{n}-{f}.fits").squeeze().astype(np.float64)
    for f in ("t-dirty", "t-psf", "V-image"):
        a, c = get("off", f), get("on", f)
        assert np.abs(c - a).max() <= 1e-4 * np.abs(a).max(), f
    for f in ("t-image", "q-t0000-image", "q-t0001-image"):
        a, c = get("off", f), get("on", f)
        assert np.abs(c - a).max() <= 5e-3 * np.abs(a).max(), f
    a, c = get("off", "t-residual"), get("on", "t-residual")
    assert np.sqrt(np.mean((c - a) ** 2)) <= 0.05 * a.std()
    a, c = get("off", "t-model"), get("on", "t-model")
    assert abs(c.sum() / a.sum() - 1) < 1e-3
