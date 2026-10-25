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


def test_hour_orca_products_naming(tmp_path, monkeypatch):
    """image_hour with orca products: one load, WSClean/orca file names, V-only product."""
    import os
    from ttcalx_imager import hour
    rng = np.random.default_rng(3)
    ant = rng.normal(0, 80, (60, 3)); ant[:, 2] *= 0.05
    b = (ant[:, None] - ant[None])[np.triu_indices(60, 1)]
    freqs = np.array([23.0e6, 23.1e6])
    lam = 299792458.0 / freqs

    def snap(i):
        u = (b[:, 0:1] / lam).ravel().astype(np.float32); v = (b[:, 1:2] / lam).ravel().astype(np.float32)
        w = (b[:, 2:3] / lam).ravel().astype(np.float32)
        n = u.size
        noise = lambda: (rng.normal(size=n) + 1j * rng.normal(size=n)).astype(np.complex64)
        return dict(path=f"s{i}", stem=f"20261024_1000{i}3_23MHz", time=100.0 + 10 * i, tstart=95.0 + 10 * i,
                    freqs=freqs, chan_width=1e5, R=np.eye(3), u=u, v=v, w=w, nw=np.ones(n, np.float32),
                    I=(5.0 + noise()).astype(np.complex64), V=noise(), flagfrac=0.0, zero_frac=0.0,
                    telescope="OVRO-LWA", observer="", field_name="")

    monkeypatch.setattr(hour, "load_all", lambda paths, ra, dec, nproc=8, want_v=True: [snap(i) for i in range(len(paths))])
    prods = bands.hour_products(["I-Deep-NoTaper-Robust-0", "V-Taper-Deep"])
    out = str(tmp_path / "23MHz")
    s = hour.image_hour(["a", "b"], 23, out, phase_centre="02h30m00s 37d12m57.057s", products=prods, qa="none")
    names = sorted(os.listdir(tmp_path))
    for suffix in ("image", "residual", "model", "dirty", "psf"):
        assert f"23MHz-I-Deep-NoTaper-Robust-0-{suffix}.fits" in names
    assert "23MHz-V-Taper-Deep-image.fits" in names
    assert not any("-I-image" in n for n in names)
    assert set(s["products"]) == {"I-Deep-NoTaper-Robust-0", "V-Taper-Deep"}
    assert "I-Deep-NoTaper-Robust-0: deconvolution (incl. major cycles)" in s["timing"]
    assert "V-Taper-Deep: dirty V" in s["timing"]
    assert os.path.exists(f"{out}-summary.json")


def test_multiscale_local_rms_only_in_auto_mask_phase():
    """radler (WSClean 3.4) uses -local-rms only while auto-masking: phase 1 thresholds use the
    minimum local RMS, phase 2 (auto-threshold) uses the MAD noise of the residual with
    unweighted peaks (WSClean logs 'Lowest RMS in image' only for the first deconvolution)."""
    from ttcalx_imager.clean import Convolver, deconvolve_multiscale, local_rms_factor, mad_std, horizon_mask
    N, sdeg, theo = 256, 0.3, 3.0
    rng = np.random.default_rng(11)
    yy, xx = np.mgrid[:N, :N] - N // 2
    psf = cp.asarray(np.exp(-0.5 * (xx ** 2 + yy ** 2) / (theo / 2.3548) ** 2), dtype=cp.float32)
    C = Convolver(N, N); K = C.kfft(psf)
    sky = cp.zeros((N, N), cp.float32); sky[100, 100] = 500.0; sky[180, 60] = 30.0; sky[60, 190] = 12.0
    dirty = (C(sky, K) + cp.asarray(rng.normal(0, 1.0, (N, N)), dtype=cp.float32)).astype(cp.float32)
    above = horizon_mask(N, sdeg, 0.0)
    seen = []

    def residual(M):
        R = (dirty - C(M, K)).astype(cp.float32)
        seen.append(mad_std(R[above]))
        return R

    params = bands.DEEP_I.with_(niter=20000)
    _, R, M, info = deconvolve_multiscale(dirty, psf, sdeg, theo, params, residual_fn=residual, beam=(0.9, 0.9, 0.0))
    majors = info["majors"]
    _, minrms0 = local_rms_factor(dirty, theo, sdeg, horizon_mask(N, sdeg, params.horizon_deg), params.local_rms_window)
    assert majors[0]["phase"] == 1 and majors[0]["sigma"] == pytest.approx(minrms0, rel=1e-5)
    p2 = [m for m in majors if m["phase"] == 2]
    assert p2, "no auto-threshold phase"
    for m in p2:
        assert any(m["sigma"] == pytest.approx(s, rel=1e-5) for s in seen)
    # deconvolved down to the auto-threshold: bright source removed from the residual
    assert float(cp.abs(R[95:106, 95:106]).max()) < 5.0


def test_snapshot_psf_includes_w_term():
    """The snapshot PSF is the image of unit visibilities with the w-term (as WSClean's PSF):
    far from the centre it matches a direct DFT with the w-term, not the w-less 2-D FT."""
    N, sdeg = 128, 0.5                                     # l up to 0.56: strong w-term
    sc = math.radians(sdeg)
    u, v, w = _samples(3000, 25.0, 25.0, seed=9)
    rng = np.random.default_rng(2)
    vis = (rng.normal(size=u.size) + 1j * rng.normal(size=u.size)).astype(np.complex64)
    r = SnapshotImager().image_arrays(u, v, w, np.ones(u.size, np.float32), vis, None, N, sdeg,
                                      bands.SNAPSHOT_CLEAN_I.with_(taper_inner_tukey=0.0))
    psf = cp.asnumpy(r["psf"])
    from ttcalx_imager.weights import briggs_weights
    wt = briggs_weights(u, v, np.ones(u.size), N, sc, 0.0, None, xp=np)
    ys, xs = np.array([10, 30, 64, 100, 120, 40]), np.array([15, 110, 20, 64, 100, 64])
    l = (xs - N // 2) * sc; m = (((N - ys) % N) - N // 2) * sc
    n1 = np.sqrt(1 - l ** 2 - m ** 2) - 1
    with_w = np.real(np.exp(2j * np.pi * (np.outer(u, l) + np.outer(v, m) - np.outer(w, n1))).T @ wt) / wt.sum()
    without_w = np.real(np.exp(2j * np.pi * (np.outer(u, l) + np.outer(v, m))).T @ wt) / wt.sum()
    assert psf[N // 2, N // 2] == 1.0
    assert np.abs(psf[ys, xs] - with_w).max() < 2e-4
    assert np.abs(with_w - without_w).max() > 1e-2          # the test is sensitive to the w-term


def test_hour_psf_includes_w_term(tmp_path, monkeypatch):
    """Hour PSF (w-snapshot mode, rotated snapshot frames) = image of a unit point source at the
    hour phase centre with the w-term, as WSClean: compare with a direct DFT in the hour frame."""
    from astropy.io import fits
    from ttcalx_imager import hour
    from ttcalx_imager.rotate import analytic_rotation
    from ttcalx_imager.weights import WeightGrid
    rng = np.random.default_rng(5)
    ant = rng.normal(0, 70, (64, 3)); ant[:, 2] *= 0.05
    b = (ant[:, None] - ant[None])[np.triu_indices(64, 1)]
    freqs = np.array([30.0e6]); lam = 299792458.0 / freqs[0]
    dec = math.radians(37.2); ra_h = 1.0
    nsnap = 12; ras = ra_h + (np.arange(nsnap) - nsnap // 2) * math.radians(1.25)   # zenith moves 1.25 deg / 5 min

    def snap(i):
        u, v, w = (b[:, k] / lam for k in range(3))
        n = u.size
        return dict(path=f"s{i}", stem=f"20261024_10{i:02d}03_30MHz", time=100.0 + 300 * i, tstart=95.0 + 300 * i,
                    freqs=freqs, chan_width=1e5, R=analytic_rotation(ras[i], dec, ra_h, dec),
                    u=u.astype(np.float32), v=v.astype(np.float32), w=w.astype(np.float32), nw=np.ones(n, np.float32),
                    I=np.ones(n, np.complex64), V=np.zeros(n, np.complex64), flagfrac=0.0, zero_frac=0.0,
                    telescope="OVRO-LWA", observer="", field_name="")

    snaps = [snap(i) for i in range(nsnap)]
    monkeypatch.setattr(hour, "load_all", lambda paths, ra, dec, nproc=8, want_v=True: [dict(s) for s in snaps])
    N, sdeg = 256, 0.4
    monkeypatch.setattr(hour.bands, "geometry", lambda band: (N, sdeg))
    prod = bands.HourProduct("t", 1, bands.DEEP_I.with_(niter=10))
    out = str(tmp_path / "30MHz")
    hour.image_hour([s["path"] for s in snaps], 30, out, phase_centre=f"{math.degrees(ra_h)} {math.degrees(dec)}",
                    products=[prod], qa="none")
    psf = fits.getdata(f"{out}-t-psf.fits").squeeze().astype(np.float64)
    sc = math.radians(sdeg)
    uh, vh, wh, wt = [], [], [], []
    WG = WeightGrid(N, sc, 0.0, 30.0, xp=np)
    rot = [s["R"] @ np.stack([s["u"], s["v"], s["w"]]).astype(np.float64) for s in snaps]
    for r in rot:
        WG.add(r[0], r[1], np.ones(r.shape[1]))
    WG.finalize()
    for r in rot:
        uh.append(r[0]); vh.append(r[1]); wh.append(r[2]); wt.append(WG.weights(r[0], r[1], np.ones(r.shape[1])))
    uh, vh, wh, wt = (np.concatenate(a).astype(np.float64) for a in (uh, vh, wh, wt))
    c = N // 2
    pts = [(c + 1, c), (c, c + 2), (c - 3, c + 1), (c + 40, c - 70), (c - 90, c + 60), (20, 40), (200, 220), (128, 10)]
    ys = np.array([p[0] for p in pts]); xs = np.array([p[1] for p in pts])
    l = (xs - c) * sc; m = (((N - ys) % N) - c) * sc; n1 = np.sqrt(1 - l ** 2 - m ** 2) - 1
    ref = np.real(np.exp(2j * np.pi * (np.outer(uh, l) + np.outer(vh, m) - np.outer(wh, n1))).T @ wt) / wt.sum()
    ref_now = np.real(np.exp(2j * np.pi * (np.outer(uh, l) + np.outer(vh, m))).T @ wt) / wt.sum()
    err = np.abs(psf[ys, xs] - ref)
    print("hour PSF |err| at core/sidelobe pixels:", np.round(err, 5), " w-term effect:", np.round(np.abs(ref - ref_now), 5))
    assert psf[c, c] == 1.0
    assert err[:3].max() < 2e-3                      # core (beam fit region)
    assert err[3:].max() < 2e-3                      # far sidelobes, where the w-term matters
    assert np.abs(ref - ref_now)[3:].max() > 1e-2    # the test is sensitive to the w-term


def test_hour_date_obs_includes_rejected_first_snapshot(tmp_path, monkeypatch):
    """orca flags QA-rejected snapshots inside the concatenated MS, so WSClean's DATE-OBS is the
    first time step of the interval even if that snapshot was rejected."""
    from astropy.io import fits
    from astropy.time import Time
    from ttcalx_imager import hour
    rng = np.random.default_rng(8)
    ant = rng.normal(0, 60, (40, 3)); ant[:, 2] *= 0.05
    b = (ant[:, None] - ant[None])[np.triu_indices(40, 1)]
    freqs = np.array([50.0e6]); lam = 299792458.0 / freqs[0]

    def snap(i):
        u, v, w = (b[:, k] / lam for k in range(3))
        n = u.size
        return dict(path=f"s{i}", stem=f"s{i}", time=86400.0 * 61337 + 10 * i + 5, tstart=86400.0 * 61337 + 10 * i,
                    freqs=freqs, chan_width=1e5, R=np.eye(3), u=u.astype(np.float32), v=v.astype(np.float32),
                    w=w.astype(np.float32), nw=np.ones(n, np.float32), I=np.ones(n, np.complex64),
                    V=np.zeros(n, np.complex64), flagfrac=0.0, zero_frac=0.0, telescope="OVRO-LWA", observer="", field_name="")

    monkeypatch.setattr(hour, "load_all", lambda paths, ra, dec, nproc=8, want_v=True: [snap(i) for i in range(3)])
    monkeypatch.setattr(hour.bands, "geometry", lambda band: (128, 0.5))
    out = str(tmp_path / "50MHz")
    hour.image_hour(["a", "b", "c"], 50, out, phase_centre="02h30m00s 37d12m57.057s",
                    products=[bands.HourProduct("t", 1, bands.DEEP_I.with_(niter=5))], qa="none", exclude=[0])
    want = Time(61337.0, format="mjd", scale="utc").isot[:21]
    assert fits.getheader(f"{out}-t-image.fits")["DATE-OBS"] == want


def test_multiscale_absolute_threshold_without_auto_mask():
    """-threshold T with auto-mask/auto-threshold off: no mask phase, the whole sky is cleaned
    down to T (residual peak < T at the end), and the iteration cap is reported."""
    from ttcalx_imager.clean import Convolver, deconvolve_multiscale, horizon_mask
    N, sdeg, theo = 256, 0.3, 3.0
    rng = np.random.default_rng(12)
    yy, xx = np.mgrid[:N, :N] - N // 2
    psf = cp.asarray(np.exp(-0.5 * (xx ** 2 + yy ** 2) / (theo / 2.3548) ** 2), dtype=cp.float32)
    C = Convolver(N, N); K = C.kfft(psf)
    sky = cp.zeros((N, N), cp.float32)
    for k in range(40):
        sky[int(rng.integers(30, 226)), int(rng.integers(30, 226))] = float(rng.uniform(2, 50))
    dirty = (C(sky, K) + cp.asarray(rng.normal(0, 0.3, (N, N)), dtype=cp.float32)).astype(cp.float32)
    residual = lambda M: (dirty - C(M, K)).astype(cp.float32)
    T = 1.5
    params = bands.DEEP_I.with_(auto_mask=0.0, auto_threshold=0.0, local_rms=False, threshold=T, niter=10 ** 6)
    _, R, M, info = deconvolve_multiscale(dirty, psf, sdeg, theo, params, residual_fn=residual, beam=(0.9, 0.9, 0.0))
    assert all(m["phase"] == 2 for m in info["majors"])
    assert not info["niter_reached"]
    inside = horizon_mask(N, sdeg, params.horizon_deg)
    assert float(cp.abs(R[inside]).max()) < T * 1.02
    deficit = float(sky.sum() - M.sum())                 # each source may keep < T in the residual
    assert -0.02 * float(sky.sum()) < deficit < 40 * T
    _, _, _, info2 = deconvolve_multiscale(dirty, psf, sdeg, theo, params.with_(niter=50), residual_fn=residual,
                                           beam=(0.9, 0.9, 0.0))
    assert info2["niter_reached"]
