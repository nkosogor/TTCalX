"""CPU-only tests: geometry, weights, rotation conventions, beam fit, FITS header, QA rule."""
import math

import numpy as np
import pytest

from ttcalx_imager import bands
from ttcalx_imager.beam import DegeneratePSFError, check_psf_support, fit_beam, local_rms_radius, to_fwhm_pa
from ttcalx_imager.qa import qa_reject
from ttcalx_imager.rotate import analytic_rotation, rotate_samples, rotation_from_uvw, uvw_frame
from ttcalx_imager.weights import WeightGrid, padded_size


# ----------------------------------------------------------------------------- bands
def test_geometry_tiers():
    assert bands.geometry("41MHz") == (2358, 0.049)        # wsclean rounds 2357 up to even
    assert bands.geometry(18) == (1508, 0.0767)
    assert bands.geometry("/x/20261023_080003_69MHz.ms") == (3122, 0.037)
    assert bands.hour_phase_centre(2) == "02h30m00s 37d12m57.057s"


def test_orca_hour_products():
    P = {p.tag: p for p in bands.hour_products(["orca"])}
    assert list(P) == list(bands.ORCA_I_PRODUCTS)
    d = P["I-Deep-Taper-Robust-0"]
    assert (d.n_intervals, d.params, d.v_tag, d.naming) == (1, bands.DEEP_I, "V-Taper-Deep", "orca")
    assert P["I-Deep-Taper-Robust-0.75"].params.robust == -0.75
    assert P["I-Deep-Taper-Robust-0.75"].params.taper_inner_tukey == 30.0
    for n, r in (("I-Deep-NoTaper-Robust-0.75", -0.75), ("I-Deep-NoTaper-Robust-0", 0.0)):
        p = P[n].params
        assert (p.robust, p.taper_inner_tukey, p.niter, p.multiscale, p.mgain) == (r, 0.0, 150000, True, 0.95)
        assert P[n].v_tag is None
    t = P["I-Taper-10min"]
    assert (t.n_intervals, t.params.niter, t.v_tag) == (6, 50000, "V-Taper-10min")
    assert P["I-Deep-Taper-Robust-0.75"].params.niter == 500000


def test_hour_products_v_only_and_legacy():
    (p,) = bands.hour_products(["V-Taper-Deep"])
    assert (p.tag, p.image_i, p.v_tag, p.params) == ("I-Deep-Taper-Robust-0", False, "V-Taper-Deep", bands.DEEP_I)
    (p,) = bands.hour_products(["I-Deep-Taper-Robust-0"])
    assert p.v_tag is None and p.image_i
    a, b = bands.hour_products(["1h", "10min"], intervals=3, niter=7, niter_10min=5, pols="I")
    assert (a.tag, a.naming, a.params.niter, a.v_tag) == ("1h", "legacy", 7, None)
    assert (b.n_intervals, b.params.niter) == (3, 5)
    assert bands.hour_products(["1h"])[0].v_tag == "V"
    with pytest.raises(ValueError):
        bands.hour_products(["I-Deep-Taper-Robust-1"])


# ----------------------------------------------------------------------------- weights
def _briggs_reference(u, v, nw, N, sc, robust, tukey):
    """Straightforward per-sample implementation of the WSClean 3.4 rules (dict of cells)."""
    npad = padded_size(N); du = 1.0 / (npad * sc)
    cells = {}
    keys = []
    for a, b, wn in zip(u, v, nw):
        if b < 0:
            a, b = -a, -b
        k = (math.floor(a / du) + npad // 2, math.floor(b / du) + npad // 2)
        if not (0 <= k[0] < npad and 0 <= k[1] < npad):
            k = None                       # outside the (padded) uv grid: weight 0
        keys.append(k)
        if k is not None:
            cells[k] = cells.get(k, 0.0) + wn
    W = np.array(list(cells.values()))
    f2 = (5 * 10 ** -robust) ** 2 / ((W ** 2).sum() / W.sum())
    out = []
    for k, wn in zip(keys, nw):
        if k is None:
            out.append(0.0); continue
        x = wn / (1 + cells[k] * f2)
        r = math.hypot((k[0] - npad // 2) * du, (k[1] - npad // 2) * du)
        if tukey and r < tukey:
            x *= 0.5 * (1 - math.cos(math.pi * r / tukey))
        out.append(x)
    return np.array(out)


def test_padded_size():
    assert padded_size(2358) == 2832 and padded_size(3122) == 3748 and padded_size(1508) == 1812


@pytest.mark.parametrize("robust,tukey", [(0.0, 30.0), (-0.75, None), (1.0, 10.0)])
def test_briggs_matches_reference(robust, tukey):
    rng = np.random.default_rng(1)
    n = 3000
    u = rng.normal(0, 25, n); v = rng.normal(0, 25, n)       # grid edge at ~57 lambda: some fall off
    u[:200] = rng.normal(0, 3, 200); v[:200] = rng.normal(0, 3, 200)      # dense core, inside the taper
    nw = rng.uniform(0.5, 2, n)
    N, sc = 256, math.radians(0.5)
    g = WeightGrid(N, sc, robust, tukey, xp=np)
    g.add(u[:1000], v[:1000], nw[:1000]); g.add(u[1000:], v[1000:], nw[1000:])     # incremental
    g.finalize()
    got = g.weights(u, v, nw)
    ref = _briggs_reference(u, v, nw, N, sc, robust, tukey)
    assert np.allclose(got, ref, rtol=2e-6, atol=1e-9)
    assert (ref == 0).sum() > 0 and (ref > 0).sum() > 2500


def test_briggs_hermitian_fold():
    g = WeightGrid(128, math.radians(1.0), 0.0, None, xp=np)
    u = np.array([10.3, -10.3]); v = np.array([5.1, -5.1]); nw = np.ones(2)
    g.add(u, v, nw); g.finalize()
    w = g.weights(u, v, nw)
    assert w[0] == w[1]                     # (u, v) and (-u, -v) share one cell


# ----------------------------------------------------------------------------- rotation
def test_rotation_from_uvw_recovers_analytic():
    rng = np.random.default_rng(2)
    xyz = rng.normal(0, 500, (60, 3))
    b = (xyz[:, None] - xyz[None]).reshape(-1, 3)
    R = analytic_rotation(0.3, 0.65, 0.5, 0.60)
    uvw0 = b @ uvw_frame(0.3, 0.65).T
    uvw1 = b @ uvw_frame(0.5, 0.60).T
    assert np.allclose(rotation_from_uvw(uvw0, uvw1), R, atol=1e-12)
    assert np.allclose(R @ R.T, np.eye(3), atol=1e-12)


def test_rephasing_convention_point_source():
    """With the gridder convention V = exp(-2 pi i (u l + v m - w (n - 1))) a point source has
    (l, m, -n) transforming like (u, v, w); rotating (uvw, vis) to a new frame must give the
    same formula evaluated in the new frame (this fixes the sign of the re-phasing)."""
    rng = np.random.default_rng(3)
    uvw0 = rng.normal(0, 40, (500, 3))
    R = analytic_rotation(0.2, 0.65, 0.35, 0.62)
    d0 = np.array([0.1, -0.2, -math.sqrt(1 - 0.05)])          # (l, m, -n) in frame 0
    V0 = np.exp(-2j * np.pi * (uvw0 @ d0 + uvw0[:, 2]))
    u1, v1, w1, V1 = rotate_samples(uvw0[:, 0], uvw0[:, 1], uvw0[:, 2], R, V0)
    d1 = R @ d0
    expect = np.exp(-2j * np.pi * (u1 * d1[0] + v1 * d1[1] + w1 * d1[2] + w1))
    assert np.allclose(V1, expect, atol=1e-9)
    # direct formula of the re-phasing factor
    assert np.allclose(V1 / V0, np.exp(2j * np.pi * (uvw0[:, 2] - w1)))


# ----------------------------------------------------------------------------- beam fit
def _gauss_ws(N, bmaj_px, bmin_px, pa_deg):
    """Peak-1 Gaussian in [y, x] order, PA from +y towards -x (FITS BPA for CDELT1 < 0)."""
    s2f = 2 * math.sqrt(2 * math.log(2))
    y, x = np.mgrid[:N, :N].astype(float)
    x -= N // 2; y -= N // 2
    pa = math.radians(pa_deg)
    a = -x * math.sin(pa) + y * math.cos(pa)
    b = x * math.cos(pa) + y * math.sin(pa)
    return np.exp(-0.5 * (a * a / (bmaj_px / s2f) ** 2 + b * b / (bmin_px / s2f) ** 2)).astype(np.float32)


@pytest.mark.parametrize("pa", [54.6, -30.0, 0.0, 89.0])
@pytest.mark.parametrize("wrap", [True, False])
def test_beam_fit_synthetic(pa, wrap):
    scale = 0.049
    img = _gauss_ws(256, 6.3, 4.0, pa)
    bmaj, bmin, bpa = fit_beam(img, scale, theoretical_beam_px=3.95, wrap=wrap)
    assert bmaj == pytest.approx(6.3 * scale, rel=1e-4)
    assert bmin == pytest.approx(4.0 * scale, rel=1e-4)
    assert ((bpa - pa + 90) % 180 - 90) == pytest.approx(0, abs=0.01)


def test_beam_fit_matches_wsclean_psf():
    """Central 64x64 of a real WSClean 3.4 PSF (41 MHz snapshot, 2358 px @ 0.049 deg) and the
    BMAJ/BMIN/BPA WSClean wrote for it; theoretical beam from its log (maxuvw = 295.801 lambda)."""
    import os
    psf = np.load(os.path.join(os.path.dirname(__file__), "data", "wsclean_psf41_crop64.npy"))
    ref = (0.314415564444161, 0.197452708801402, 54.2707995116071)
    theo = (1 / 295.801) / math.radians(0.049)
    bmaj, bmin, bpa = fit_beam(psf, 0.049, theo)
    assert bmaj == pytest.approx(ref[0], rel=1e-5) and bmin == pytest.approx(ref[1], rel=1e-5)
    assert bpa == pytest.approx(ref[2], abs=1e-3)
    full = fit_beam(psf, 0.049, theo, wrap=False)        # naive full-box fit is ~1 % off
    assert abs(full[1] / ref[1] - 1) > 3e-3


def _dft_psf(u, v, H, scale_deg):
    """Natural-weight w-less PSF [y, x] of (u, v) [lambda] on a (2H)^2 grid, peak 1."""
    j = (np.arange(2 * H) - H) * math.radians(scale_deg)
    ph = 2 * np.pi * (u[:, None, None] * -j[None, None, :] + v[:, None, None] * j[None, :, None])
    p = np.cos(ph).sum(0)
    return (p / p[H, H]).astype(np.float32)


def test_degenerate_psf_is_rejected_before_the_fit():
    """20261031_101739_32MHz: after the peel's flag_zero only 28 baselines of one 8-antenna group
    remain (672 samples, max|uvw| 21.7 lambda): theoretical beam 34.4 px at 0.0767 deg, local-RMS
    radius < 0 (the old code: overflow in exp in the beam fit, then a zero-size min in
    local_rms_factor).  Synthetic: the same kind of uv coverage."""
    scale, window = 0.0767, 25.0
    rng = np.random.default_rng(1)
    ang = rng.uniform(0, 2 * np.pi, 28); r = rng.uniform(1.2, 21.7, 28); r[0] = 21.7
    u, v = r * np.cos(ang), r * np.sin(ang)
    theo = 1.0 / np.hypot(u, v).max() / math.radians(scale)
    assert theo > 30 and local_rms_radius(scale, theo, window) < 0
    psf = _dft_psf(u, v, 48, scale)
    assert psf.max() == pytest.approx(1.0) and (psf > 0.5).sum() > 300     # main lobe ~25 px wide (normal: ~8 px)
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                    # no numpy overflow on the way
        with pytest.raises(DegeneratePSFError, match="degenerate PSF: max.uvw. = 21.7 lambda over 672 samples"):
            check_psf_support(scale, theo, window, npts=672)
    with pytest.raises(ValueError):                                       # still a ValueError (v2 callers)
        check_psf_support(scale, float("nan"), window)
    assert DegeneratePSFError.permanent_input_failure is True


@pytest.mark.parametrize("band,maxuvw", [(32, 123.546), (41, 295.801), (82, 900.0), (32, 31.9)])
def test_normal_psf_support_passes(band, maxuvw):
    """Normal snapshots (max|uvw| 123.5 lambda at 32 MHz on 2026-10-31) pass untouched; the limit
    at 32 MHz is max|uvw| ~ 31.85 lambda."""
    N, scale = bands.geometry(band)
    theo = 1.0 / maxuvw / math.radians(scale)
    assert check_psf_support(scale, theo, 25.0) is None
    assert local_rms_radius(scale, theo) > 0


def test_fit_beam_rejects_an_empty_psf():
    with pytest.raises(DegeneratePSFError, match="centre pixel"):
        fit_beam(np.zeros((64, 64), np.float32), 0.0767, 6.0, strict=True)
    with pytest.raises(DegeneratePSFError, match="centre pixel"):
        fit_beam(np.full((64, 64), np.nan, np.float32), 0.0767, 6.0, strict=True)


def _fit_beam_86afd89(psf, scale_deg, theo):
    """fit_beam of 86afd89 (no checks)."""
    from ttcalx_imager.beam import fit_psf_wsclean
    psf = np.asarray(psf, dtype=np.float32)
    H, W = psf.shape
    c = psf[H // 2, W // 2]
    if c != 0:
        psf = psf / c
    maj, mnr, pa = fit_psf_wsclean(psf, max(float(theo), 1.0))
    return maj * scale_deg, mnr * scale_deg, math.degrees(pa)


def _outcome(fn, *a):
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return ("ok", np.asarray(fn(*a), np.float64).tobytes())
    except Exception as e:                       # noqa: BLE001
        return ("raised", type(e).__name__)


@pytest.mark.parametrize("kind", ["zeros", "nan", "spike"])
def test_fit_beam_default_is_86afd89(kind, monkeypatch):
    """Default (TTCALX_STRICT_BEAM unset): the hour multiscale cleaner fits the beam as before, whatever
    the PSF -- no DegeneratePSFError turns a previously written hour image into a failed hour job."""
    monkeypatch.delenv("TTCALX_STRICT_BEAM", raising=False)
    psf = {"zeros": np.zeros((64, 64), np.float32), "nan": np.full((64, 64), np.nan, np.float32),
           "spike": np.pad(np.ones((1, 1), np.float32), 31)[:64, :64]}[kind]
    got = _outcome(fit_beam, psf, 0.0767, 6.0)
    assert got == _outcome(_fit_beam_86afd89, psf, 0.0767, 6.0)
    assert got[1] != "DegeneratePSFError"
    monkeypatch.setenv("TTCALX_STRICT_BEAM", "1")
    if kind != "spike":
        assert _outcome(fit_beam, psf, 0.0767, 6.0) == ("raised", "DegeneratePSFError")


def test_to_fwhm_pa_circular():
    s2f = 2 * math.sqrt(2 * math.log(2))
    maj, mnr, pa = to_fwhm_pa(2.0, 2.0, 0.0)
    assert maj == pytest.approx(2 * s2f) and mnr == pytest.approx(2 * s2f)


# ----------------------------------------------------------------------------- FITS
def test_fits_header(tmp_path):
    from astropy.io import fits
    from astropy.wcs import WCS
    from ttcalx_imager.fits import make_header, write_image
    N, sdeg = 2358, 0.049
    freqs = 41e6 + np.arange(48) * 95703.125
    h = make_header(N, sdeg, math.radians(40.59), math.radians(37.2158), freqs, np.full(48, 95703.125),
                    5.3e9, beam=(0.3075, 0.1978, 54.56), observer="lwams.py", obj="zenith")
    assert h["CRPIX1"] == 1180.0 and h["CDELT1"] == -0.049 and h["CDELT3"] == 4593750.0
    assert h["CRVAL3"] == pytest.approx(0.5 * (freqs[0] + freqs[-1]))
    assert (h["BMAJ"], h["BMIN"], h["BPA"]) == (0.3075, 0.1978, 54.56)
    assert len(h["DATE-OBS"]) == 21 and h["DATE-OBS"][10] == "T"
    p = write_image(str(tmp_path / "x-V-image.fits"), np.zeros((N, N)), h, stokes="V")
    hh = fits.getheader(p)
    assert hh["NAXIS"] == 4 and hh["NAXIS1"] == N and hh["CRVAL4"] == 4.0
    w = WCS(hh).celestial
    ra, dec = w.wcs_pix2world([[N // 2, N // 2]], 0)[0]        # 0-based centre pixel = CRVAL
    assert ra == pytest.approx(40.59) and dec == pytest.approx(37.2158)
    ra1, _ = w.wcs_pix2world([[N // 2 - 10, N // 2]], 0)[0]    # left = east = larger RA
    assert ra1 > ra



def test_write_images_header_per_stokes(tmp_path):
    """A dirty V image is written with its own HISTORY (not the I CLEAN line), as the snapshot and hour writers do."""
    from astropy.io import fits
    from ttcalx_imager.fits import make_header, write_images
    hdr = lambda hist: make_header(64, 1.0, 0.0, 0.6, [55e6], [1e5], 5.3e9, history=[hist])
    h = {"I": hdr("ttcalx-image snapshot: clean I niter=50000"), "V": hdr("ttcalx-image snapshot: dirty V (no CLEAN)")}
    pi, pv = write_images([(str(tmp_path / "x-I-image.fits"), np.zeros((64, 64)), "I"),
                           (str(tmp_path / "x-V-image.fits"), np.zeros((64, 64)), "V")], h)
    hi, hv = fits.getheader(pi), fits.getheader(pv)
    assert hi["CRVAL4"] == 1.0 and "clean I" in str(hi["HISTORY"]) and "dirty V" not in str(hi["HISTORY"])
    assert hv["CRVAL4"] == 4.0 and "dirty V (no CLEAN)" in str(hv["HISTORY"]) and "clean I" not in str(hv["HISTORY"])
    # one header for every product still works (the MS hour job / older callers)
    (p1,) = write_images([(str(tmp_path / "y-V-image.fits"), np.zeros((64, 64)), "V")], h["I"])
    assert fits.getheader(p1)["CRVAL4"] == 4.0

# ----------------------------------------------------------------------------- QA
def test_qa_reject():
    r = np.r_[np.full(20, 0.89) + np.linspace(-0.005, 0.005, 20), 1.2, 0.3]
    bad, med, sd = qa_reject(r)
    assert bad == [20, 21] and med == pytest.approx(0.89, abs=0.01)
