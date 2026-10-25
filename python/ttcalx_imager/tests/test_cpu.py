"""CPU-only tests: geometry, weights, rotation conventions, beam fit, FITS header, QA rule."""
import math

import numpy as np
import pytest

from ttcalx_imager import bands
from ttcalx_imager.beam import fit_beam, to_fwhm_pa
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


# ----------------------------------------------------------------------------- QA
def test_qa_reject():
    r = np.r_[np.full(20, 0.89) + np.linspace(-0.005, 0.005, 20), 1.2, 0.3]
    bad, med, sd = qa_reject(r)
    assert bad == [20, 21] and med == pytest.approx(0.89, abs=0.01)
