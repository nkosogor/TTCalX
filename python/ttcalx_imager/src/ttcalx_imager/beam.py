"""Restoring-beam fit: a port of WSClean 3.4's PSF fit (schaapcommon fitters/gaussianfitter.cc,
``Fit2DGaussianCentred`` as called from wsclean math/imageoperations.cpp ``FitBeamSize``).

Algorithm (default WSClean settings: -beam-fitting-size 10, fit with negatives):
  1. beam estimate b0 [px] = theoretical beam / pixel scale, theoretical beam = 1 / max|uvw|
     [rad, lambda over unflagged samples], at least one pixel.
  2. box = max(ceil(10), ceil(10 b0)) pixels, rounded up to even, centred on pixel (W/2, H/2).
  3. Least-squares fit (Levenberg-Marquardt, GSL lmsder in WSClean; MINPACK lmder here) of
         G(x, y) = exp(-x^2 / (2 sx^2) + beta x y / (sx sy) - y^2 / (2 sy^2))
     to the box pixels (peak fixed to 1, centre fixed, negative values included), with
     x = (xi - box/2) / s, y = (yi - box/2) / s, s = box size, starting from
     sx = sy = b0 / (s * 2 sqrt(2 ln 2)), beta = 0.
     Quirk reproduced on purpose: the C++ computes (xi - x_mid) with xi unsigned (size_t), so for
     xi < x_mid (or yi < y_mid) the coordinate wraps to ~1e19, the model is exactly 0 there and
     those pixels do not constrain the fit; effectively only the quadrant x >= 0, y >= 0 of the
     box is fitted.  With this emulation the port reproduces WSClean's BMAJ/BMIN to ~1e-6 and
     BPA to ~1e-5 deg on WSClean's own PSFs; fitting the full box instead differs by 1-2 %.
  4. If 0.8 * 10 * FWHM_major >= box, repeat with box = ceil(10 FWHM_major) (even) and
     b0 = max(FWHM_major, b0), at most 5 times.
  5. (sx, sy, beta) -> covariance -> eigen decomposition -> FWHM major/minor and
     PA = -atan2(vx, vy) of the major-axis eigenvector (aocommon EigenValuesAndVectors, whose
     eigenvector sign convention fixes the PA branch), i.e. measured from +y (north) towards -x
     (east for RA---SIN with CDELT1 < 0).

The PSF must be in FITS/WSClean memory order [y, x] (row y = 0 at the bottom), peak-normalised.
The prototype's fit (centred Gaussian on the main lobe only, pixels > 5 % of the peak) gave beams
1-3 % wider than WSClean's.
"""
import math
import os

import numpy as np

SIGMA_TO_FWHM = 2.0 * math.sqrt(2.0 * math.log(2.0))
WRAP = True    # emulate the unsigned index arithmetic of the C++ fitter (see _fit_box)


class DegeneratePSFError(ValueError):
    """The snapshot's PSF / restoring beam is unusable: the unflagged uv coverage is too small for
    the WSClean-equivalent clean (e.g. a snapshot whose data are almost all exactly zero, flagged by
    the peel's flag_zero, leaving a few short baselines).  A property of the input data, so a
    retry gives the same result: callers treat it as a permanent input failure (skip the snapshot)."""
    permanent_input_failure = True


def local_rms_radius(scale_deg, theo_pix, window=25.0):
    """Radius [px] inside which local_rms_factor takes the minimum RMS: the horizon (r = 1) less
    1.5 RMS-window widths (3 sigma of the window Gaussian, FWHM = window * theo_pix)."""
    return 1.0 / math.radians(scale_deg) - 3 * window * theo_pix / 2.3548


def check_psf_support(scale_deg, theo_pix, window=25.0, npts=None):
    """Raise DegeneratePSFError when the theoretical beam (1 / max|uvw|) is so wide that the
    local-RMS window covers the whole sky: local_rms_factor would have no pixel to take its minimum
    over (local_rms_radius <= 0; the centre pixel is always inside the horizon, so the selection is
    empty exactly then), and the restoring-beam fit runs on a PSF wider than its fitting box.
    Snapshots that pass are imaged exactly as before."""
    R = local_rms_radius(scale_deg, theo_pix, window)
    if not R > 0:
        sc = math.radians(scale_deg)
        maxuvw = 1.0 / (theo_pix * sc) if theo_pix and math.isfinite(theo_pix) else float("nan")
        raise DegeneratePSFError(
            f"degenerate PSF: max|uvw| = {maxuvw:.4g} lambda{'' if npts is None else f' over {npts} samples'} "
            f"gives a theoretical beam of {theo_pix:.4g} px ({theo_pix * scale_deg:.3g} deg); the local-RMS "
            f"window ({window:g} beams) covers the whole sky (radius {R:.4g} px <= 0): too few / too short "
            f"unflagged baselines (e.g. a snapshot whose data are almost all exactly zero)")


def _gauss(x, y, sx, sy, beta):
    return np.exp(-x * x / (2.0 * sx * sx) + beta * x * y / (sx * sy) - y * y / (2.0 * sy * sy))


def _eigen_aocommon(m):
    """aocommon::EigenValuesAndVectors for a real 2x2 [m0 m1; m2 m3] (row-major)."""
    tr = m[0] + m[3]
    d = m[0] * m[3] - m[1] * m[2]
    term = math.sqrt(max(tr * tr * 0.25 - d, 0.0))
    e1 = tr * 0.5 + term
    e2 = tr * 0.5 - term
    limit = min(abs(e1), abs(e2)) * 1e-6
    if abs(m[2]) > limit:
        v1 = (m[3] - e1, -m[2]); v2 = (m[3] - e2, -m[2])
    elif abs(m[1]) > limit:
        v1 = (-m[1], m[0] - e1); v2 = (-m[1], m[0] - e2)
    else:
        if (m[0] - e1) ** 2 < (m[0] - e2) ** 2:
            v1, v2 = (1.0, 0.0), (0.0, 1.0)
        else:
            v1, v2 = (0.0, 1.0), (1.0, 0.0)
    return e1, e2, v1, v2


def to_fwhm_pa(sx, sy, beta):
    """schaapcommon ToAnglesAndFwhm: (sx, sy, beta) -> (major, minor, pa [rad]) (units of sx)."""
    bf = 1.0 - beta * beta
    cov = (sx * sx / bf, beta * sx * sy / bf, beta * sx * sy / bf, sy * sy / bf)
    e1, e2, v1, v2 = _eigen_aocommon(cov)
    if math.isfinite(e1):
        maj = math.sqrt(abs(e1)) * SIGMA_TO_FWHM
        mnr = math.sqrt(abs(e2)) * SIGMA_TO_FWHM
        if maj < mnr:
            maj, mnr = mnr, maj
            v1 = v2
        return maj, mnr, -math.atan2(v1[0], v1[1])
    s = math.sqrt(abs(sx)) * SIGMA_TO_FWHM
    return s, s, 0.0


def _fit_box(img, beam_est, wrap=WRAP):
    """SingleFit2DGaussianCentred on a (h, w) array; returns (major_px, minor_px, pa_rad)."""
    from scipy.optimize import least_squares
    h, w = img.shape
    s = (w + h) // 2                       # data.scale_factor (integer, as size_t in C++)
    x = (np.arange(w) - w // 2) / s
    y = (np.arange(h) - h // 2) / s
    X, Y = np.meshgrid(x, y)               # img[yi, xi]
    d = img.astype(np.float64)
    # C++: (size_t xi - int x_mid) is unsigned, so x < 0 wraps to ~1.8e19 / s and the model and
    # its derivatives are exactly 0 there: only the quadrant x >= 0, y >= 0 constrains the fit.
    live = (X >= 0) & (Y >= 0) if wrap else np.ones_like(X, bool)

    def res(p):
        return (np.where(live, _gauss(X, Y, p[0], p[1], p[2]), 0.0) - d).ravel()

    def jac(p):
        sx, sy, beta = p
        e = np.where(live, _gauss(X, Y, sx, sy, beta), 0.0)
        dsx = (beta * X * Y / (sx * sx * sy) + X * X / sx ** 3) * e
        dsy = (beta * X * Y / (sy * sy * sx) + Y * Y / sy ** 3) * e
        db = X * Y / (sx * sy) * e
        return np.stack([dsx.ravel(), dsy.ravel(), db.ravel()], axis=1)

    p0 = beam_est / (s * SIGMA_TO_FWHM)
    r = least_squares(res, [p0, p0, 0.0], jac=jac, method="lm", xtol=1e-10, ftol=1e-12, gtol=1e-12, max_nfev=2000)
    maj, mnr, pa = to_fwhm_pa(*r.x)
    return maj * s, mnr * s, pa


def fit_psf_wsclean(psf, beam_est_px, box_scale=10.0, wrap=WRAP, info=None):
    """WSClean Fit2DGaussianCentred: (major_px, minor_px, pa_rad) of a peak-normalised PSF.
    info (dict): gets "max_box", the largest fitting box used."""
    psf = np.asarray(psf)
    H, W = psf.shape
    if info is not None:
        info["max_box"] = 0
    pref = max(int(math.ceil(box_scale)), int(math.ceil(beam_est_px * box_scale)))
    pref += pref % 2
    if not (pref < W or pref < H):
        if info is not None:
            info["max_box"] = max(W, H)
        return _fit_box(psf, beam_est_px, wrap)
    for _ in range(5):
        bw, bh = min(pref, W), min(pref, H)
        x0, y0 = (W - bw) // 2, (H - bh) // 2
        maj, mnr, pa = _fit_box(psf[y0:(H + bh) // 2, x0:x0 + bw], beam_est_px, wrap)
        if info is not None:
            info["max_box"] = max(info["max_box"], bw, bh)
        ok = ((maj * box_scale * 0.8 < bw or bw >= W) and (maj * box_scale * 0.8 < bh or bh >= H))
        if ok:
            break
        pref = max(int(math.ceil(box_scale)), int(math.ceil(maj * box_scale)))
        pref += pref % 2
        beam_est_px = max(maj, beam_est_px)
    return maj, mnr, pa


def strict_beam():
    """TTCALX_STRICT_BEAM=1: fit_beam raises DegeneratePSFError for an unusable PSF / beam fit.  Off by
    default (86afd89 behaviour: the fit's values are used as they come, e.g. a NaN beam in the header);
    a degenerate snapshot is caught before the fit by check_psf_support in either case."""
    return os.environ.get("TTCALX_STRICT_BEAM", "0") == "1"


def fit_beam(psf, scale_deg, theoretical_beam_px, wrap=WRAP, strict=None, window=None):
    """Restoring beam (BMAJ, BMIN, BPA) in degrees, as written by WSClean.

    psf: [y, x] array (numpy or cupy), normalised here by its centre pixel (H/2, W/2) as WSClean.
    theoretical_beam_px: 1 / max|uvw| / scale (WSClean's initial value; >= 1 pixel)."""
    full = psf
    win = int(os.environ.get("TTCALX_FIT_WINDOW", "0")) if window is None else int(window)
    H0, W0 = psf.shape
    if win and win % 2 == 0 and win < min(H0, W0) and H0 % 2 == 0 and W0 % 2 == 0:
        # TTCALX_FIT_WINDOW=<even px> (default 0): only the centred win x win part of the PSF is copied and
        # normalised (the fitting boxes are centred and much smaller; a fit whose box reaches the window is
        # redone on the whole PSF), so the fit sees the same pixels
        psf = psf[H0 // 2 - win // 2:H0 // 2 + win // 2, W0 // 2 - win // 2:W0 // 2 + win // 2]
    else:
        win = 0
    if hasattr(psf, "get"):
        psf = psf.get()
    psf = np.asarray(psf, dtype=np.float32)
    H, W = psf.shape
    c = psf[H // 2, W // 2]
    if c != 0:
        psf = psf / c
    strict = strict_beam() if strict is None else strict
    if strict and not (c != 0 and math.isfinite(float(c))):
        raise DegeneratePSFError(f"degenerate PSF: centre pixel is {float(c)!r} (no usable uv samples)")
    info = {}
    maj, mnr, pa = fit_psf_wsclean(psf, max(float(theoretical_beam_px), 1.0), wrap=wrap, info=info)
    if win and info["max_box"] >= win:
        return fit_beam(full, scale_deg, theoretical_beam_px, wrap=wrap, strict=strict, window=0)
    if strict and not (math.isfinite(maj) and math.isfinite(mnr) and math.isfinite(pa) and maj > 0 and mnr > 0):
        raise DegeneratePSFError(f"degenerate restoring beam: fit gave major {maj!r} px, minor {mnr!r} px, "
                                 f"PA {pa!r} rad (theoretical beam {float(theoretical_beam_px):.3g} px)")
    return maj * scale_deg, mnr * scale_deg, math.degrees(pa)
