"""Exact phase-centre rotation of snapshot visibilities (equivalent to WSClean ``chgcentre``).

For each snapshot the new UVW are computed with casacore measures for the target J2000 direction
(per-antenna uvw from ``me.to_uvw``; baseline uvw = A[a1] - A[a2], the sign convention of the
MS UVW column), and the 3x3 matrix R with ``uvw_new = R @ uvw_stored`` is obtained by least
squares over all baselines (a rigid rotation to ~1e-9).  This reproduces chgcentre including
precession/nutation/aberration, which a purely geometric (ra, dec) -> (ra', dec') rotation does
not.  Visibilities are then re-phased with

    vis_new = vis * exp(+2 pi i (w_stored - w_new) / lambda)

(sign verified against chgcentre on our MSes; with the gridder convention
I(l,m) = sum V exp(+2 pi i (u l + v m + w (n-1))) used here, see gridder.py).

Because (l, m, -n) transforms like (u, v, w), the same R maps sky directions between the
snapshot frame and the target frame (used by reproject.py).
"""
import numpy as np


def parse_phase_centre(pc):
    """'02h30m00s 37d12m57.057s' (or 'ra_deg dec_deg') -> (ra, dec) rad."""
    from astropy.coordinates import Angle
    a, b = pc.split()
    ra = Angle(a, unit="hourangle") if any(c in a for c in "hm:") else Angle(float(a), unit="deg")
    return ra.rad, Angle(b, unit="deg").rad


def measures_uvw(ant_pos, time_s, ra, dec):
    """Per-antenna J2000 uvw [m] (nant, 3) of ITRF antenna positions at MJD seconds time_s."""
    from casacore.measures import measures
    from casacore.quanta import quantity as q
    me = measures()
    me.do_frame(me.epoch("UTC", q(time_s, "s")))
    me.do_frame(me.position("ITRF", *[q(x, "m") for x in ant_pos.mean(0)]))
    me.do_frame(me.direction("J2000", q(ra, "rad"), q(dec, "rad")))
    b = me.baseline("ITRF", q(ant_pos[:, 0], "m"), q(ant_pos[:, 1], "m"), q(ant_pos[:, 2], "m"))
    return np.array(me.to_uvw(b)["xyz"].get_value()).reshape(-1, 3)


def parse_phase_centre_chgcentre(pc):
    """(ra, dec) rad of '05h30m00s 37d12m57.057s' as the PHASE_DIR chgcentre writes, bit for bit.

    It can differ from :func:`parse_phase_centre` in the last bit (05h30m00s: 1 ulp in RA), so a
    snapshot rotated in memory gets chgcentre's image header.  Matched against /opt/bin/chgcentre
    for all 24 LST-hour centres (gpu-subband-imaging scripts/v3_engine/chgcentre_check.py
    --phase-dirs): RA = (h/24 + m/1440 + s/86400) 2 pi and Dec = (d/360 + m/21600 + s/1296000) 2 pi
    in x87 long double (aocommon RaDecCoord), rounded to double, then through casacore's
    direction cosines: ra = atan2(y, x) (wrapped to (-pi, pi]), dec = asin(z).  Where long double
    is plain double (aarch64 macOS) a result can be 1 ulp off.  Only the 'XXhYYmZZs [+-]XXdYYmZZs'
    form."""
    import math
    import re
    a, b = pc.split()
    m = re.fullmatch(r"(\d+)h(\d+)m([\d.]+)s", a)
    n = re.fullmatch(r"([+-]?)(\d+)d(\d+)m([\d.]+)s", b)
    if not m or not n:
        raise ValueError(f"not a chgcentre phase centre: {pc!r}")
    ld = np.longdouble
    two_pi = ld("6.28318530717958647692528676655900577")
    ra = float((ld(m[1]) / 24 + ld(m[2]) / (24 * 60) + ld(float(m[3])) / (24 * 3600)) * two_pi)
    dec = float((ld(n[2]) / 360 + ld(n[3]) / (360 * 60) + ld(float(n[4])) / (360 * 3600)) * two_pi)
    dec = -dec if n[1] == "-" else dec
    x, y, z = math.cos(ra) * math.cos(dec), math.sin(ra) * math.cos(dec), math.sin(dec)
    return math.atan2(y, x), math.asin(z)


def chgcentre_uvw(ant_pos, a1, a2, time_s, ra, dec):
    """The UVW chgcentre writes for a snapshot: A[a1] - A[a2] with the per-antenna J2000 uvw of
    :func:`measures_uvw` at the snapshot's TIME (all rows, autocorrelations 0)."""
    A = measures_uvw(ant_pos, time_s, ra, dec)
    return A[a1] - A[a2]


def rotation_from_uvw(uvw_stored, uvw_new):
    """Least-squares R (3x3) with uvw_new ~= R @ uvw_stored (rows are baselines)."""
    return np.linalg.lstsq(uvw_stored, uvw_new, rcond=None)[0].T


def snapshot_rotation(uvw_stored, a1, a2, ant_pos, time_s, ra, dec):
    """R for one snapshot so that R @ uvw_stored == chgcentre(ra, dec) uvw."""
    A = measures_uvw(ant_pos, time_s, ra, dec)
    return rotation_from_uvw(uvw_stored, A[a1] - A[a2])


def rotate_samples(u, v, w, R, vis=None, xp=np):
    """Apply R to per-sample (u, v, w) in wavelengths and re-phase vis (chgcentre).

    Works with numpy or cupy arrays (``xp``).  Returns (u', v', w', vis')."""
    un = R[0, 0] * u + R[0, 1] * v + R[0, 2] * w
    vn = R[1, 0] * u + R[1, 1] * v + R[1, 2] * w
    wn = R[2, 0] * u + R[2, 1] * v + R[2, 2] * w
    if vis is not None:
        vis = vis * xp.exp(2j * np.pi * (w - wn)).astype(vis.dtype)
    return un, vn, wn, vis


# ---------------------------------------------------------------- analytic frames (tests, docs)
def uvw_frame(ra, dec):
    """Rows: unit vectors of the (u, v, w) axes for phase centre (ra, dec), in equatorial xyz."""
    sa, ca, sd, cd = np.sin(ra), np.cos(ra), np.sin(dec), np.cos(dec)
    return np.array([[-sa, ca, 0.0],
                     [-sd * ca, -sd * sa, cd],
                     [cd * ca, cd * sa, sd]])


def analytic_rotation(ra0, dec0, ra1, dec1):
    """Geometric R (no precession etc.) for a phase-centre change (ra0, dec0) -> (ra1, dec1)."""
    return uvw_frame(ra1, dec1) @ uvw_frame(ra0, dec0).T
