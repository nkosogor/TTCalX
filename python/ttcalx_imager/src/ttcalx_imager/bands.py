"""Per-band image geometry and the main-pipeline (orca) imaging presets.

Geometry tiers are orca ``get_pixel_size`` / ``get_pixel_scale``.  WSClean rounds an odd
``-size`` up to the next even number (2357 -> 2358), so the actual image is even-sized and
its reference pixel is N/2 + 1 (FITS, 1-based).
"""
import re
from dataclasses import dataclass, replace

# (lowest band MHz, highest band MHz, orca -size, -scale [deg])
PIXEL_TIERS = [
    (18, 36, 1507, 0.0767),
    (41, 59, 2357, 0.049),
    (64, 82, 3122, 0.037),
]


def band_mhz(band):
    """'41MHz', '41', 41, 41.0 or a path containing '_41MHz' -> 41."""
    if isinstance(band, (int, float)):
        return int(round(band))
    m = re.search(r"(\d+)\s*MHz", str(band)) or re.fullmatch(r"\s*(\d+)\s*", str(band))
    if not m:
        raise ValueError(f"cannot parse band from {band!r}")
    return int(m.group(1))


def orca_size_scale(band):
    """orca (-size, -scale) for a band; bands above the last tier use the last tier (as orca)."""
    b = band_mhz(band)
    for lo, hi, n, s in PIXEL_TIERS:
        if lo <= b <= hi:
            return n, s
    return PIXEL_TIERS[-1][2], PIXEL_TIERS[-1][3]


def wsclean_size(n):
    """WSClean image size for a requested -size n (odd sizes are rounded up)."""
    return n + (n % 2)


def geometry(band):
    """(N, scale_deg) of the image WSClean actually produces for this band."""
    n, s = orca_size_scale(band)
    return wsclean_size(n), s


@dataclass(frozen=True)
class CleanParams:
    """Subset of WSClean options this imager emulates (names follow the WSClean flags)."""
    niter: int = 50000
    gain: float = 0.1
    mgain: float = 0.9999
    auto_mask: float = 5.0
    auto_threshold: float = 1.0
    local_rms: bool = True
    local_rms_window: float = 25.0      # WSClean default (-local-rms-window), in synthesized beams
    horizon_deg: float = 10.0           # -horizon-mask 10deg
    robust: float = 0.0                 # -weight briggs 0
    taper_inner_tukey: float = 30.0     # -taper-inner-tukey 30 (lambda)
    multiscale: bool = False
    multiscale_scale_bias: float = 0.8
    nmajor: int = 2                     # snapshot only: fixed number of GPU major cycles

    def with_(self, **kw):
        return replace(self, **kw)


# orca SNAPSHOT_CLEAN_I_PARAMS: -pol I -niter 50000 -mgain 0.9999 -auto-mask 5 -auto-threshold 1
#   -local-rms -horizon-mask 10deg -taper-inner-tukey 30 -weight briggs 0
SNAPSHOT_CLEAN_I = CleanParams()

# orca IMAGING_STEPS 'I-Deep-Taper-Robust-0': -multiscale -multiscale-scale-bias 0.8 -niter 500000
#   -mgain 0.95 -horizon-mask 10deg -auto-threshold 0.5 -auto-mask 3 -local-rms
#   -taper-inner-tukey 30 -weight briggs 0
DEEP_I = CleanParams(niter=500000, mgain=0.95, auto_mask=3.0, auto_threshold=0.5, multiscale=True)

# orca 'I-Taper-10min': as DEEP_I with -niter 50000 -intervals-out 6
TENMIN_I = DEEP_I.with_(niter=50000)
TENMIN_INTERVALS = 6

# Dirty Stokes V (pilot snapshots, V-Taper-Deep, V-Taper-10min): same weighting, -niter 0.


def hour_phase_centre(lst_hour):
    """orca hour phase centre: '<LST>h30m00s 37d12m57.057s' (OVRO-LWA latitude)."""
    return f"{int(lst_hour):02d}h30m00s 37d12m57.057s"
