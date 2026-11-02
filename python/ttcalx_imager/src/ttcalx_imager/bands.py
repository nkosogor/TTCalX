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
    auto_mask: float = 5.0              # 0 = off: no auto-mask phase, clean the whole sky (as WSClean)
    auto_threshold: float = 1.0         # 0 = off (WSClean: option not given)
    threshold: float = 0.0              # absolute stopping threshold [Jy] (WSClean -threshold); 0 = none
    fits_mask: str = None               # clean mask FITS on the image grid (WSClean -fits-mask): all scales
    local_rms: bool = True
    local_rms_window: float = 25.0      # WSClean default (-local-rms-window), in synthesized beams
    horizon_deg: float = 10.0           # -horizon-mask 10deg
    robust: float = 0.0                 # -weight briggs 0
    taper_inner_tukey: float = 30.0     # -taper-inner-tukey 30 (lambda)
    multiscale: bool = False
    multiscale_scale_bias: float = 0.8
    multiscale_max_scales: int = 0      # WSClean -multiscale-max-scales (number of scales incl. 0); 0 = unlimited
    divergence_limit: float = 4.0       # WSClean -divergence-limit: stop when a peak grows this much; 0 = off
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

# orca 'I-Deep-Taper-Robust-0.75': as DEEP_I with -weight briggs -0.75
DEEP_I_R075 = DEEP_I.with_(robust=-0.75)
# orca 'I-Deep-NoTaper-Robust-0.75' / '-0': no -taper-inner-tukey, -niter 150000
DEEP_I_NOTAPER_R075 = DEEP_I.with_(robust=-0.75, taper_inner_tukey=0.0, niter=150000)
DEEP_I_NOTAPER_R0 = DEEP_I.with_(taper_inner_tukey=0.0, niter=150000)


@dataclass(frozen=True)
class HourProduct:
    """One imaging configuration of ``image_hour``.

    tag: name of the I product (file stem part); n_intervals: 1 (1 h) or e.g. 6 (10 min);
    params: weighting + CLEAN settings; image_i: make the CLEANed I image; v_tag: also make
    a dirty V image with the same weighting under this name (None: no V).  naming="orca":
    files <prefix>-<tag>[-tNNNN]-{image,residual,model,dirty,psf}.fits and
    <prefix>-<v_tag>[-tNNNN]-image.fits as WSClean writes them for the orca steps;
    naming="legacy": <prefix>-<tag>[-tNNNN]-{I-image,...,V-image}.fits.
    """
    tag: str
    n_intervals: int
    params: CleanParams
    image_i: bool = True
    v_tag: str = None
    naming: str = "orca"


# orca IMAGING_STEPS (subband_config.py): I name -> (intervals, params); V name -> I config
# whose weighting it shares (briggs 0, inner Tukey 30).
ORCA_I_PRODUCTS = {
    "I-Deep-Taper-Robust-0.75": (1, DEEP_I_R075),
    "I-Deep-Taper-Robust-0": (1, DEEP_I),
    "I-Deep-NoTaper-Robust-0.75": (1, DEEP_I_NOTAPER_R075),
    "I-Deep-NoTaper-Robust-0": (1, DEEP_I_NOTAPER_R0),
    "I-Taper-10min": (TENMIN_INTERVALS, TENMIN_I),
}
ORCA_V_PRODUCTS = {"V-Taper-Deep": "I-Deep-Taper-Robust-0", "V-Taper-10min": "I-Taper-10min"}
ORCA_ALL = list(ORCA_I_PRODUCTS) + list(ORCA_V_PRODUCTS)


def hour_products(names, intervals=TENMIN_INTERVALS, niter=0, niter_10min=0, pols="IV"):
    """Product names -> [HourProduct].

    Names: orca step names (ORCA_ALL, or 'orca' for all of them), or the legacy tags '1h'
    (DEEP_I) and '10min' (TENMIN_I, ``intervals``), whose V follows ``pols``.  A V product
    is made together with the I product of the same weighting; a V product requested
    without that I product gets a V-only configuration (no CLEAN).  niter / niter_10min
    override the legacy tags only."""
    names = [n.strip() for n in names if n.strip()]
    if "orca" in names:
        names = [n for n in names if n != "orca"] + [n for n in ORCA_ALL if n not in names]
    unknown = [n for n in names if n not in ORCA_ALL and n not in ("1h", "10min")]
    if unknown:
        raise ValueError(f"unknown hour product(s) {unknown}; known: {ORCA_ALL + ['1h', '10min']}")
    out = []
    for n in names:
        if n == "1h":
            out.append(HourProduct("1h", 1, DEEP_I.with_(niter=niter or DEEP_I.niter),
                                   v_tag="V" if "V" in pols else None, naming="legacy"))
        elif n == "10min":
            out.append(HourProduct("10min", intervals, TENMIN_I.with_(niter=niter_10min or TENMIN_I.niter),
                                   v_tag="V" if "V" in pols else None, naming="legacy"))
        elif n in ORCA_I_PRODUCTS:
            nint, params = ORCA_I_PRODUCTS[n]
            v = [vn for vn, i_name in ORCA_V_PRODUCTS.items() if i_name == n and vn in names]
            out.append(HourProduct(n, nint, params, v_tag=v[0] if v else None))
    for vn, i_name in ORCA_V_PRODUCTS.items():
        if vn in names and i_name not in names:
            nint, params = ORCA_I_PRODUCTS[i_name]
            out.append(HourProduct(i_name, nint, params, image_i=False, v_tag=vn))
    return out


def hour_phase_centre(lst_hour):
    """orca hour phase centre: '<LST>h30m00s 37d12m57.057s' (OVRO-LWA latitude)."""
    return f"{int(lst_hour):02d}h30m00s 37d12m57.057s"
