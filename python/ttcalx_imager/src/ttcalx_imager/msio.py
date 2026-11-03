"""Measurement-set reading for single-time snapshot MSes (OVRO-LWA, linear feeds, one SPW).

Stokes conventions (verified against ``wsclean -pol IV`` on our data):
    I = (XX + YY) / 2
    V = i (YX - XY) / 2            (correlation order XX, XY, YX, YY)
A sample enters Stokes I *and* V when both XX and YY are unflagged (WSClean flags a Stokes-I
sample if any contributing correlation is flagged; we use the same mask for V so that I and V
share one set of imaging weights).  Natural weight = (W_XX + W_YY) / 2 from the WEIGHT column
(equal per channel).  Autocorrelations are dropped.

This module needs only numpy + python-casacore (no GPU).
"""
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

C = 299792458.0


@dataclass
class MSData:
    path: str
    uvw: np.ndarray            # (nrow, 3) float64 metres, cross-correlations only
    a1: np.ndarray
    a2: np.ndarray
    visI: np.ndarray           # (nrow, nchan) complex64
    visV: Optional[np.ndarray]
    ok: np.ndarray             # (nrow, nchan) bool, XX and YY unflagged
    natw: np.ndarray           # (nrow,) float32 natural weight per row
    freqs: np.ndarray          # (nchan,) Hz
    chan_width: np.ndarray
    phase_dir: tuple           # (ra, dec) rad, J2000
    ant_pos: np.ndarray        # (nant, 3) ITRF metres
    time: float                # TIME of the first row [MJD s, UTC]
    tstart: float
    tmid: float
    telescope: str = "OVRO-LWA"
    observer: str = ""
    field_name: str = ""
    zero_frac: float = 0.0     # fraction of unflagged I samples with XX == YY == 0 exactly
    extra: dict = field(default_factory=dict)

    @property
    def flag_frac(self):
        return float(1.0 - self.ok.mean()) if self.ok.size else 1.0

    @property
    def stem(self):
        import os
        return os.path.basename(self.path.rstrip("/")).rsplit(".ms", 1)[0]


def _subtable_col(path, sub, col, default=None):
    from casacore.tables import table
    try:
        t = table(f"{path}/{sub}", ack=False)
        v = t.getcol(col)
        t.close()
        return v
    except Exception:
        return default


def msdata_from_arrays(path, a1, a2, uvw, data, flag, wcol, tcol, freqs, chan_width, phase_dir, ant_pos,
                       telescope="OVRO-LWA", observer="", field_name="", want_v=True):
    """:class:`MSData` from the main-table columns of a single-SPW snapshot (all rows, autos
    included, MS row order) and its subtable values.

    a1, a2: ANTENNA1/2; uvw (nrow, 3); data (nrow, nchan, 4) complex64; flag (nrow, nchan, 4)
    bool; wcol: WEIGHT (nrow, 4); tcol: TIME (nrow,); freqs, chan_width: (nchan,) Hz;
    phase_dir: (ra, dec) rad; ant_pos (nant, 3) ITRF m.  These are the expressions
    :func:`read_ms` has always used, so an in-memory snapshot (gsi v3 SnapArrays) and the MS
    written from it give identical arrays."""
    keep = a1 != a2
    a1 = a1[keep]; a2 = a2[keep]; uvw = uvw[keep]; data = data[keep]; flag = flag[keep]; wcol = wcol[keep]
    ok = ~(flag[:, :, 0] | flag[:, :, 3])
    visI = (0.5 * (data[:, :, 0] + data[:, :, 3])).astype(np.complex64)
    visV = (0.5j * (data[:, :, 2] - data[:, :, 1])).astype(np.complex64) if want_v else None
    nok = int(ok.sum())
    zero = ((data[:, :, 0] == 0) & (data[:, :, 3] == 0) & ok).sum()
    return MSData(path=path, uvw=uvw.astype(np.float64), a1=a1, a2=a2, visI=visI, visV=visV, ok=ok,
                  natw=(0.5 * (wcol[:, 0] + wcol[:, 3])).astype(np.float32),
                  freqs=np.asarray(freqs, np.float64), chan_width=np.asarray(chan_width, np.float64),
                  phase_dir=(float(phase_dir[0]), float(phase_dir[1])), ant_pos=np.asarray(ant_pos, np.float64),
                  time=float(tcol[0]), tstart=float(tcol.min()), tmid=float(tcol.mean()),
                  telescope=telescope, observer=observer, field_name=field_name,
                  zero_frac=float(zero / nok) if nok else 1.0,
                  extra=dict(n_v_only_flag=int((ok & (flag[:, :, 1] | flag[:, :, 2])).sum())))


def read_ms(path, want_v=True, column="DATA"):
    """Read a single-SPW snapshot MS into an :class:`MSData`."""
    from casacore.tables import table
    t = table(path, ack=False)
    a1 = t.getcol("ANTENNA1"); a2 = t.getcol("ANTENNA2")
    uvw = t.getcol("UVW"); data = t.getcol(column); flag = t.getcol("FLAG"); wcol = t.getcol("WEIGHT")
    tcol = t.getcol("TIME")
    t.close()
    freqs = _subtable_col(path, "SPECTRAL_WINDOW", "CHAN_FREQ")[0]
    chw = _subtable_col(path, "SPECTRAL_WINDOW", "CHAN_WIDTH")[0]
    pdir = _subtable_col(path, "FIELD", "PHASE_DIR")[0, 0]
    names = _subtable_col(path, "FIELD", "NAME", [""])
    pos = _subtable_col(path, "ANTENNA", "POSITION")
    tel = _subtable_col(path, "OBSERVATION", "TELESCOPE_NAME", ["OVRO-LWA"])
    obs = _subtable_col(path, "OBSERVATION", "OBSERVER", [""])
    return msdata_from_arrays(path, a1, a2, uvw, data, flag, wcol, tcol, freqs, chw, pdir, pos,
                              telescope=str(tel[0]) if len(tel) else "OVRO-LWA",
                              observer=str(obs[0]) if len(obs) else "",
                              field_name=str(names[0]) if len(names) else "", want_v=want_v)


def is_empty(ms, zero_limit=0.999):
    """True if a snapshot has no usable data (everything flagged or (almost) all exactly zero)."""
    return (not ms.ok.any()) or ms.zero_frac >= zero_limit


def expand_coord(uvw, k, freqs, ok, xp=np):
    """Per-kept-sample coordinate k (0 = u, 1 = v, 2 = w) in wavelengths, float32: the
    load_samples expression, from per-row uvw [m] (nrow, 3), freqs (nchan,) Hz and the
    (nrow, nchan) kept mask.  One float64 multiply and a cast, so numpy and CuPy agree bitwise."""
    lamf = freqs / C
    return (uvw[:, k:k + 1] * lamf[None])[ok].astype(xp.float32)


def expand_nw(natw, nchan, ok, xp=np):
    """Per-kept-sample natural weight (float32) from the per-row natw (nrow,)."""
    return (natw[:, None] * xp.ones((1, nchan)))[ok].astype(xp.float32)


def samples_from_msdata(ms, ra=None, dec=None, want_v=True, compact=False):
    """Kept-sample arrays of one snapshot (:class:`MSData`) for the hour imager.

    u, v, w are in wavelengths (float32) in the *stored* (snapshot phase centre) frame; R is the
    3x3 matrix with uvw_hour = R @ uvw_stored (chgcentre-exact, see rotate.py), identity if no
    target phase centre is given.

    compact=True (gsi v3 hour record): instead of u, v, w, nw it returns the per-row arrays they
    are expanded from -- uvw (nrow, 3) float64, natw (nrow,) float32, ok (nrow, nchan) bool -- plus
    n (number of kept samples), a1, a2, ant_pos, tmid and extra; :func:`expand_coord` / :func:`expand_nw`
    rebuild u, v, w, nw bit-identically.  I and V are the same arrays in both forms."""
    R = np.eye(3)
    if ra is not None:
        from .rotate import snapshot_rotation
        R = snapshot_rotation(ms.uvw, ms.a1, ms.a2, ms.ant_pos, ms.time, ra, dec)
    ok = ms.ok
    out = dict(path=ms.path, stem=ms.stem, time=ms.time, tstart=ms.tstart, freqs=ms.freqs, chan_width=ms.chan_width, pd=ms.phase_dir, R=R.astype(np.float64),
               pd_new=(ra, dec), nrow=int(ms.uvw.shape[0]), nchan=len(ms.freqs), flagfrac=ms.flag_frac,
               zero_frac=ms.zero_frac, telescope=ms.telescope, observer=ms.observer, field_name=ms.field_name)
    if compact:
        out.update(uvw=ms.uvw, natw=ms.natw, ok=ok, n=int(ok.sum()), a1=ms.a1, a2=ms.a2, ant_pos=ms.ant_pos, tmid=ms.tmid,
                   extra=dict(ms.extra))
    else:
        out["u"] = expand_coord(ms.uvw, 0, ms.freqs, ok)
        out["v"] = expand_coord(ms.uvw, 1, ms.freqs, ok)
        out["w"] = expand_coord(ms.uvw, 2, ms.freqs, ok)
        out["nw"] = expand_nw(ms.natw, len(ms.freqs), ok)
    out["I"] = ms.visI[ok]
    if want_v:
        out["V"] = ms.visV[ok]
    return out


def load_samples(path, ra=None, dec=None, want_v=True):
    """Kept-sample arrays of one snapshot MS for the hour imager (picklable; runs in worker
    processes).  = samples_from_msdata(read_ms(path)), see there."""
    return samples_from_msdata(read_ms(path, want_v=want_v), ra, dec, want_v)
