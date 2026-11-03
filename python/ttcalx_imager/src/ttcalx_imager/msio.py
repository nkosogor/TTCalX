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
    keep = a1 != a2
    a1 = a1[keep]; a2 = a2[keep]; uvw = uvw[keep]; data = data[keep]; flag = flag[keep]; wcol = wcol[keep]
    ok = ~(flag[:, :, 0] | flag[:, :, 3])
    visI = (0.5 * (data[:, :, 0] + data[:, :, 3])).astype(np.complex64)
    visV = (0.5j * (data[:, :, 2] - data[:, :, 1])).astype(np.complex64) if want_v else None
    nok = int(ok.sum())
    zero = ((data[:, :, 0] == 0) & (data[:, :, 3] == 0) & ok).sum()
    return MSData(path=path, uvw=uvw.astype(np.float64), a1=a1, a2=a2, visI=visI, visV=visV, ok=ok,
                  natw=(0.5 * (wcol[:, 0] + wcol[:, 3])).astype(np.float32),
                  freqs=np.asarray(freqs, np.float64), chan_width=np.asarray(chw, np.float64),
                  phase_dir=(float(pdir[0]), float(pdir[1])), ant_pos=np.asarray(pos, np.float64),
                  time=float(tcol[0]), tstart=float(tcol.min()), tmid=float(tcol.mean()),
                  telescope=str(tel[0]) if len(tel) else "OVRO-LWA", observer=str(obs[0]) if len(obs) else "",
                  field_name=str(names[0]) if len(names) else "",
                  zero_frac=float(zero / nok) if nok else 1.0,
                  extra=dict(n_v_only_flag=int((ok & (flag[:, :, 1] | flag[:, :, 2])).sum())))


def is_empty(ms, zero_limit=0.999):
    """True if a snapshot has no usable data (everything flagged or (almost) all exactly zero)."""
    return (not ms.ok.any()) or ms.zero_frac >= zero_limit


def load_samples(path, ra=None, dec=None, want_v=True):
    """Kept-sample arrays of one snapshot for the hour imager (picklable; runs in worker processes).

    u, v, w are in wavelengths (float32) in the *stored* (snapshot phase centre) frame; R is the
    3x3 matrix with uvw_hour = R @ uvw_stored (chgcentre-exact, see rotate.py), identity if no
    target phase centre is given.
    """
    ms = read_ms(path, want_v=want_v)
    R = np.eye(3)
    if ra is not None:
        from .rotate import snapshot_rotation
        R = snapshot_rotation(ms.uvw, ms.a1, ms.a2, ms.ant_pos, ms.time, ra, dec)
    ok = ms.ok
    lamf = ms.freqs / C
    out = dict(path=path, stem=ms.stem, time=ms.time, tstart=ms.tstart, freqs=ms.freqs, chan_width=ms.chan_width, pd=ms.phase_dir, R=R.astype(np.float64),
               pd_new=(ra, dec), nrow=int(ms.uvw.shape[0]), nchan=len(ms.freqs), flagfrac=ms.flag_frac,
               zero_frac=ms.zero_frac, telescope=ms.telescope, observer=ms.observer, field_name=ms.field_name)
    out["u"] = (ms.uvw[:, 0:1] * lamf[None])[ok].astype(np.float32)
    out["v"] = (ms.uvw[:, 1:2] * lamf[None])[ok].astype(np.float32)
    out["w"] = (ms.uvw[:, 2:3] * lamf[None])[ok].astype(np.float32)
    out["nw"] = (ms.natw[:, None] * np.ones((1, len(ms.freqs))))[ok].astype(np.float32)
    out["I"] = ms.visI[ok]
    if want_v:
        out["V"] = ms.visV[ok]
    return out
