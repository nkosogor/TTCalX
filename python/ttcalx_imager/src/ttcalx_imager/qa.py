"""Snapshot QA rejection (orca analyze_snapshot_quality semantics), CPU only.

A snapshot is rejected when the std of the central 1024 x 1024 pixels of its dirty Stokes V
image is > median + 3 * MAD-std over the hour, or < 0.5 * median (dead / zero data).
"""
import numpy as np

MAD_TO_STD = 1.482602218505602     # astropy.stats.mad_std


def mad_std(x):
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    return float(MAD_TO_STD * np.median(np.abs(x - np.median(x)))) if x.size else float("nan")


def qa_reject(v_rms, nsigma=3.0, low_frac=0.5):
    """Indices of rejected snapshots and the (median, mad_std) used."""
    r = np.asarray(v_rms, float)
    med = float(np.nanmedian(r)); sd = mad_std(r)
    bad = [i for i, x in enumerate(r) if not np.isfinite(x) or x > med + nsigma * sd or x < low_frac * med]
    return bad, med, sd
