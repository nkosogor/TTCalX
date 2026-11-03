"""A tiny single-SPW, single-time snapshot MS (casacore) for CPU tests of the MS readers."""
import numpy as np


def snapshot_arrays(nant=7, nchan=5, seed=0, t0=5.2995e9, flag_frac=0.3):
    """Main-table columns of one time step, autos included, in MS row order (i <= j)."""
    rng = np.random.default_rng(seed)
    a1, a2 = (np.array(x, np.int32) for x in zip(*[(i, j) for i in range(nant) for j in range(i, nant)]))
    nrow = len(a1)
    pos = rng.normal(0, 200, (nant, 3)) + np.array([-2409150.0, -4478573.0, 3838617.0])
    uvw = (pos[a1] - pos[a2]) @ np.linalg.qr(rng.normal(size=(3, 3)))[0]
    data = (rng.normal(size=(nrow, nchan, 4)) + 1j * rng.normal(size=(nrow, nchan, 4))).astype(np.complex64)
    data[2, 1, :] = 0                                     # an exactly-zero sample (zero_frac)
    flag = rng.random((nrow, nchan, 4)) < flag_frac
    flag[1] = True                                        # a fully flagged row
    flag[2, 1, [0, 3]] = False                            # the zero sample is kept (counts in zero_frac)
    flag[3, :, 1] = True                                  # V-only flags (XY flagged, XX/YY fine)
    weight = rng.uniform(0.5, 2.0, (nrow, 4)).astype(np.float32)
    time = np.full(nrow, t0)
    freqs = 40e6 + 23925.78125 * 4 * np.arange(nchan)
    return dict(a1=a1, a2=a2, uvw=uvw, data=data, flag=flag, weight=weight, time=time, freqs=freqs,
                chan_width=np.full(nchan, 23925.78125 * 4), phase_dir=(1.234, 0.649), ant_pos=pos)


def make_ms(path, **kw):
    """Write snapshot_arrays(**kw) as an MS; returns the arrays."""
    from casacore.tables import default_ms, makearrcoldesc, maketabdesc, table
    d = snapshot_arrays(**kw)
    nrow, nchan = d["data"].shape[:2]
    t = default_ms(str(path))
    t.addcols(maketabdesc(makearrcoldesc("DATA", 0j, shape=[nchan, 4], valuetype="complex")))
    t.addrows(nrow)
    t.putcol("ANTENNA1", d["a1"]); t.putcol("ANTENNA2", d["a2"])
    t.putcol("UVW", d["uvw"]); t.putcol("DATA", d["data"]); t.putcol("FLAG", d["flag"])
    t.putcol("WEIGHT", d["weight"]); t.putcol("SIGMA", 1 / np.sqrt(d["weight"]))
    t.putcol("TIME", d["time"]); t.putcol("TIME_CENTROID", d["time"])
    t.close()
    a = table(f"{path}/ANTENNA", readonly=False, ack=False)
    a.addrows(len(d["ant_pos"])); a.putcol("POSITION", d["ant_pos"]); a.close()
    s = table(f"{path}/SPECTRAL_WINDOW", readonly=False, ack=False)
    s.addrows(1); s.putcell("CHAN_FREQ", 0, d["freqs"]); s.putcell("CHAN_WIDTH", 0, d["chan_width"])
    s.putcell("NUM_CHAN", 0, nchan); s.close()
    f = table(f"{path}/FIELD", readonly=False, ack=False)
    f.addrows(1); f.putcell("PHASE_DIR", 0, np.array([d["phase_dir"]])); f.putcell("NAME", 0, "zenith"); f.close()
    o = table(f"{path}/OBSERVATION", readonly=False, ack=False)
    o.addrows(1); o.putcell("TELESCOPE_NAME", 0, "OVRO-LWA"); o.putcell("OBSERVER", 0, "test"); o.close()
    return d
