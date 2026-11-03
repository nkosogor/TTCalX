"""CPU tests of the pieces the gsi v3 engine uses (gpu-subband-imaging workers/gsi_v3_engine.py):
chgcentre's phase-centre parse, the chgcentre re-phasing, the in-memory hour run, lazy_snaps."""
import importlib
import math
import os
import sys

import numpy as np
import pytest

from ttcalx_imager import gsiv, rotate
from ttcalx_imager.msio import C, msdata_from_arrays, samples_from_msdata

sys.path.insert(0, os.path.dirname(__file__))

# PHASE_DIR written by /opt/bin/chgcentre (wsclean chgcentre) for <h>h30m00s 37d12m57.057s,
# measured on a calim CPU node on 2026-11-01 (gpu-subband-imaging scripts/v3_engine/chgcentre_check.py)
CHG_RA = [0.13089969389957473, 0.39269908169872414, 0.6544984694978736, 0.916297857297023,
          1.1780972450961724, 1.439896632895322, 1.7016960206944713, 1.9634954084936207,
          2.2252947962927703, 2.48709418409192, 2.748893571891069, 3.0106929596902186,
          -3.0106929596902186, -2.748893571891069, -2.4870941840919194, -2.2252947962927707,
          -1.9634954084936205, -1.7016960206944713, -1.4398966328953222, -1.1780972450961722,
          -0.9162978572970231, -0.6544984694978739, -0.3926990816987239, -0.13089969389957481]
CHG_DEC = 0.6495391018839214
X87 = np.finfo(np.longdouble).eps < np.finfo(np.float64).eps


@pytest.mark.parametrize("h", range(24))
def test_parse_phase_centre_chgcentre(h):
    ra, dec = rotate.parse_phase_centre_chgcentre(f"{h:02d}h30m00s 37d12m57.057s")
    if X87:
        assert (ra, dec) == (CHG_RA[h], CHG_DEC)
    else:                                     # long double = double: within 1 ulp
        assert abs(ra - CHG_RA[h]) <= 2 * np.spacing(abs(CHG_RA[h])) and abs(dec - CHG_DEC) <= 2e-16
    a = rotate.parse_phase_centre(f"{h:02d}h30m00s 37d12m57.057s")
    assert abs(math.remainder(a[0] - ra, 2 * math.pi)) < 1e-15 and abs(a[1] - dec) < 1e-15


def test_parse_phase_centre_chgcentre_rejects_other_forms():
    with pytest.raises(ValueError):
        rotate.parse_phase_centre_chgcentre("82.5 37.2")
    ra, dec = rotate.parse_phase_centre_chgcentre("05h30m00s -37d12m57.057s")
    assert dec == pytest.approx(-CHG_DEC, abs=1e-15)


def _snapshot():
    hour_cpu = pytest.importorskip("hour_cpu")
    hour_cpu.import_hour()                                   # cupy stand-in where there is no GPU
    return importlib.import_module("ttcalx_imager.snapshot")


def test_rephase_matches_rotate_samples_convention():
    snapshot = _snapshot()
    rng = np.random.default_rng(1)
    nrow, nchan = 50, 6
    freqs = 50e6 + 1e5 * np.arange(nchan)
    data = (rng.normal(size=(nrow, nchan, 4)) + 1j * rng.normal(size=(nrow, nchan, 4))).astype(np.complex64)
    w0, w1 = rng.uniform(-30, 30, nrow), rng.uniform(-30, 30, nrow)
    out = snapshot.rephase(data, w0, w1, freqs, xp=np)
    assert out.dtype == np.complex64 and out.shape == data.shape
    lam = freqs / C
    ref = (data.astype(np.complex128) * np.exp(2j * np.pi * (w0[:, None] - w1[:, None]) * lam[None])[:, :, None])
    assert np.array_equal(out, ref.astype(np.complex64))
    # the convention of rotate_samples (per-sample w in wavelengths; R with w' = 0 gives
    # vis * exp(2 pi i w)) to float32 rounding
    _, _, _, v = rotate.rotate_samples(np.zeros(nrow * nchan), np.zeros(nrow * nchan), (w0[:, None] * lam).ravel(),
                                       np.diag([1.0, 1.0, 0.0]), data[:, :, 0].ravel().astype(np.complex128))
    out0 = snapshot.rephase(data, w0, np.zeros(nrow), freqs, xp=np)
    assert np.allclose(out0[:, :, 0].ravel(), v, rtol=0, atol=1e-5 * np.abs(v).max())
    assert np.array_equal(snapshot.rephase(data, w0, w0, freqs, xp=np), data)


def _msdata(seed, nant=8, nchan=5, time=5.3e9):
    rng = np.random.default_rng(seed)
    a1, a2 = np.triu_indices(nant)
    nrow = len(a1)
    uvw = rng.normal(size=(nrow, 3)) * 100; uvw[a1 == a2] = 0
    data = (rng.normal(size=(nrow, nchan, 4)) + 1j * rng.normal(size=(nrow, nchan, 4))).astype(np.complex64)
    flag = rng.random((nrow, nchan, 4)) < 0.2
    return msdata_from_arrays(f"/x/20261031_1010{seed:02d}_41MHz.ms", a1.astype(np.int32), a2.astype(np.int32), uvw,
                              data, flag, np.ones((nrow, 4), np.float32), np.full(nrow, time + 10 * seed),
                              40e6 + 1e5 * np.arange(nchan), np.full(nchan, 1e5), (1.3, 0.6),
                              rng.normal(size=(nant, 3)) * 1e6)


def test_lazy_snaps_uses_or_recomputes_R(monkeypatch):
    recs = []
    for s in (2, 0, 1):
        c = samples_from_msdata(_msdata(s), compact=True)
        c.update(R=np.eye(3) * (1 + s), pd_new=(1.3, 0.65))
        meta, arrays = gsiv.record_from_samples(c)
        recs.append((meta, arrays, f"mem:{s}"))
    snaps = gsiv.lazy_snaps(recs, 1.3, 0.65)
    assert [s["stem"] for s in snaps] == [f"20261031_1010{s:02d}_41MHz" for s in (0, 1, 2)]
    assert np.array_equal(snaps[0]["R"], np.eye(3))
    monkeypatch.setattr(rotate, "snapshot_rotation", lambda *a: np.full((3, 3), 7.0))
    other = gsiv.lazy_snaps(recs, 1.2, 0.65, want_v=False)
    assert np.array_equal(other[0]["R"], np.full((3, 3), 7.0)) and "V" not in other[0]


def test_run_hour_snaps_equals_cli_archive(tmp_path, monkeypatch):
    """The in-memory hour run passes image_hour_snaps exactly what the CLI's --archive path does."""
    hour_cpu = pytest.importorskip("hour_cpu")
    hour = hour_cpu.import_hour()
    from ttcalx_imager import cli
    calls = []
    monkeypatch.setattr(hour, "image_hour_snaps", lambda snaps, band, out, ra, dec, **kw:
                        calls.append(dict(stems=[s["stem"] for s in snaps], band=band, out=out, ra=ra, dec=dec,
                                          kw={k: v for k, v in kw.items() if k not in ("timing", "t0")}))
                        or dict(rejected=[], timing={}, products={}))
    monkeypatch.setattr(rotate, "snapshot_rotation", lambda *a: np.eye(3))
    recs = []
    for s in (0, 1, 2):
        c = samples_from_msdata(_msdata(s), compact=True)
        c.update(R=np.eye(3), pd_new=rotate.parse_phase_centre("04h30m00s 37d12m57.057s"))
        recs.append(gsiv.record_from_samples(c))
    p = tmp_path / "a.gsiv"
    w = gsiv.RecordWriter(p, dict(mode="iv"), {k: recs[0][1][k] for k in gsiv.SHARED_ARRAYS})
    for meta, arrays in recs:
        w.append(meta, arrays)
    w.close()
    args = ["--band", "41", "--out", str(tmp_path / "o" / "41MHz"), "--lst-hour", "4", "--qa", "none",
            "--products", "I-Deep-Taper-Robust-0,V-Taper-Deep", "--exclude-stems", "20261031_101001_41MHz", "-v"]
    from ttcalx_imager import FAST_ENV

    def unset_fast():
        for k in FAST_ENV:
            monkeypatch.setenv(k, "x"); monkeypatch.delenv(k)     # absent, and absent again after the test
    unset_fast()
    cli.main(["hour", "--archive", str(p)] + args)
    assert {k: os.environ.get(k) for k in FAST_ENV} == FAST_ENV
    unset_fast()
    got = {}
    cli.run_hour_snaps(args, lambda ra, dec, want_v: got.update(ra=ra, dec=dec, want_v=want_v) or
                       gsiv.lazy_snaps([(m, a, "mem") for m, a in recs], ra, dec, want_v))
    assert {k: os.environ.get(k) for k in FAST_ENV} == FAST_ENV       # the v3 entry points: all fast settings
    unset_fast()
    assert calls[0] == calls[1] and got["want_v"] is True
    assert calls[1]["kw"]["exclude_stems"] == ("20261031_101001_41MHz",)
    with pytest.raises(ValueError):
        cli.run_hour_snaps(["--archive", str(p)] + args, lambda *a: [])
    with pytest.raises(ValueError):
        cli.run_hour_snaps(["--band", "41", "--out", str(tmp_path / "x")], lambda *a: [])


def test_rotate_rephase_is_the_snapshot_rephase():
    snapshot = _snapshot()
    rng = np.random.default_rng(2)
    data = (rng.normal(size=(9, 3, 4)) + 1j * rng.normal(size=(9, 3, 4))).astype(np.complex64)
    w0, w1, f = rng.normal(size=9) * 10, rng.normal(size=9) * 10, 4e7 + np.arange(3) * 1e5
    assert np.array_equal(rotate.rephase(data, w0, w1, f), snapshot.rephase(data, w0, w1, f, xp=np))
