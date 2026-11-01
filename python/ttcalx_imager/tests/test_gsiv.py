"""CPU tests of the gsi v3 archive path: msio refactor, GSIV round trip / torn tail / crc,
load_archive == load_samples bitwise, LazySnap, and image_hour_snaps on the CPU (hour_cpu)."""
import json
import os

import numpy as np
import pytest

from ttcalx_imager import bands, gsiv
from ttcalx_imager.msio import (expand_coord, expand_nw, load_samples, msdata_from_arrays, read_ms,
                                samples_from_msdata)

from msmaker import make_ms, snapshot_arrays

KEYS = ("path", "stem", "time", "tstart", "freqs", "chan_width", "pd", "R", "pd_new", "nrow", "nchan",
        "flagfrac", "zero_frac", "telescope", "observer", "field_name", "u", "v", "w", "nw", "I", "V")

try:
    import casacore.tables  # noqa: F401
    CASACORE = True
except Exception:
    CASACORE = False
needs_casacore = pytest.mark.skipif(not CASACORE, reason="needs python-casacore")


def same(a, b):
    """Bit-identical (arrays: dtype, shape, bytes; floats: IEEE bits; tuples elementwise)."""
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        a, b = np.asarray(a), np.asarray(b)
        return a.dtype == b.dtype and a.shape == b.shape and a.tobytes() == b.tobytes()
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(same(x, y) for x, y in zip(a, b))
    if isinstance(a, float):
        return isinstance(b, float) and np.float64(a).tobytes() == np.float64(b).tobytes()
    return type(a) is type(b) and a == b


def assert_same_samples(a, b, keys=KEYS):
    bad = [k for k in keys if (k in a) != (k in b) or (k in a and not same(a[k], b[k]))]
    assert not bad, bad


def msdata(seed=0, path="/x/20261031_101007_41MHz.ms", **kw):
    d = snapshot_arrays(seed=seed, **kw)
    return msdata_from_arrays(path, d["a1"], d["a2"], d["uvw"], d["data"], d["flag"], d["weight"], d["time"],
                              d["freqs"], d["chan_width"], d["phase_dir"], d["ant_pos"], observer="test",
                              field_name="zenith")


def archive(tmp_path, mss, R=None, name="a.gsiv", pd_new=(1.3, 0.65)):
    """GSIV of MSData objects with a given R (no casacore needed)."""
    p = tmp_path / name
    w = None
    for ms in mss:
        c = samples_from_msdata(ms, compact=True)
        c["R"] = np.eye(3) if R is None else R
        c["pd_new"] = pd_new
        meta, arrays = gsiv.record_from_samples(c)
        if w is None:
            w = gsiv.RecordWriter(p, dict(mode="iv", pd_new=list(pd_new)), {k: arrays[k] for k in gsiv.SHARED_ARRAYS})
        w.append(meta, arrays)
    w.close()
    return p


def reference(ms, R, pd_new):
    s = samples_from_msdata(ms)
    s["R"] = R; s["pd_new"] = pd_new
    return s


# ----------------------------------------------------------------------------- msio refactor
def test_compact_expands_to_load_samples_layout():
    ms = msdata()
    a = samples_from_msdata(ms)
    c = samples_from_msdata(ms, compact=True)
    assert c["n"] == len(a["u"]) == int(ms.ok.sum())
    for k in "uvw":
        assert same(expand_coord(c["uvw"], "uvw".index(k), c["freqs"], c["ok"]), a[k])
    assert same(expand_nw(c["natw"], c["nchan"], c["ok"]), a["nw"])
    assert same(c["I"], a["I"]) and same(c["V"], a["V"])
    assert a["zero_frac"] > 0 and 0 < a["flagfrac"] < 1


@needs_casacore
def test_read_ms_equals_msdata_from_arrays(tmp_path):
    p = str(tmp_path / "20261031_101007_41MHz.ms")
    d = make_ms(p)
    ms = read_ms(p)
    ref = msdata_from_arrays(p, d["a1"], d["a2"], d["uvw"], d["data"], d["flag"], d["weight"], d["time"],
                             d["freqs"], d["chan_width"], d["phase_dir"], d["ant_pos"], telescope="OVRO-LWA",
                             observer="test", field_name="zenith")
    for k in ("uvw", "a1", "a2", "visI", "visV", "ok", "natw", "freqs", "chan_width", "ant_pos"):
        assert same(np.asarray(getattr(ms, k)), np.asarray(getattr(ref, k))), k
    for k in ("phase_dir", "time", "tstart", "tmid", "telescope", "observer", "field_name", "zero_frac", "extra"):
        assert getattr(ms, k) == getattr(ref, k), k
    assert_same_samples(load_samples(p), samples_from_msdata(ref))


# ----------------------------------------------------------------------------- format
def test_roundtrip_bitwise_both_backends(tmp_path):
    R = np.array([[0.9, 0.1, 0.0], [-0.1, 0.9, 0.2], [0.0, -0.2, 0.98]])
    mss = [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in (3, 1, 2)]
    p = archive(tmp_path, mss, R)
    for backend in ("ram", "memmap"):
        snaps = gsiv.load_archive(str(p), 1.3, 0.65, backend=backend)
        assert [s["stem"] for s in snaps] == [ms.stem for ms in mss]     # equal times: stable, file order
        for ms in mss:
            s = next(x for x in snaps if x["stem"] == ms.stem)
            assert_same_samples(reference(ms, R, (1.3, 0.65)), s)
            assert set(s.keys()) == set(KEYS) and s.n == len(s["u"])
    # stored R for ra=None, V dropped for want_v=False
    s = gsiv.load_archive(p, want_v=False)[0]
    assert "V" not in s and same(s["R"], R) and s["pd_new"] == (1.3, 0.65)


def test_shared_arrays_stored_once(tmp_path):
    p = archive(tmp_path, [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in range(3)])
    sc = gsiv.scan(p)
    assert set(sc.header_arrays) >= {"a1", "a2", "ant_pos"} and len(sc.entries) == 3
    hm, ha, blocks = gsiv.read_blocks(p)
    for meta, arrays, e in blocks:
        assert same(arrays["a1"], ha["a1"])
    fd = os.open(str(p), os.O_RDONLY)
    try:
        h = gsiv.decode_header(gsiv._pread(fd, 64, sc.entries[0]["offset"]))
        js = json.loads(gsiv._pread(fd, h["meta_len"], sc.entries[0]["offset"] + 64))
    finally:
        os.close(fd)
    names = {t["name"] for t in js["arrays"]}
    assert "a1" not in names and {"I", "V", "uvw", "natw", "ok_bits", "R"} <= names
    assert all(t["offset"] % 64 == 0 for t in js["arrays"])
    assert all(e["offset"] % 64 == 0 and e["nbytes"] % 64 == 0 for e in sc.entries)


def test_torn_tail_detected_and_truncated(tmp_path):
    mss = [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in range(3)]
    p = archive(tmp_path, mss)
    sc = gsiv.scan(p)
    e3 = sc.entries[2]
    with open(p, "r+b") as fh:                                  # crash in the middle of record 3
        fh.truncate(e3["offset"] + e3["nbytes"] // 2)
    sc = gsiv.scan(p)
    assert len(sc.entries) == 2 and sc.torn_bytes == e3["nbytes"] // 2 and sc.idx_dropped == 1
    assert [s["stem"] for s in gsiv.load_archive(p)] == [mss[0].stem, mss[1].stem]
    w = gsiv.RecordWriter(p, dict(mode="iv"))                   # resume: truncate, rewrite index, append
    assert len(w) == 2
    meta, arrays = gsiv.record_from_samples(dict(samples_from_msdata(mss[2], compact=True), pd_new=None))
    w.append(meta, arrays); w.close()
    sc = gsiv.scan(p, verify=True)
    assert sc.clean and len(sc.entries) == 3 and sc.size == sc.end


def test_record_without_index_line_is_recovered(tmp_path):
    p = archive(tmp_path, [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in range(3)])
    idx = gsiv.idx_path(p)
    lines = idx.read_text().splitlines()
    idx.write_text("\n".join(lines[:2]) + "\n" + lines[2][:17])   # crash while writing the 3rd line
    sc = gsiv.scan(p)
    assert len(sc.entries) == 3 and sc.unindexed == 1 and sc.idx_dropped == 1 and sc.torn_bytes == 0
    assert len(gsiv.load_archive(p)) == 3
    gsiv.repair(p)
    assert gsiv.scan(p).clean and len(idx.read_text().splitlines()) == 3


def test_crc_failure(tmp_path):
    p = archive(tmp_path, [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in range(3)])
    e2 = gsiv.scan(p).entries[1]
    with open(p, "r+b") as fh:                                  # flip one bit in record 2's I samples
        fh.seek(e2["offset"] + e2["nbytes"] - 1000)
        b = fh.read(1); fh.seek(-1, 1); fh.write(bytes([b[0] ^ 1]))
    assert gsiv.scan(p, verify=True).bad_crc == [2]
    with pytest.raises(gsiv.GSIVError, match="crc"):
        gsiv.load_archive(p)
    assert len(gsiv.load_archive(p, strict=False)) == 2
    assert len(gsiv.load_archive(p, verify=False)) == 3        # unverified read does not notice
    gsiv.repair(p)                                             # checkpoint semantics: keep the valid prefix
    assert len(gsiv.scan(p, verify=True).entries) == 1


def test_bad_file_header(tmp_path):
    p = archive(tmp_path, [msdata()])
    with open(p, "r+b") as fh:
        fh.seek(100); fh.write(b"\xff")
    with pytest.raises(gsiv.GSIVError, match="header"):
        gsiv.scan(p)
    q = tmp_path / "junk.gsiv"; q.write_bytes(b"\0" * 200)
    with pytest.raises(gsiv.GSIVError):
        gsiv.load_archive(q)


def test_multiple_files_and_duplicates(tmp_path):
    a = archive(tmp_path, [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in (0, 1)], name="b0.gsiv")
    b = archive(tmp_path, [msdata(seed=s, path=f"/x/20261031_1010{s:02d}_41MHz.ms") for s in (1, 2)], name="b1.gsiv")
    snaps = gsiv.load_archive(f"{a},{b}")
    assert [s["stem"] for s in snaps] == [f"20261031_1010{s:02d}_41MHz" for s in (0, 1, 2)]
    lst = tmp_path / "l.txt"; lst.write_text(f"{a}\n{b}\n")
    assert len(gsiv.load_archive(f"@{lst}")) == 3


# ----------------------------------------------------------------------------- LazySnap
def test_lazysnap_mapping(tmp_path):
    p = archive(tmp_path, [msdata()], R=np.eye(3))
    s = gsiv.load_archive(p, 1.3, 0.65)[0]
    assert "u" in s and "I" in s and "tidx" not in s
    s["tidx"] = 4
    assert s["tidx"] == 4 and "tidx" in s and "tidx" in list(s)
    d = s.derive(R=2 * np.eye(3), R_hour=np.eye(3))
    assert same(d["R"], 2 * np.eye(3)) and same(s["R"], np.eye(3)) and d["tidx"] == 4
    assert d.arrays is s.arrays and d.n == s.n
    assert same(s.expand("w", np), s["w"]) and s.nbytes() > 0
    assert s.uvmax() == float(np.sqrt(s["u"] ** 2 + s["v"] ** 2 + s["w"] ** 2).max()) == d.uvmax()
    ok = s.ok()
    assert ok.dtype == np.bool_ and ok.sum() == s.n


@needs_casacore
def test_load_archive_recomputes_R_for_another_phase_centre(tmp_path):
    p = str(tmp_path / "20261031_101007_41MHz.ms")
    make_ms(p)
    ra, dec = 1.3, 0.65
    try:
        ref = load_samples(p, ra, dec)
    except Exception as e:                                   # no casacore measures data here
        pytest.skip(f"casacore measures: {e}")
    gsiv.write_archive(tmp_path / "x.gsiv", [p], ra=0.1, dec=0.2)
    s = gsiv.load_archive(tmp_path / "x.gsiv", ra, dec)[0]   # archive R is for (0.1, 0.2): recomputed
    assert_same_samples(ref, s)
    gsiv.write_archive(tmp_path / "y.gsiv", [p], ra=ra, dec=dec)
    assert_same_samples(ref, gsiv.load_archive(tmp_path / "y.gsiv", ra, dec)[0])


@needs_casacore
def test_write_archive_from_ms_bitwise(tmp_path):
    ps = []
    for k in range(3):
        p = str(tmp_path / f"20261031_1010{k:02d}_41MHz.ms"); make_ms(p, seed=k); ps.append(p)
    gsiv.write_archive(tmp_path / "a.gsiv", ps)
    snaps = gsiv.load_archive(tmp_path / "a.gsiv")
    for p, s in zip(ps, snaps):
        assert_same_samples(load_samples(p), s)


# ----------------------------------------------------------------------------- image_hour_snaps (CPU)
def _hour_snaps(tmp_path, n=8):
    mss = [msdata(seed=s, path=f"/x/20261031_10{10 + s:02d}07_41MHz.ms", nant=9, nchan=6) for s in range(n)]
    th = np.radians(3.0)
    R = np.array([[np.cos(th), 0, np.sin(th)], [0, 1, 0], [-np.sin(th), 0, np.cos(th)]])
    dicts = []
    for k, ms in enumerate(mss):
        d = samples_from_msdata(ms); d["R"] = R; d["pd_new"] = (1.3, 0.65)
        d["time"] = float(d["time"] + 10 * k); d["tstart"] = d["time"]
        dicts.append(d)
    p = tmp_path / "h.gsiv"
    w = None
    for d, ms in zip(dicts, mss):
        c = samples_from_msdata(ms, compact=True); c.update(R=R, pd_new=(1.3, 0.65), time=d["time"], tstart=d["tstart"])
        meta, arrays = gsiv.record_from_samples(c)
        w = w or gsiv.RecordWriter(p, dict(mode="iv"), {k: arrays[k] for k in gsiv.SHARED_ARRAYS})
        w.append(meta, arrays)
    w.close()
    return dicts, p


@pytest.mark.parametrize("cache,free_gb", [("auto", 1000), ("auto", 0), ("store", 1000), ("store", 0), ("off", 0)])
def test_image_hour_snaps_cpu_archive_equals_dicts(tmp_path, cache, free_gb):
    hour_cpu = pytest.importorskip("hour_cpu")
    prods = bands.hour_products(["I-Deep-Taper-Robust-0", "V-Taper-Deep", "I-Taper-10min", "V-Taper-10min"])
    dicts, p = _hour_snaps(tmp_path)
    kw = dict(N=64, exclude_stems=(dicts[3]["stem"],), qa="none")
    ref_s, ref = hour_cpu.run([dict(d) for d in dicts], "41MHz", str(tmp_path / "ref"), 1.3, 0.65, prods,
                              cache="auto", free_gb=1000, **kw)
    s, got = hour_cpu.run(gsiv.load_archive(p, 1.3, 0.65), "41MHz", str(tmp_path / "ref"), 1.3, 0.65, prods,
                          cache=cache, free_gb=free_gb, **kw)
    assert hour_cpu.comparable(s) == hour_cpu.comparable(ref_s)
    assert ref_s["rejected"] == [3] and len(ref.images) == len(got.images) > 0
    # the gridder sees the same inputs in every cache mode (the cached calls are skipped, not changed)
    key = lambda c: [x for x in c if x[0] in ("grid", "degrid", "grid_psf")]
    assert key(got.calls) == key(ref.calls)
    assert sorted(got.images) == sorted(ref.images)
    for k in ref.images:
        assert same(got.images[k], ref.images[k]), k
