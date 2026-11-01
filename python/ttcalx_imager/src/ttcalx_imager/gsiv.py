"""GSIV v1 archive reader for the hour imager: ``load_archive(path, ra, dec) -> [LazySnap]``.

A GSIV file holds one compact record per peeled snapshot of a band-hour (gsi v3,
gpu-subband-imaging docs/v3_design.md 3.4 and 4): I and V (complex64) per kept sample, and per
cross-correlation row uvw (float64), natw (float32) and the kept mask, plus R, time, tstart,
freqs, chan_width, pd, zero_frac, flagfrac and stem.  The records are what
``msio.samples_from_msdata(..., compact=True)`` returns; :class:`LazySnap` rebuilds u, v, w, nw
with :func:`msio.expand_coord` / :func:`msio.expand_nw`, the expressions of
``msio.load_samples``, so ``load_archive`` gives bit-identical sample arrays (and the hour
products of ``hour.image_hour_snaps`` equal those of ``hour.image_hour`` on the MSes).

The format is defined (and written) by ``gpu_subband_imaging/v3/hourstore.py``; the section
between the BEGIN/END markers below is a verbatim copy of it (the pipeline tests compare the
two), so this module can also write files (tests, rebuilding an archive from MSes).

numpy only (+ casacore when R has to be recomputed for a different phase centre).
"""
from __future__ import annotations

import json
import os
import struct
import threading
import time
import zlib
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np

from .msio import expand_coord, expand_nw

# ============================================================================= GSIV v1 codec
# BEGIN shared section: copied verbatim into ttcalx_imager/gsiv.py (tests check they match)
FORMAT = "GSIV"
VERSION = 1
MAGIC_HDR = b"GSIVHDR1"
MAGIC_REC = b"GSIVREC1"
ALIGN = 64
BLOCK_HEADER = 64
_BH = struct.Struct("<8sIIQQI")          # 36 bytes, then hcrc (u32) and 24 zero bytes
_HCRC = struct.Struct("<I")
MODES = ("iv",)                          # "full" (full-pol + flags + weights) is not implemented yet
# per-record arrays that are left out when they equal the file header's (same for every snapshot)
SHARED_ARRAYS = ("a1", "a2", "ant_pos")
# metadata copied into each index line (the rest is in the record JSON)
INDEX_KEYS = ("stem", "time", "tstart", "zero_frac", "peel")

PathLike = Union[str, Path]


class GSIVError(Exception):
    pass


def _pad(n: int) -> int:
    return (-n) % ALIGN


def idx_path(path: PathLike) -> Path:
    return Path(f"{path}.idx.jsonl")


# ----------------------------------------------------------------------------- block codec
def _le(a: np.ndarray) -> np.ndarray:
    a = np.ascontiguousarray(a)
    if a.dtype.byteorder == ">":
        a = a.astype(a.dtype.newbyteorder("<"))
    return a


def encode_block(magic: bytes, seq: int, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]
                 ) -> Tuple[bytes, List[Any], int, int]:
    """(64-B header, payload pieces, total nbytes, crc) of one block; nothing is copied."""
    table, data = [], []
    off = 0
    for name, a in arrays.items():
        if a is None:
            continue
        a = _le(np.asarray(a))
        if a.dtype == np.bool_:
            raise GSIVError(f"array {name}: store bool masks bit-packed (pack_mask)")
        if a.dtype.hasobject:
            raise GSIVError(f"array {name}: object dtype")
        table.append(dict(name=name, dtype=a.dtype.str, shape=list(a.shape), offset=off, nbytes=int(a.nbytes)))
        data.append(a)
        off += a.nbytes + _pad(a.nbytes)
    m = dict(meta)
    m["arrays"] = table
    js = json.dumps(m, separators=(",", ":")).encode()
    pieces: List[Any] = [js, b"\0" * _pad(len(js))]
    for a in data:
        pieces.append(memoryview(a.reshape(-1).view(np.uint8)))
        pieces.append(b"\0" * _pad(a.nbytes))
    crc = 0
    for p in pieces:
        crc = zlib.crc32(p, crc)
    nbytes = BLOCK_HEADER + len(js) + _pad(len(js)) + off
    h = _BH.pack(magic, len(js), len(table), nbytes, seq, crc)
    header = h + _HCRC.pack(zlib.crc32(h)) + b"\0" * (BLOCK_HEADER - _BH.size - _HCRC.size)
    return header, pieces, nbytes, crc


def decode_header(b: bytes) -> Dict[str, Any]:
    if len(b) < BLOCK_HEADER:
        raise GSIVError("short block header")
    h = bytes(b[:_BH.size])
    (hcrc,) = _HCRC.unpack(bytes(b[_BH.size:_BH.size + _HCRC.size]))
    if zlib.crc32(h) != hcrc:
        raise GSIVError("block header checksum mismatch")
    magic, meta_len, n_arrays, nbytes, seq, crc = _BH.unpack(h)
    if magic not in (MAGIC_HDR, MAGIC_REC):
        raise GSIVError(f"bad block magic {magic!r}")
    if nbytes % ALIGN or nbytes < BLOCK_HEADER + meta_len:
        raise GSIVError("bad block length")
    return dict(magic=magic, meta_len=meta_len, n_arrays=n_arrays, nbytes=nbytes, seq=seq, crc=crc)


def decode_block(buf, off: int = 0, hdr: Optional[Dict[str, Any]] = None
                 ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """(meta, arrays) of the block at ``off`` of ``buf`` (bytes / numpy uint8 / memmap).
    The arrays are read-only views into ``buf``."""
    mv = memoryview(buf).cast("B") if not isinstance(buf, np.ndarray) else buf
    if hdr is None:
        hdr = decode_header(bytes(mv[off:off + BLOCK_HEADER]))
    js = bytes(mv[off + BLOCK_HEADER: off + BLOCK_HEADER + hdr["meta_len"]])
    meta = json.loads(js)
    d0 = off + BLOCK_HEADER + hdr["meta_len"] + _pad(hdr["meta_len"])
    arrays = {}
    for t in meta.pop("arrays"):
        dt = np.dtype(t["dtype"])
        cnt = int(np.prod(t["shape"], dtype=np.int64)) if t["shape"] else 1
        a = np.frombuffer(buf, dtype=dt, count=cnt, offset=d0 + t["offset"]).reshape(t["shape"])
        a.flags.writeable = False
        arrays[t["name"]] = a
    return meta, arrays


def block_crc(buf, off: int, nbytes: int) -> int:
    mv = memoryview(buf).cast("B") if not isinstance(buf, np.ndarray) else memoryview(buf)
    return zlib.crc32(mv[off + BLOCK_HEADER: off + nbytes])


def pack_mask(ok: np.ndarray) -> np.ndarray:
    return np.packbits(np.ascontiguousarray(ok, dtype=np.bool_).reshape(-1), bitorder="little")


def unpack_mask(bits: np.ndarray, shape: Sequence[int]) -> np.ndarray:
    n = int(np.prod(shape, dtype=np.int64))
    return np.unpackbits(bits, count=n, bitorder="little").reshape(tuple(shape)).astype(np.bool_)


def _pread(fd: int, n: int, off: int) -> bytes:
    out = []
    while n > 0:
        b = os.pread(fd, n, off)
        if not b:
            break
        out.append(b)
        n -= len(b); off += len(b)
    return b"".join(out)


def _preadinto(fd: int, buf: np.ndarray, off: int) -> int:
    mv = memoryview(buf).cast("B")
    got = 0
    while got < len(mv):
        k = os.preadv(fd, [mv[got:]], off + got)
        if k <= 0:
            break
        got += k
    return got


# ----------------------------------------------------------------------------- scan / repair
@dataclass
class Scan:
    """Structure of a GSIV file: the valid records and what is wrong after them."""
    path: str
    size: int
    header_meta: Dict[str, Any]
    header_arrays: Dict[str, np.ndarray]
    header_nbytes: int
    entries: List[Dict[str, Any]]           # index entries of the valid records, file order
    end: int                                # end of the last valid record
    torn_bytes: int = 0                     # bytes after `end` that are not a valid record
    unindexed: int = 0                      # valid records found by scanning (index line missing)
    idx_dropped: int = 0                    # index lines that were unparsable or did not match
    bad_crc: List[int] = field(default_factory=list)   # seq of indexed records with a bad crc

    @property
    def clean(self) -> bool:
        return not (self.torn_bytes or self.unindexed or self.idx_dropped or self.bad_crc)

    @property
    def stems(self) -> List[str]:
        return [e["stem"] for e in self.entries]


def _entry(meta: Dict[str, Any], arrays: Dict[str, np.ndarray], seq: int, off: int, nbytes: int, crc: int
           ) -> Dict[str, Any]:
    e = dict(seq=seq, offset=off, nbytes=nbytes, crc=crc, n_kept=int(meta.get("n", 0)))
    for k in INDEX_KEYS:
        if k in meta:
            e[k] = meta[k]
    e["flag_frac"] = meta.get("flagfrac")
    if "R" in arrays:
        e["R"] = np.asarray(arrays["R"], np.float64).tolist()
    return e


def _read_index(path: PathLike) -> Tuple[List[Dict[str, Any]], int]:
    """Index lines up to the first unparsable one; (entries, number of lines dropped)."""
    p = idx_path(path)
    if not p.exists():
        return [], 0
    lines = p.read_bytes().split(b"\n")
    out = []
    for i, ln in enumerate(lines):
        if not ln.strip():
            continue
        try:
            out.append(json.loads(ln))
        except ValueError:
            return out, sum(1 for x in lines[i:] if x.strip())
    return out, 0


def scan(path: PathLike, verify: bool = False, workers: int = 8) -> Scan:
    """Validate the header, the index and the blocks after it (see the module docstring).

    verify=True also checks the crc of every indexed record (reads the whole file); records found
    by scanning past the index are always crc-checked."""
    path = str(path)
    fd = os.open(path, os.O_RDONLY)
    try:
        size = os.fstat(fd).st_size
        hb = _pread(fd, BLOCK_HEADER, 0)
        try:
            h = decode_header(hb)
        except GSIVError as e:
            raise GSIVError(f"{path}: not a GSIV file ({e})") from None
        if h["magic"] != MAGIC_HDR or h["seq"] != 0 or h["nbytes"] > size:
            raise GSIVError(f"{path}: bad file header")
        hbuf = _pread(fd, h["nbytes"], 0)
        if block_crc(hbuf, 0, h["nbytes"]) != h["crc"]:
            raise GSIVError(f"{path}: file header crc mismatch")
        hmeta, harr = decode_block(hbuf, 0, h)
        if hmeta.get("format") != FORMAT or int(hmeta.get("version", 0)) != VERSION:
            raise GSIVError(f"{path}: unsupported format {hmeta.get('format')} v{hmeta.get('version')}")
        sc = Scan(path=path, size=size, header_meta=hmeta, header_arrays=harr, header_nbytes=h["nbytes"],
                  entries=[], end=h["nbytes"])
        idx, sc.idx_dropped = _read_index(path)
        # 1. index entries, while they describe consecutive complete blocks
        for i, e in enumerate(idx):
            ok = (isinstance(e, dict) and e.get("offset") == sc.end and e.get("seq") == len(sc.entries) + 1
                  and sc.end + int(e.get("nbytes", 0)) <= size)
            if ok:
                try:
                    bh = decode_header(_pread(fd, BLOCK_HEADER, sc.end))
                    ok = (bh["magic"] == MAGIC_REC and bh["nbytes"] == e["nbytes"] and bh["crc"] == e["crc"]
                          and bh["seq"] == e["seq"])
                except GSIVError:
                    ok = False
            if not ok:
                sc.idx_dropped += len(idx) - i
                break
            sc.entries.append(e)
            sc.end += int(e["nbytes"])
        # 2. complete blocks after the index (crash between record fsync and index line)
        while sc.end + BLOCK_HEADER <= size:
            try:
                bh = decode_header(_pread(fd, BLOCK_HEADER, sc.end))
            except GSIVError:
                break
            if bh["magic"] != MAGIC_REC or bh["seq"] != len(sc.entries) + 1 or sc.end + bh["nbytes"] > size:
                break
            buf = _pread(fd, bh["nbytes"], sc.end)
            if len(buf) != bh["nbytes"] or block_crc(buf, 0, bh["nbytes"]) != bh["crc"]:
                break
            meta, arrays = decode_block(buf, 0, bh)
            sc.entries.append(_entry(meta, arrays, bh["seq"], sc.end, bh["nbytes"], bh["crc"]))
            sc.unindexed += 1
            sc.end += bh["nbytes"]
        sc.torn_bytes = size - sc.end
        # 3. crc of the indexed records
        if verify:
            n_idx = len(sc.entries) - sc.unindexed

            def check(e):
                b = np.empty(int(e["nbytes"]), np.uint8)
                if _preadinto(fd, b, int(e["offset"])) != len(b):
                    return e["seq"]
                return None if block_crc(b, 0, len(b)) == e["crc"] else e["seq"]
            with ThreadPoolExecutor(max(1, workers)) as ex:
                sc.bad_crc = [s for s in ex.map(check, sc.entries[:n_idx]) if s is not None]
        return sc
    finally:
        os.close(fd)


def _fsync_dir(p: Path) -> None:
    try:
        fd = os.open(str(p.parent), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except OSError:
        pass


def _write_index(path: PathLike, entries: Sequence[Dict[str, Any]]) -> None:
    p = idx_path(path)
    tmp = Path(f"{p}.tmp{os.getpid()}")
    with open(tmp, "w") as fh:
        for e in entries:
            fh.write(json.dumps(e, separators=(",", ":")) + "\n")
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(str(tmp), str(p))
    _fsync_dir(p)


def repair(path: PathLike, log=None) -> Scan:
    """Make the file and its index consistent: truncate at the first bad record (torn tail or
    bad crc), rewrite the index for the valid prefix.  Returns the scan of the repaired file."""
    sc = scan(path, verify=True)
    if sc.bad_crc:
        first = min(sc.bad_crc)
        keep = [e for e in sc.entries if e["seq"] < first]
        sc.end = keep[-1]["offset"] + keep[-1]["nbytes"] if keep else sc.header_nbytes
        sc.entries = keep
    if not sc.clean:
        if log:
            log(f"GSIV repair {path}: {len(sc.entries)} valid records, torn {sc.torn_bytes} B, "
                f"unindexed {sc.unindexed}, index lines dropped {sc.idx_dropped}, bad crc {sc.bad_crc}"
                f" -> truncate to {sc.end}")
        with open(path, "r+b") as fh:
            fh.truncate(sc.end)
            fh.flush()
            os.fsync(fh.fileno())
        _write_index(path, sc.entries)
        sc = scan(path, verify=False)
    return sc


def read_blocks(path: PathLike, backend: str = "ram", verify: bool = True, strict: bool = True,
                workers: int = 8, log=None) -> Tuple[Dict[str, Any], Dict[str, np.ndarray],
                                                     List[Tuple[Dict[str, Any], Dict[str, np.ndarray], Dict[str, Any]]]]:
    """(header meta, header arrays, [(meta, arrays, index entry)]) of the valid records of a
    GSIV file.  ram: one read per record (threads) into its own buffer, crc computed on that
    buffer; memmap: views of one read-only np.memmap.  A bad crc raises (strict) or drops the
    record.  Records without a1/a2/ant_pos get the header's."""
    sc = scan(path, verify=False)
    entries = sc.entries
    if backend == "memmap":
        mm = np.memmap(str(path), dtype=np.uint8, mode="r")
        bufs = [(mm, int(e["offset"])) for e in entries]
    elif backend == "ram":
        fd = os.open(str(path), os.O_RDONLY)
        try:
            def rd(e):
                b = np.empty(int(e["nbytes"]), np.uint8)
                if _preadinto(fd, b, int(e["offset"])) != len(b):
                    raise GSIVError(f"{path}: short read of record {e['seq']}")
                return b
            with ThreadPoolExecutor(max(1, workers)) as ex:
                bufs = [(b, 0) for b in ex.map(rd, entries)]
        finally:
            os.close(fd)
    else:
        raise ValueError("backend must be 'ram' or 'memmap'")
    bad = []
    if verify:
        with ThreadPoolExecutor(max(1, workers)) as ex:
            crcs = list(ex.map(lambda be: block_crc(be[0][0], be[0][1], int(be[1]["nbytes"])), zip(bufs, entries)))
        bad = [e["seq"] for e, c in zip(entries, crcs) if c != e["crc"]]
        if bad and strict:
            raise GSIVError(f"{path}: crc mismatch in record(s) {bad}")
        if bad and log:
            log(f"GSIV {path}: dropped record(s) with a bad crc: {bad}")
    out = []
    for (b, off), e in zip(bufs, entries):
        if e["seq"] in bad:
            continue
        meta, arrays = decode_block(b, off)
        for k in SHARED_ARRAYS:
            if k not in arrays and k in sc.header_arrays:
                arrays[k] = sc.header_arrays[k]
        out.append((meta, arrays, e))
    return sc.header_meta, sc.header_arrays, out


# ----------------------------------------------------------------------------- writer
class RecordWriter:
    """Append-only GSIV writer (one process per file).

    A new file gets ``header_meta`` (+ ``header_arrays``); an existing one is resumed: repaired
    (torn tail truncated) and appended to.  ``append`` returns the index entry once the record
    and its index line are on disk."""

    def __init__(self, path: PathLike, header_meta: Optional[Dict[str, Any]] = None,
                 header_arrays: Optional[Dict[str, np.ndarray]] = None, fsync: bool = True,
                 resume: bool = True, log=None):
        self.path = Path(path)
        self.fsync = fsync
        self._lock = threading.Lock()
        self._f = None
        self._idx = None
        if self.path.exists() and self.path.stat().st_size > 0:
            if not resume:
                raise GSIVError(f"{self.path} exists (resume=False)")
            sc = repair(self.path, log=log)
            if header_meta is not None:
                for k in ("mode", "phase_centre"):
                    if k in header_meta and sc.header_meta.get(k) != header_meta.get(k):
                        raise GSIVError(f"{self.path}: {k} {sc.header_meta.get(k)!r} != {header_meta.get(k)!r}")
            self.header_meta, self.header_arrays = sc.header_meta, sc.header_arrays
            self.entries = list(sc.entries)
            self._end = sc.end
        else:
            if header_meta is None:
                raise GSIVError(f"{self.path}: new file needs header_meta")
            hm = dict(header_meta)
            hm.setdefault("format", FORMAT); hm.setdefault("version", VERSION); hm.setdefault("mode", "iv")
            if hm["mode"] not in MODES:
                raise GSIVError(f"GSIV mode {hm['mode']!r} not implemented (have {MODES})")
            self.path.parent.mkdir(parents=True, exist_ok=True)
            hdr, pieces, nbytes, _ = encode_block(MAGIC_HDR, 0, hm, dict(header_arrays or {}))
            with open(self.path, "wb") as fh:
                fh.write(hdr)
                for p in pieces:
                    fh.write(p)
                fh.flush()
                os.fsync(fh.fileno())
            _write_index(self.path, [])
            self.header_meta = hm
            self.header_arrays = {k: np.asarray(v) for k, v in (header_arrays or {}).items()}
            self.entries = []
            self._end = nbytes
        self._f = open(self.path, "r+b")
        self._idx = open(idx_path(self.path), "a")
        self.stems = {e.get("stem") for e in self.entries}

    def __len__(self) -> int:
        return len(self.entries)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    def _strip_shared(self, arrays: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = dict(arrays)
        for k in SHARED_ARRAYS:
            if k in out and k in self.header_arrays:
                a, h = np.asarray(out[k]), self.header_arrays[k]
                if a.shape == h.shape and a.dtype == h.dtype and np.array_equal(a, h):
                    del out[k]
        return out

    def append(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray]) -> Dict[str, Any]:
        with self._lock:
            if self._f is None:
                raise GSIVError("writer closed")
            seq = len(self.entries) + 1
            arrays = self._strip_shared(arrays)
            hdr, pieces, nbytes, crc = encode_block(MAGIC_REC, seq, meta, arrays)
            off = self._end
            self._f.seek(off)
            self._f.write(hdr)
            for p in pieces:
                self._f.write(p)
            self._f.flush()
            if self.fsync:
                os.fsync(self._f.fileno())
            e = _entry(meta, arrays, seq, off, nbytes, crc)
            self._idx.write(json.dumps(e, separators=(",", ":")) + "\n")
            self._idx.flush()
            if self.fsync:
                os.fsync(self._idx.fileno())
            self.entries.append(e)
            self.stems.add(e.get("stem"))
            self._end = off + nbytes
            return e

    def close(self) -> None:
        with self._lock:
            for f in (self._f, self._idx):
                if f is not None:
                    f.close()
            self._f = self._idx = None


# END shared section


# ----------------------------------------------------------------------------- records <-> samples
def record_from_samples(c: Dict[str, Any], peel: Optional[Dict[str, Any]] = None
                        ) -> Tuple[Dict[str, Any], Dict[str, np.ndarray]]:
    """(meta, arrays) of one record from ``msio.samples_from_msdata(..., compact=True)``
    (= gpu_subband_imaging.v3.hourstore.record_from_compact)."""
    pd_new = c.get("pd_new")
    meta = dict(kind="iv", stem=c["stem"], path=str(c["path"]), time=float(c["time"]), tstart=float(c["tstart"]),
                tmid=float(c["tmid"]), pd=[float(x) for x in c["pd"]],
                pd_new=None if pd_new is None or pd_new[0] is None else [float(x) for x in pd_new],
                nrow=int(c["nrow"]), nchan=int(c["nchan"]), n=int(c["n"]), flagfrac=float(c["flagfrac"]),
                zero_frac=float(c["zero_frac"]), telescope=str(c["telescope"]), observer=str(c["observer"]),
                field_name=str(c["field_name"]), extra=dict(c.get("extra") or {}),
                ok_shape=[int(x) for x in c["ok"].shape])
    if peel:
        meta["peel"] = peel
    arrays = dict(I=np.asarray(c["I"], np.complex64))
    if c.get("V") is not None:
        arrays["V"] = np.asarray(c["V"], np.complex64)
    arrays.update(uvw=np.asarray(c["uvw"], np.float64), natw=np.asarray(c["natw"], np.float32),
                  ok_bits=pack_mask(c["ok"]), R=np.asarray(c["R"], np.float64),
                  freqs=np.asarray(c["freqs"], np.float64), chan_width=np.asarray(c["chan_width"], np.float64),
                  a1=np.asarray(c["a1"], np.int32), a2=np.asarray(c["a2"], np.int32),
                  ant_pos=np.asarray(c["ant_pos"], np.float64))
    return meta, arrays


def write_archive(path, ms_paths, ra=None, dec=None, phase_centre=None, want_v=True, header=None):
    """Archive of snapshot MSes (``ttcalx-image gsiv-build``; tests).  Returns the index entries."""
    from .msio import read_ms, samples_from_msdata
    w = None
    for p in sorted(ms_paths, key=lambda x: os.path.basename(str(x).rstrip("/"))):
        meta, arrays = record_from_samples(samples_from_msdata(read_ms(str(p), want_v=want_v), ra, dec, want_v,
                                                               compact=True))
        if w is None:
            hm = dict(header or {}, phase_centre=phase_centre, pd_new=None if ra is None else [ra, dec])
            w = RecordWriter(path, hm, {k: arrays[k] for k in SHARED_ARRAYS})
        if meta["stem"] not in w.stems:
            w.append(meta, arrays)
    if w is None:
        return []
    w.close()
    return list(w.entries)


_SAMPLE_KEYS = ("path", "stem", "time", "tstart", "freqs", "chan_width", "pd", "R", "pd_new", "nrow", "nchan",
                "flagfrac", "zero_frac", "telescope", "observer", "field_name")
_EXPANDED = ("u", "v", "w", "nw")


class LazySnap(Mapping):
    """One archived snapshot as the mapping ``msio.load_samples`` returns (same keys, same
    values), with I and V views into the record and u, v, w, nw expanded on access (not kept).

    ``s.n`` is the number of kept samples (no expansion); ``s.expand(k, xp)`` expands u, v, w or
    nw with numpy or CuPy (bit-identical); assignments (``s["tidx"] = i``) go to an overlay, and
    ``s.derive(R=...)`` is a copy with more overlay keys that shares the arrays (``dict(s)`` would
    expand everything)."""

    def __init__(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray], R: np.ndarray, pd_new,
                 source: str = "", overlay: Optional[Dict[str, Any]] = None):
        self.meta = meta
        self.arrays = arrays
        self.source = source
        self._o = dict(overlay or {})
        self._base = dict(path=meta["path"], stem=meta["stem"], time=meta["time"], tstart=meta["tstart"],
                          freqs=arrays["freqs"], chan_width=arrays["chan_width"], pd=tuple(meta["pd"]),
                          R=R, pd_new=tuple(pd_new) if pd_new is not None else (None, None),
                          nrow=meta["nrow"], nchan=meta["nchan"], flagfrac=meta["flagfrac"],
                          zero_frac=meta["zero_frac"], telescope=meta["telescope"], observer=meta["observer"],
                          field_name=meta["field_name"])
        self._keys = _SAMPLE_KEYS + _EXPANDED + tuple(k for k in ("I", "V") if k in arrays)
        self._memo = {}                                  # shared with derive() copies

    @property
    def n(self) -> int:
        return int(self.meta["n"])

    def ok(self, xp=np):
        m = unpack_mask(self.arrays["ok_bits"], self.meta["ok_shape"])
        return m if xp is np else xp.asarray(m)

    def expand(self, k: str, xp=np):
        ok = self.ok(xp)
        if k == "nw":
            return expand_nw(xp.asarray(self.arrays["natw"]), int(self.meta["nchan"]), ok, xp)
        return expand_coord(xp.asarray(self.arrays["uvw"]), "uvw".index(k), xp.asarray(self.arrays["freqs"]), ok, xp)

    def uvmax(self) -> float:
        """max |(u, v, w)| [lambda], the float32 expression of hour.theoretical_beam_pix."""
        if "uvmax" not in self._memo:
            u, v, w = (self.expand(k) for k in "uvw")
            self._memo["uvmax"] = float(np.sqrt(u ** 2 + v ** 2 + w ** 2).max())
        return self._memo["uvmax"]

    def __getitem__(self, k):
        if k in self._o:
            return self._o[k]
        if k in _EXPANDED:
            return self.expand(k)
        if k in ("I", "V"):
            return self.arrays[k]
        return self._base[k]

    def __setitem__(self, k, v):
        self._o[k] = v

    def __contains__(self, k):
        return k in self._o or k in self._keys

    def __iter__(self):
        return iter(self._keys + tuple(k for k in self._o if k not in self._keys))

    def __len__(self):
        return len(self._keys) + sum(1 for k in self._o if k not in self._keys)

    def derive(self, **kw) -> "LazySnap":
        s = LazySnap.__new__(LazySnap)
        s.meta, s.arrays, s.source, s._base, s._keys = self.meta, self.arrays, self.source, self._base, self._keys
        s._memo = self._memo
        s._o = {**self._o, **kw}
        return s

    def nbytes(self) -> int:
        return int(sum(a.nbytes for a in self.arrays.values()))

    def __repr__(self):
        return f"LazySnap({self.meta['stem']}, n={self.n}, {self.source})"


def _split_paths(paths) -> List[str]:
    if isinstance(paths, (str, Path)):
        paths = [paths]
    out = []
    for p in paths:
        p = str(p)
        if p.startswith("@"):
            out += [l.strip() for l in open(p[1:]) if l.strip() and not l.startswith("#")]
        else:
            out += [x for x in p.split(",") if x]
    return out


def archive_phase_centre(paths) -> Tuple[float, float]:
    """PHASE_DIR of the middle snapshot (stem order), as hour.middle_zenith does for MSes."""
    st = []
    for p in _split_paths(paths):
        sc = scan(p, verify=False)
        st += [(e["stem"], p, e) for e in sc.entries]
    st.sort(key=lambda x: x[0])
    stem, p, e = st[len(st) // 2]
    fd = os.open(p, os.O_RDONLY)
    try:
        hb = _pread(fd, BLOCK_HEADER, int(e["offset"]))
        h = decode_header(hb)
        meta = json.loads(_pread(fd, h["meta_len"], int(e["offset"]) + BLOCK_HEADER))
    finally:
        os.close(fd)
    return float(meta["pd"][0]), float(meta["pd"][1])


def load_archive(paths, ra=None, dec=None, want_v=True, backend="ram", verify=True, strict=True, workers=8,
                 log=None) -> List[LazySnap]:
    """Snapshots of one or more GSIV files (list, comma-separated string or @listfile) as
    :class:`LazySnap` mappings, in time order (a stem present twice: the first one is kept).

    (ra, dec): the hour phase centre.  A record's R was computed for the header's phase centre
    (``pd_new``); for another (ra, dec) it is recomputed from the record's uvw, a1, a2, ant_pos
    and time with rotate.snapshot_rotation (casacore), exactly as load_samples would.  ra=None:
    the stored R."""
    log = log or (lambda *a: None)
    snaps, seen = [], set()
    t0 = time.perf_counter()
    nbytes = 0
    for p in _split_paths(paths):
        hm, ha, blocks = read_blocks(p, backend=backend, verify=verify, strict=strict, workers=workers, log=log)
        nrot = 0
        for meta, arrays, e in blocks:
            if meta["stem"] in seen:
                log(f"load_archive: duplicate {meta['stem']} in {p} ignored")
                continue
            seen.add(meta["stem"])
            pd_rec = meta.get("pd_new")
            if ra is None:
                R, pd_new = arrays["R"], pd_rec
            elif pd_rec is not None and float(pd_rec[0]) == float(ra) and float(pd_rec[1]) == float(dec):
                R, pd_new = arrays["R"], (ra, dec)
            else:
                from .rotate import snapshot_rotation
                R = snapshot_rotation(arrays["uvw"], arrays["a1"], arrays["a2"], arrays["ant_pos"], meta["time"],
                                      ra, dec).astype(np.float64)
                pd_new = (ra, dec); nrot += 1
            if not want_v:
                arrays = {k: v for k, v in arrays.items() if k != "V"}
            snaps.append(LazySnap(meta, arrays, R, pd_new, source=p))
            nbytes += sum(a.nbytes for a in arrays.values())
        if nrot:
            log(f"load_archive: {p}: R recomputed for {nrot} record(s) (phase centre differs from the archive's)")
    snaps.sort(key=lambda s: s["time"])
    log(f"load_archive: {len(snaps)} snapshots, {sum(s.n for s in snaps) / 1e6:.1f} M samples, "
        f"{nbytes / 1e9:.2f} GB ({backend}), {time.perf_counter() - t0:.1f} s")
    return snaps
