"""Command line: ``ttcalx-image snapshot ...`` and ``ttcalx-image hour ...``."""
import argparse
import json
import os
import sys


def _beam(s):
    if not s:
        return None
    b = [float(x) for x in s.split(",")]
    if len(b) != 3:
        raise argparse.ArgumentTypeError("--beam BMAJ,BMIN,BPA (deg, deg, deg)")
    return tuple(b)


def _read_list(items):
    out = []
    for it in items:
        if it.startswith("@"):
            out += [l.strip() for l in open(it[1:]) if l.strip() and not l.startswith("#")]
        else:
            out.append(it)
    return out


def cmd_snapshot(a):
    from . import bands
    from .snapshot import SnapshotImager
    params = bands.SNAPSHOT_CLEAN_I.with_(nmajor=a.nmajor, niter=a.niter)
    im = SnapshotImager(eps=a.eps, svd_tol=a.svd_tol, dft_max=a.dft_max, verbose=a.verbose)
    os.makedirs(a.out_dir, exist_ok=True)
    rc = 0
    if not a.no_pipeline:
        reqs = []
        for ms in _read_list(a.ms):
            prefix = os.path.join(a.out_dir, os.path.basename(ms.rstrip("/")).rsplit(".ms", 1)[0])
            if a.skip_existing and os.path.exists(f"{prefix}-qa.json"):
                continue
            kw = dict(N=a.size, scale_deg=a.scale) if a.size else dict(band=a.band or ms)
            reqs.append(dict(ms_path=ms, out_prefix=prefix, pols=a.pols, params=params, beam=_beam(a.beam),
                             write_aux=a.keep_aux, write_dirty=a.write_dirty, write_psf=a.write_psf, **kw))
        for rq, info in zip(reqs, im.image_many(reqs, free_pool=False)):
            stem = os.path.basename(rq["out_prefix"])
            if info["status"] == "error":
                rc = 1
                print(json.dumps(dict(stem=stem, status="error", error=info.get("error"))), flush=True)
            else:
                print(json.dumps(dict(stem=stem, status=info["status"], beam=info.get("beam"), niter=info.get("niter"),
                                      V_rms_box=info.get("V_rms_box"), total_s=info["timing"]["TOTAL"]), default=float),
                      flush=True)
        return rc
    for ms in _read_list(a.ms):
        stem = os.path.basename(ms.rstrip("/")).rsplit(".ms", 1)[0]
        prefix = os.path.join(a.out_dir, stem)
        if a.skip_existing and os.path.exists(f"{prefix}-qa.json"):
            continue
        try:
            kw = dict(N=a.size, scale_deg=a.scale) if a.size else dict(band=a.band or ms)
            info = im.image(ms, out_prefix=prefix, pols=a.pols, params=params, beam=_beam(a.beam),
                            write_aux=a.keep_aux, write_dirty=a.write_dirty, write_psf=a.write_psf, **kw)
            print(json.dumps(dict(stem=stem, status=info["status"], beam=info.get("beam"), niter=info.get("niter"),
                                  V_rms_box=info.get("V_rms_box"), total_s=info["timing"]["TOTAL"]), default=float), flush=True)
        except Exception as e:      # keep going over a list; report and return non-zero at the end
            rc = 1
            print(json.dumps(dict(stem=stem, status="error", error=repr(e))), flush=True)
    return rc


def _stems(spec):
    if not spec:
        return ()
    return tuple(x.strip() for x in (_read_list([spec]) if spec.startswith("@") else spec.split(",")) if x.strip())


def cmd_hour(a):
    import time
    from . import bands
    from .hour import image_hour, image_hour_snaps, _want_v
    if bool(a.ms) == bool(a.archive):
        raise SystemExit("ttcalx-image hour: give snapshot MSes or --archive, not both / neither")
    prods = bands.hour_products(a.products, intervals=a.intervals, niter=a.niter,
                                niter_10min=a.niter_10min, pols=a.pols)
    pc = a.phase_centre or (bands.hour_phase_centre(a.lst_hour) if a.lst_hour is not None else None)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    kw = dict(products=prods, mode=a.mode, block=a.block, pols=a.pols, qa=a.qa, qa_dir=a.qa_dir,
              exclude=[int(x) for x in a.exclude.split(",")] if a.exclude else (),
              exclude_stems=_stems(a.exclude_stems), beam=_beam(a.beam), write_aux=not a.no_aux, verbose=a.verbose)
    if a.archive:
        from .gsiv import archive_phase_centre, load_archive
        # archive snapshots: device chunk cache when it fits, else rebuilt on the GPU from the
        # records (no host-side u, v, w duplicates); TTCALX_CHUNK_CACHE overrides
        os.environ.setdefault("TTCALX_CHUNK_CACHE", "store")
        from .rotate import parse_phase_centre
        t0 = time.perf_counter()
        ra, dec = parse_phase_centre(pc) if pc else archive_phase_centre(a.archive)
        log = (lambda *x: print(*x, flush=True)) if a.verbose else None
        if a.no_verify:
            print("WARNING: --no-verify: the archive record crc are not checked", file=sys.stderr, flush=True)
        snaps = load_archive(a.archive, ra, dec, want_v=_want_v(prods, a.pols, a.qa), backend=a.archive_backend,
                             verify=not a.no_verify, workers=a.nproc, log=log, band=a.band)
        T = {"load + rotation (CPU)": time.perf_counter() - t0}
        s = image_hour_snaps(snaps, a.band, a.out, ra, dec, timing=T, t0=t0, **kw)
    else:
        s = image_hour(_read_list(a.ms), a.band, a.out, phase_centre=pc, nproc=a.nproc, **kw)
    print(json.dumps(dict(rejected=s["rejected"], timing=s["timing"],
                          products={k: [dict(name=o["name"], n_snap=o["n_snap"], beam=o.get("beam"), nit=o.get("nit"),
                                             nmajor=o.get("nmajor")) for o in v] for k, v in s["products"].items()}),
                     default=float), flush=True)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="ttcalx-image", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("snapshot", help="10-s snapshot: clean I + dirty V + QA JSON per MS")
    s.add_argument("ms", nargs="+", help="snapshot MS paths (or @listfile)")
    s.add_argument("--out-dir", required=True)
    s.add_argument("--band", help="band for the orca pixel tier (default: parsed from the MS name)")
    s.add_argument("--size", type=int, help="image size (overrides --band; use an even number)")
    s.add_argument("--scale", type=float, help="pixel scale [deg] (with --size)")
    s.add_argument("--pols", default="IV", choices=["I", "IV"])
    s.add_argument("--nmajor", type=int, default=2)
    s.add_argument("--niter", type=int, default=50000)
    s.add_argument("--keep-aux", action="store_true", help="also write I-residual and I-model")
    s.add_argument("--write-dirty", action="store_true")
    s.add_argument("--write-psf", action="store_true")
    s.add_argument("--beam", help="fixed restoring beam BMAJ,BMIN,BPA [deg] instead of the fit")
    s.add_argument("--skip-existing", action="store_true")
    s.add_argument("--no-pipeline", action="store_true",
                   help="image one MS at a time (default: read the next MS / write FITS while the GPU works)")
    s.add_argument("--eps", type=float, default=1e-4, help="cuFINUFFT tolerance")
    s.add_argument("--svd-tol", type=float, default=1e-4, help="w-term factorisation tolerance")
    s.add_argument("--dft-max", type=int, default=8000, help="max model components for DFT predict")
    s.add_argument("-v", "--verbose", action="store_true")
    s.set_defaults(fn=cmd_snapshot)

    h = sub.add_parser("hour", help="1-h and/or N x 10-min I + V from peeled snapshot MSes")
    h.add_argument("ms", nargs="*", help="snapshot MS paths (or @listfile)")
    h.add_argument("--archive", help="GSIV record file(s) of the band-hour instead of MSes: path[,path] or "
                                     "@listfile (gsi v3, ttcalx_imager.gsiv)")
    h.add_argument("--archive-backend", default="ram", choices=["ram", "memmap"],
                   help="read the archive into RAM (default) or memory-map it")
    h.add_argument("--no-verify", action="store_true", help="skip the archive record crc check")
    h.add_argument("--band", required=True)
    h.add_argument("--out", required=True, help="output prefix")
    g = h.add_mutually_exclusive_group()
    g.add_argument("--phase-centre", help="'02h30m00s 37d12m57.057s' (default: zenith of the middle snapshot)")
    g.add_argument("--lst-hour", type=int, help="orca convention: phase centre <LST>h30m00s 37d12m57.057s")
    h.add_argument("--products", default="1h,10min",
                   help="comma list: orca step names (I-Deep-Taper-Robust-0, I-Deep-Taper-Robust-0.75, "
                        "I-Deep-NoTaper-Robust-0.75, I-Deep-NoTaper-Robust-0, I-Taper-10min, V-Taper-Deep, "
                        "V-Taper-10min; 'orca' = all, written with WSClean's orca file names) or the legacy "
                        "tags 1h, 10min. All products are made from one load of the data.")
    h.add_argument("--intervals", type=int, default=6, help="number of intervals for the legacy 10min product")
    h.add_argument("--niter", type=int, default=0, help="legacy 1h niter (default 500000)")
    h.add_argument("--niter-10min", type=int, default=0, help="legacy 10min niter (default 50000)")
    h.add_argument("--mode", default="wsnap", choices=["wsnap", "vis"])
    h.add_argument("--block", type=int, default=6, help="snapshots per w-snapshot block")
    h.add_argument("--pols", default="IV", choices=["I", "IV"], help="legacy tags: also make V")
    h.add_argument("--qa", default="auto", choices=["auto", "json", "compute", "none"],
                   help="snapshot QA source: json from --qa-dir, compute here, or both (auto)")
    h.add_argument("--qa-dir", help="directory with <stem>-qa.json from 'ttcalx-image snapshot'")
    h.add_argument("--exclude", default="", help="extra snapshot indices (time order) to drop")
    h.add_argument("--exclude-stems", default="", help="snapshot stems to drop: s1,s2 or @listfile")
    h.add_argument("--nproc", type=int, default=8, help="MS reader processes (archive: reader threads)")
    h.add_argument("--beam", help="fixed restoring beam BMAJ,BMIN,BPA [deg] instead of the fit")
    h.add_argument("--no-aux", action="store_true", help="do not write residual/model/dirty/psf")
    h.add_argument("-v", "--verbose", action="store_true")
    h.set_defaults(fn=cmd_hour)

    a = ap.parse_args(argv)
    if a.cmd == "hour":
        a.products = [p.strip() for p in a.products.split(",") if p.strip()]
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
