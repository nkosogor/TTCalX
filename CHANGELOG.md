# Changelog

## v0.5.0 (2026-11-24, gsi v5)

Two speed paths, both **off by default**: with default settings the peeler and the imager behave as in v0.4.0
(the peeler bitwise as 748884a). The gsi v5 config turns both on.

### Peeling: `zest_sky_rfi!(...; peel_fast=true)` (`src/gpu/peel_fast.jl`)

- What: the same rk4 peel (sky, then near-field RFI) rearranged for the GPU: ComplexF32 visibilities as read from
  the MS, rows without data compacted out, fused add-back / subtract, host-mapped slot memory. ~3x faster per
  snapshot. Results are another rounding of the default: Float32 level at 55 / 82 MHz; ~1e-3 relative at 23 MHz,
  where an RFI divergence-guard decision occasionally flips (as with `fast_kernels!` on vs off).
- Default: `peel_fast=false`; with `false` the code path is the v0.4.0 one (ComplexF32 visibilities are rejected).
- Enable (Julia): `fast_kernels!(true)` once, then
  `zest_sky_rfi!(vis, meta, mjd, sky, rfi; peel_fast=true, workspace=PeelWorkspace(), solver=:rk4,
  precision=:mixed, rfi_mode=:after, cache_model_sq=true, maxiter=5)`.
  Unsupported options throw `ArgumentError("peel_fast needs ...")`: diagonal gains, a phase prior (`reg`),
  `smooth_flux` / `rfi_smooth_flux`, `restore`, `rfi_mode=:joint`, `initial_gains` (warm start), CPU
  visibilities. One fast peel per process at a time.
- gsi: `peel: {peel_fast: true}` (gsi v5; also needs `fast_kernels`, `workspace`, `precision: mixed`,
  `solver: rk4`).
- Tests: `test/test_peel_fast.jl` (in `test/runtests.jl`; GPU only).

### Imager: `TTCALX_V5=1` (`python/ttcalx_imager`)

- What: `TTCALX_V5=1` in the environment of a process that calls `ttcalx_imager.use_fast_settings()` (the gsi v3
  engine) sets, where not already set, every variable of `ttcalx_imager.v5_env()`:
  - snapshot (`SNAP_V5_ENV`): `TTCALX_SNAP_ZEN=1`, zenith-frame snapshot gridding with a per-snapshot router
    (`zenith.py`); snapshot time x2.2-2.5 at 55-82 MHz over an hour (x3-3.6 at the hour ends). Same products to
    the gridding accuracy, headers equal except BMAJ/BMIN/BPA at the 1e-5 / 1e-3 deg level. Needs the stored UVW
    (gsi `small["uvw_stored"]`); falls back to the production path for snapshots far from their zenith
    (`ZenFar`) and with `GSI_SVD_QUANTUM > 0`.
  - hour stage (`HOUR_V5_ENV`): `TTCALX_REPROJ_FAST=1`, `TTCALX_SVD_QUANTUM=0.25`, `TTCALX_FUSED_CAK=1`,
    `TTCALX_WMAX_MEMO=1`, `TTCALX_UVMAX_GPU=1`; band-hour x1.14-1.20 (55 / 73 MHz), 0 fails vs production.
- Default: unset; none of these are in `FAST_ENV`, so v0.4.0 behaviour. An explicit value of any variable wins.
- Opt-in only, not in the v5 set: `TTCALX_PINNED_H2D` (no measured speedup), `TTCALX_MS_INCR` (leave off: moves
  73 MHz I-Deep-NoTaper-Robust-0).
- Enable: gsi `imaging.gpu_env: {TTCALX_V5: "1"}`; standalone `TTCALX_V5=1` before `use_fast_settings()`, or
  `ttcalx_imager.use_hour_v5_settings()` for the hour stage only.
- Details, tuning knobs and measurements: [python/ttcalx_imager/README.md](python/ttcalx_imager/README.md)
  ("v5 switches", "Known issues").
- Router timing fix (after the gsi v5 e2e test): the router's hour-table rank K_h now goes through the same SVD
  prefetch as production's `snapshot_svd` (`TTCALX_SVD_PREFETCH`, in `FAST_ENV`; keys one and two 0.25-lambda steps
  ahead). Before, the zenith path computed each new hour key's SVD synchronously: +0.3 s on every other snapshot
  at 32 / 36 MHz (alternating 0.25 / 0.55 s in the e2e) and +0.5-1 s on nearly every snapshot at 73-82 MHz. Same
  factors, same router decisions, same images.
- Tests: `python/ttcalx_imager/tests/test_zen.py`, `test_hour_v5.py`.

### Versions

`Project.toml` 0.4.0 -> 0.5.0; the `ttcalx_imager` Python package 0.1.0 -> 0.5.0 (now follows the TTCalX
release number).

## v0.4.0 (gsi v4)

The imager's v4 speed switches in `ttcalx_imager.FAST_ENV`, turned on by the gsi v3 engine; same images as
v0.3.1.
