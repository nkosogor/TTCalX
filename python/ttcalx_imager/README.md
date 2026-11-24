# ttcalx-imager

GPU CLEAN imager for OVRO-LWA all-sky data (CuPy + cuFINUFFT + python-casacore) that
reproduces the main-pipeline WSClean products for our settings:

* `ttcalx-image snapshot` — 10-s snapshot: CLEANed Stokes I + dirty Stokes V + QA JSON
  (orca `SNAPSHOT_CLEAN_I_PARAMS` + pilot V).
* `ttcalx-image hour` — 1-h and 6 x 10-min Stokes I (multiscale CLEAN, `I-Deep-Taper-Robust-0`,
  `I-Taper-10min`) and dirty V from a list of peeled snapshot MSes, with the orca snapshot QA.

```bash
pip install -e 'python/ttcalx_imager[gpu,test]'
ttcalx-image snapshot /data/20261023_080003_41MHz.ms --out-dir out/
ttcalx-image hour @list41.txt --band 41 --out out/41MHz --lst-hour 2
ttcalx-image hour --archive 41MHz-2026-10-31-02h-peeled-b000.gsiv --band 41 --out out/41MHz --lst-hour 2
pytest python/ttcalx_imager/tests        # GPU tests are skipped without CuPy/CUDA
```

Full documentation: [docs/src/gpu-clean-imager.md](../../docs/src/gpu-clean-imager.md).

## Tuning environment variables

All are optional.  **The defaults are 748884a's behaviour** for the MS hour job and the snapshot
imager: no chunk cache, cuFFT plans cleared per interval, LAPACK SVD, FFT local RMS, full-size scale
FFTs, FFT residual updates.  The gsi v3 entry points (`hour --archive`, `cli.run_hour_snaps`, the gsi v3
engine) call `ttcalx_imager.use_fast_settings()`, which sets every variable of `FAST_ENV` that is not
already set (value in the third column).  Always on (not gated): the compact w-term factors
(float32 interpolation of the 1-D tables) and the new Hogbom kernels; images are within 748884a's
run-to-run GPU floor (82 MHz MS hour job vs 748884a, 2026-11-02; 2026-10-27 regression set PASS with 86afd89).

| variable | default | `FAST_ENV` | meaning |
|---|---|---|---|
| `TTCALX_CHUNK_CACHE` | `off` | `store` | hour: keep loaded chunks on the GPU (`auto`, if free memory allows, else host), on the host (`host`), on the GPU (`device`) or not at all (`off`). `store`: on the GPU if it fits, else rebuilt per use from the archive records. The gridder gets the same arrays in every mode. |
| `TTCALX_CACHE_MARGIN_GB` | `5` | | chunk cache: free GPU memory to leave. |
| `TTCALX_CACHE_RESERVE` | `0` | `auto` | chunk cache: GPU memory to leave for the multiscale CLEAN of each interval (`auto` = `hour.deconv_reserve_bytes`, ~9 GB at 3122 px, or GB); the larger of this and the margin counts. |
| `TTCALX_CACHE_OOM_FALLBACK` | `0` | `1` | hour: a GPU OOM in an interval with chunks cached on the GPU images it again without the device cache (`cache_fallback` in the summary). `0`: raise. |
| `TTCALX_KEEP_FFT_PLANS` / `TTCALX_FFT_PLAN_CACHE_GB` | `0` / `1.5` | `1` | hour: keep cuFFT plans across intervals in a plan cache of this size (process wide). |
| `TTCALX_SVD_METHOD` | `lapack` | `gram` | w-term factorisation: `np.linalg.svd` or the top eigenpairs of M M^H (2-4x faster, same rank, factors equal to rounding). |
| `TTCALX_RMS_FILTER` | `fft` | `circulant` | local RMS smoothing: complex FFT pair or two SGEMMs with the circulant 1-D kernel (no Bluestein plan; ~1e-6 of the max). |
| `TTCALX_SCALE_FFT` | `full` | `small` | multiscale: scale convolutions on the full P x P FFT size or a smaller one (same linear convolution, rounding). |
| `TTCALX_SPARSE_UPDATE_MAX` | `0` | `64` | max components for the direct (non-FFT) residual update (0 = always FFT; capped at 64). |
| `TTCALX_STRICT_BEAM` | `0` | | `1`: the restoring-beam fit raises `DegeneratePSFError` for a zero / non-finite PSF centre or fit (default: the fit's values as before). A degenerate snapshot is caught before the fit by `check_psf_support` either way. |
| `TTCALX_HOGBOM_MB_BPS` | `2` | | blocks per SM of the cooperative Hogbom kernel (a refused launch or a compile error falls back to the single-block kernel, same components). |
| `TTCALX_HOGBOM_MATRIX_MAX_K` | `4096` | | largest candidate set for the G-matrix Hogbom path. |
| `TTCALX_FAST_EXPAND` | `0` | `1` | archive / v3 records: kept mask unpacked on the GPU, u, v, w expanded with one transfer (same arrays). |
| `TTCALX_FAST_AK` | `0` | `1` | a_k(w) of the w-term with one `searchsorted` per (de)gridding call and one fused kernel (cupy_interp's arithmetic, per-cell slopes precomputed; equal to `cp.interp`). `TTCALX_FAST_AK_FUSED=0`: CuPy's own interp kernel on the shared index. |
| `TTCALX_HOUR_MULTI` | `0` | `1` | hour: dirty I, PSF and dirty V of an I product in one pass over the chunks (same images). |
| `TTCALX_PSF_MEMO` | `0` | `1` | hour: each snapshot's PSF visibilities computed once per band-hour and kept on the host (~8 B/sample). |
| `TTCALX_HOUR_SETUP_FAST` | `0` | `1` | hour: the density grid of a snapshot set re-used by products with the same snapshots; weights, their sum and wmax from one pass. |
| `TTCALX_SVD_PREFETCH` | `0` | `1` | snapshot: w-term factors of the neighbouring 0.25-lambda keys computed in a background thread (the same factors). |
| `TTCALX_FAST_COORDS` | `0` | `1` | reprojection coordinates in one kernel with the same float64 operations (each rounded separately: same bits). |
| `TTCALX_FIT_WINDOW` | `0` | `512` | restoring-beam fit on a centred window of this many pixels (even) instead of copying the whole PSF; falls back to the whole PSF if a fitting box reaches the window (same fit). |
| `TTCALX_FAST_UPSAMPLE` | `0` | `1` | hour: the 2x band-limited model upsampling as four N-point inverse FFTs (polyphase identity) instead of one zero-padded 2N transform, a slow Bluestein one at the hour sizes (values to float32 rounding; 82 MHz hour -13 s). |
| `TTCALX_NUFFT_PRUNED` | unset | `1` | `1` (every scope) or a list of `snap,hour`: NUFFTs as cuFINUFFT spread / interpolation only plus our own fine-grid FFT (row FFTs of the occupied band, tiled transposes, contiguous row FFTs; `pruned.py`). Same transform to float32 rounding (5e-7); 82 MHz 8.2 -> 3.4 ms (type 1), 7.7 -> 2.3 ms (type 2); 82 MHz band-hour -29 %. |
| `TTCALX_FUSED_WACC` | `0` | `1` | pruned type-1 transforms (`TTCALX_NUFFT_PRUNED`): each w-term accumulated inside the last mode gather (`PrunedPlan.execute_acc`) instead of writing the N x N modes and re-reading them in `acc_re`. The same float operations (images at the rerun floor, 0 CLEAN flips in 60 snapshots); 82 MHz snapshot imaging -8 %, 82 MHz band-hour snapshot stage -4.4 % (2026-11-04). |
| `TTCALX_REPROJ_FAST` | `0` | (`HOUR_V5_ENV`: `1`) | hour w-snapshot reprojection (order 3 / 5): B-spline prefilter as one zero-padded real-FFT division by the B-spline transfer function, interpolation in one float32 kernel, instead of cupyx `spline_filter` + `map_coordinates`. Equal to float32 rounding where the image is zero within ~10 px of its edge (the production geometries end 7-12 px beyond the horizon). |
| `TTCALX_SVD_QUANTUM` | `2` | (`HOUR_V5_ENV`: `0.25`) | hour: w-term factorisations for wmax rounded up to this multiple [lambda] (0.25 = the snapshot imager's): each block gets the rank its own w range needs. Other factors within the 1e-4 tolerance. |
| `TTCALX_MS_INCR` | `0` | (`HOUR_V5_ENV`: `16`) | multiscale CLEAN (with `TTCALX_SPARSE_UPDATE_MAX`): after a sparse residual update with <= n components the scale images are updated with the same components (S_s * S_k * PSF kernels) instead of recomputed by FFT. Kernels capped at `MultiScaleClean.INCR_CACHE_GB` (1) per interval and made only with `INCR_FREE_GB` (1) of the GPU free, else the FFT path. |
| `TTCALX_FUSED_CAK` | `0` | (`HOUR_V5_ENV`: `1`) | `SampleSet.grid` / `degrid` (with `TTCALX_FAST_AK`; grid also `TTCALX_FUSED_WACC` + pruned plans): a_k(w) evaluated inside the product kernel (no a_k / conj / product temporaries); the same float operations (degrid bitwise). |
| `TTCALX_WMAX_MEMO` | `0` | (`HOUR_V5_ENV`: `1`) | hour: a w-snapshot block's max abs(w) re-used by every product / interval of the call (bit-identical). |
| `TTCALX_UVMAX_GPU` | `0` | (`HOUR_V5_ENV`: `1`) | hour: the archive snapshots' max abs(u, v, w) computed on the GPU right after the QA drop instead of a host-side expansion per snapshot (bit-identical). |

Hour-stage v5 switches: `ttcalx_imager.use_hour_v5_settings()` sets `FAST_ENV` and `HOUR_V5_ENV` (an explicit
value wins); they are not in `FAST_ENV`, so the gsi entry points leave them off unless the caller sets them
(`TTCALX_V5=1`, below, includes them).
`summary["hour_v5"]` lists the ones in effect.

## v5 switches (default off)

Not in `FAST_ENV`: production behaviour is unchanged unless they are set.  `TTCALX_V5=1` in the environment of
a process that calls `use_fast_settings()` (the gsi v3 engine) sets all of them -- the snapshot switches below and
the hour-stage `HOUR_V5_ENV` rows above (`ttcalx_imager.v5_env()`; an explicit value wins).  For the nightly: `imaging.gpu_env: {TTCALX_V5: "1"}` in the gsi config (the engine also
needs gsi's `small["uvw_stored"]`, branch `exp/gsi-v5-img`; without it the snapshot router has nothing to route).

| variable | default | `TTCALX_V5` | meaning |
|---|---|---|---|
| `TTCALX_SNAP_ZEN` | `0` | `1` | snapshot (gsi v3 engine path): grid dirty I, PSF, the two residuals and dirty V in the snapshot's stored (zenith) frame, K ~ 4-8 w-layers instead of the hour frame's 17-61, reprojected to the production hour grid (`zenith.py`).  Weights, prediction (DFT / degrid switch at `dft_max`), CLEAN, beam fit, restore, QA and FITS are the production code.  `1` / `auto`: per-snapshot router (zenith path when K_hour >= K_zen + `TTCALX_SNAP_ZEN_DK`), `force`: always.  2026-11-24, 2024-12-20 LST 01h, 6 snapshots per band: snapshot time x3.0-3.6 at the hour ends, x1.0-1.1 mid-hour, x2.5 (82 MHz) / x2.2 (55) / x1.4 (23) over the 6; headers equal except BMAJ/BMIN (<= 2e-5 rel.) and BPA (<= 3e-3 deg); I above 10 deg <= 0.007 sigma rms, below 10 deg incl. the horizon <= 0.006 sigma rms / 0.05 sigma max except where CLEAN near-ties occur; dirty I up to 0.67 sigma at the brightest peaks (~1e-4 relative: production itself moves 0.2 sigma for 1e-4 rms per-antenna gain errors; vs an exact DFT the zenith path is 5-40x closer than production); CLEAN near-ties (niter differs) in 7 of 17 zenith-path snapshots, restored max up to 1.7 sigma (55 MHz, one source 1 %), rms 0.013-0.019 sigma in those two; V QA rms -1e-4 relative. Not used with GSI_SVD_QUANTUM > 0 (production path). |
| `TTCALX_SNAP_ZEN_DK` | `4` | | router margin in w-term layers. |
| `TTCALX_SNAP_ZEN_GZ` / `_PSFGZ` | `1` / `1.5` | | zenith grid oversampling of the I/V/residual passes / of the PSF pass.  GPU peak per snapshot (82 / 55 / 23 MHz): production 3.4 / 2.3 / 1.3 GB, zenith path 6.5 / 3.9 / 1.8 GB (the 1.5x PSF grid's plan and buffers); PSFGZ 1: 3.0 GB at 82 MHz, ~-10 % time, beam 3-10x further off (BMAJ/BMIN <= 1.1e-4 rel., BPA <= 0.007 deg). |
| `TTCALX_SNAP_ZEN_RING_DEG` / `_RING_PX` | `6` / `8` | | horizon ring: hour pixels below this zenith-frame elevation, or within this many zenith-grid pixels of the zenith-frame horizon, are evaluated per w-layer (exact w-term at the pixel's direction) instead of from the reprojected zenith image (whose w-term factor varies too fast there: 1-2 sigma near bright sources at el ~ 0 without it). |
| `TTCALX_SNAP_ZEN_AK_LOSS` | `1` | | `1`: multiply the zenith-path image by production's per-pixel a_k / b_k linear-interpolation gain (`zenith.gain_map`), so that the products stay production's.  `0`: the loss-free image (see Known issues). |
| `TTCALX_PINNED_H2D` | `0` | `1` | snapshot DATA / FLAG to the GPU through one page-locked staging buffer (`snapshot.h2d`; the gsi engine's `prepare` uses it when present): the same bytes, 82 MHz 107 MB 13.2 -> 8.5 ms. |

### Known issues

* **a_k interpolation flux loss (production, not fixed here).**  The snapshot hour-frame gridder interpolates the
  a_k(w) table (600 points over [0, wmax]) and the b_k(x') table (2001 points) linearly; at the hour ends
  (K 45-61) this lowers the image by up to ~0.3 % (55 MHz) / ~0.5 % (73-82 MHz), direction dependent
  (`zenith.gain_map`, 2nd order: 1 - 2 pi^2 [t(1-t) (dw x')^2 + s(1-s) (dx w)^2]).  The zenith path reproduces it
  (`TTCALX_SNAP_ZEN_AK_LOSS=1`) so that v5 is identity-first.  Fixing it is a separate, product-changing step:
  `TTCALX_SNAP_ZEN_AK_LOSS=0` (zenith path without the loss, unvalidated as a product change) and, for the
  production path, finer tables (not implemented).
* The zenith path needs the stored UVW (`small["uvw_stored"]`); the MS-reading paths (`image`, `image_many`) and
  `v_rms_from_samples` always use the production gridding.

`SnapshotImager.image_many` reads with a spawned helper process: scripts calling it need an
`if __name__ == "__main__":` guard (or `reader="thread"`). Set `CUPY_CACHE_DIR` if the home
quota is small.
