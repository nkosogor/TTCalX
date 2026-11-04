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
| `TTCALX_FAST_EXPAND` | `0` | | archive / v3 records: kept mask unpacked on the GPU, u, v, w expanded with one transfer (same arrays). |
| `TTCALX_FAST_AK` | `0` | | a_k(w) of the w-term with one `searchsorted` per (de)gridding call and one fused kernel (cupy_interp's arithmetic, per-cell slopes precomputed; equal to `cp.interp`). `TTCALX_FAST_AK_FUSED=0`: CuPy's own interp kernel on the shared index. |
| `TTCALX_HOUR_MULTI` | `0` | | hour: dirty I, PSF and dirty V of an I product in one pass over the chunks (same images). |
| `TTCALX_PSF_MEMO` | `0` | | hour: each snapshot's PSF visibilities computed once per band-hour and kept on the host (~8 B/sample). |
| `TTCALX_HOUR_SETUP_FAST` | `0` | | hour: the density grid of a snapshot set re-used by products with the same snapshots; weights, their sum and wmax from one pass. |
| `TTCALX_SVD_PREFETCH` | `0` | | snapshot: w-term factors of the neighbouring 0.25-lambda keys computed in a background thread (the same factors). |
| `TTCALX_FAST_COORDS` | `0` | | reprojection coordinates in one kernel with the same float64 operations (each rounded separately: same bits). |
| `TTCALX_FIT_WINDOW` | `0` | | restoring-beam fit on a centred window of this many pixels (even) instead of copying the whole PSF; falls back to the whole PSF if a fitting box reaches the window (same fit). |
| `TTCALX_FAST_UPSAMPLE` | `0` | | hour: the 2x band-limited model upsampling as four N-point inverse FFTs (polyphase identity) instead of one zero-padded 2N transform, a slow Bluestein one at the hour sizes (values to float32 rounding; 82 MHz hour -13 s). |
| `TTCALX_NUFFT_PRUNED` | unset | | `1` (every scope) or a list of `snap,hour`: NUFFTs as cuFINUFFT spread / interpolation only plus our own fine-grid FFT (row FFTs of the occupied band, tiled transposes, contiguous row FFTs; `pruned.py`). Same transform to float32 rounding (5e-7); 82 MHz 8.2 -> 3.4 ms (type 1), 7.7 -> 2.3 ms (type 2); 82 MHz band-hour -29 %. |
| `TTCALX_FUSED_WACC` | `0` | | pruned type-1 transforms (`TTCALX_NUFFT_PRUNED`): each w-term accumulated inside the last mode gather (`PrunedPlan.execute_acc`) instead of writing the N x N modes and re-reading them in `acc_re`. The same float operations (images at the rerun floor, 0 CLEAN flips in 60 snapshots); 82 MHz snapshot imaging -8 %, 82 MHz band-hour snapshot stage -4.4 % (2026-11-04). |

`SnapshotImager.image_many` reads with a spawned helper process: scripts calling it need an
`if __name__ == "__main__":` guard (or `reader="thread"`). Set `CUPY_CACHE_DIR` if the home
quota is small.
