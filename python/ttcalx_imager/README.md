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
pytest python/ttcalx_imager/tests        # GPU tests are skipped without CuPy/CUDA
```

Full documentation: `docs/src/gpu-clean-imager.md`.

## Tuning environment variables

All are optional; the defaults are what the regression tests use.

| variable | default | meaning |
|---|---|---|
| `TTCALX_CHUNK_CACHE` | `auto` | hour job: keep loaded chunks on the GPU (`auto`, if free memory allows), on the host (`host`) or not at all (`off`). Use `host`/`off` on a GPU shared with other daemons. |
| `TTCALX_CACHE_MARGIN_GB` | `5` | `auto` chunk cache: free GPU memory to leave. |
| `TTCALX_KEEP_FFT_PLANS` / `TTCALX_FFT_PLAN_CACHE_GB` | `1` / `1.5` | hour job: keep cuFFT plans across intervals in a plan cache of this size (set process wide, not restored). |
| `TTCALX_HOGBOM_MB_BPS` | `2` | blocks per SM of the cooperative Hogbom kernel (falls back to the single-block kernel if the launch is refused). |
| `TTCALX_HOGBOM_MATRIX_MAX_K` | `4096` | largest candidate set for the G-matrix Hogbom path. |
| `TTCALX_SPARSE_UPDATE_MAX` | `64` | max components for the direct (non-FFT) residual update; capped at 64. |
| `TTCALX_SCALE_FFT` | `small` | multiscale: scale convolutions on a small FFT size (`small`) or the full P x P size (anything else). |

`SnapshotImager.image_many` reads with a spawned helper process: scripts calling it need an
`if __name__ == "__main__":` guard (or `reader="thread"`). Set `CUPY_CACHE_DIR` if the home
quota is small.
