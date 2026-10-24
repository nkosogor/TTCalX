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
