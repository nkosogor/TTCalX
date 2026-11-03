# GPU CLEAN imager (`ttcalx_imager`)

`python/ttcalx_imager` (CuPy + cuFINUFFT + python-casacore) makes the imaging products of the
OVRO-LWA main (orca) pipeline on one GPU from peeled snapshot measurement sets, emulating WSClean
with the orca settings.

| product | command | WSClean equivalent (orca) |
|---|---|---|
| 10-s snapshot: CLEAN I, dirty V, QA JSON | `ttcalx-image snapshot` | `SNAPSHOT_CLEAN_I_PARAMS`, pilot V, `analyze_snapshot_quality` |
| 1-h and 6 x 10-min I (multiscale CLEAN) and V (dirty) | `ttcalx-image hour` | `I-Deep-Taper-Robust-0`, `I-Taper-10min`, `V-Taper-Deep`, `V-Taper-10min` (`--products orca` adds the other deep presets) |

The hour command replaces concat, chgcentre, pilot snapshots and the WSClean runs of a band-hour.

## Usage

```bash
pip install numpy==1.26.4 scipy astropy python-casacore cupy-cuda12x==13.6.0 cufinufft==2.5.1
pip install -e 'python/ttcalx_imager[test]'
export NUMPY_MADVISE_HUGEPAGE=0              # calim nodes: avoids multi-second numpy stalls
export CUPY_CACHE_DIR=/somewhere/with/space

ttcalx-image snapshot /data/41MHz/*.ms --out-dir snap41/
#   -> <stem>-I-image.fits, <stem>-V-image.fits, <stem>-qa.json
ttcalx-image hour @list41.txt --band 41 --out hour41/41MHz --lst-hour 2 --qa-dir snap41/ --products orca
#   -> WSClean file names: 41MHz-I-Deep-Taper-Robust-0-image.fits, ...-I-Taper-10min-t0000-image.fits, ...
ttcalx-image hour --archive a.gsiv --band 41 --out hour41/41MHz --lst-hour 2   # gsi v3 band-hour archive
```

Python: `ttcalx_imager.snapshot.SnapshotImager().image(ms, out_prefix=...)`,
`ttcalx_imager.hour.image_hour(paths, band, out_prefix, ...)`. Image sizes follow the orca tiers
(`bands.py`): 18-36 MHz 1508 px at 0.0767 deg, 41-59 MHz 2358 px at 0.049 deg, 64-82 MHz 3122 px
at 0.037 deg.

## Method

- **Snapshot:** I = (XX+YY)/2 and V = i(YX-XY)/2; Briggs weights on the GPU; dirty image and PSF
  by cuFINUFFT with the w-term split into a few separable terms (SVD, max error 1e-4); Clark/Högbom
  CLEAN with WSClean's auto-mask, auto-threshold and local RMS, two exact major cycles; restore with
  a port of WSClean's PSF fit.
- **Hour:** exact rotation of every snapshot to the hour phase centre (as chgcentre); the orca QA
  rule (V rms > median + 3 MAD or < 0.5 median) rejects snapshots; one Briggs grid over all accepted
  samples; blocks of 6 snapshots are gridded in their middle snapshot's frame (small w) and
  resampled to the hour grid; multiscale CLEAN with major cycles done in the block frames.

| WSClean | here |
|---|---|
| `-weight briggs R`, `-taper-inner-tukey L` | `weights.WeightGrid` (padded density grid, f² = (5·10⁻ᴿ)² / (ΣW²/ΣW)) |
| gridding, w-stacking | cuFINUFFT type 1/2, eps 1e-4, SVD w-term (`gridder.py`) |
| `-niter`, `-mgain`, `-auto-mask`, `-auto-threshold`, `-local-rms` | `clean.py` (snapshot: 2 major cycles; hour: multiscale, bias 0.8) |
| `-horizon-mask 10deg` | components only above 10° elevation |
| `chgcentre` | `rotate.py` (casacore-measures UVW, least-squares rotation, re-phasing) |
| PSF fit (`aocommon`) | `beam.py` (BMAJ/BMIN equal to WSClean's to ~1e-6 on WSClean's PSFs) |

Conventions: kernel exp(+2πi(ul + vm − w(n−1))) (WSClean/ducc0); images in FITS order with RA to the
left, centre pixel N/2; Stokes V = i(YX − XY)/2 (checked against `wsclean -pol IV`).

## Validation (RTX A4000, against WSClean 3.4 with the orca parameters on the same MSes)

| | 41 MHz | 69 MHz |
|---|---|---|
| snapshot restored difference / noise (median) | 0.070 | 0.061 |
| snapshot source flux ratio | 1.0000 | 1.0003 |
| 1-h restored difference / noise (12 snapshots) | 0.091 | 0.105 |
| full hour (288 snapshots) vs the main-pipeline image | 0.073 σ, flux ratio 1.0002 ± 0.0024 | |
| time per snapshot (I CLEAN + V + QA) | 0.55 s (WSClean 3.9 s) | 0.80 s (WSClean 7-11 s) |
| 1 h + 6 x 10 min, I + V | 216 s for 288 snapshots (main pipeline: 42 min) | |

These differences are the size of CLEAN's own run-to-run scatter; dirty images agree to ~1e-4.

## Limitations

- Stokes I (CLEAN) and V (dirty) only; no MFS, cubes, primary-beam correction or dewarping.
- Validated against WSClean only for the orca deep (Briggs 0, Tukey 30) and snapshot settings; the
  robust −0.75 and no-taper presets use the same code but are not compared yet.
- No AOFlagger on the concatenated hour (changed the 1-h difference from 0.111 to 0.114 σ in a test).
- cuFINUFFT adds atomically, so reruns agree only to float32 rounding (CLEAN can amplify this to
  ~1e-4-1e-3 σ).
