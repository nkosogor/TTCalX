# GPU CLEAN imager (`ttcalx_imager`)

`python/ttcalx_imager` is a Python package (CuPy + cuFINUFFT + python-casacore) that makes the
OVRO-LWA imaging products of the main (orca) pipeline on one GPU, from peeled snapshot
measurement sets:

| Product | Command | WSClean equivalent (orca) |
|---|---|---|
| 10-s snapshot, Stokes I CLEAN | `ttcalx-image snapshot` | `SNAPSHOT_CLEAN_I_PARAMS` |
| 10-s snapshot, Stokes V dirty + QA | `ttcalx-image snapshot` | `SNAPSHOT_PARAMS` (pilot V) + `analyze_snapshot_quality` |
| 1-h Stokes I, multiscale CLEAN | `ttcalx-image hour` | `I-Deep-Taper-Robust-0` |
| 6 x 10-min Stokes I | `ttcalx-image hour` | `I-Taper-10min` |
| 1-h / 10-min Stokes V dirty | `ttcalx-image hour` | `V-Taper-Deep`, `V-Taper-10min` |

The hour command replaces concat, chgcentre, pilot snapshots and the WSClean runs; it does not
run AOFlagger on the concatenated hour (see Limitations).

## Installation and usage

```bash
python -m venv venv && . venv/bin/activate
pip install numpy==1.26.4 scipy astropy python-casacore cupy-cuda12x==13.6.0 cufinufft==2.5.1
pip install -e python/ttcalx_imager[test]
export NUMPY_MADVISE_HUGEPAGE=0     # calim nodes: numpy THP madvise causes multi-second stalls
export CUPY_CACHE_DIR=/somewhere/with/space   # not $HOME if the home quota is full

# 10-s snapshots (one persistent process: plans and w-term tables are cached per geometry)
ttcalx-image snapshot /data/41MHz/*.ms --out-dir snap41/ [--keep-aux] [--write-psf]
#   -> snap41/<stem>-I-image.fits, <stem>-V-image.fits, <stem>-qa.json

# one band-hour: 1 h + 6 x 10 min, I and V, orca phase centre for LST hour 2
ttcalx-image hour @list41.txt --band 41 --out hour41/41MHz --lst-hour 2 --qa-dir snap41/
#   -> hour41/41MHz-1h-{I-image,I-residual,I-model,I-dirty,psf,V-image}.fits
#      hour41/41MHz-10min-t000N-...fits, hour41/41MHz-summary.json
```

All main-pipeline hourly products from one load of the data, with WSClean's file names
(`<prefix>-I-Deep-Taper-Robust-0-image.fits`, `-residual`, `-model`, `-dirty`, `-psf`,
`<prefix>-I-Taper-10min-t0000-image.fits`, `<prefix>-V-Taper-Deep-image.fits`, ...):

```bash
ttcalx-image hour @list41.txt --band 41 --out hour41/41MHz --qa compute \
    --products I-Deep-Taper-Robust-0,I-Deep-Taper-Robust-0.75,I-Deep-NoTaper-Robust-0.75,I-Deep-NoTaper-Robust-0,I-Taper-10min,V-Taper-Deep,V-Taper-10min
#   (--products orca = the same list)
```

| `--products` name | weighting | niter | intervals |
|---|---|---|---|
| `I-Deep-Taper-Robust-0` (+ `V-Taper-Deep`) | briggs 0, inner Tukey 30 | 500000 | 1 |
| `I-Deep-Taper-Robust-0.75` | briggs -0.75, inner Tukey 30 | 500000 | 1 |
| `I-Deep-NoTaper-Robust-0.75` | briggs -0.75 | 150000 | 1 |
| `I-Deep-NoTaper-Robust-0` | briggs 0 | 150000 | 1 |
| `I-Taper-10min` (+ `V-Taper-10min`) | briggs 0, inner Tukey 30 | 50000 | 6 |

All use multiscale (bias 0.8), mgain 0.95, auto-mask 3, auto-threshold 0.5, local RMS and the
10 deg horizon mask (orca `IMAGING_STEPS`). A V product requested without its I product is made
without CLEAN. Snapshot QA always uses the pilot weighting (briggs 0, Tukey 30). The summary JSON
times every step per product (`"<product>: <step>"`).

The geometry follows the orca tiers (`bands.py`): 18-36 MHz 1507 px at 0.0767 deg, 41-59 MHz
2357 px at 0.049 deg, 64-82 MHz 3122 px at 0.037 deg; as in WSClean, odd sizes are rounded up
to even (2357 -> 2358).

Python API: `ttcalx_imager.snapshot.SnapshotImager().image(ms, out_prefix=...)`,
`ttcalx_imager.hour.image_hour(paths, band, out_prefix, ...)`.

## What it does

**Snapshot.** Read the MS (casacore), form I = (XX+YY)/2 and V = i(YX-XY)/2 with the
XX|YY flag mask, compute Briggs weights on the GPU, grid dirty I and the PSF with cuFINUFFT
(w-term by SVD factorisation, see below), run Clark/Hogbom CLEAN with WSClean's auto-mask,
auto-threshold and local-RMS logic and two exact major cycles (DFT prediction of the sparse
model, degridding above 8000 components), restore with the WSClean beam fit, and grid dirty V
with the same weights. QA (`<stem>-qa.json`): std of the central 1024 x 1024 pixels of V (the
orca statistic), V peak, residual rms/MAD, flag and exactly-zero fractions, beam, iteration
counts, timings. Snapshots with no unflagged or only zero data get `status: "empty"` and no
images.

**Hour.** Load all snapshots in worker processes and compute each one's exact rotation to the
hour phase centre; reject snapshots by the orca QA rule (V rms > median + 3 MAD-std or
< 0.5 median; V rms from the snapshot QA JSON when available, otherwise computed); build one
Briggs density grid over all accepted samples (the weighting WSClean applies to the
concatenated MS); grid by *w-snapshots*: blocks of 6 consecutive snapshots are gridded in the
frame of their middle snapshot, where |w| is small, and reprojected to the hour SIN grid;
multiscale CLEAN with WSClean-style major cycles in which the model is reprojected to each block
frame and degridded there; restore with the WSClean beam fit of the hour PSF. The 10-min images
use `-intervals-out 6` boundaries on the original time-step index.

## Mapping to WSClean options

| WSClean option | Implementation |
|---|---|
| `-weight briggs R` | `weights.WeightGrid`: density grid on the *padded* image (ceil(1.2 N) rounded up to a multiple of 4), cell 1/(N_pad * scale), half-plane fold (v < 0 -> -u, -v), floor binning; f^2 = (5 * 10^-R)^2 / (sum W^2 / sum W) |
| `-taper-inner-tukey L` | 0.5 (1 - cos(pi r / L)) for r < L, r = uv distance of the weight-grid cell corner |
| (gridding) | cuFINUFFT type 1 / type 2, eps 1e-4, full w-term; pixels beyond the horizon set to 0 |
| `-niter`, `-gain 0.1` | Hogbom iterations on candidate pixel sets (one CUDA block, exact PSF subtraction) |
| `-mgain 0.9999` (snapshots) | 2 major cycles (what WSClean does with 0.9999 on our data) |
| `-mgain 0.95` (hour) | minor loop stops at max(threshold, 0.05 peak), then major cycle |
| `-auto-mask A` | phase 1 to A * sigma; component positions (per scale) become the mask |
| `-auto-threshold T` | phase 2 inside the mask to T * sigma |
| `-local-rms` | radler RmsImage: sqrt(Gaussian-smoothed residual^2), FWHM 25 theoretical beams; peak search on residual * min(rms)/rms |
| `-horizon-mask 10deg` | components only where elevation > 10 deg |
| `-multiscale -multiscale-scale-bias 0.8` | scales 0, 4 x theoretical beam, doubling to N/2; tapered-quadratic kernels; bias 0.8^-k |
| beam fit | port of WSClean's `Fit2DGaussianCentred` (see below) |
| `chgcentre` | `rotate.py`: casacore measures UVW, least-squares R, re-phasing exp(+2 pi i (w_old - w_new)) |

## Conventions

* **Gridding kernel**: I(l, m) = Re sum c exp(+2 pi i (u l + v m - w (n - 1))), prediction is the
  adjoint (WSClean/ducc0 convention, `divide_by_n=False`). Consequently (l, m, -n) transforms
  like (u, v, w) under a phase-centre rotation, which is what the w-snapshot reprojection uses.
* **Orientation**: raw NUFFT arrays are [ix(l), iy(m)]; images are stored in WSClean/FITS memory
  order [y, x] with x = ix and y = (N - iy) mod N (RA increases to the left, CDELT1 < 0). The
  image centre is pixel (N/2, N/2) (0-based), CRPIX = N/2 + 1.
* **w-term**: exp(-2 pi i w (n - 1)) = exp(i pi w) exp(-2 pi i w x'), x' = n - 1/2, and the second
  factor is approximated by sum_k a_k(w) b_k(x') from an SVD on a (w, x') grid with max error
  1e-4. Each term is one 2-D NUFFT; K ~ 6-8 for a snapshot or a w-snapshot block, ~25-37 if the
  whole hour were gridded in the hour frame (`--mode vis`).
* **Phase rotation**: uvw_new = R uvw_stored with R fitted to casacore-measures UVW
  (A[a1] - A[a2], the MS sign convention; includes precession/nutation/aberration as
  chgcentre), vis_new = vis exp(+2 pi i (w_stored - w_new) / lambda).
* **Reprojection flux**: dirty/residual images (Jy/beam) are resampled as-is; a model (Jy/pixel)
  moved from frame A to frame B is multiplied by the SIN pixel solid-angle Jacobian n_A / n_B.
* **Stokes V** = i (YX - XY) / 2 for XX, XY, YX, YY correlations (verified against `wsclean -pol IV`).
* **Restoring beam**: BPA from north (+y) through east (-x), as in the FITS header.

### Restoring-beam fit

`beam.py` ports WSClean 3.4's PSF fit: box = 10 x theoretical beam (1 / max|uvw|) rounded up to
even and grown until it exceeds 8 x the fitted FWHM; Levenberg-Marquardt fit of a peak-1 centred
elliptical Gaussian (sigma_x, sigma_y, beta) including negative pixels; eigen decomposition with
aocommon's eigenvector convention for the PA. The C++ evaluates pixel offsets as unsigned
integers, so pixels left of or below the centre get a model value of exactly 0 and only one
quadrant of the box constrains the fit; the port reproduces this deliberately. On WSClean's own
PSFs the port returns WSClean's BMAJ/BMIN to ~1e-6 and BPA to ~1e-5 deg (a full-box fit would be
1-2 % off). The prototype's main-lobe fit was 1-3 % wider than WSClean's.

## Validation (calim01, RTX A4000, 2026-10-23 08h UTC, peeled with TTCalX `feature/cpu-matched-zest`)

Metrics: restored-image difference rms / reference residual rms over elevation > 10 deg
("diff/sigma"), restoring-beam ratio, median peak-flux ratio of the (up to) 50 brightest
sources, residual-rms ratio. References are stock WSClean 3.4 runs (`/opt/bin/wsclean -j 16`)
with the orca parameters on the same input MSes.

**Snapshots** (12 per band, 5 min apart; `ttcalx-image snapshot` vs WSClean
`SNAPSHOT_CLEAN_I_PARAMS` and pilot V):

| | 41 MHz (2358 px) | 69 MHz (3122 px) |
|---|---|---|
| restored diff/sigma, min / median / max | 0.049 / 0.070 / 0.082 | 0.054 / 0.061 / 0.095 |
| same with the prototype's beam fit | 0.052 / 0.080 / 0.091 | 0.057 / 0.063 / 0.095 |
| BMAJ, BMIN ratio to WSClean | 1.000002, 1.000007 | 1.00014, 1.00021 |
| BPA difference | < 3e-5 deg | 0.002 deg |
| prototype beam fit, BMAJ, BMIN ratio | 1.011, 1.032 | 1.018, 1.016 |
| source flux ratio (median over snapshots) | 1.0000 (0.998-1.0015) | 1.0003 (1.0001-1.0005) |
| residual rms ratio | 0.991-0.996 | 0.995-0.998 |
| dirty V: diff rms / V rms | 0.009-0.021 | 0.010-0.012 |
| QA statistic (central-box V std) ratio | 0.99999 | 0.99995 |
| time per snapshot, I clean + V + QA, incl. MS read and FITS write | 0.55 s | 0.80 s |
| WSClean clean I + pilot V (-j 16) | 2.6 s + 1.3 s | 5-8 s + 2.1-2.6 s |

The first snapshot of a geometry takes ~3.2 s (cuFINUFFT plans, w-term tables, CUDA kernel
compilation). Steady-state breakdown at 69 MHz: MS read 0.18 s, gridding of dirty I / V 0.08 s
each, CLEAN with 2 major cycles and restore 0.30 s, FITS write 0.05 s.

**Hour mode on the same 12 snapshots** (one per 5 min; WSClean on a virtualconcat + chgcentre of
the same MSes, `I-Deep-Taper-Robust-0`, `I-Taper-10min`, `V-Taper-Deep`, `V-Taper-10min`; no QA
rejection on either side):

| | 41 MHz | 69 MHz |
|---|---|---|
| 1-h restored diff/sigma | 0.091 | 0.105 |
| same with the prototype's beam fit | 0.095 | 0.119 |
| 1-h BMAJ, BMIN ratio; BPA difference | 1.00001, 1.00000; 0.001 deg | 1.00001, 1.00000; 0.000 deg |
| 1-h source flux ratio | 0.9998 +- 0.0042 (36 sources) | 1.0004 +- 0.0018 (50) |
| 1-h residual rms ratio | 0.990 | 1.0005 |
| 1-h V diff / V rms | 0.014 | 0.013 |
| 10-min (6 x 2 snapshots) diff/sigma | 0.117-0.140 | 0.088-0.139 |
| 10-min beam ratios | within 1.3e-4 | within 1e-4 |
| 10-min V diff / V rms | 0.011-0.022 | 0.016-0.020 |
| GPU: 1 h + 6 x 10 min, I + V, incl. load and QA | 36 s | 52 s |
| WSClean: deep I + V, 10-min I + V (-j 16) | 40 + 7 s, 245 + 9 s | 451 + 10 s, 511 + 16 s |

With `--qa compute` the orca QA rule rejected 2 of the 12 41 MHz snapshots (08:45:12 and
08:55:13, the latter with a 30x higher residual rms) and none at 69 MHz.

**Full hour, 41 MHz.** The 308 snapshots of 2026-10-23 08h that the prototype study used were
re-prepared from the night-time averaged archive (same bad-antenna lists and BP/XY tables) and
re-peeled with TTCalX CPU-matched zest (RFI on). With the main pipeline's 20 QA rejections
(288 snapshots), against the main-pipeline WSClean image (concat, chgcentre, AOFlagger, pilot QA,
`I-Deep-Taper-Robust-0`; sigma = its residual rms 0.819 Jy/beam):

| | package | prototype (original peel) |
|---|---|---|
| restored diff/sigma, el > 10 / > 60 / 30-60 / 10-30 deg | 0.073 / 0.090 / 0.061 / 0.074 | 0.111 / 0.159 / 0.090 / 0.083 |
| max abs diff / sigma | 2.5 | 15.8 |
| BMAJ, BMIN ratio; BPA difference | 1.0008, 1.0006; 0.007 deg | 1.018, 1.031; -1.9 deg |
| source flux ratio (50 brightest) | 1.0002 +- 0.0024 | 1.0026 +- 0.0069 |
| residual rms ratio | 0.993 | 0.992 |

(The prototype reached 0.072 sigma only when restored with WSClean's beam.) Timing: 118 s for
the 1-h I + V (load and rotation 28 s, weights and setup 9 s, dirty I 10 s, multiscale CLEAN with
3 major cycles 57 s, dirty V 10 s); 216 s for 1 h + 6 x 10 min, I + V, with on-the-fly QA. The
main pipeline took 42 min for the same products on the same node (-j 44: concat 9.4 min,
chgcentre 1.8, AOFlagger 4.5, pilot snapshots 8.6, imaging 17.6). Without AOFlagger, our QA
rejected 9 snapshots, all among the main pipeline's 20 (median V rms 0.887 Jy/beam, MAD-std
0.0057).

**Package vs the validated prototype** (same inputs, restoring beam forced to the prototype's):
dirty images agree to ~3e-7 sigma rms; restored snapshots to 2e-7-2e-4 sigma rms (worst pixel
0.09 sigma) and 1-h images to 8e-5 (41 MHz) and 7e-4 sigma rms (69 MHz). These are float32
rounding differences amplified by CLEAN decisions: two runs of the prototype itself differ by
the same amount (e.g. 1.3e-4 sigma rms for a 69 MHz snapshot, 9e-5 sigma for the 69 MHz hour)
because cuFINUFFT spreads with atomic additions.

## Limitations

* Stokes I (CLEAN) and V (dirty) only; no Q/U, no MFS / multi-frequency deconvolution
  (one spectral window imaged as a single channel) and no spectral cubes.
* No primary-beam correction and no dewarping; outputs correspond to WSClean `-image` files.
* Only the orca deep configuration (Briggs 0, inner Tukey 30 lambda, multiscale bias 0.8,
  auto-mask 3 / auto-threshold 0.5, 1 h and 10 min) and the snapshot configuration are
  validated against WSClean. The robust -0.75 and no-taper presets use the same code paths
  (the Briggs weights for robust -0.75 and no taper are unit-tested) but are not yet
  compared with WSClean images.
* The hour mode does not run AOFlagger on the concatenated hour; inputs are the peeled
  snapshots as they are (in the 2026-10-23 prototype test, skipping AOFlagger changed the
  1-h difference to the main pipeline from 0.111 to 0.114 sigma).
* The snapshot clean always does 2 major cycles (no `-mgain` convergence test).
* The hour mode needs the peeled snapshot MSes; the GPU subband-imaging worker currently deletes
  each peeled MS after imaging it.
* cuFINUFFT uses atomic additions, so results are reproducible only to float32 rounding
  (~1e-7 of the noise for dirty images; CLEAN can amplify this to ~1e-4-1e-3 sigma rms).
* python-casacore's bundled measures tables may be older than the data (it logs "IERS table"
  warnings); UT1-UTC is then taken as 0, an error of < 1 s in time (< 15 arcsec, < 0.1 pixel).
