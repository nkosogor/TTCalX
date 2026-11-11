# Peeling

For each bright source TTCalX solves antenna Jones gains ``J_i`` (per channel) that minimise

```math
\sum_{i \neq j} \left\| V_{ij} - J_i \, M_{ij} \, J_j^\dagger \right\|^2 ,
```

with ``V`` the data and ``M`` the source's model, then subtracts ``J_i M_{ij} J_j^\dagger``. A peel pass
puts each source back with its last gains, re-solves, and subtracts it again; `--peeliter` passes are
made over all sources. The gain update is StEFCal: with all other antennas fixed, the best ``J_i`` is a
linear least-squares solution, ``J_i \leftarrow (\sum_j V_{ij} Z_{ij}^\dagger)(\sum_j Z_{ij} Z_{ij}^\dagger)^{-1}``,
``Z_{ij} = M_{ij} J_j^\dagger``.

## Two peeling paths

| | `bin/ttcal_gpu.jl` (`peel_gpu!`, `zest_gpu!`, ...) | `bin/zest_sky_rfi.jl` (`zest_sky_rfi!`) |
|---|---|---|
| solver | damped StEFCal: ``g \leftarrow (g_\text{old} + g_\text{new})/2`` until `--tolerance` or `--maxiter` | RK4 of the StEFCal step, per-channel stopping, as CPU TTCal's `solve!` (`--solver=damped` available) |
| model, UVW | kernel model, MS UVW | exact model (`genvis_exact!`), geometric UVW (`geometric_uvw!`) |
| guard | none | an update is kept only if it lowers the residual power and \|g\| ≤ `max_gain` (`guarded_zest!`) |
| extras | | RFI emitters, phase prior (`--reg`), gain smoothing (`--smooth-flux`), restore of point sources (`--restore`), mixed precision, warm start |
| defaults | maxiter 30, tolerance 1e-4, minuvw 10, peeliter 3 | maxiter 5, tolerance 1e-4, minuvw 5, peeliter 3 |

`zest_sky_rfi.jl` reproduces CPU TTCal and is what the OVRO-LWA GPU pipeline uses. All options:
[CLI Reference](cli.md). Both paths run on CPU arrays too (automatic without CUDA; `zest_sky_rfi.jl --cpu`
with `julia -t N`), with the same results to rounding.

## Modes

| mode | Jones | gains per | wrapper / function |
|---|---|---|---|
| `peel` | diagonal (2) | antenna, channel | `GPUPeelingSource`, `peel_gpu!` |
| `zest` | full 2×2 (4) | antenna, channel | `GPUZestingSource`, `zest_gpu!` (recommended for OVRO-LWA) |
| `shave` | diagonal | antenna, subband | `GPUShavingSource`, `shave_gpu!` (not validated against CPU TTCal) |
| `prune` | full 2×2 | antenna, subband | `GPUPruningSource`, `prune_gpu!` (not validated against CPU TTCal) |

## Source models

A JSON array (TTCal format). Included: `sources/sources_7src_tauA.json` (Cyg A, Cas A, Tau A, Vir A,
Her A, Hya A, Per B; the production sky model) and `sources/rfi_43.2_ver20251101.json` (RFI emitters).

| field | required | meaning |
|---|---|---|
| `name` | ✓ | source name |
| `ra`, `dec` | ✓ | `"19h59m29.990s"`, `"+40d43m57.53s"` (a JSON number is read as radians) |
| `I`, `freq`, `index` | ✓ | Stokes I (Jy) at `freq` (Hz); log-polynomial spectral index ``[\alpha_1, \alpha_2, \ldots]`` |
| `Q`, `U`, `V` | | other Stokes parameters (default 0) |
| `major-fwhm`, `minor-fwhm`, `position-angle` | | Gaussian source (arcsec, arcsec, deg) |
| `components` | | sub-sources of a multi-component source (then no other field is needed) |

```json
[{"name": "Cas A", "ra": "23h23m27.94s", "dec": "+58d48m42.4s", "I": 22000.0, "freq": 1.0e6, "index": [0.3, -0.2]}]
```

The spectrum is ``\log_{10} S(\nu) = \log_{10} I + \alpha_1 x + \alpha_2 x^2 + \ldots``, ``x = \log_{10}(\nu/\nu_0)``
(`GPUPowerLaw`). Sources named exactly `Sun`, `Moon` or `Jupiter` (and `az`/`el` positions) are read
as a placeholder at (0, 0); give moving sources a J2000 position and another name, e.g.
`"Jupiter (ephemeris)"`.

**RFI emitters** (`GPURFISource`, peeled by `zest_sky_rfi!` / `--rfi`): entries with `rfi-frequencies`
(Hz), `rfi-I` (Jy per frequency, optional `rfi-Q/U/V`) and a position: `sys: "WGS84"` with `long`,
`lat` (deg) and height `el` (m), or `sys: "ITRF"` with X, Y, Z in `el`, `long`, `lat`.
