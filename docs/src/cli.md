# CLI Reference

Both scripts take one sources file and any number of MSes; pass all MSes in one call so that only
the first pays the Julia compilation. They read and write the same column (`--column`, default
`CORRECTED_DATA`) in place. Exit code 0 on success, 1 on invalid arguments or missing
python-casacore. Environment: [Installation](install.md). The imager CLI (`ttcalx-image`) is
described in [GPU CLEAN imager](gpu-clean-imager.md).

## `bin/zest_sky_rfi.jl`

```
julia bin/zest_sky_rfi.jl [options] <sources.json> <ms1> [ms2] ...
```

Peels the sky sources, then the near-field RFI emitters (`zest_sky_rfi_ms!`): RK4 StefCal as CPU
TTCal, exact sky model, geometric UVW, residual-power guard. The defaults are the production
settings.

| option | default | description |
|---|---|---|
| `--rfi=FILE` | none | near-field RFI model (JSON entries with `rfi-frequencies`) |
| `--rfi-mode=MODE` | `after` | `after` (separate pass after the sky) or `joint` (extra directions) |
| `--solver=S` | `rk4` | `rk4` or `damped` |
| `--maxiter=N` | 5 | max StefCal iterations per solve |
| `--tolerance=T` | 1e-4 | convergence tolerance |
| `--minuvw=M` | 5.0 | min baseline length (wavelengths) |
| `--peeliter=P` | 3 | peeling passes |
| `--max-gain=G` | 100 | reject solutions with \|g\| > G |
| `--min-elevation=E` | 0 | skip sky sources below E degrees |
| `--reg=O,T[,L]` | off | Tukey phase prior: max offset O deg, taper T deg, weight L; not with `--smooth-flux` |
| `--smooth-flux=S` | 0 (off) | directions fainter than ~S Jy get one band gain instead of per-channel gains (suggested 1000) |
| `--rfi-smooth-flux=S` | `--smooth-flux` | the same for the RFI emitters |
| `--smooth-exempt-flux=F` | none | sky sources with catalog flux ≥ F Jy keep per-channel gains (suggested 1e4) |
| `--restore=FILE` | none | point sources (e.g. Jupiter at its ephemeris position) given back after peeling; the name must not be exactly `Sun`, `Moon` or `Jupiter` |
| `--restore-below-mhz=F` | always | restore only when the band starts below F MHz |
| `--restore-offset=D` | 0 | search radius (deg) for the restored position on Stokes V (0.5 for Jupiter) |
| `--precision=P` | `double` | `mixed`: single-precision normal equations (~3-4x faster on GPUs with slow FP64; tolerance floored at 1e-6) |
| `--warm-start` | off | start from the previous MS's gains (same channels, ≤ 60 s apart) |
| `--peel` | off | diagonal Jones instead of full Jones |
| `--keep-zeros` | off | do not flag exactly-zero visibilities |
| `--cpu` | off | CPU threads instead of the GPU (`julia -t N`) |
| `--column=COL` | `CORRECTED_DATA` | MS column to read and write |
| `--verbose` / `--quiet` / `--help` | | every source update / errors only / help |

```bash
julia bin/zest_sky_rfi.jl --rfi=sources/rfi_43.2_ver20251101.json sources/sources_7src_tauA.json *.ms
julia -t 24 bin/zest_sky_rfi.jl --cpu --reg=0.5,0.5 sources.json data.ms
julia bin/zest_sky_rfi.jl --rfi=rfi.json --smooth-flux=1000 --smooth-exempt-flux=1e4 \
      --restore=jupiter.json --restore-below-mhz=41 --restore-offset=0.5 sources.json *.ms
```

## `bin/ttcal_gpu.jl`

```
julia bin/ttcal_gpu.jl <command> [options] <sources.json> <ms1> [ms2] ...
```

Damped StefCal with the kernel sky model and the MS UVW (`peel_gpu!` and friends).

| command | Jones | frequency |
|---|---|---|
| `peel` | diagonal | per channel |
| `zest` | full | per channel |
| `shave` | diagonal | wideband (one per subband) |
| `prune` | full | wideband (one per subband) |

| option | default | description |
|---|---|---|
| `--maxiter=N` | 30 | max StefCal iterations per solve |
| `--tolerance=T` | 1e-4 | convergence tolerance |
| `--minuvw=M` | 10.0 | min baseline length (wavelengths) |
| `--peeliter=P` | 3 | peeling passes |
| `--column=COL` | `CORRECTED_DATA` | MS column to read and write |
| `--verbose` / `--quiet` / `--help` | | per-source output / errors only / help |

```bash
julia bin/ttcal_gpu.jl zest sources/sources_7src_tauA.json *.ms
julia bin/ttcal_gpu.jl peel --maxiter=50 --minuvw=15 --column=DATA sources.json data.ms
```

The script's defaults differ from the library's (`peel_gpu!` defaults: `maxiter=20`,
`tolerance=1e-3`, `minuvw=0`).
