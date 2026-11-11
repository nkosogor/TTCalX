# API Reference

!!! note
    The docs build cannot load `TTCalX` (CUDA, PyCall), so this page is written by hand; the
    docstrings in `src/gpu/*.jl` are the full reference (`?zest_sky_rfi!` in the REPL).

## Main Module (`TTCalX`)

### GPU Availability

```julia
is_gpu_available() → Bool
```
Check if a CUDA GPU is available and functional.

```julia
get_gpu_backend() → Symbol
```
Returns `:cuda` if CUDA is functional, `:cpu` otherwise.

---

## Data Types

### Visibilities

```julia
GPUVisibilities{T,F}
```
GPU-friendly visibility storage using Struct-of-Arrays layout. Fields: `xx`, `xy`, `yx`, `yy` (complex arrays), `flags` (boolean array). Shape: `(Nbase, Nfreq)`.

### Calibration Gains

```julia
GPUCalibration{T,F}
```
GPU-friendly calibration gain storage. Fields: `xx`, `xy`, `yx`, `yy`, `flags`, `is_diagonal`. Shape: `(Nant, Nfreq)`.

### Metadata

```julia
GPUMetadata{T,V,I}
```
Metadata container: antenna positions, baseline mapping, channel frequencies, UVW coordinates, phase center.

```julia
create_gpu_metadata(antenna_positions, baselines, channels, phase_center_ra, phase_center_dec, uvw; gpu=true)
```
Convenience constructor that transfers to GPU when available.

### Jones Matrices

```julia
GPUJonesMatrixArray{T}          # Full 2×2 Jones: xx, xy, yx, yy
GPUDiagonalJonesMatrixArray{T}  # Diagonal Jones: xx, yy only
GPUHermitianJonesMatrixArray{T} # Hermitian Jones
```

---

## Source Models

```julia
read_gpu_sources(filename::String) → Vector{AbstractGPUPeelingSource}
```
Read sources from a JSON file in TTCal format.

### Source Types

```julia
GPUPointSource     # Unresolved point source
GPUGaussianSource  # Elliptical Gaussian
GPUMultiSource     # Multi-component source
```

### Peeling Wrappers

```julia
GPUPeelingSource   # Diagonal Jones, per-channel
GPUShavingSource   # Diagonal Jones, wideband
GPUZestingSource   # Full Jones, per-channel
GPUPruningSource   # Full Jones, wideband
```

### Source Utilities

```julia
get_name(source) → String
unwrap(source) → GPUSource
is_diagonal(source) → Bool
is_wideband(source) → Bool
source_direction_lmn(source, phase_center_ra, phase_center_dec, lst) → (l, m, n)
```

---

## Peeling Functions

```julia
peel_gpu!(vis, meta, sources; peeliter=3, maxiter=20, tolerance=1e-3, minuvw=0.0,
          phase_center_ra=0.0, phase_center_dec=0.0, lst=0.0) → calibrations
zest_gpu!(vis, meta, sources; kwargs...)    # same keywords
shave_gpu!(vis, meta, sources; kwargs...)
prune_gpu!(vis, meta, sources; kwargs...)
```

Damped-StefCal peeling (`bin/ttcal_gpu.jl`, which passes `maxiter=30`, `tolerance=1e-4`,
`minuvw=10`). Each wraps the sources in the matching peeling type and runs the peel loop.

---

## Model Visibilities

```julia
gpu_genvis(meta, source; phase_center_ra=0.0, phase_center_dec=0.0, lst=0.0) → GPUVisibilities
```
Generate model visibilities for a source (allocates output).

```julia
gpu_genvis!(vis, meta, source, phase_center_ra, phase_center_dec, lst)
```
Generate model visibilities and add to existing data (in-place).

---

## Corrupt / Apply Calibration

```julia
gpu_corrupt!(vis, cal, meta)
```
Apply Jones gains: ``V_{ij} \leftarrow J_i \, V_{ij} \, J_j^\dagger``

```julia
gpu_applycal!(vis, cal, meta)
```
Apply inverse calibration: ``V_{ij} \leftarrow J_i^{-1} \, V_{ij} \, (J_j^{-1})^\dagger``

---

## StEFCal Solver

```julia
gpu_stefcal!(calibration, measured, model, meta; maxiter=20, tolerance=1e-3, minuvw=0.0)
```
Damped StEFCal (half steps, one global convergence test). `solve_gains!` below is the general solver.

---

## MS I/O

```julia
init_pycasacore() → Bool
```
Initialize python-casacore via PyCall. Must be called before any MS operations.

```julia
read_ms_to_gpu(ms_path; gpu=true, column="DATA")
    → (vis, cal, meta, baseline_dict, Nrows)
```
Read a Measurement Set into GPU-friendly data structures.

```julia
write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column="DATA")
```
Write GPU visibilities back to a Measurement Set.

```julia
read_ms_fast(ms_path; gpu=true, column="DATA") → (vis, cal, meta, baseline_dict, Nrows)
write_ms_fast!(ms_path, vis, baseline_dict, Nrows; column="DATA")
ms_mjd(ms_path) → Float64
```
Faster MS read/write (row gather in numpy, same results; used by `zest_sky_rfi_ms!`) and the mean `TIME` as MJD (UTC).

---

## Sky + RFI peeling (`bin/zest_sky_rfi.jl`)

```julia
zest_sky_rfi_ms!(ms, sky, rfi=GPUSource[]; column="CORRECTED_DATA", gpu=true, kwargs...)
zest_sky_rfi!(vis, meta, mjd, sky, rfi=GPUSource[]; kwargs...)
    → (peeled, skipped, flagged_zeros, restored, gains)
```
`zest_sky_rfi_ms!` reads `column`, sets geometric UVW, calls `zest_sky_rfi!` and writes back.
Keywords (defaults as the CLI): `rfi_mode=:after`, `solver=:rk4`, `maxiter=5`, `tolerance=1e-4`,
`minuvw=5.0`, `peeliter=3`, `max_gain=100.0`, `min_elevation=0.0`, `diagonal=false`, `reg=nothing`,
`flag_zeros=true`, `report=nothing`, `smooth_flux=0.0`, `rfi_smooth_flux=smooth_flux`,
`smooth_exempt_flux=Inf`, `smooth_exempt_names=()`, `restore=GPUSource[]`,
`restore_below_mhz=Inf`, `restore_offset_deg=0.0`, `precision=:double`, `initial_gains=nothing`
(a previous result's `gains`), `warm_max_age_s=60.0`, `cache_model_sq=(precision == :mixed)`,
`workspace=nothing`.

```julia
PeelWorkspace()                       # reusable buffers: zest_sky_rfi!(...; workspace=ws), bitwise same results
workspace_bytes(ws); free_workspace!(ws)
fast_kernels!(on::Bool); fast_kernels() → Bool   # 0.3.0 GPU kernels, off by default; equal to rounding
```

Building blocks:

| function | purpose |
|---|---|
| `solve_gains!(cal, measured, model, meta; solver=:rk4, maxiter=5, tolerance=1e-4, minuvw=0.0, reg=nothing, smooth_flux=0.0, precision=:double, workspace=nothing)` | one direction's gains (RK4 or damped) |
| `guarded_zest!(vis, meta, coherencies, calibrations, solve; peeliter=3, max_gain=Inf, require_convergence=false, report=nothing, workspace=nothing)` | CPU TTCal's peel loop with the residual-power guard |
| `genvis_exact!(vis, meta, source, mjd)`, `exact_model_tables` | exact sky / RFI model (as CPU TTCal) |
| `geometric_uvw!(meta, mjd)`, `uvw_geometry(meta)`, `j2000_to_itrf_matrix(mjd)` | geometric UVW and frames |
| `restore_point_sources!(vis, before, meta, mjd, sources; others=[], minuvw=5.0, offset_deg=0.0)` | give back what peeling took from point sources |
| `phase_regularization(meta, max_offset_deg, taper_deg; lambda=1.0)`, `PhaseRegularization`, `tukey_prior` | Tukey phase prior (`reg`) |
| `GPURFISource`, `GPURFISpectrum`, `is_rfi`, `wgs84_to_itrf` | near-field RFI emitters |
| `flag_zero_visibilities!`, `residual_power`, `max_gain_amplitude`, `source_elevation_deg` | guard helpers |
| `mask_short_baselines!`, `stefcal_steps!`, `host_makesquare!` | solver internals (GPU and host arrays) |

---

## Memory Utilities

```julia
to_gpu(x)   # Transfer to GPU (no-op if CUDA unavailable)
to_cpu(x)   # Transfer to CPU
```

---

## Logging

```julia
set_verbosity(level::Symbol)  # :quiet, :normal, or :verbose
log_header(title)
log_section(title)
log_step(msg)
log_substep(msg)
log_detail(msg)              # verbose only
log_success(msg)
log_warning(msg)
log_error(msg)
```
