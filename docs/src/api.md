# API Reference

!!! note
    Full auto-generated API docs (from docstrings) require loading the `TTCalX` module, which depends on CUDA and PyCall. The reference below is maintained manually from the source code docstrings.

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
peel_gpu!(vis, meta, sources; maxiter=30, tolerance=1e-4, minuvw=10.0, peeliter=3, ...)
zest_gpu!(vis, meta, sources; maxiter=30, tolerance=1e-4, minuvw=10.0, peeliter=3, ...)
shave_gpu!(vis, meta, sources; maxiter=30, tolerance=1e-4, minuvw=10.0, peeliter=3, ...)
prune_gpu!(vis, meta, sources; maxiter=30, tolerance=1e-4, minuvw=10.0, peeliter=3, ...)
```

Main calibration entry points. Each wraps sources in the appropriate peeling type and runs the iterative peel loop.

---

## Model Visibilities

```julia
gpu_genvis(meta, source; ...) → GPUVisibilities
```
Generate model visibilities for a source (allocates output).

```julia
gpu_genvis!(vis, meta, source, ...) 
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
gpu_stefcal!(calibration, measured, model, meta; maxiter=30, tolerance=1e-4, ...)
```
Run the StEFCal solver to find antenna-based Jones gains.

---

## MS I/O

```julia
init_pycasacore() → Bool
```
Initialize python-casacore via PyCall. Must be called before any MS operations.

```julia
read_ms_to_gpu(ms_path; gpu=true, column="CORRECTED_DATA")
    → (vis, cal, meta, baseline_dict, Nrows)
```
Read a Measurement Set into GPU-friendly data structures.

```julia
write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column="CORRECTED_DATA")
```
Write GPU visibilities back to a Measurement Set.

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
