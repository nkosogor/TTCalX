# Measurement Set I/O

TTCalX reads and writes CASA Measurement Sets using `python-casacore` via Julia's `PyCall.jl`.

## Overview

The MS bridge (`src/gpu/pycall_ms_bridge.jl`) provides two main functions:

- **`read_ms_to_gpu`** — Read an MS into GPU-friendly data structures
- **`write_gpu_to_ms!`** — Write calibrated visibilities back to the MS

## Reading Data

```
read_ms_to_gpu(ms_path; gpu=true, column="CORRECTED_DATA")
    → (vis, cal, meta, baseline_dict, Nrows)
```

### What it reads

| MS Column | Julia Field | Description |
|-----------|------------|-------------|
| `DATA` or `CORRECTED_DATA` | `vis` | Complex visibility data |
| `FLAG` | `vis.flags` | Boolean flag array |
| `ANTENNA1`, `ANTENNA2` | `meta.baselines` | Baseline → antenna mapping |
| `UVW` | `meta.uvw` | Baseline UVW coordinates |
| `CHAN_FREQ` (spectral window) | `meta.channels` | Channel frequencies in Hz |
| Antenna positions | `meta.antenna_positions` | ITRF XYZ positions |
| Phase center | `meta.phase_center_ra/dec` | Field pointing direction |

### Data Layout

- Visibilities are stored as Struct-of-Arrays (SoA): separate `xx`, `xy`, `yx`, `yy` arrays of shape `(Nbase, Nfreq)`
- Auto-correlations are excluded
- Baselines are indexed via `baseline_dict::Dict{Tuple{Int,Int}, Int}`

### GPU Transfer

When `gpu=true` and CUDA is functional, all arrays are transferred to GPU memory as `CuArray`. Otherwise, standard CPU `Array` is used — the rest of the pipeline works identically.

## Writing Data

```
write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column="CORRECTED_DATA")
```

Transfers data from GPU to CPU (if needed), reconstructs the `(Nrows, Nfreq, 4)` layout expected by casacore, and writes back to the specified column using `python-casacore`.

## Python Initialization

The `init_pycasacore()` function lazily imports `casacore.tables` and `numpy` via PyCall. This must be called before any MS operations:

```julia
if !init_pycasacore()
    error("python-casacore not available. Install with: pip install python-casacore")
end
```

## Dependencies

- **PyCall.jl** — Julia ↔ Python interop
- **python-casacore** — Python bindings for casacore (MS access)
- **numpy** — Array conversion between Python and Julia
