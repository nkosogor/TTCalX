[![CI](https://github.com/nkosogor/TTCalX/actions/workflows/ci.yml/badge.svg)](https://github.com/nkosogor/TTCalX/actions/workflows/ci.yml)
[![Documentation](https://github.com/nkosogor/TTCalX/actions/workflows/docs.yml/badge.svg)](https://nkosogor.github.io/TTCalX/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# TTCalX

GPU-accelerated direction-dependent calibration developed for the OVRO LWA.
TTCalX is an extended version of the previous TTCal package by Michael Eastwood (see https://github.com/mweastwood/TTCal.jl).


## Table of Contents

- [Overview](#overview)
  - [Calibration Modes](#calibration-modes)
- [Requirements](#requirements)
  - [Server (Linux with NVIDIA GPU)](#server-linux-with-nvidia-gpu)
- [Installation](#installation)
  - [Using pre-installed version on calim servers](#using-pre-installed-version-on-calim-servers)
  - [Installation from scratch](#installation-from-scratch)
    - [1. Python environment with casacore](#1-python-environment-with-casacore)
    - [2. Install Julia](#2-install-julia)
    - [3. Clone and install Julia packages](#3-clone-and-install-julia-packages)
    - [4. Verify installation](#4-verify-installation)
- [Usage](#usage)
  - [Basic Command](#basic-command)
  - [Commands](#commands)
  - [Options](#options)
  - [Examples](#examples)
  - [Batch Processing (Recommended)](#batch-processing-recommended)
- [Tutorial](docs/TUTORIAL.md)
- [Source File Format](#source-file-format)
- [CPU-TTCal-matched zesting (sky + RFI)](#cpu-ttcal-matched-zesting-sky--rfi)
- [GPU CLEAN imager (Python)](#gpu-clean-imager-python)
- [Tests](#tests)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
  - [CUDA not found](#cuda-not-found)

## Overview

TTCalX performs direction-dependent calibration to remove bright radio sources from interferometric data. All heavy computation runs on CUDA when available.

### Calibration Modes

This software is under active development. Peel and zest have undergone limited testing; shave and prune are currently untested and should be used with caution. Supported modes: 

| Mode | Jones Matrix | Frequency | Use Case |
|------|-------------|-----------|----------|
| **peel** | Diagonal (2 params) | Per-channel | Phase/amplitude calibration |
| **zest** | Full (4 params) | Per-channel | + Polarization leakage |
| **shave** | Diagonal | Wideband | Fast diagonal calibration |
| **prune** | Full | Wideband | Fast full Jones calibration |



## Requirements

### Server (Linux with NVIDIA GPU)
- **Julia 1.10+** (tested on 1.10.4)
- **NVIDIA GPU** (tested on NVIDIA RTX A4000 16 GB)
- **CUDA Toolkit 11+**
- **Python 3** with `python-casacore` for MS I/O (via PyCall)

## Installation 

### Using pre-installed version on calim servers

The package is pre-installed at `/opt/devel/nkosogor/nkosogor/TTCalX`.

```bash
# Activate conda environment with casacore
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor

# Add Julia to PATH
export PATH="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin:$PATH"

# Run TTCalX (example: zest 3 sources from an MS file)
julia --project=/opt/devel/nkosogor/nkosogor/TTCalX \
    /opt/devel/nkosogor/nkosogor/TTCalX/bin/ttcal_gpu.jl \
    zest /opt/devel/nkosogor/nkosogor/TTCalX/sources.json your_data.ms
```

For convenience, you can add these to your `~/.bashrc`:

```bash
# TTCalX setup
alias ttcalx_env='conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor && export PATH="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin:$PATH" && export JULIA_DEPOT_PATH="/tmp/julia_${USER}:/opt/devel/nkosogor/nkosogor/julia_depot"'
alias ttcalx='julia --project=/opt/devel/nkosogor/nkosogor/TTCalX /opt/devel/nkosogor/nkosogor/TTCalX/bin/ttcal_gpu.jl'

# Then simply run:
# ttcalx_env
# ttcalx zest /opt/devel/nkosogor/nkosogor/TTCalX/sources.json data.ms

```

> **Note:** The `ttcalx_env` alias sets `JULIA_DEPOT_PATH` with two paths: `/tmp/julia_$USER` (writable, for compiled cache) and `/opt/devel/nkosogor/nkosogor/julia_depot` (read-only, for pre-installed packages). This way no per-user package installation is needed — only the maintainer needs to run `Pkg.instantiate()` once.

> **Important:** Do **not** add Julia 1.10.4 to your PATH globally in `~/.bashrc`. Only the alias should set it. Running `ttcalx_env` modifies the current terminal session's PATH and `JULIA_DEPOT_PATH`, so if you also use original (https://github.com/mweastwood/TTCal.jl) TTCal (in `julia060`), use a **separate terminal** for each:
> - **Terminal 1** (production): `conda activate julia060` → `ttcal.jl peel ...`
> - **Terminal 2** (TTCalX): `ttcalx_env` → `ttcalx zest ...`

### Installation from scratch

#### 1. Python environment with casacore

On OVRO-LWA servers you can use the existing environment (see https://github.com/ovro-lwa/distributed-pipeline/)

```bash
# Use existing conda environment with casacore
conda activate <your_casacore_env>

# Verify python-casacore works
python -c "from casacore.tables import table; print('casacore OK')"
```

Or create a new environment:

```bash
# Create new conda environment
conda create -n ttcal python=3.10 -y
conda activate ttcal

# Install python-casacore
pip install python-casacore
```

#### 2. Install Julia

Download and extract Julia 1.10+:

```bash
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz
tar -xzf julia-1.10.4-linux-x86_64.tar.gz
export PATH="$PWD/julia-1.10.4/bin:$PATH"
```

#### 3. Clone and install Julia packages

```bash
git clone https://github.com/nkosogor/TTCalX.git
cd TTCalX

# Install dependencies and build PyCall with current Python
julia -e '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
ENV["PYTHON"] = Sys.which("python")
Pkg.build("PyCall")
'
```

#### 4. Verify installation

```bash
# Check CUDA is working
julia --project=. -e "using CUDA; println(\"CUDA functional: \", CUDA.functional())"

# Test TTCalX loads and shows help
julia --project=. bin/ttcal_gpu.jl --help
```

## Usage

For a detailed walkthrough with example data, see the [Tutorial](docs/TUTORIAL.md).

### Basic Command

```bash
julia bin/ttcal_gpu.jl <command> [options] <sources.json> <ms_file(s)>
```

### Commands

- `peel` - Diagonal Jones, per-channel
- `zest` - Full Jones, per-channel
- `shave` - Diagonal Jones, wideband
- `prune` - Full Jones, wideband

### Options

| Option | Default | Description |
|--------|---------|-------------|
| `--maxiter=N` | 30 | Max StEFCal iterations |
| `--tolerance=T` | 1e-4 | Convergence tolerance |
| `--minuvw=M` | 10.0 | Min baseline length (wavelengths) |
| `--peeliter=P` | 3 | Number of peeling iterations |
| `--column=COL` | CORRECTED_DATA | MS data column |
| `--verbose` | | Show detailed diagnostic output |
| `--quiet` | | Suppress all output except errors |
| `--help` | | Show help message |

### Examples

```bash
# Peel sources from a single MS
julia bin/ttcal_gpu.jl peel sources.json my_data.ms

# Zest with custom parameters
julia bin/ttcal_gpu.jl zest --maxiter=50 --minuvw=15 sources.json data.ms

# Process multiple MS files (batch mode)
julia bin/ttcal_gpu.jl peel sources.json *.ms

# Use specific data column
julia bin/ttcal_gpu.jl peel --column=DATA sources.json data.ms
```

### Batch Processing (Recommended)

> **Performance Tip:** Always use batch mode when processing multiple MS files! The first MS file includes JIT compilation overhead (~0.5-1 min), but subsequent files run at full speed (~10-12s each). Processing files individually means paying the JIT cost every time.

```bash
# Process all MS files in a directory (RECOMMENDED)
julia bin/ttcal_gpu.jl peel sources.json /path/to/*.ms
```

## Source File Format

The sources file is a JSON array of source objects (see [sources.json](sources.json) in this repo).

## CPU-TTCal-matched zesting (sky + RFI)

The library also provides the pieces to zest exactly like the main pipeline's CPU TTCal
(`ttcal.jl zest`, then `ttcal_dev` for the RFI model), on the GPU or on CPU threads.
They are opt-in: the existing `ttcal_gpu.jl` and `peel_gpu!`/`zest_gpu!` code paths are not
changed, except for two fixes in `sources.jl` that affect every caller (see below).

| Piece | What it does |
|---|---|
| `geometric_uvw!(meta, ms_mjd(ms))` | Replaces the MS UVW (OVRO-LWA: r₁−r₂, unprecessed) with precessed ITRF baselines, so the far-field model follows the apparent sky as in CPU TTCal |
| `genvis_exact!(vis, meta, source, mjd)` | CPU TTCal's sky model: per-component horizon, 3-D Gaussian projection on the local north/east axes, `xy = U − iV` |
| `GPURFISource` (JSON entries with `rfi-frequencies`) | Near-field RFI emitter, as ttcal_dev's `RFISource`; model via `gpu_genvis!` |
| `solve_gains!(cal, data, model, meta; solver=:rk4)` | CPU TTCal's solver (RK4 of the StefCal step, per-channel convergence); `solver=:damped` is `gpu_stefcal!`'s iteration; optional Tukey phase prior `reg=phase_regularization(...)` |
| Host arrays (`read_ms_to_gpu(ms; gpu=false)`) | The same operations on CPU threads (`julia -t N`), same results as the GPU |
| `read_ms_fast` / `write_ms_fast!` | Same results as `read_ms_to_gpu` / `write_gpu_to_ms!`, with the row gather/scatter in numpy |

A zest of one source with CPU TTCal's settings:

```julia
vis, _, meta, bl, nrows = read_ms_fast(ms)
mjd = ms_mjd(ms)
geometric_uvw!(meta, mjd)
model = GPUVisibilities(meta.Nbase, meta.Nfreq)
genvis_exact!(model, meta, source, mjd)
model.flags .= vis.flags                       # flagged data out of both sides of the solve
cal = calibration_type(GPUZestingSource(source), meta.Nant, meta.Nfreq)
solve_gains!(cal, vis, model, meta; solver=:rk4, maxiter=5, tolerance=1e-4, minuvw=5.0)
gpu_corrupt!(model, cal, meta)
gpu_subsrc!(vis, model)
write_ms_fast!(ms, vis, bl, nrows)
```

**Behaviour changes for all callers** (`sources.jl`):
- `GPUPowerLaw` uses TTCal's base-10 logs. Natural logs agreed only for single-index
  spectra, so curved spectra such as Cyg A's were several times too faint.
- Source JSON entries with `rfi-frequencies` are now read as `GPURFISource` (near-field
  RFI emitters) instead of failing to parse.

## GPU CLEAN imager (Python)

`python/ttcalx_imager` is a pip-installable Python package (CuPy + cuFINUFFT + python-casacore)
that makes the main-pipeline imaging products from peeled snapshot MSes on one GPU, emulating
WSClean with the orca settings:

- `ttcalx-image snapshot`: 10-s CLEANed Stokes I (`SNAPSHOT_CLEAN_I_PARAMS`) + dirty Stokes V
  (pilot snapshot) + QA JSON, ~0.6 s (41 MHz) / ~0.9 s (69 MHz) per snapshot on an RTX A4000;
- `ttcalx-image hour`: 1-h and 6 x 10-min Stokes I (multiscale `I-Deep-Taper-Robust-0`,
  `I-Taper-10min`) and dirty V, with chgcentre-exact phase rotation and the orca snapshot QA.

The restoring beam is a port of WSClean's PSF fit (BMAJ/BMIN within 1e-5 of WSClean's).

```bash
pip install -e 'python/ttcalx_imager[gpu,test]'
ttcalx-image snapshot /data/41MHz/*.ms --out-dir snap41/
ttcalx-image hour @list41.txt --band 41 --out hour41/41MHz --lst-hour 2 --qa-dir snap41/
pytest python/ttcalx_imager/tests          # GPU tests are skipped without CuPy/CUDA
```

See [docs/src/gpu-clean-imager.md](docs/src/gpu-clean-imager.md) for the conventions, the
mapping to WSClean options, validation numbers and limitations.

## Tests

```bash
julia --project=. -t 4 test/runtests.jl     # unit tests (CPU; GPU checks run if CUDA works)
julia --project=. test/quick_test.jl        # end-to-end smoke test on the example MS
```

## Project Structure

```
TTCalX/
├── bin/
│   └── ttcal_gpu.jl              # Main CLI script
├── src/
│   ├── TTCalX.jl                 # Main module entry point
│   └── gpu/
│       ├── types.jl              # GPU data types
│       ├── sources.jl            # Source models
│       ├── peel_gpu.jl           # Peeling implementation
│       ├── pycall_ms_bridge.jl   # MS I/O via python-casacore
│       ├── geometry.jl           # Precession/nutation/GAST, geometric UVW
│       ├── rfi.jl                # Near-field RFI sources
│       ├── genvis_exact.jl       # CPU-TTCal-exact sky model (GPU kernel + host threads)
│       ├── regularization.jl     # Tukey phase prior
│       ├── solver.jl             # solve_gains!: RK4 / damped StefCal, GPU + host
│       ├── host_backend.jl       # Multithreaded CPU versions of the peel operations
│       ├── ms_io.jl              # Faster MS read/write, ms_mjd
│       ├── memory.jl             # Memory utilities
│       ├── logging.jl            # Progress bars & verbosity
│       └── kernels/              # CUDA kernels
│           ├── utils.jl
│           ├── stefcal.jl
│           ├── corrupt.jl
│           └── genvis.jl
├── test/
│   ├── runtests.jl               # Unit tests
│   └── quick_test.jl             # Smoke test on the example MS
├── python/ttcalx_imager/         # GPU CLEAN imager (Python package, `ttcalx-image` CLI)
│   ├── src/ttcalx_imager/        # msio, weights, gridder, rotate, clean, beam, snapshot, hour, ...
│   └── tests/                    # pytest (GPU tests skipped without CUDA)
├── examples/
│   ├── sources.json              # Example sources file
│   └── 20240524_090003_73MHz_ch0to5.ms  # Test dataset (6 ch, ~22 MB)
├── Project.toml                  # Julia dependencies
├── README.md                     # This file
└── LICENSE.md
```

## Troubleshooting

### CUDA not found

```bash
# Check GPU visibility
nvidia-smi

# In Julia, test CUDA
julia --project=/opt/devel/nkosogor/nkosogor/TTCalX -e "using CUDA; println(CUDA.functional())"
```

### "Package CUDA is required but does not seem to be installed"

Make sure `JULIA_DEPOT_PATH` includes the shared depot:

```bash
export JULIA_DEPOT_PATH="/tmp/julia_${USER}:/opt/devel/nkosogor/nkosogor/julia_depot"
```

This is already included in the `ttcalx_env` alias. The first path (`/tmp`) is writable for compiled cache, the second path reads pre-installed packages from the shared depot.




