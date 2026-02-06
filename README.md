# TTCalX

GPU-accelerated direction-dependent calibration developed for the OVRO LWA.
TTCalX is an extended version of the previous TTCal package by Michael Eastwood (see https://github.com/mweastwood/TTCal.jl).

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

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
- [Source File Format](#source-file-format)
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
    zest /home/pipeline/sources.json your_data.ms
```

For convenience, you can add these to your `~/.bashrc`:

```bash
# TTCalX setup
alias ttcalx_env='conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor && export PATH="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin:$PATH"'
alias ttcalx='julia --project=/opt/devel/nkosogor/nkosogor/TTCalX /opt/devel/nkosogor/nkosogor/TTCalX/bin/ttcal_gpu.jl'

# Then simply run:
# ttcalx_env
# ttcalx zest /home/pipeline/sources.json data.ms

```

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
git clone https://github.com/nkosogor/TTCal-GPU.git
cd TTCal-GPU

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
julia -e '
push!(LOAD_PATH, "src")
include("src/gpu/GPUTTCal.jl")
using .GPUTTCal
println("CUDA available: ", is_gpu_available())
init_pycasacore()
'
```

## Usage

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

The sources file is a JSON array of source objects (see [sources.json](sources.json) in this repo or `/home/pipeline/sources.json` on calim servers).




## Project Structure

```
TTCal.jl/
├── bin/
│   └── ttcal_gpu.jl          # Main CLI script
├── src/
│   └── gpu/
│       ├── GPUTTCal.jl       # Main module
│       ├── types.jl          # GPU data types
│       ├── sources.jl        # Source models
│       ├── peel_gpu.jl       # Peeling implementation
│       ├── pycall_ms_bridge.jl  # MS I/O via python-casacore
│       ├── memory.jl         # Memory utilities
│       └── kernels/          # CUDA kernels
│           ├── utils.jl
│           ├── stefcal.jl
│           ├── corrupt.jl
│           └── genvis.jl
├── examples/
│   └── sources.json          # Example sources file
├── README.md                 # This file
└── LICENSE.md
```

## Troubleshooting

### CUDA not found

```bash
# Check GPU visibility
nvidia-smi

# In Julia, test CUDA
julia -e 'using CUDA; println(CUDA.functional())'
```




