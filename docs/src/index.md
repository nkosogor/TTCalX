# TTCalX

*GPU-accelerated direction-dependent calibration for the OVRO-LWA.*

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

## Overview

TTCalX performs direction-dependent calibration to remove bright radio sources from interferometric data. All heavy computation runs on CUDA GPUs with automatic CPU fallback.

It is an extended, GPU-accelerated version of the original [TTCal](https://github.com/mweastwood/TTCal.jl) by Michael Eastwood.

## Calibration Modes

| Mode      | Jones Matrix     | Frequency   | Use Case                      |
|-----------|-----------------|-------------|-------------------------------|
| **peel**  | Diagonal (2 params) | Per-channel | Phase/amplitude calibration   |
| **zest**  | Full (4 params)     | Per-channel | + Polarization leakage        |
| **shave** | Diagonal            | Wideband    | Fast diagonal calibration     |
| **prune** | Full                | Wideband    | Fast full Jones calibration   |

## Quick Start

```bash
# Activate environment (on calim servers)
ttcalx_env

# Run zest on a measurement set
ttcalx zest /home/pipeline/sources.json data.ms --verbose
```

For a full walkthrough, see the [Tutorial](@ref tutorial).

## Contents

```@contents
Pages = [
    "tutorial.md",
    "calibration.md",
    "sources.md",
    "kernels.md",
    "msio.md",
    "cli.md",
    "api.md",
]
Depth = 2
```

## Requirements

- **Julia 1.10+**
- **NVIDIA GPU** with CUDA 11+ (falls back to CPU if unavailable)
- **Python 3** with `python-casacore` (for Measurement Set I/O via PyCall)

## Installation

```bash
git clone https://github.com/nkosogor/TTCalX.git
cd TTCalX

julia -e '
using Pkg
Pkg.activate(".")
Pkg.instantiate()
ENV["PYTHON"] = Sys.which("python")
Pkg.build("PyCall")
'
```
