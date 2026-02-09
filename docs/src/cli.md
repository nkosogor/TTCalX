# CLI Reference

TTCalX is invoked via the `ttcal_gpu.jl` script (or the `ttcalx` shell alias on calim servers).

## Usage

```
julia bin/ttcal_gpu.jl <command> [options] <sources.json> <ms1> [ms2] ...
```

Or with the alias:

```
ttcalx <command> [options] <sources.json> <ms1> [ms2] ...
```

## Commands

| Command | Description |
|---------|-------------|
| `peel` | Diagonal Jones matrices, per frequency channel |
| `zest` | Full Jones matrices, per frequency channel |
| `shave` | Diagonal Jones matrices, wideband (one per subband) |
| `prune` | Full Jones matrices, wideband (one per subband) |

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `--maxiter=N` | 30 | Maximum StEFCal iterations per source per peel iteration |
| `--tolerance=T` | 1e-4 | Relative convergence tolerance |
| `--minuvw=M` | 10.0 | Minimum baseline length in wavelengths |
| `--peeliter=P` | 3 | Number of peeling iterations over all sources |
| `--column=COL` | `CORRECTED_DATA` | MS data column to read and write |
| `--verbose` | | Show detailed diagnostic output (source-by-source progress) |
| `--quiet` | | Suppress all output except errors |
| `--help` | | Show help message and exit |

## Examples

### Basic peeling

```bash
julia bin/ttcal_gpu.jl peel sources.json data.ms
```

### Zest with custom parameters

```bash
julia bin/ttcal_gpu.jl zest --maxiter=50 --tolerance=1e-5 sources.json data.ms
```

### Batch processing (recommended)

```bash
julia bin/ttcal_gpu.jl zest sources.json *.ms --minuvw=15
```

### Using DATA column (for split/calibrated data)

```bash
julia bin/ttcal_gpu.jl zest --column=DATA sources.json test.ms
```

## Exit Codes

| Code | Meaning |
|------|---------|
| 0 | Success |
| 1 | Error (invalid arguments, missing files, missing python-casacore) |

## Environment Setup

On calim servers, use the convenience alias:

```bash
# Add to ~/.bashrc
alias ttcalx_env='conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor && export PATH="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin:$PATH" && export JULIA_DEPOT_PATH="/tmp/julia_${USER}:/home/pipeline/.julia"'
alias ttcalx='julia --project=/opt/devel/nkosogor/nkosogor/TTCalX /opt/devel/nkosogor/nkosogor/TTCalX/bin/ttcal_gpu.jl'
```

Then:

```bash
ttcalx_env
ttcalx zest /home/pipeline/sources.json data.ms --verbose
```
