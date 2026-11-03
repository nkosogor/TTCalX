# Installation

Requirements: Julia 1.10+ (tested on 1.10.4), an NVIDIA GPU with CUDA Toolkit 11+ (tested on an
RTX A4000, 16 GB), and Python 3 with `python-casacore` (MS I/O through PyCall).

## Pre-installed on the calim servers

```bash
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor
export PATH="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin:$PATH"
export JULIA_DEPOT_PATH="/tmp/julia_${USER}:/opt/devel/nkosogor/nkosogor/julia_depot"
julia --project=/opt/devel/nkosogor/nkosogor/TTCalX /opt/devel/nkosogor/nkosogor/TTCalX/bin/ttcal_gpu.jl --help
```

`JULIA_DEPOT_PATH` lists a writable cache in `/tmp` and the read-only shared depot with the
installed packages, so no per-user installation is needed. Set this in an alias rather than in
`~/.bashrc` globally; if you also use the original TTCal (`julia060`), use a separate terminal.

The shared install is the version the production pipeline runs (tag `gsi-v2`, 748884a). To run
another version (e.g. `main`, 0.3.0) with the same depot, clone it and copy the shared install's
`Manifest.toml` into the clone (the repository has no manifest; without one Julia reports
"Package CUDA is required but does not seem to be installed"):

```bash
git clone https://github.com/nkosogor/TTCalX.git && cp /opt/devel/nkosogor/nkosogor/TTCalX/Manifest.toml TTCalX/
julia --project=TTCalX TTCalX/bin/zest_sky_rfi.jl --help
```

## From scratch

```bash
# 1. Python with casacore (or an existing casacore environment)
conda create -n ttcal python=3.10 -y && conda activate ttcal
pip install python-casacore

# 2. Julia
wget https://julialang-s3.julialang.org/bin/linux/x64/1.10/julia-1.10.4-linux-x86_64.tar.gz
tar -xzf julia-1.10.4-linux-x86_64.tar.gz && export PATH="$PWD/julia-1.10.4/bin:$PATH"

# 3. TTCalX and its dependencies (PyCall built against the active Python)
git clone https://github.com/nkosogor/TTCalX.git && cd TTCalX
julia -e 'using Pkg; Pkg.activate("."); Pkg.instantiate(); ENV["PYTHON"] = Sys.which("python"); Pkg.build("PyCall")'

# 4. Check
julia --project=. -e 'using CUDA; println("CUDA functional: ", CUDA.functional())'
julia --project=. bin/ttcal_gpu.jl --help
```

The GPU imager is a separate Python package: `pip install -e 'python/ttcalx_imager[gpu,test]'`
(see [GPU CLEAN imager](gpu-clean-imager.md) for the pinned CuPy / cuFINUFFT versions).
