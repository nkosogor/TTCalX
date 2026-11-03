[![CI](https://github.com/nkosogor/TTCalX/actions/workflows/ci.yml/badge.svg)](https://github.com/nkosogor/TTCalX/actions/workflows/ci.yml)
[![Documentation](https://github.com/nkosogor/TTCalX/actions/workflows/docs.yml/badge.svg)](https://nkosogor.github.io/TTCalX/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

# TTCalX

GPU direction-dependent calibration (peeling) and a GPU CLEAN imager for the OVRO-LWA.
TTCalX extends [TTCal](https://github.com/mweastwood/TTCal.jl) by Michael Eastwood.
Documentation: <https://nkosogor.github.io/TTCalX/>.

**Versions used by the OVRO-LWA GPU pipeline (gsi):** `gsi-v1` (deployed until 2026-10-25),
`gsi-v2` = 748884a (current production), `v0.3.0` = `main` (gsi v3). With default settings 0.3.0
peels bitwise like 748884a; the faster paths are opt-in.

## Install

On the calim servers TTCalX is pre-installed at `/opt/devel/nkosogor/nkosogor/TTCalX` (the `gsi-v2`
version; to run `main`, see [install.md](docs/src/install.md)):

```bash
conda activate /opt/devel/pipeline/envs/py38_orca_nkosogor
export PATH="/opt/devel/nkosogor/nkosogor/julia-1.10.4/bin:$PATH"
export JULIA_DEPOT_PATH="/tmp/julia_${USER}:/opt/devel/nkosogor/nkosogor/julia_depot"
julia --project=/opt/devel/nkosogor/nkosogor/TTCalX /opt/devel/nkosogor/nkosogor/TTCalX/bin/ttcal_gpu.jl --help
```

Keep this Julia out of your global `PATH` if you also use the original TTCal (`julia060`); use a
separate terminal for each. Installation from scratch: [docs/src/install.md](docs/src/install.md).
Walkthrough (peel, then image): [tutorial](docs/src/tutorial.md).
Requirements: Julia 1.10+, an NVIDIA GPU with CUDA 11+ (tested on RTX A4000), Python 3 with
`python-casacore`.

## Peeling

Two command-line entry points:

| | `bin/ttcal_gpu.jl` | `bin/zest_sky_rfi.jl` |
|---|---|---|
| peels | sky sources | sky sources, then near-field RFI emitters (or jointly) |
| solver | damped StefCal | RK4 StefCal, as CPU TTCal |
| model, UVW | kernel model, MS UVW | exact model, geometric UVW, divergence guard |
| use | quick peeling | reproduces CPU TTCal (production gsi) |

```bash
julia bin/ttcal_gpu.jl zest sources.json *.ms          # modes: peel, zest, shave, prune
julia bin/zest_sky_rfi.jl --column=DATA --rfi=sources/rfi_43.2_ver20251101.json sources.json *.ms
```

Pass many MSes in one call: the first pays ~1 min of compilation, the rest take seconds each.
`--help` lists every option. Main `bin/zest_sky_rfi.jl` options (all off by default):

| option | what it does |
|---|---|
| `--rfi=FILE`, `--rfi-mode=after\|joint` | peel near-field RFI emitters after the sky, or as extra directions |
| `--solver=rk4\|damped`, `--peel`, `--cpu` | solver; diagonal instead of full Jones; CPU threads (`julia -t N`, same results) |
| `--reg=OFFSET,TAPER` | Tukey phase prior on the gains (degrees) |
| `--smooth-flux=S` | directions much fainter than S Jy get one gain per band, so they cannot absorb other sources |
| `--restore=FILE`, `--restore-offset=0.5` | give back what peeling took from known point sources (e.g. Jupiter; name must not be exactly "Jupiter"), position searched on Stokes V |
| `--precision=mixed`, `--warm-start` | single-precision normal equations; start from the previous snapshot's gains (~4 s → ~1 s per snapshot) |

From Julia: `zest_sky_rfi_ms!(ms, sky, rfi; kwargs...)`, or `zest_sky_rfi!` on visibilities in
memory. Opt-in speed paths for long-running processes: `fast_kernels!(true)` (faster GPU kernels,
equal to rounding) and `zest_sky_rfi!(...; workspace=PeelWorkspace())` (reuses ~2 GB of GPU buffers
across calls, bitwise the same results). The building blocks (`solve_gains!`, `genvis_exact!`,
`geometric_uvw!`, `restore_point_sources!`, `read_ms_fast`) are described in their docstrings and
in the [documentation](https://nkosogor.github.io/TTCalX/).

Source models are JSON arrays ([sources.json](sources.json)). Included:
[sources/sources_7src_tauA.json](sources/sources_7src_tauA.json) (Cyg A, Cas A, Tau A, Vir A, Her A,
Hya A, Per B) and [sources/rfi_43.2_ver20251101.json](sources/rfi_43.2_ver20251101.json) (RFI
emitters; entries with `rfi-frequencies`).

## GPU CLEAN imager

`python/ttcalx_imager` (CuPy + cuFINUFFT) makes the main-pipeline imaging products on one GPU,
emulating WSClean with the orca settings: 10-s snapshots (CLEAN I, dirty V, QA) at 0.5-0.8 s each,
and the 1-h / 10-min products of a band-hour in a few minutes, within 0.06-0.10 sigma of WSClean.

```bash
pip install -e 'python/ttcalx_imager[gpu,test]'
ttcalx-image snapshot /data/41MHz/*.ms --out-dir snap41/
ttcalx-image hour @list41.txt --band 41 --out hour41/41MHz --lst-hour 2 --qa-dir snap41/
```

Details, the mapping to WSClean options and validation: [docs/src/gpu-clean-imager.md](docs/src/gpu-clean-imager.md).

## Tests

```bash
julia --project=. -t 4 test/runtests.jl     # GPU checks run when CUDA works
pytest python/ttcalx_imager/tests           # GPU tests skipped without CuPy/CUDA
```

## Troubleshooting

- `CUDA.functional()` false: check `nvidia-smi`, then
  `julia --project=. -e "using CUDA; println(CUDA.functional())"`.
- "Package CUDA is required but does not seem to be installed": `JULIA_DEPOT_PATH` must include
  the shared depot (see Install), and a clone needs a `Manifest.toml` ([install.md](docs/src/install.md)).
