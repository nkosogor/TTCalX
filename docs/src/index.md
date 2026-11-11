# TTCalX

GPU direction-dependent calibration (peeling) and a GPU CLEAN imager for the OVRO-LWA.
TTCalX extends [TTCal](https://github.com/mweastwood/TTCal.jl) by Michael Eastwood.
Source: <https://github.com/nkosogor/TTCalX>.

| part | entry points |
|---|---|
| peeling, RK4 StefCal, exact model, RFI emitters (production) | `bin/zest_sky_rfi.jl`, `zest_sky_rfi_ms!`, `zest_sky_rfi!` |
| peeling, damped StefCal (peel / zest / shave / prune) | `bin/ttcal_gpu.jl`, `peel_gpu!`, `zest_gpu!`, ... |
| GPU CLEAN imager (Python) | `ttcalx-image snapshot`, `ttcalx-image hour` |

## Versions

| tag | commit | used by the OVRO-LWA GPU pipeline (gsi) |
|---|---|---|
| `gsi-v1` | 62f8c54 | until 2026-10-25 |
| `gsi-v2` | 748884a | current production; the pre-installed copy on the calim servers |
| `v0.3.0`, `v0.3.1` | `main` | gsi v3 (0.3.1: same code, updated docs); with default settings it peels bitwise like 748884a, the faster paths are opt-in |

## Pages

```@contents
Pages = [
    "install.md",
    "tutorial.md",
    "peeling.md",
    "gpu-clean-imager.md",
    "cli.md",
    "api.md",
    "internals.md",
]
Depth = 1
```
