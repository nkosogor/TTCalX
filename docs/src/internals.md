# Internals

## Measurement Set I/O

MSes are read and written with `python-casacore` through PyCall (`src/gpu/pycall_ms_bridge.jl`;
`init_pycasacore()` imports it on first use).

```
read_ms_to_gpu(ms_path; gpu=true, column="DATA") -> (vis, cal, meta, baseline_dict, Nrows)
write_gpu_to_ms!(ms_path, vis, baseline_dict, Nrows; column="DATA")
```

`read_ms_fast` / `write_ms_fast!` (`src/gpu/ms_io.jl`) take the same arguments and give the same
results with the row gather/scatter in numpy; `zest_sky_rfi_ms!` uses them. `ms_mjd(ms_path)` is the
mean `TIME` as MJD (UTC). The CLIs pass `--column` (default `CORRECTED_DATA`); the functions default
to `DATA`.

| MS | Julia | |
|---|---|---|
| `DATA` / `CORRECTED_DATA`, `FLAG` | `vis.xx, xy, yx, yy` (Nbase × Nfreq), `vis.flags` | cross-correlations only |
| `ANTENNA1`, `ANTENNA2`, `UVW` | `meta.baselines`, `meta.uvw`, `baseline_dict` | |
| `CHAN_FREQ`, antenna positions, phase centre | `meta.channels`, `meta.antenna_positions`, `meta.phase_center_ra/dec` | |

With `gpu=true` and a working CUDA the arrays are `CuArray`s, otherwise `Array`s; the rest of the code
works on either.

## GPU kernels

| file | contents |
|---|---|
| `src/gpu/kernels/utils.jl` | 2×2 Jones algebra (`jones_multiply`, `jones_inv`, ...), thread indexing |
| `src/gpu/kernels/genvis.jl` | model visibilities: point, Gaussian (and disk, kernel only) sources |
| `src/gpu/kernels/corrupt.jl` | corrupt ``J_i V_{ij} J_j^\dagger`` and applycal ``J_i^{-1} V_{ij} J_j^{-\dagger}`` |
| `src/gpu/kernels/stefcal.jl` | baseline → antenna×antenna matrices (split real/imaginary atomics), StEFCal step per (antenna, channel) |
| `src/gpu/genvis_exact.jl`, `solver.jl`, `guard.jl` | sky + RFI path: exact model, RK4 / damped `solve_gains!` with the normal-equation kernels, guarded peel loop |
| `src/gpu/rfi.jl`, `geometry.jl`, `workspace.jl`, `host_backend.jl` | RFI emitters, precession / geometric UVW, `PeelWorkspace`, the same code on host arrays |

`fast_kernels!(true)` switches on the 0.3.0 kernels (tiled full-Jones normal equations, RFI fringes on
the device, fused corrupt-subtract). They are off by default, so the GPU runs the 748884a kernels; on,
results agree to rounding (summation order). Every kernel has a host version used when the arrays
are not `CuArray`s, with the same results.
