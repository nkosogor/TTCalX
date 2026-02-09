# GPU Kernels

TTCalX uses CUDA.jl kernels for all performance-critical operations. Each kernel has a matching CPU fallback that produces identical results.

## Architecture

The GPU code is organized in `src/gpu/kernels/`:

| File | Purpose |
|------|---------|
| `utils.jl` | Jones matrix algebra, thread indexing helpers |
| `genvis.jl` | Model visibility generation (point, Gaussian, disk sources) |
| `corrupt.jl` | Corrupt/apply calibration to visibilities |
| `stefcal.jl` | StEFCal gain solver iterations |

## Jones Matrix Operations (`utils.jl`)

All Jones operations are `@inline` device-side functions:

| Function | Operation |
|----------|-----------|
| `jones_multiply(A, B)` | ``C = A \cdot B`` |
| `jones_multiply_conjtrans(A, B)` | ``C = A \cdot B^\dagger`` |
| `jones_conjtrans(A)` | ``A^\dagger`` |
| `jones_det(A)` | ``\det(A)`` |
| `jones_inv(A)` | ``A^{-1}`` |
| `jones_add(A, B)` | ``A + B`` |
| `jones_norm(A)` | ``\|A\|_F`` (Frobenius norm) |
| `diag_jones_multiply(d, A)` | Diagonal × Full multiplication |
| `jones_multiply_diag_conjtrans(A, d)` | Full × Diagonal† multiplication |
| `diag_jones_inv(d)` | Inverse of diagonal Jones |

## Model Visibility Generation (`genvis.jl`)

Generates model visibilities for a source at all baselines and frequencies.

### Point Sources
Computes geometric delay fringes using UVW coordinates:
```math
\phi = 2\pi(u \cdot l + v \cdot m + w \cdot (n - 1)) / \lambda
```

### Gaussian Sources
Same as point sources but with baseline coherency attenuation from the Gaussian envelope (Fourier transform of the sky brightness).

### Disk Sources
Uses Bessel function ``J_1`` for the coherency of a uniform disk model.

## Corrupt & Applycal (`corrupt.jl`)

### Corrupt
Apply Jones gains to model visibilities:
```math
V_{ij}^{\text{corrupted}} = J_i \, V_{ij} \, J_j^\dagger
```

### Applycal
Apply inverse calibration:
```math
V_{ij}^{\text{corrected}} = J_i^{-1} \, V_{ij} \, (J_j^{-1})^\dagger
```

Both handle diagonal and full Jones matrices with separate kernel paths.

## StEFCal Solver (`stefcal.jl`)

### Makesquare
Reorganizes baseline-indexed visibilities into antenna×antenna matrices for the StEFCal update. Uses split real/imaginary atomic additions on GPU (CUDA does not support atomic `ComplexF64`).

### StEFCal Step
One iteration of the gain update. Each CUDA thread handles one `(antenna, frequency)` pair.

**Diagonal:**
```math
g_j^{\text{new}} = \frac{\sum_i (G_i M_{ij})^\dagger V_{ij}}{\sum_i (G_i M_{ij})^\dagger (G_i M_{ij})}
```

**Full 2×2:**
Explicit matrix inversion at each antenna.

## CPU Fallback

When `CUDA.functional()` returns `false`, all operations automatically use CPU implementations defined in `peel_gpu.jl`. The CPU paths are maintained as exact mirrors of the GPU kernels to ensure numerical equivalence.

### Detection Pattern
```julia
use_gpu = _is_gpu(vis)  # checks if arrays are CuArrays
if use_gpu
    # Launch CUDA kernel
else
    # Run CPU loop
end
```
