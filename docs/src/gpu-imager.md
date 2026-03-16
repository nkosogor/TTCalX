# GPU Wide-Field Imager

GPU-accelerated wide-field imaging with w-term correction using w-stacking,
integrated into the TTCalX calibration pipeline.

## Overview

The GPU imager creates dirty images from visibility data, with full support for:

- **W-stacking**: Corrects w-term phase errors for wide-field imaging
- **GPU acceleration**: CUDA kernels for gridding and w-correction (with CPU fallback)
- **Peel-and-image**: Direct pipeline from peeled visibilities to image without intermediate MS files
- **Weighting schemes**: Natural, uniform, and Briggs (robust) weighting
- **Full Stokes**: Image storage for I, Q, U, V (currently Stokes I implemented)

## Algorithm

The w-stacking algorithm (Offringa et al. 2014):

1. Bin visibilities by w-value into discrete w-layers
2. Grid visibilities onto each w-layer's UV grid (nearest-neighbor)
3. IFFT each w-layer to image domain
4. Apply w-correction phase: $\exp(2\pi i \cdot w_\text{layer} \cdot (\sqrt{1-l^2-m^2} - 1))$ per pixel
5. Sum corrected layers → dirty image

Phase convention (matching CASA / wsclean):
```
V(u,v,w) = ∫∫ I(l,m) · exp(-2πi(ul + vm + w(n-1))) dl dm / n
I(l,m) ≈ Σ_k V_k · exp(+2πi(u_k·l + v_k·m + w_k·(n-1)))
```

## Usage

### Basic imaging (no peeling)

```julia
using TTCalX
using TTCalX.GPUImager

# Load data
init_pycasacore()
vis, cal, meta, baseline_dict, Nrows = read_ms_to_gpu("observation.ms")

# Configure imager
config = GPUImagerConfig(
    image_size = 1024,
    cell_size  = deg2rad(1.0/60.0),  # 1 arcmin
    w_layers   = 16,
    weighting  = :natural
)

# Or auto-configure from observation parameters
config = auto_configure_imager(meta, vis, image_size=1024)

# Create dirty image
img = make_image(vis, meta, config)

# Access image data
stokes_I = img.stokes_I  # CuArray on GPU, or Array on CPU
```

### Peel and image

```julia
# Load sources
sources_raw = read_gpu_sources("sources.json")
sources = [GPUPeelingSource(s) for s in sources_raw]

# Peel and image in one step (all on GPU)
img = peel_and_image(vis, meta, sources, config;
    peeliter = 3,
    maxiter  = 20,
    tolerance = 1e-3,
    minuvw   = 30.0,
    phase_center_ra  = meta.phase_center_ra,
    phase_center_dec = meta.phase_center_dec,
    lst = 0.0
)
```

### Command-line interface

```bash
# Peel sources and image residuals  
julia bin/peel_and_image_gpu.jl peel sources.json observation.ms

# Zest with Briggs weighting
julia bin/peel_and_image_gpu.jl zest --weight=briggs --robust=0.5 sources.json observation.ms

# Image only (no peeling)
julia bin/peel_and_image_gpu.jl image --size=2048 observation.ms

# Run demo with synthetic data
julia bin/peel_and_image_gpu.jl --demo
```

## Configuration

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `image_size` | Int | 512 | Image dimensions (square, must be even) |
| `cell_size` | Float64 | 1 arcmin | Pixel angular size in radians |
| `w_layers` | Int | 1 | Number of w-stacking layers |
| `padding_factor` | Float64 | 1.0 | FFT zero-padding factor |
| `weighting` | Symbol | `:natural` | Weighting scheme |
| `robust` | Float64 | 0.0 | Briggs robust parameter (-2 to 2) |
| `oversampling` | Int | 8 | Gridding convolution oversampling |
| `support` | Int | 3 | Gridding kernel half-width |
| `w_max` | Float64 | 0.0 | Max |w| in wavelengths (0 = auto) |

## Data Types

### `GPUImagerConfig`
Imaging configuration (see table above).

### `GPUGrid`
UV grid for w-stacking. Contains:
- `data`: Complex visibility grid `(Nu, Nv, Nw_layers)`
- `weights`: Sampling density weights `(Nu, Nv, Nw_layers)`
- `w_values`: w-value at center of each layer

### `GPUImage`
Image storage with full Stokes:
- `stokes_I`, `stokes_Q`, `stokes_U`, `stokes_V`: 2D Float64 arrays

## Key Functions

| Function | Description |
|----------|-------------|
| `make_image(vis, meta, config)` | Create dirty image from visibilities |
| `peel_and_image(vis, meta, sources, config)` | Peel + image in one step |
| `grid_visibilities!(grid, vis, meta, config)` | Grid visibilities onto UV grid |
| `grid_to_image!(img, grid, config)` | FFT grid to image with w-correction |
| `auto_configure_imager(meta, vis)` | Auto-configure from observation |
| `field_of_view(config)` | Compute FoV in radians |
| `image_coordinates(img, config)` | Get (l, m) coordinate arrays |

## CUDA Kernels

| Kernel | Description |
|--------|-------------|
| `grid_nn_kernel!` | Nearest-neighbor visibility gridding |
| `grid_convolve_kernel!` | Convolutional gridding with PSWF |
| `w_correction_kernel!` | Per-pixel w-phase correction |
| `normalize_grid_kernel!` | Grid normalization by weights |

## Output Files

The CLI produces these output files:

| File | Format | Description |
|------|--------|-------------|
| `*_dirty.bin` | Binary (Int32 header + Float64 data) | Raw image data |
| `*_dirty.csv` | CSV | Image as comma-separated values |
| `*_dirty.pgm` | ASCII PGM | 64x64 thumbnail for preview |

Load binary files in Python:
```python
import numpy as np
def load_image(path):
    with open(path, 'rb') as f:
        nx, ny = np.fromfile(f, np.int32, 2)
        return np.fromfile(f, np.float64).reshape(nx, ny)
```

## Performance

Typical performance for OVRO-LWA 73 MHz data (352 antennas, 6 channels):

| Operation | CPU | GPU | Speedup |
|-----------|-----|-----|---------|
| Peeling (6 sources) | ~40s | ~6.6s | 6x |
| Imaging (1024x1024) | ~5s | ~0.9s | 6x |
| **Total** | ~46s | ~7.5s | **6x** |

## References

1. Offringa et al. (2014) "WSCLEAN: An implementation of a fast, generic wide-field imager"
2. Cornwell et al. (2008) "The Noncoplanar Baselines Effect in Radio Interferometry"
3. Salvini & Wijnholds (2014) "StEFCal" (calibration algorithm)
4. Briggs (1995) "High Fidelity Deconvolution of Moderately Resolved Sources"
5. Thompson, Moran & Swenson, "Interferometry and Synthesis in Radio Astronomy"
