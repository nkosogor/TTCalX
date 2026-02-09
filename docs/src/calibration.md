# Calibration

TTCalX implements the **StEFCal** (Statistically Efficient and Fast Calibration) algorithm on GPU for direction-dependent calibration of radio interferometric data.

## Algorithm Overview

For each bright source, TTCalX solves for antenna-based Jones gains ``J_i`` that minimize:

```math
\sum_{i \neq j} \left\| V_{ij} - J_i \, M_{ij} \, J_j^\dagger \right\|^2
```

where ``V_{ij}`` are the measured visibilities and ``M_{ij}`` are the model visibilities for that source.

## Peeling Loop

The peeling procedure iterates over sources multiple times (`--peeliter`):

1. **Put-back**: Add the current best model (corrupted by the last gains) back into the residual visibilities.
2. **Solve**: Run StEFCal to find the best-fit Jones gains for this source.
3. **Subtract**: Remove the source model (corrupted by the new gains) from the data.

After all peeling iterations, the residual visibilities contain only the sky signal minus the bright sources.

## StEFCal Solver

The StEFCal update equation for diagonal Jones gains is:

```math
g_j^{\text{new}} = \frac{\sum_i (G_i M_{ij})^\dagger V_{ij}}{\sum_i (G_i M_{ij})^\dagger (G_i M_{ij})}
```

For full 2×2 Jones matrices, the update involves a matrix solve per antenna per frequency channel.

### Convergence

The solver iterates until either:
- The relative change falls below `--tolerance`: ``\|\delta g\| < \text{tol} \times \|g\|``
- The maximum number of iterations `--maxiter` is reached

Half-step damping is applied to improve stability: ``g \leftarrow \frac{1}{2}(g_{\text{old}} + g_{\text{new}})``

## Calibration Modes

### Peel (diagonal, per-channel)
- 2 complex parameters per antenna per frequency channel
- Best for amplitude + phase calibration
- Fastest mode

### Zest (full Jones, per-channel)
- 4 complex parameters per antenna per frequency channel
- Handles polarization leakage and Faraday rotation
- Recommended for OVRO-LWA data at low frequencies

### Shave (diagonal, wideband)
- 2 complex parameters per antenna per subband
- Fewer degrees of freedom than peel
- ⚠️ Currently untested

### Prune (full Jones, wideband)
- 4 complex parameters per antenna per subband
- Fewer degrees of freedom than zest
- ⚠️ Currently untested

## GPU vs CPU

TTCalX uses CUDA for all heavy computation:
- **Model visibility generation** (genvis): per-baseline fringe computation
- **Corrupt / applycal**: Jones matrix multiplication per baseline
- **StEFCal iterations**: gain updates per antenna per frequency
- **Makesquare**: baseline→antenna-antenna matrix reorganization

When CUDA is unavailable, all operations fall back to equivalent CPU implementations automatically. The CPU path produces identical results but is significantly slower.

## Key Parameters

| Parameter | Flag | Default | Description |
|-----------|------|---------|-------------|
| Max iterations | `--maxiter` | 30 | Maximum StEFCal iterations per source per peel iteration |
| Tolerance | `--tolerance` | 1e-4 | Relative convergence threshold |
| Min UVW | `--minuvw` | 10.0 | Minimum baseline length in wavelengths (flags shorter baselines) |
| Peel iterations | `--peeliter` | 3 | Number of peeling passes over all sources |
