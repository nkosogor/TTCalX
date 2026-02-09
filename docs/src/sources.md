# Source Models

TTCalX reads source models from a JSON file and converts them to GPU-friendly data structures.

## Source File Format

The source file is a JSON array of source objects. Each source has:

| Field | Required | Description |
|-------|----------|-------------|
| `name` | ✓ | Source name (e.g., `"Cyg A"`) |
| `ra` | ✓ | Right ascension — degrees or `"HHhMMmSS.SSs"` |
| `dec` | ✓ | Declination — degrees or `"+DDdMMmSS.SSs"` |
| `I` | ✓ | Stokes I flux density (Jy) at reference frequency |
| `freq` | | Reference frequency (Hz) for the spectral model |
| `index` | | Spectral index coefficients `[α₁, α₂, ...]` |
| `Q`, `U`, `V` | | Stokes parameters (default: 0) |
| `major-fwhm` | | Major axis FWHM in arcseconds (Gaussian sources) |
| `minor-fwhm` | | Minor axis FWHM in arcseconds (Gaussian sources) |
| `position-angle` | | Position angle in degrees (Gaussian sources) |
| `components` | | Array of sub-components for multi-component sources |

## Example

```json
[
  {
    "name": "Cyg A",
    "components": [
      {
        "name": "1",
        "ra": "19h59m29.990s",
        "dec": "+40d43m57.53s",
        "I": 43170.55,
        "freq": 1.0e6,
        "index": [0.085, -0.178],
        "major-fwhm": 127.87,
        "minor-fwhm": 22.46,
        "position-angle": -74.50
      },
      {
        "name": "2",
        "ra": "19h59m24.316s",
        "dec": "+40d44m50.70s",
        "I": 6374.46,
        "freq": 1.0e6,
        "index": [0.085, -0.178],
        "major-fwhm": 183.43,
        "minor-fwhm": 141.44,
        "position-angle": 43.45
      }
    ]
  },
  {
    "name": "Cas A",
    "ra": "23h23m27.94s",
    "dec": "+58d48m42.4s",
    "I": 22000.0,
    "freq": 1.0e6,
    "index": [0.3, -0.2]
  }
]
```

## Spectral Model

Source flux follows a log-polynomial model:

```math
\log S(\nu) = \log S_0 + \alpha_1 \log\!\left(\frac{\nu}{\nu_0}\right) + \alpha_2 \left[\log\!\left(\frac{\nu}{\nu_0}\right)\right]^2 + \cdots
```

This is implemented in `GPUPowerLaw` which stores:
- `I, Q, U, V` — Stokes parameters at the reference frequency
- `reference_frequency` — ``\nu_0``
- `index` — coefficients ``[\alpha_1, \alpha_2, \ldots]``

## Source Types

### Point Source (`GPUPointSource`)
Unresolved source. Model visibilities are pure fringes with flux scaling.

### Gaussian Source (`GPUGaussianSource`)
Resolved source with an elliptical Gaussian shape. The baseline coherency is attenuated by the Fourier transform of the Gaussian envelope.

### Multi-component Source (`GPUMultiSource`)
A source composed of multiple sub-components (each a point or Gaussian source). Model visibilities are the sum of all component contributions.

## Peeling Wrappers

Each source is wrapped in one of four peeling types that controls the calibration strategy:

| Wrapper | Jones type | Frequency | Function |
|---------|-----------|-----------|----------|
| `GPUPeelingSource` | Diagonal | Per-channel | `peel_gpu!` |
| `GPUShavingSource` | Diagonal | Wideband | `shave_gpu!` |
| `GPUZestingSource` | Full 2×2 | Per-channel | `zest_gpu!` |
| `GPUPruningSource` | Full 2×2 | Wideband | `prune_gpu!` |
