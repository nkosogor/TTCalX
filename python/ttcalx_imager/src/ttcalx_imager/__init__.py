"""GPU CLEAN imager for OVRO-LWA all-sky data (CuPy + cuFINUFFT + python-casacore).

Products (see docs/src/gpu-clean-imager.md):
  * 10-s snapshot: CLEANed Stokes I + dirty Stokes V + QA JSON     (ttcalx_imager.snapshot)
  * 1-h and N x 10-min Stokes I (multiscale CLEAN) + V (dirty)      (ttcalx_imager.hour)

The modules that do not need a GPU (bands, msio, weights, rotate, beam, fits) can be
imported without CuPy installed.
"""
__version__ = "0.1.0"
