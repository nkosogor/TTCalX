"""FITS output with WSClean-like headers.

Arrays are written as (1, 1, N, N) float32 in WSClean memory order [y, x] (y = 0 is the bottom
row, x increases to the west because CDELT1 < 0).  Keyword order and values follow aocommon
FitsWriter: CRPIX = N/2 + 1, CDELT1 = -scale, CRVAL3 = band centre (f_first + f_last) / 2,
CDELT3 = total bandwidth, Stokes axis 1 (I) or 4 (V), DATE-OBS = start of the first integration.
WSClean's WSC* bookkeeping keywords are not reproduced; ORIGIN says which imager wrote the file.
"""
import math

import numpy as np
from astropy.io import fits
from astropy.time import Time

STOKES = {"I": 1.0, "Q": 2.0, "U": 3.0, "V": 4.0}


def make_header(N, scale_deg, ra_rad, dec_rad, freqs, chan_width, tstart_mjd_s, beam=None,
                stokes="I", telescope="OVRO-LWA", observer="", obj="", history=(), extra=None):
    f = np.asarray(freqs, float)
    h = fits.Header()
    h["BSCALE"] = 1.0
    h["BZERO"] = 0.0
    h["BUNIT"] = ("JY/BEAM", "Units are in Jansky per beam")
    if beam is not None:
        h["BMAJ"], h["BMIN"], h["BPA"] = (float(b) for b in beam)
    h["EQUINOX"] = (2000.0, "J2000")
    h["LONPOLE"] = 180.0
    h["BTYPE"] = "Intensity"
    h["TELESCOP"] = telescope
    if observer:
        h["OBSERVER"] = observer
    if obj:
        h["OBJECT"] = obj
    h["ORIGIN"] = ("ttcalx-imager", "GPU CLEAN imager (CuPy + cuFINUFFT), TTCalX")
    h["CTYPE1"] = ("RA---SIN", "Right ascension angle cosine")
    h["CRPIX1"] = N / 2.0 + 1.0
    h["CRVAL1"] = math.degrees(ra_rad) % 360.0
    h["CDELT1"] = -scale_deg
    h["CUNIT1"] = "deg"
    h["CTYPE2"] = ("DEC--SIN", "Declination angle cosine")
    h["CRPIX2"] = N / 2.0 + 1.0
    h["CRVAL2"] = math.degrees(dec_rad)
    h["CDELT2"] = scale_deg
    h["CUNIT2"] = "deg"
    h["CTYPE3"] = ("FREQ", "Central frequency")
    h["CRPIX3"] = 1.0
    h["CRVAL3"] = 0.5 * (f[0] + f[-1])
    h["CDELT3"] = float(np.sum(chan_width))
    h["CUNIT3"] = "Hz"
    h["CTYPE4"] = "STOKES"
    h["CRPIX4"] = 1.0
    h["CRVAL4"] = STOKES[stokes]
    h["CDELT4"] = 1.0
    h["CUNIT4"] = ""
    h["SPECSYS"] = "TOPOCENT"
    h["DATE-OBS"] = Time(tstart_mjd_s / 86400.0, format="mjd", scale="utc").isot[:21]
    for k, v in (extra or {}).items():
        h[k] = v
    for line in history:
        h.add_history(line)
    return h


def write_image(path, arr, header, stokes=None):
    """Write a 2-D image (numpy or cupy) as (1, 1, N, N) float32."""
    if hasattr(arr, "get"):
        arr = arr.get()
    hh = header.copy()
    if stokes is not None:
        hh["CRVAL4"] = STOKES[stokes]
    fits.writeto(path, np.asarray(arr, np.float32)[None, None], hh, overwrite=True)
    return path
