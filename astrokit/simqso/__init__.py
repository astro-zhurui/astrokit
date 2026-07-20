"""Patched simqso bundled in astrokit for mock quasar spectra and SEDs.

Upstream: https://github.com/imcgreer/simqso (BSD-3-Clause)
Patched for Python 3.12+ / scipy / astropy compatibility.
"""

__version__ = '1.2.5dev'

from .sqrun import qsoSimulation, buildSpectraBulk, buildQsoSpectrum
from . import sqbase, sqgrids, sqrun, sqphoto, sqmodels, sqanalysis, lumfun

__all__ = [
    'qsoSimulation',
    'buildSpectraBulk',
    'buildQsoSpectrum',
    'sqbase',
    'sqgrids',
    'sqrun',
    'sqphoto',
    'sqmodels',
    'sqanalysis',
    'lumfun',
]
