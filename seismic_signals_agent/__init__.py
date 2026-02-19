"""
Seismic Signals Agent Module

Earthquake signal processing using eqsig and pyrotd.

Current analyses:
  - Response spectrum (Nigam-Jennings via eqsig)
  - Intensity measures (Arias intensity, CAV, significant duration)
  - Rotated spectral acceleration (RotD50/RotD100 via pyrotd)
  - Signal processing (filtering + baseline correction via eqsig)

References:
    - eqsig: https://github.com/eng-tools/eqsig
    - pyrotd: https://github.com/arkottke/pyrotd
    - Nigam & Jennings (1969). "Calculation of Response Spectra from
      Strong-Motion Earthquake Records." BSSA 59(2).
"""

from seismic_signals_agent.results import (
    ResponseSpectrumResult,
    IntensityMeasuresResult,
    RotDSpectrumResult,
    SignalProcessingResult,
)
from seismic_signals_agent.response_spectrum import analyze_response_spectrum
from seismic_signals_agent.intensity_measures import analyze_intensity_measures
from seismic_signals_agent.rotd_spectrum import analyze_rotd_spectrum
from seismic_signals_agent.signal_processing import analyze_signal_processing
from seismic_signals_agent.signal_utils import has_eqsig, has_pyrotd

__all__ = [
    'analyze_response_spectrum',
    'analyze_intensity_measures',
    'analyze_rotd_spectrum',
    'analyze_signal_processing',
    'ResponseSpectrumResult',
    'IntensityMeasuresResult',
    'RotDSpectrumResult',
    'SignalProcessingResult',
    'has_eqsig',
    'has_pyrotd',
]
