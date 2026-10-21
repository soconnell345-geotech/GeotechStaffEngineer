"""
The covariance models the module draws random fields with.

GSTools' own model classes, with one fix: GSTools 1.7.0's two-dimensional
``Exponential.spectral_rad_ppf`` returns an infinite wavenumber for every
uniform draw ``u`` within 1e-8 of zero (it tests ``np.isclose(u, 0)``,
whose absolute tolerance is 1e-8), where the exact inverse,
``sqrt(1/u^2 - 1) / len``, is large but finite. One infinite mode makes
the whole randomization-method field NaN. The draw is rare (of GSTools'
65,535 random substreams only 12837 draws a ``u`` that small among a
field's 1,000 modes; master seeds 47863 and 137080 reach it below 200,000)
and it is what put a NaN realization in the owner's 2026-10-15 field
session: realization 863 of a run seeded 47000, i.e. seed 47863 (GM-91,
GM-96). Here only ``u == 0`` exactly is out of range, and it is clipped to
the smallest uniform draw, 2**-53, so every wavenumber is finite and every
other draw gives GSTools' own number, bit for bit.

Kriging and variogram fitting never call the ppf; the module uses
:func:`model_class` wherever it draws a field (``conditional``,
``random_field``).
"""

import numpy as np

#: The smallest positive draw of ``RandomState.uniform`` (53-bit floats).
_U_MIN = 2.0 ** -53


def _exponential_class():
    import gstools as gs

    class Exponential(gs.Exponential):
        """GSTools' Exponential model with a finite 2-D spectral ppf."""

        def spectral_rad_ppf(self, u):
            if self.dim != 2:
                return super().spectral_rad_ppf(u)
            u = np.maximum(np.asarray(u, dtype=np.double), _U_MIN)
            return np.sqrt(1.0 / u ** 2 - 1.0) / self.len_rescaled

    return Exponential


_FIXED = {}


def model_class(model_type):
    """The model class to draw fields with: GSTools' own, or this module's
    fixed copy where GSTools' has a defect (``Exponential`` in 2-D)."""
    import gstools as gs

    if model_type == "Exponential":
        if "Exponential" not in _FIXED:
            _FIXED["Exponential"] = _exponential_class()
        return _FIXED["Exponential"]
    return getattr(gs, model_type)
