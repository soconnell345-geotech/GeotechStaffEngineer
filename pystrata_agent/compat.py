"""The soil types the site-response tools build, on pystrata 0.5 or 0.6.

pystrata 0.6.0 (2026-11-05, Python >= 3.13 only) renamed the classes
``pystrata_agent`` and ``opensees_agent``'s curve checks build on
(BACKLOG INF-68):

=============================  =========================================
pystrata 0.5.x                 pystrata 0.6.x
=============================  =========================================
``site.DarendeliSoilType``     ``factories.darendeli_soil_type``
``site.MenqSoilType``          ``factories.menq_soil_type``
  (``uniformity_coeff=``)        (``coef_unif=``)
``site.NonlinearProperty``     ``site.ModulusReductionCurve`` /
  (one class, ``param=``)        ``site.DampingCurve``
``site.SoilType``              ``site.SoilType`` (unchanged arguments)
EQL ``tolerance`` in PERCENT   EQL ``tolerance`` in DECIMAL
  (signed change)                (absolute change)
=============================  =========================================

0.6's factories are a port of 0.5's ``ModifiedHyperbolicSoilType``: the
same Darendeli (2001) and Menq (2003) constants, the same Masing damping
correction and the same default strains (``logspace(-6, -1.5, 20)``), so
both versions give the same curves (checked to 1e-12 in
``pystrata_agent/tests/test_compat.py``). This module is the one place
that knows which API is installed; it picks by what the installed
pystrata has, not by its version string.

0.6's curve classes also clip interpolated values to (0.001, 1) for
G/Gmax and (0, 0.49) for damping; 0.5 did not clip. A custom curve whose
own points sit inside those ranges is unaffected.
"""

import numpy as np


def has_factories(pystrata) -> bool:
    """True when the installed pystrata has the 0.6 API (``factories``)."""
    return not hasattr(pystrata.site, "DarendeliSoilType")


def eql_tolerance(pystrata, fraction):
    """The ``tolerance`` argument of ``EquivalentLinearCalculator`` for a
    relative change given as a fraction (0.01 = 1 %).

    pystrata 0.5 compares a layer's relative change in PERCENT; 0.6
    compares it in DECIMAL (``IterativeValue.relative_error``). Passing the
    0.5 value to 0.6 asks for a 100 % change, so the iteration stops after
    one pass.
    """
    return fraction if has_factories(pystrata) else fraction * 100.0


def _factories(pystrata):
    import pystrata.factories as factories  # noqa: F401 (0.6 only)
    return pystrata.factories


def darendeli_soil_type(pystrata, unit_wt, plas_index=0.0, ocr=1.0,
                        stress_mean=101.3, strains=None):
    """Darendeli (2001) soil type (G/Gmax and damping) on either API."""
    if not has_factories(pystrata):
        return pystrata.site.DarendeliSoilType(
            unit_wt=unit_wt, plas_index=plas_index, ocr=ocr,
            stress_mean=stress_mean, strains=strains)
    return _factories(pystrata).darendeli_soil_type(
        unit_wt=unit_wt, stress_mean=stress_mean, plas_index=plas_index,
        ocr=ocr, strains=strains)


def menq_soil_type(pystrata, unit_wt, uniformity_coeff=10.0, diam_mean=5.0,
                   stress_mean=101.3, strains=None):
    """Menq (2003) soil type (G/Gmax and damping) on either API."""
    if not has_factories(pystrata):
        return pystrata.site.MenqSoilType(
            unit_wt=unit_wt, uniformity_coeff=uniformity_coeff,
            diam_mean=diam_mean, stress_mean=stress_mean, strains=strains)
    return _factories(pystrata).menq_soil_type(
        unit_wt=unit_wt, stress_mean=stress_mean,
        coef_unif=uniformity_coeff, diam_mean=diam_mean, strains=strains)


def linear_soil_type(pystrata, unit_wt, damping, name="Linear"):
    """A linear soil (no modulus reduction, constant damping)."""
    return pystrata.site.SoilType(
        name=name, unit_wt=unit_wt, mod_reduc=None, damping=damping)


def custom_soil_type(pystrata, unit_wt, strains, mod_reduc, damping,
                     name="Custom"):
    """A soil from user G/Gmax and damping points at the given strains."""
    strains = np.asarray(strains, dtype=float)
    mod_reduc = np.asarray(mod_reduc, dtype=float)
    damping = np.asarray(damping, dtype=float)
    site = pystrata.site
    if hasattr(site, "ModulusReductionCurve"):
        mr = site.ModulusReductionCurve(
            name="Custom G/Gmax", strains=strains, values=mod_reduc)
        dm = site.DampingCurve(
            name="Custom Damping", strains=strains, values=damping)
    else:
        mr = site.NonlinearProperty(
            name="Custom G/Gmax", strains=strains, values=mod_reduc,
            param="mod_reduc")
        dm = site.NonlinearProperty(
            name="Custom Damping", strains=strains, values=damping,
            param="damping")
    return site.SoilType(name=name, unit_wt=unit_wt, mod_reduc=mr,
                         damping=dm)
