"""
Soil classification — USCS and AASHTO methods.

Uses the geolysis library for USCS (Unified Soil Classification System)
and AASHTO (American Association of State Highway and Transportation Officials)
soil classification based on index properties.
"""

from geolysis_agent.geolysis_utils import import_soil_classifier
from geolysis_agent.results import ClassificationResult


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_classification_inputs(liquid_limit, plastic_limit, fines=None):
    """Validate common classification inputs."""
    if liquid_limit is not None:
        if not 0 <= liquid_limit <= 200:
            raise ValueError(f"liquid_limit must be 0-200, got {liquid_limit}")

    if plastic_limit is not None:
        if not 0 <= plastic_limit <= 200:
            raise ValueError(f"plastic_limit must be 0-200, got {plastic_limit}")

    if liquid_limit is not None and plastic_limit is not None:
        if plastic_limit > liquid_limit:
            raise ValueError(
                f"plastic_limit ({plastic_limit}) cannot exceed "
                f"liquid_limit ({liquid_limit})"
            )

    if fines is not None:
        if not 0 <= fines <= 100:
            raise ValueError(f"fines must be 0-100%, got {fines}")


def _validate_uscs_inputs(liquid_limit, plastic_limit, fines, sand, d_10, d_30, d_60):
    """Validate USCS-specific inputs."""
    _validate_classification_inputs(liquid_limit, plastic_limit, fines)

    if sand is not None:
        if not 0 <= sand <= 100:
            raise ValueError(f"sand must be 0-100%, got {sand}")

    # d_10, d_30, d_60 can be None (for non-granular soils)
    for d_name, d_val in [("d_10", d_10), ("d_30", d_30), ("d_60", d_60)]:
        if d_val is not None and d_val < 0:
            raise ValueError(f"{d_name} must be >= 0, got {d_val}")


def _validate_aashto_inputs(liquid_limit, plastic_limit, fines):
    """Validate AASHTO-specific inputs."""
    _validate_classification_inputs(liquid_limit, plastic_limit, fines)

    if fines is not None:
        if not 0 <= fines <= 100:
            raise ValueError(f"fines must be 0-100%, got {fines}")


# ---------------------------------------------------------------------------
# Classification functions
# ---------------------------------------------------------------------------

def classify_uscs(
    liquid_limit=None,
    plastic_limit=None,
    fines=None,
    sand=None,
    d_10=None,
    d_30=None,
    d_60=None,
    organic=False,
):
    """
    Classify soil using USCS (Unified Soil Classification System).

    Parameters
    ----------
    liquid_limit : float, optional
        Liquid limit (%). If None, assumes non-plastic or granular.
    plastic_limit : float, optional
        Plastic limit (%). If None, assumes non-plastic.
    fines : float, optional
        Fines content (% passing #200 sieve). If None, assumes 0.
    sand : float, optional
        Sand content (% passing #4 and retained on #200). If None, assumes 0.
    d_10 : float, optional
        Effective size (mm). If None, gradation-based classification unavailable.
    d_30 : float, optional
        Particle size at 30% passing (mm).
    d_60 : float, optional
        Particle size at 60% passing (mm).
    organic : bool, default False
        True if soil is organic (peat, Pt).

    Returns
    -------
    ClassificationResult
        Contains USCS symbol, description, and input properties.

    Notes
    -----
    - All percentages are 0-100, not decimals
    - geolysis may return dual symbols like 'SW-SC,SP-SC' for borderline cases
    - Requires geolysis library (pip install geolysis)
    """
    _validate_uscs_inputs(liquid_limit, plastic_limit, fines, sand, d_10, d_30, d_60)

    soil_classifier = import_soil_classifier()

    # geolysis expects None for missing values
    clf = soil_classifier.create_uscs_classifier(
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
        fines=fines,
        sand=sand,
        d_10=d_10,
        d_30=d_30,
        d_60=d_60,
        organic=organic,
    )

    result = clf.classify()

    # Compute plasticity index
    if liquid_limit is not None and plastic_limit is not None:
        pi = liquid_limit - plastic_limit
    else:
        pi = None

    return ClassificationResult(
        system="uscs",
        symbol=result.symbol,
        description=result.description,
        group_index=None,  # USCS doesn't use group index
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
        plasticity_index=pi,
        fines=fines,
        sand=sand,
    )


def classify_aashto(liquid_limit=None, plastic_limit=None, fines=None):
    """
    Classify soil using AASHTO (American Association of State Highway and
    Transportation Officials) system.

    Parameters
    ----------
    liquid_limit : float, optional
        Liquid limit (%). If None, assumes non-plastic.
    plastic_limit : float, optional
        Plastic limit (%). If None, assumes non-plastic.
    fines : float, optional
        Fines content (% passing #200 sieve). If None, assumes 0.

    Returns
    -------
    ClassificationResult
        Contains AASHTO symbol (e.g., 'A-7-6(20)'), description, and
        group index.

    Notes
    -----
    - All percentages are 0-100, not decimals
    - AASHTO group index appears in parentheses in the symbol
    - Requires geolysis library (pip install geolysis)
    """
    _validate_aashto_inputs(liquid_limit, plastic_limit, fines)

    soil_classifier = import_soil_classifier()

    clf = soil_classifier.create_aashto_classifier(
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
        fines=fines,
    )

    result = clf.classify()

    # Compute plasticity index
    if liquid_limit is not None and plastic_limit is not None:
        pi = liquid_limit - plastic_limit
    else:
        pi = None

    return ClassificationResult(
        system="aashto",
        symbol=result.symbol,
        description=result.description,
        group_index=result.group_index,
        liquid_limit=liquid_limit,
        plastic_limit=plastic_limit,
        plasticity_index=pi,
        fines=fines,
        sand=None,  # AASHTO doesn't use sand content separately
    )
