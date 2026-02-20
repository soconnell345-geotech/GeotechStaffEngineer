"""
Utility functions for geolysis agent.

Provides optional-dependency guards and import helpers following
the same pattern as opensees_agent, pystrata_agent, and liquepy_agent.
"""


def has_geolysis():
    """Return True if geolysis is installed and importable."""
    try:
        import geolysis  # noqa: F401
        return True
    except ImportError:
        return False


def import_geolysis():
    """Import and return the geolysis package, raising RuntimeError if missing."""
    try:
        import geolysis
        return geolysis
    except ImportError:
        raise RuntimeError(
            "geolysis is not installed. Install with: pip install geolysis"
        )


def import_soil_classifier():
    """Import and return geolysis.soil_classifier module."""
    import_geolysis()
    import geolysis.soil_classifier as soil_classifier
    return soil_classifier


def import_spt():
    """Import and return geolysis.spt module."""
    import_geolysis()
    import geolysis.spt as spt
    return spt


def import_bearing_capacity():
    """Import and return geolysis.bearing_capacity modules."""
    import_geolysis()
    import geolysis.bearing_capacity.abc as abc
    import geolysis.bearing_capacity.ubc as ubc
    return abc, ubc
