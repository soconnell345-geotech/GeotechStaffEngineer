"""
Utility functions for the seismic signals agent module.

Provides runtime dependency checking and import helpers for eqsig and pyrotd,
following the same pattern as pystrata_agent/pystrata_utils.py.
"""

# Gravitational acceleration constant for unit conversion (m/s² ↔ g)
_G = 9.81


def has_eqsig() -> bool:
    """Return True if eqsig is importable."""
    try:
        import eqsig  # noqa: F401
        return True
    except ImportError:
        return False


def has_pyrotd() -> bool:
    """Return True if pyrotd is importable."""
    try:
        import pyrotd  # noqa: F401
        return True
    except ImportError:
        return False


def import_eqsig():
    """Import and return eqsig with a helpful error message.

    Returns
    -------
    module
        The eqsig module.

    Raises
    ------
    ImportError
        If eqsig is not installed.
    """
    try:
        import eqsig
        return eqsig
    except ImportError:
        raise ImportError(
            "eqsig is required for response spectrum and intensity measures. "
            "Install with: pip install eqsig"
        )


def import_pyrotd():
    """Import and return pyrotd with a helpful error message.

    Returns
    -------
    module
        The pyrotd module.

    Raises
    ------
    ImportError
        If pyrotd is not installed.
    """
    try:
        import pyrotd
        return pyrotd
    except ImportError:
        raise ImportError(
            "pyrotd is required for rotated spectral acceleration. "
            "Install with: pip install pyrotd"
        )
