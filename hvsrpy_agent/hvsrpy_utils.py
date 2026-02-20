"""
Utility functions for hvsrpy agent.

Provides optional dependency check and import helpers.
"""


def has_hvsrpy():
    """Check if hvsrpy is installed and importable."""
    try:
        import hvsrpy  # noqa: F401
        return True
    except ImportError:
        return False


def import_hvsrpy():
    """Import and return the hvsrpy module.

    Raises
    ------
    ImportError
        If hvsrpy is not installed.
    """
    try:
        import hvsrpy
        return hvsrpy
    except ImportError:
        raise ImportError(
            "hvsrpy is not installed. Install with: pip install hvsrpy"
        )
