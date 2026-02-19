"""
Utility functions for pygef agent.

Provides optional-dependency guards and import helpers.
"""


def has_pygef():
    """Return True if pygef is installed and importable."""
    try:
        import pygef  # noqa: F401
        return True
    except ImportError:
        return False


def import_pygef():
    """Import and return the pygef package, raising RuntimeError if missing."""
    try:
        import pygef
        return pygef
    except ImportError:
        raise RuntimeError(
            "pygef is not installed. Install with: pip install pygef"
        )
