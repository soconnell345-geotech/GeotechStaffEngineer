"""
Utility functions for GSTools agent.
"""


def has_gstools():
    """Check if gstools is installed and importable."""
    try:
        import gstools  # noqa: F401
        return True
    except ImportError:
        return False


def import_gstools():
    """Import and return the gstools module."""
    try:
        import gstools
        return gstools
    except ImportError:
        raise ImportError(
            "gstools is not installed. Install with: pip install gstools"
        )
