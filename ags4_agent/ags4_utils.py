"""
Utility functions for AGS4 agent.
"""


def has_ags4():
    """Check if python-ags4 is installed and importable."""
    try:
        import python_ags4  # noqa: F401
        return True
    except ImportError:
        return False


def import_ags4():
    """Import and return the AGS4 class from python-ags4."""
    try:
        from python_ags4 import AGS4
        return AGS4
    except ImportError:
        raise ImportError(
            "python-ags4 is not installed. Install with: pip install python-ags4"
        )
