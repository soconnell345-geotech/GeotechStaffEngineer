"""
Utility functions for SALib agent.
"""


def has_salib():
    """Check if SALib is installed and importable."""
    try:
        import SALib  # noqa: F401
        return True
    except ImportError:
        return False


def import_salib_sobol_sample():
    """Import and return SALib.sample.sobol."""
    try:
        from SALib.sample import sobol
        return sobol
    except ImportError:
        raise ImportError("SALib is not installed. Install with: pip install SALib")


def import_salib_sobol_analyze():
    """Import and return SALib.analyze.sobol."""
    try:
        from SALib.analyze import sobol
        return sobol
    except ImportError:
        raise ImportError("SALib is not installed. Install with: pip install SALib")


def import_salib_morris_sample():
    """Import and return SALib.sample.morris."""
    try:
        from SALib.sample import morris
        return morris
    except ImportError:
        raise ImportError("SALib is not installed. Install with: pip install SALib")


def import_salib_morris_analyze():
    """Import and return SALib.analyze.morris."""
    try:
        from SALib.analyze import morris
        return morris
    except ImportError:
        raise ImportError("SALib is not installed. Install with: pip install SALib")
