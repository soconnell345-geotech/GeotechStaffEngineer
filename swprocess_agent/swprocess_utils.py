"""
Utility functions for swprocess agent.
"""


def has_swprocess():
    """Check if swprocess is installed and importable."""
    try:
        import swprocess  # noqa: F401
        return True
    except ImportError:
        return False


def import_swprocess():
    """Import and return key swprocess classes."""
    try:
        from swprocess import ActiveTimeSeries, Sensor1C, Source, Array1D
        from swprocess.wavefieldtransforms import PhaseShift, FK, FDBF
        return {
            "ActiveTimeSeries": ActiveTimeSeries,
            "Sensor1C": Sensor1C,
            "Source": Source,
            "Array1D": Array1D,
            "PhaseShift": PhaseShift,
            "FK": FK,
            "FDBF": FDBF,
        }
    except ImportError:
        raise ImportError(
            "swprocess is not installed. Install with: pip install swprocess"
        )
