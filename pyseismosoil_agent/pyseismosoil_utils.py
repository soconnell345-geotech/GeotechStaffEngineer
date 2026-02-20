"""
Utility functions for PySeismoSoil agent.
"""


def has_pyseismosoil():
    """Check if PySeismoSoil is installed and importable."""
    try:
        import PySeismoSoil  # noqa: F401
        return True
    except ImportError:
        return False


def import_pyseismosoil():
    """Import and return key PySeismoSoil classes."""
    try:
        from PySeismoSoil.class_curves import MKZ_Param, HH_Param
        from PySeismoSoil.class_Vs_profile import Vs_Profile
        return MKZ_Param, HH_Param, Vs_Profile
    except ImportError:
        raise ImportError(
            "PySeismoSoil is not installed. Install with: pip install PySeismoSoil"
        )
