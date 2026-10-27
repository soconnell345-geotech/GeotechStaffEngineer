"""Utility functions for the PyNite agent."""

from geotech_common.optional_packages import import_error


def pynite_import_error():
    """Why PyNiteFEA cannot be imported (not installed, or its import's own
    error), or None when it can (BACKLOG INF-54)."""
    return import_error("Pynite", "PyNiteFEA")


def has_pynite():
    """Check if PyNiteFEA is installed and importable."""
    return pynite_import_error() is None


def import_pynite():
    """Import and return Pynite's FEModel3D."""
    error = pynite_import_error()
    if error:
        raise ImportError(error)
    from Pynite import FEModel3D
    return FEModel3D


#: support presets -> (DX, DY, DZ, RX, RY, RZ) restrained flags
SUPPORT_PRESETS = {
    "fixed":    (True, True, True, True, True, True),
    "pinned":   (True, True, True, False, False, False),
    "roller_y": (False, True, False, False, False, False),  # vertical roller
    "roller_x": (True, False, False, False, False, False),
    "free":     (False, False, False, False, False, False),
}
