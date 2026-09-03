"""Utility functions for the PyNite agent."""


def has_pynite():
    """Check if PyNiteFEA is installed and importable."""
    try:
        from Pynite import FEModel3D  # noqa: F401
        return True
    except ImportError:
        return False


def import_pynite():
    """Import and return Pynite's FEModel3D."""
    try:
        from Pynite import FEModel3D
        return FEModel3D
    except ImportError:
        raise ImportError(
            "PyNiteFEA is not installed. Install with: pip install PyNiteFEA"
        )


#: support presets -> (DX, DY, DZ, RX, RY, RZ) restrained flags
SUPPORT_PRESETS = {
    "fixed":    (True, True, True, True, True, True),
    "pinned":   (True, True, True, False, False, False),
    "roller_y": (False, True, False, False, False, False),  # vertical roller
    "roller_x": (True, False, False, False, False, False),
    "free":     (False, False, False, False, False, False),
}
