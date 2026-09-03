"""Utility functions for the concrete properties agent."""


def has_concreteproperties():
    """Check if concreteproperties is installed and importable."""
    try:
        import concreteproperties  # noqa: F401
        return True
    except ImportError:
        return False


def import_concreteproperties():
    """Import and return the concreteproperties pieces the wrapper uses."""
    try:
        from concreteproperties.concrete_section import ConcreteSection
        from concreteproperties.material import Concrete, SteelBar
        from concreteproperties import stress_strain_profile as ssp
        from sectionproperties.pre.library import concrete_rectangular_section
        return ConcreteSection, Concrete, SteelBar, ssp, \
            concrete_rectangular_section
    except ImportError:
        raise ImportError(
            "concreteproperties is not installed (requires Python >= 3.12). "
            "Install with: pip install concreteproperties"
        )
