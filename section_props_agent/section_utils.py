"""Utility functions for the section properties agent."""


def has_sectionproperties():
    """Check if sectionproperties is installed and importable."""
    try:
        import sectionproperties  # noqa: F401
        return True
    except ImportError:
        return False


def import_sectionproperties():
    """Import and return the sectionproperties library modules."""
    try:
        from sectionproperties.analysis import Section
        from sectionproperties.pre import library
        return Section, library
    except ImportError:
        raise ImportError(
            "sectionproperties is not installed. "
            "Install with: pip install sectionproperties"
        )
