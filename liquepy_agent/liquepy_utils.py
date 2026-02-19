"""
Utility functions for liquepy agent.

Provides optional-dependency guards and import helpers following
the same pattern as opensees_agent and pystrata_agent.
"""


def has_liquepy():
    """Return True if liquepy is installed and importable."""
    try:
        import liquepy  # noqa: F401
        return True
    except ImportError:
        return False


def import_liquepy():
    """Import and return the liquepy package, raising RuntimeError if missing."""
    try:
        import liquepy
        return liquepy
    except ImportError:
        raise RuntimeError(
            "liquepy is not installed. Install with: pip install liquepy"
        )


def import_liquepy_trigger():
    """Import and return liquepy.trigger module."""
    import_liquepy()
    import liquepy.trigger as trigger
    return trigger


def import_liquepy_field():
    """Import and return liquepy.field module."""
    import_liquepy()
    import liquepy.field as field
    return field
