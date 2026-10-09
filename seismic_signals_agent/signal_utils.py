"""
Utility functions for the seismic signals agent module.

Provides runtime dependency checking and import helpers for eqsig and pyrotd,
following the same pattern as pystrata_agent/pystrata_utils.py.
"""

# Gravitational acceleration constant for unit conversion (m/s² ↔ g)
_G = 9.81


def has_eqsig() -> bool:
    """Return True if eqsig is importable."""
    try:
        import eqsig  # noqa: F401
        return True
    except ImportError:
        return False


def has_pyrotd() -> bool:
    """Return True if pyrotd is importable.

    False covers two different situations; ``pyrotd_import_error()`` says
    which (not installed, or installed but failing to import).
    """
    return pyrotd_import_error() is None


def pyrotd_import_error():
    """Why pyrotd cannot be imported, or None when it can.

    Distinguishes "not installed" from "installed but its import fails".
    pyrotd 0.6.x reads its own version through ``pkg_resources``, which
    setuptools 81+ no longer ships, so on a fresh environment the package
    is present yet ``import pyrotd`` raises ``ModuleNotFoundError: No module
    named 'pkg_resources'`` (live smoke G10, 2026-10-08). The rotated
    spectrum does not depend on it: ``analyze_rotd_spectrum`` falls back to
    the numpy implementation in ``rotd_native``.
    """
    import importlib.util
    try:
        installed = importlib.util.find_spec("pyrotd") is not None
    except (ImportError, ValueError):
        installed = False
    if not installed:
        return "pyrotd is not installed"
    try:
        import pyrotd  # noqa: F401
        return None
    except Exception as exc:  # ImportError, or anything its import raises
        msg = (f"pyrotd is installed but cannot be imported: "
               f"{type(exc).__name__}: {exc}")
        if "pkg_resources" in str(exc):
            msg += (" (pyrotd 0.6 reads its version through pkg_resources, "
                    "which setuptools 81 and later no longer provide)")
        return msg


def import_eqsig():
    """Import and return eqsig with a helpful error message.

    Returns
    -------
    module
        The eqsig module.

    Raises
    ------
    ImportError
        If eqsig is not installed.
    """
    try:
        import eqsig
        return eqsig
    except ImportError:
        raise ImportError(
            "eqsig is required for response spectrum and intensity measures. "
            "Install with: pip install eqsig"
        )


def import_pyrotd():
    """Import and return pyrotd with a helpful error message.

    Returns
    -------
    module
        The pyrotd module.

    Raises
    ------
    ImportError
        If pyrotd is not installed or its import fails; the message says
        which, with the underlying error.
    """
    err = pyrotd_import_error()
    if err is None:
        import pyrotd
        return pyrotd
    if err == "pyrotd is not installed":
        raise ImportError(
            "pyrotd is not installed. Install with: pip install pyrotd")
    raise ImportError(err)
