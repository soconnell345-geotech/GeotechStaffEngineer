"""
Why an optional package cannot be imported.

A package can be missing, or installed and failing to import: pandas 3 on
numpy 1.26 raised inside SALib's import, and the check reported "SALib is
not installed" (BACKLOG INF-54). ``import_error`` tells the two apart and
gives the import's own error for the second, so a tool and the app's
diagnostics say what is actually wrong. Stdlib only.
"""

import importlib
from typing import Optional

#: The words that open a message for a package that is not installed.
NOT_INSTALLED = "is not installed"

#: The words that open a message for a package whose import fails.
CANNOT_IMPORT = "is installed but cannot be imported"


def import_error(import_name: str, package: Optional[str] = None) -> Optional[str]:
    """Why ``import_name`` cannot be imported, or None when it can.

    ``package`` is the name pip installs (``PyNiteFEA`` for ``Pynite``);
    the import name by default. The package counts as not installed only
    when the module missing is ``import_name`` itself or a parent of it; a
    module missing inside its import (a dependency, a submodule) and any
    other error its import raises mean it is installed and cannot be
    imported, and the message carries that error.
    """
    package = package or import_name
    try:
        importlib.import_module(import_name)
        return None
    except ModuleNotFoundError as exc:
        if exc.name and (import_name == exc.name
                         or import_name.startswith(exc.name + ".")):
            return (f"{package} {NOT_INSTALLED}. "
                    f"Install with: pip install {package}")
        error = exc
    except Exception as exc:  # ImportError, or anything its import raises
        error = exc
    return f"{package} {CANNOT_IMPORT}: {type(error).__name__}: {error}"
