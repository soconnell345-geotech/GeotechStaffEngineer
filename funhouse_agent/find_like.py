"""Moved to ``planlens.review.find_like`` (BACKLOG PL-29, the app split's step 9).

This name is the same module, so an import or a patch by either name
reaches the one copy of the code. The app's host (its look, signer, file
confinement) is handed over by :mod:`funhouse_agent.document_tools`,
imported first. The old name goes at step 16.
"""

import sys as _sys

from funhouse_agent import document_tools as _binding  # noqa: F401 - the host
from planlens.review import find_like as _moved

_sys.modules[__name__] = _moved
