"""FHWA NHI-06-088/089 Soils and Foundations Reference Manual adapter.

Volumes I and II of the NHI Soils and Foundations Reference Manual (FHWA,
December 2006): stress and strain in soils, subsurface explorations, soil and
rock description and classification, laboratory testing, slope stability
(Taylor's and Janbu's charts, Figs. 6-14 to 6-19), approach roadway
deformations, shallow foundations, deep foundations (driven piles and drilled
shafts), earth retaining structures and geotechnical reports. Text only: the
chapters are the print's own wording with its page numbers, and the figures
are in the figure catalog (``figure_db``). Chapters 1-7 are Volume I
(FHWA-NHI-06-088), chapters 8-12 Volume II (FHWA-NHI-06-089).
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    # Text only, no Python lookup functions yet
    registry, info = build_lookup_registry([])
    add_text_retrieval(registry, info, "fhwa_soils_foundations",
                       "FHWA-NHI-06-088/089")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
