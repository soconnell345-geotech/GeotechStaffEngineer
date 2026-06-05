"""California Trenching and Shoring Manual reference adapter (Caltrans).

Geotech / excavation-engineering content from the Caltrans Trenching and Shoring
Manual (June 2011, Revision 2 - July 2025): Cal/OSHA Type A/B/C soil
classification and maximum allowable temporary slopes (Ch 2), soil properties
and Ka/equivalent fluid weight (Ch 3), Rankine/Coulomb/Bell earth pressure
coefficients and log-spiral passive Kp (Ch 4), apparent earth pressure (AEP)
envelopes for braced/anchored walls (Ch 8), soldier-pile arching and cantilever
design (Ch 7), structural overstress/lagging (Ch 6), and special conditions -
bottom heave, piping, slope stability (Ch 10). Native US customary units.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.california_trenching import equations, tables
    registry, info = build_lookup_registry([
        (tables, "Caltrans T&S Tables", "Caltrans T&S Manual"),
        (equations, "Caltrans T&S Equations", "Caltrans T&S Manual"),
    ])
    add_text_retrieval(registry, info, "california_trenching", "Caltrans T&S Manual")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
