"""FHWA-NHI-05-037 "Geotechnical Aspects of Pavements" reference adapter.

Geotechnical inputs for pavement design from FHWA-NHI-05-037 (FHWA, May 2006):
resilient modulus Mr (default values by AASHTO/USCS class and CBR/R-value/DCP/
plasticity correlations, the stress-dependent granular Mr model, seasonal and
backcalculated-to-design adjustment), typical CBR by soil class, soil
suitability as a pavement material, drainage modifier/coefficient and
permeability, frost-susceptibility classification (F1-F4), swell potential, and
compaction characteristics. Units follow the source (Mr in psi, CBR/R-value in
%, unit weight in pcf). DISTINCT from the UFC 3-250-01 roads/parking pavement-
design module (``ufc_pavement``).
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.fhwa_pavements import equations, tables
    registry, info = build_lookup_registry([
        (tables, "FHWA Pavements Tables", "FHWA-NHI-05-037"),
        (equations, "FHWA Pavements Equations", "FHWA-NHI-05-037"),
    ])
    add_text_retrieval(registry, info, "fhwa_pavements", "FHWA-NHI-05-037")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
