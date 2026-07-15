"""Eurocode 7 Part 1 reference adapter (EN 1997-1:2004).

Geotechnical design general rules: the Annex A normative partial-factor system
(EQU/STR/GEO action sets A1/A2, material sets M1/M2, resistance sets R1-R4 for
spread foundations / driven / bored / CFA piles / anchorages / retaining
structures / slopes, UPL/HYD sets, pile correlation factors xi, and the Design
Approach DA1-C1/DA1-C2/DA2/DA3 set combinations), Annex C sample earth-pressure
procedures, Annex D sample analytical bearing resistance (drained + undrained),
Annex E pressuremeter bearing resistance, Annex F adjusted-elasticity
settlement, Annex G presumed rock bearing, Annex H limiting movements.
SI units throughout.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.eurocode_7_1 import equations, tables
    registry, info = build_lookup_registry([
        (tables, "EC7-1 Tables", "EN 1997-1:2004"),
        (equations, "EC7-1 Equations", "EN 1997-1:2004"),
    ])
    add_text_retrieval(registry, info, "eurocode_7_1", "EN 1997-1:2004")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
