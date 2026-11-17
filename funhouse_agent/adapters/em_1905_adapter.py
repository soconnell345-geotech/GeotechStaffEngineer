"""EM 1110-1-1905 reference adapter (USACE, shallow foundations on soils).

Geotechnical Design of Shallow Foundations on Soils (U.S. Army Corps of
Engineers, 31 July 2025; it supersedes EM 1110-1-1904 (1990) and EM
1110-1-1905 (1992)): Table 2-1 minimum factors of safety; Chapter 5
ultimate limit state (bearing capacity factors, shape, depth, inclination
and undrained factors of Tables 5-2 to 5-4, in-situ bearing factors of
Table 5-5, Eqs. 5-1 to 5-36: effective dimensions, drained and undrained
ultimate bearing capacity, punching, sliding, uplift); Chapter 6 stress
distribution (Eqs. 6-1 to 6-13) and Chapter 7 settlement (Eqs. 7-1 to 7-47:
consolidation, constrained modulus, Menard, Schmertmann, Burland and
Burbidge, time rate, secondary compression; Table 7-6; the table printed on
Figure 7-16). US customary units as printed, each parameter naming its
unit. A lookup refuses an input outside what the print covers; where the
page prints no condition the message says the refusal is the library's
reading. Chapter text by the printed paragraph numbers ('5-5a(2)(b)';
'B-1b' in an appendix).
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)

_REF = "EM 1110-1-1905 (2025)"

#: Names the table/equation-number stripping would give that another lookup
#: could equally claim: "time_factor" (Burland and Burbidge's C3, Eq. 7-34,
#: or the consolidation time factor of Figure 7-16), "shape_factor" (Eq.
#: 7-32's, or the bearing capacity shape factors of Table 5-2) and
#: "factor_of_safety" (bearing, Eq. 5-23, or sliding and uplift). Taken
#: out; curated names stand in (``_reference_common._CURATED_ALIASES``).
_AMBIGUOUS_ALIASES = ("time_factor", "shape_factor", "factor_of_safety")


def _build():
    from geotech_references.em_1905 import (
        equations, figures, settlement, tables,
    )
    registry, info = build_lookup_registry([
        (tables, "EM 1905 Tables (Ch 2, 5, 7)", _REF),
        (equations, "EM 1905 Ultimate Limit State (Ch 5)", _REF),
        (settlement, "EM 1905 Stress and Settlement (Ch 6, 7)", _REF),
        (figures, "EM 1905 Figures", _REF),
    ])
    for alias in _AMBIGUOUS_ALIASES:
        if info.get(alias, {}).get("alias_of"):
            registry.pop(alias)
            info.pop(alias)
    add_text_retrieval(registry, info, "em_1905", _REF)
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
