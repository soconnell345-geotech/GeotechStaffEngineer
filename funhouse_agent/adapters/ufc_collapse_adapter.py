"""UFC 4-023-03 reference adapter (DoD progressive collapse design).

Design of Buildings to Resist Progressive Collapse (14 Jul 2009, through
Change 4, 10 Jun 2024): direct design requirements (not code modifications)
for Tie Forces (TF), Alternate Path (AP), and Enhanced Local Resistance
(ELR). Chapter 1/2 applicability thresholds and the Risk-Category TF/AP/ELR
combination table (applicability); Section 3-1 tie-force equations —
floor load, internal/peripheral/vertical tie forces, tie-strength/rebar-
area check (tie_forces); Section 3-2 Alternate Path — LRFD check, removal-
location rules, load/dynamic increase factor tables, acceptance criteria
(alternate_path); Section 3-3 ELR LRFD check + column shear demand
(enhanced_local_resistance); Chapter 4 RC modeling-parameter/m-factor
tables replacing ASCE 41 (reinforced_concrete); Chapter 5 steel m-factor
and modeling-parameter tables by connection type (structural_steel);
Chapters 6-8 masonry/wood/CFS numeric factors (masonry_wood_cfs); Appendix
H IBC 2015 Ch 16/17 modifications (ibc_modifications). US customary units.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.ufc_collapse import (
        alternate_path, applicability, enhanced_local_resistance,
        ibc_modifications, masonry_wood_cfs, reinforced_concrete,
        structural_steel, tie_forces,
    )
    registry, info = build_lookup_registry([
        (applicability, "UFC 4-023-03 Applicability (Ch 1-2)", "UFC 4-023-03"),
        (tie_forces, "UFC 4-023-03 Tie Forces (Sec 3-1)", "UFC 4-023-03"),
        (alternate_path, "UFC 4-023-03 Alternate Path (Sec 3-2)", "UFC 4-023-03"),
        (enhanced_local_resistance,
         "UFC 4-023-03 Enhanced Local Resistance (Sec 3-3)", "UFC 4-023-03"),
        (reinforced_concrete, "UFC 4-023-03 Reinforced Concrete (Ch 4)",
         "UFC 4-023-03"),
        (structural_steel, "UFC 4-023-03 Structural Steel (Ch 5)",
         "UFC 4-023-03"),
        (masonry_wood_cfs, "UFC 4-023-03 Masonry/Wood/CFS (Ch 6-8)",
         "UFC 4-023-03"),
        (ibc_modifications, "UFC 4-023-03 IBC Modifications (App H)",
         "UFC 4-023-03"),
    ])
    add_text_retrieval(registry, info, "ufc_collapse", "UFC 4-023-03")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
