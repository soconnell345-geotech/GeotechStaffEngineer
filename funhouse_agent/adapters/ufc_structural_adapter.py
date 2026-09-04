"""UFC 3-301-01 reference adapter (DoD structural engineering).

Structural Engineering (11 Apr 2023, through Change 4, 3 Jun 2025): DoD
modifications to the adopted 2024 IBC / ASCE 7-22 baseline ONLY (base
civilian-code content this UFC does not itself modify is not reprinted).
Chapter 1 modification-action definitions and progressive-collapse/
cybersecurity pointers (general_provisions); Chapter 2 risk category Table
2-2 (adds DoD Risk Category V + Sea Level Rise column), wind deflection
Table 2-1, wind-speed conversions, and Appendix E's full Table E-1 minimum
live-load table (risk_category_and_loads); the full Table 3-1 seismic
force-resisting-system replacement for ASCE 7-22 Table 12.2-1 (~85 systems)
plus the Chapter 7 healthcare and Appendix B Risk-Category-IV variants
(seismic_force_resisting_systems); additional vertical-ground-motion
seismic load combinations and the healthcare structural-separation check
(seismic_load_combinations); Chapter 4 performance-objective lookups and
retrofit-trigger thresholds (evaluation_retrofit); Chapters 6/7 healthcare
masonry/configuration criteria (healthcare_modifications); Chapter 5
nonbuilding-structure standard pointers (nonbuilding_structures); Appendix
C rigid-pipe span tables + elevator/partition seismic criteria
(nonstructural_seismic); Appendix G GFRP material/design limits (gfrp);
Appendix A best-practice numeric criteria (best_practices). Does NOT
reprint IBC/ASCE 7-22 member-design equations. US customary units.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.ufc_structural import (
        best_practices, evaluation_retrofit, general_provisions, gfrp,
        healthcare_modifications, nonbuilding_structures,
        nonstructural_seismic, risk_category_and_loads,
        seismic_force_resisting_systems, seismic_load_combinations,
    )
    registry, info = build_lookup_registry([
        (general_provisions, "UFC 3-301-01 General Provisions (Ch 1)",
         "UFC 3-301-01"),
        (risk_category_and_loads, "UFC 3-301-01 Risk Category/Loads (Ch 2)",
         "UFC 3-301-01"),
        (seismic_force_resisting_systems,
         "UFC 3-301-01 Seismic Force-Resisting Systems (Table 3-1/7-1/B-1)",
         "UFC 3-301-01"),
        (seismic_load_combinations, "UFC 3-301-01 Seismic Load Combinations",
         "UFC 3-301-01"),
        (evaluation_retrofit, "UFC 3-301-01 Evaluation/Retrofit (Ch 4)",
         "UFC 3-301-01"),
        (healthcare_modifications, "UFC 3-301-01 Healthcare (Ch 6-7)",
         "UFC 3-301-01"),
        (nonbuilding_structures, "UFC 3-301-01 Nonbuilding Structures (Ch 5)",
         "UFC 3-301-01"),
        (nonstructural_seismic, "UFC 3-301-01 Nonstructural Seismic (App C)",
         "UFC 3-301-01"),
        (gfrp, "UFC 3-301-01 GFRP (App G)", "UFC 3-301-01"),
        (best_practices, "UFC 3-301-01 Best Practices (App A)", "UFC 3-301-01"),
    ])
    add_text_retrieval(registry, info, "ufc_structural", "UFC 3-301-01")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
