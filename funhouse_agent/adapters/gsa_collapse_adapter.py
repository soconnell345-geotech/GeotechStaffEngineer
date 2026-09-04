"""GSA Alternate Path reference adapter (civilian progressive collapse).

GSA Alternate Path Analysis and Design Guidelines for Progressive Collapse
Resistance (24 Oct 2013, Rev 1, 28 Jan 2016) — the federal-civilian sibling
of UFC 4-023-03 (`ufc_collapse`): Alternate Path (AP) ONLY (Tie Forces and
Enhanced Local Resistance are removed), triggered by Facility Security
Level (FSL) rather than Risk Category. Section 2.3 FSL applicability flow
chart + story-count exclusions (applicability); Section 3.2 general LRFD
check, action classification, FSL-keyed removal-location rules, the
existing-building disproportionate-collapse allowance, load/dynamic
increase factor tables, acceptance criteria (alternate_path); Section 3.4
Redundancy Requirements — load-redistribution-system count and +/-30%
strength/stiffness uniformity checks, NOT present in UFC 4-023-03
(redundancy); Chapter 4 RC modeling-parameter/m-factor tables (reinforced_
concrete, cross-validated against ufc_collapse with documented deltas);
Chapter 5 steel m-factor tables (structural_steel); Chapters 6-8 masonry/
wood/CFS numeric factors (masonry_wood_cfs). US customary units.
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.gsa_collapse import (
        alternate_path, applicability, masonry_wood_cfs, redundancy,
        reinforced_concrete, structural_steel,
    )
    registry, info = build_lookup_registry([
        (applicability, "GSA Collapse Applicability (Sec 2.3)", "GSA Alternate Path"),
        (alternate_path, "GSA Collapse Alternate Path (Sec 3.2)", "GSA Alternate Path"),
        (redundancy, "GSA Collapse Redundancy (Sec 3.4)", "GSA Alternate Path"),
        (reinforced_concrete, "GSA Collapse Reinforced Concrete (Ch 4)",
         "GSA Alternate Path"),
        (structural_steel, "GSA Collapse Structural Steel (Ch 5)",
         "GSA Alternate Path"),
        (masonry_wood_cfs, "GSA Collapse Masonry/Wood/CFS (Ch 6-8)",
         "GSA Alternate Path"),
    ])
    add_text_retrieval(registry, info, "gsa_collapse", "GSA Alternate Path")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
