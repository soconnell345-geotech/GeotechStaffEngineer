"""Wood Handbook reference adapter (USDA FPL-GTR-282, wood engineering).

Wood Handbook: Wood as an Engineering Material (USDA Forest Products
Laboratory, 2021 edition). Digitizes the design-usable core of four
chapters (NDS design values/adjustment factors are OUT OF SCOPE — only the
handbook's own clear-wood data/equations are implemented). Chapter 4
moisture relations — equilibrium moisture content, shrinkage-moisture-
specific gravity relations, density at any moisture content, thermal
conductivity (moisture_relations); Chapter 5 clear-wood mechanical
property table (27-species structural subset), moisture-content and
temperature adjustment relations (mechanical_properties); Chapter 8
fastenings — nail/screw/lag-screw/bolt withdrawal and lateral resistance
(pre-1991 empirical and post-1991 yield-limit models), Hankinson
bearing-at-an-angle (fastenings); Chapter 9 structural analysis equations
— deformation (structural_deformation), stress incl. size effect and
notch crack initiation (structural_stress), and stability — Euler/FPL/
Ylinen column buckling, beam lateral-torsional buckling, biaxial
beam-column interaction (structural_stability). SI units (native).
"""

from funhouse_agent.adapters._reference_common import (
    build_lookup_registry, add_text_retrieval,
)


def _build():
    from geotech_references.wood_handbook import (
        fastenings, mechanical_properties, moisture_relations,
        structural_deformation, structural_stability, structural_stress,
    )
    registry, info = build_lookup_registry([
        (moisture_relations, "Wood Handbook Moisture Relations (Ch 4)",
         "USDA Wood Handbook FPL-GTR-282"),
        (mechanical_properties, "Wood Handbook Mechanical Properties (Ch 5)",
         "USDA Wood Handbook FPL-GTR-282"),
        (fastenings, "Wood Handbook Fastenings (Ch 8)",
         "USDA Wood Handbook FPL-GTR-282"),
        (structural_deformation, "Wood Handbook Structural Deformation (Ch 9)",
         "USDA Wood Handbook FPL-GTR-282"),
        (structural_stress, "Wood Handbook Structural Stress (Ch 9)",
         "USDA Wood Handbook FPL-GTR-282"),
        (structural_stability, "Wood Handbook Structural Stability (Ch 9)",
         "USDA Wood Handbook FPL-GTR-282"),
    ])
    add_text_retrieval(registry, info, "wood_handbook",
                       "USDA Wood Handbook FPL-GTR-282")
    return registry, info


METHOD_REGISTRY, METHOD_INFO = _build()
