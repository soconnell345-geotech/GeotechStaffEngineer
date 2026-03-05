"""Ground improvement adapter — aggregate piers, wick drains, vibro, surcharge."""

from funhouse_agent.adapters import clean_result
from ground_improvement import (
    analyze_aggregate_piers, analyze_wick_drains, design_drain_spacing,
    analyze_surcharge_preloading, analyze_vibro_compaction, evaluate_feasibility,
)


def _run_aggregate_piers(p): return clean_result(analyze_aggregate_piers(**p).to_dict())
def _run_wick_drains(p): return clean_result(analyze_wick_drains(**p).to_dict())
def _run_design_drain_spacing(p): return clean_result(design_drain_spacing(**p).to_dict())
def _run_surcharge_preloading(p): return clean_result(analyze_surcharge_preloading(**p).to_dict())
def _run_vibro_compaction(p): return clean_result(analyze_vibro_compaction(**p).to_dict())
def _run_feasibility(p): return clean_result(evaluate_feasibility(**p).to_dict())


METHOD_REGISTRY = {
    "aggregate_piers": _run_aggregate_piers,
    "wick_drains": _run_wick_drains,
    "design_drain_spacing": _run_design_drain_spacing,
    "surcharge_preloading": _run_surcharge_preloading,
    "vibro_compaction": _run_vibro_compaction,
    "feasibility": _run_feasibility,
}

METHOD_INFO = {
    "aggregate_piers": {
        "category": "Aggregate Piers",
        "brief": "Aggregate pier (stone column) design: settlement reduction, capacity increase.",
        "parameters": {
            "area_replacement_ratio": {"type": "float", "required": True, "description": "Area ratio As/A (0.1-0.35 typical)."},
            "column_modulus": {"type": "float", "required": True, "description": "Stone column modulus (kPa)."},
            "soil_modulus": {"type": "float", "required": True, "description": "Native soil modulus (kPa)."},
            "applied_pressure": {"type": "float", "required": True, "description": "Applied foundation pressure (kPa)."},
            "treatment_depth": {"type": "float", "required": True, "description": "Treatment depth (m)."},
        },
        "returns": {"stress_concentration_ratio": "n = sigma_c/sigma_s.", "settlement_improvement_factor": "Settlement reduction."},
    },
    "wick_drains": {
        "category": "Wick Drains",
        "brief": "Prefabricated vertical drain (PVD) consolidation analysis.",
        "parameters": {
            "ch": {"type": "float", "required": True, "description": "Horizontal coefficient of consolidation (m2/yr)."},
            "drain_spacing": {"type": "float", "required": True, "description": "Drain spacing (m)."},
            "drain_diameter": {"type": "float", "required": False, "default": 0.05, "description": "Equivalent drain diameter (m)."},
            "mandrel_diameter": {"type": "float", "required": False, "default": 0.07, "description": "Mandrel diameter (m)."},
            "time_years": {"type": "float", "required": True, "description": "Time period (years)."},
            "pattern": {"type": "str", "required": False, "default": "triangular", "description": "triangular or square."},
        },
        "returns": {"U_total_percent": "Total degree of consolidation (%)."},
    },
    "design_drain_spacing": {
        "category": "Wick Drains",
        "brief": "Find required drain spacing for target consolidation.",
        "parameters": {
            "ch": {"type": "float", "required": True, "description": "Horizontal coeff of consolidation (m2/yr)."},
            "target_U": {"type": "float", "required": True, "description": "Target consolidation degree (0-1)."},
            "time_years": {"type": "float", "required": True, "description": "Time available (years)."},
        },
        "returns": {"required_spacing_m": "Required drain spacing."},
    },
    "surcharge_preloading": {
        "category": "Surcharge",
        "brief": "Surcharge preloading settlement and time analysis.",
        "parameters": {
            "surcharge_pressure": {"type": "float", "required": True, "description": "Surcharge pressure (kPa)."},
            "design_pressure": {"type": "float", "required": True, "description": "Design foundation pressure (kPa)."},
            "cv": {"type": "float", "required": True, "description": "Vertical coeff of consolidation (m2/yr)."},
            "Hdr": {"type": "float", "required": True, "description": "Drainage path (m)."},
            "Cc": {"type": "float", "required": True, "description": "Compression index."},
            "e0": {"type": "float", "required": True, "description": "Initial void ratio."},
            "H_compressible": {"type": "float", "required": True, "description": "Compressible layer thickness (m)."},
            "sigma_v0": {"type": "float", "required": True, "description": "Initial effective overburden (kPa)."},
        },
        "returns": {"time_to_target_years": "Time to achieve target consolidation."},
    },
    "vibro_compaction": {
        "category": "Vibro Compaction",
        "brief": "Vibro compaction (vibroflotation) feasibility and densification.",
        "parameters": {
            "D50_mm": {"type": "float", "required": True, "description": "Median grain size (mm)."},
            "fines_content": {"type": "float", "required": True, "description": "Fines content (%)."},
            "initial_Dr": {"type": "float", "required": True, "description": "Initial relative density (%)."},
            "target_Dr": {"type": "float", "required": True, "description": "Target relative density (%)."},
            "treatment_depth": {"type": "float", "required": True, "description": "Treatment depth (m)."},
        },
        "returns": {"is_feasible": "Whether vibro compaction is suitable."},
    },
    "feasibility": {
        "category": "General",
        "brief": "Ground improvement method feasibility screening.",
        "parameters": {
            "soil_type": {"type": "str", "required": True, "description": "Soil type (sand/silt/clay/organic)."},
            "fines_content": {"type": "float", "required": False, "description": "Fines content (%)."},
            "treatment_depth": {"type": "float", "required": True, "description": "Required treatment depth (m)."},
        },
        "returns": {"recommended_methods": "List of feasible methods.", "is_feasible": "Overall feasibility."},
    },
}
