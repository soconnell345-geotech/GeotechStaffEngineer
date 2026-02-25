"""
Reusable geotechnical problem definitions for test scenarios.

All values are SI: meters, kPa, kN, kN/m3, degrees.
Each scenario includes the source reference and expected answer.
"""


# ============================================================================
# Bearing Capacity Scenarios
# ============================================================================

BEARING_STRIP_SAND = {
    "description": "Strip footing on sand",
    "params": {
        "width": 1.5,
        "depth": 1.0,
        "shape": "strip",
        "cohesion": 0.0,
        "friction_angle": 35.0,
        "unit_weight": 17.8,
    },
}

BEARING_SQUARE_SAND = {
    "description": "Square footing on sand",
    "params": {
        "width": 2.0,
        "length": 2.0,
        "depth": 1.5,
        "shape": "square",
        "cohesion": 0.0,
        "friction_angle": 30.0,
        "unit_weight": 18.0,
    },
}

BEARING_STRIP_CLAY = {
    "description": "Strip footing on clay (undrained, phi=0)",
    "params": {
        "width": 2.0,
        "depth": 1.0,
        "shape": "strip",
        "cohesion": 100.0,
        "friction_angle": 0.0,
        "unit_weight": 18.0,
    },
}

BEARING_WITH_GWT = {
    "description": "Footing with groundwater at base level",
    "params": {
        "width": 2.0,
        "length": 2.0,
        "depth": 1.5,
        "shape": "square",
        "cohesion": 0.0,
        "friction_angle": 30.0,
        "unit_weight": 18.0,
        "gwt_depth": 1.5,
    },
}


# ============================================================================
# Settlement Scenarios
# ============================================================================

ELASTIC_SETTLEMENT_BASIC = {
    "description": "Basic elastic settlement on sand",
    "params": {
        "q_net": 150.0,
        "B": 2.0,
        "Es": 10000.0,
        "nu": 0.3,
    },
}

CONSOLIDATION_NC_CLAY = {
    "description": "Normally consolidated clay layer",
    "params": {
        "layers": [
            {
                "thickness": 3.0,
                "depth_to_center": 5.0,
                "e0": 1.1,
                "Cc": 0.4,
                "Cr": 0.08,
                "sigma_v0": 80.0,
            }
        ],
        "delta_sigma": 50.0,
    },
}


# ============================================================================
# Axial Pile Scenarios
# ============================================================================

DRIVEN_PILE_SAND = {
    "description": "Steel H-pile in sand",
    "params": {
        "pile_type": "h_pile",
        "pile_length": 15.0,
        "designation": "HP14x89",
        "layers": [
            {
                "thickness": 20.0,
                "soil_type": "cohesionless",
                "unit_weight": 18.5,
                "friction_angle": 32.0,
            }
        ],
        "gwt_depth": 5.0,
    },
}

DRIVEN_PILE_CLAY = {
    "description": "Pipe pile in clay",
    "params": {
        "pile_type": "pipe_closed",
        "pile_length": 20.0,
        "diameter": 0.356,
        "wall_thickness": 0.0127,
        "layers": [
            {
                "thickness": 25.0,
                "soil_type": "cohesive",
                "unit_weight": 17.5,
                "cohesion": 75.0,
            }
        ],
        "gwt_depth": 2.0,
    },
}


# ============================================================================
# Drilled Shaft Scenarios
# ============================================================================

DRILLED_SHAFT_CLAY = {
    "description": "Drilled shaft in clay (GEC-10 alpha method)",
    "params": {
        "diameter": 0.914,
        "shaft_length": 15.0,
        "layers": [
            {
                "thickness": 20.0,
                "soil_type": "cohesive",
                "unit_weight": 18.0,
                "cu": 100.0,
            }
        ],
        "gwt_depth": 3.0,
    },
}

DRILLED_SHAFT_SAND = {
    "description": "Drilled shaft in sand (GEC-10 beta method)",
    "params": {
        "diameter": 0.914,
        "shaft_length": 15.0,
        "layers": [
            {
                "thickness": 20.0,
                "soil_type": "cohesionless",
                "unit_weight": 18.5,
                "phi": 35.0,
                "N60": 30,
            }
        ],
        "gwt_depth": 5.0,
    },
}


# ============================================================================
# Seismic Scenarios
# ============================================================================

SITE_CLASS_D = {
    "description": "Site Class D from Vs30",
    "params": {
        "vs30": 250.0,
        "Ss": 1.0,
        "S1": 0.4,
    },
}

MO_PRESSURE = {
    "description": "Mononobe-Okabe active pressure",
    "params": {
        "phi": 30.0,
        "kh": 0.1,
        "delta": 20.0,
    },
}

LIQUEFACTION_CHECK = {
    "description": "SPT-based liquefaction evaluation",
    "params": {
        "depths": [3.0, 6.0, 9.0, 12.0],
        "N160": [10.0, 15.0, 20.0, 25.0],
        "FC": [5.0, 10.0, 5.0, 5.0],
        "gamma": [18.0, 18.0, 19.0, 19.0],
        "amax_g": 0.25,
        "gwt_depth": 2.0,
        "magnitude": 7.5,
    },
}


# ============================================================================
# Retaining Wall Scenarios
# ============================================================================

CANTILEVER_WALL = {
    "description": "Cantilever retaining wall",
    "params": {
        "wall_height": 6.0,
        "gamma_backfill": 18.0,
        "phi_backfill": 30.0,
        "base_width": 4.2,
        "toe_length": 1.2,
        "stem_thickness_top": 0.3,
        "stem_thickness_base": 0.6,
        "base_thickness": 0.6,
        "phi_foundation": 30.0,
        "c_foundation": 0.0,
    },
}


# ============================================================================
# Sheet Pile Scenarios
# ============================================================================

CANTILEVER_SHEET_PILE = {
    "description": "Cantilever sheet pile in sand",
    "params": {
        "excavation_depth": 4.0,
        "layers": [
            {
                "thickness": 15.0,
                "unit_weight": 18.0,
                "friction_angle": 30.0,
                "cohesion": 0.0,
            }
        ],
        "FOS_passive": 1.5,
    },
}


# ============================================================================
# SOE (Support of Excavation) Scenarios
# ============================================================================

SOE_BRACED_SAND = {
    "description": "2-level braced excavation in sand (Terzaghi-Peck)",
    "params": {
        "excavation_depth": 10.0,
        "layers": [
            {
                "thickness": 20.0,
                "unit_weight": 18.0,
                "friction_angle": 30.0,
                "cohesion": 0.0,
                "soil_type": "sand",
            }
        ],
        "supports": [
            {"depth": 2.0, "support_type": "strut", "spacing": 3.0},
            {"depth": 6.0, "support_type": "strut", "spacing": 3.0},
        ],
        "surcharge": 10.0,
    },
}

SOE_CANTILEVER_SAND = {
    "description": "Cantilever excavation wall in sand",
    "params": {
        "excavation_depth": 3.0,
        "layers": [
            {
                "thickness": 15.0,
                "unit_weight": 18.0,
                "friction_angle": 30.0,
                "cohesion": 0.0,
                "soil_type": "sand",
            }
        ],
    },
}


# ============================================================================
# Slope Stability Scenarios
# ============================================================================

SIMPLE_SLOPE = {
    "description": "Simple cohesive slope with known circle",
    "params": {
        "xc": 15.0,
        "yc": 18.0,
        "radius": 13.0,
        "surface_points": [[0, 10], [10, 10], [20, 5], [30, 5]],
        "soil_layers": [
            {
                "name": "Clay",
                "top_elevation": 10.0,
                "bottom_elevation": 0.0,
                "gamma": 18.0,
                "phi": 10.0,
                "c_prime": 25.0,
            }
        ],
    },
}


# ============================================================================
# SPT / Classification Scenarios
# ============================================================================

SPT_CORRECTION = {
    "description": "SPT energy + overburden correction",
    "params": {
        "recorded_spt_n_value": 25,
        "eop": 100.0,
        "energy_percentage": 0.6,
        "hammer_type": "safety",
        "sampler_type": "standard",
        "opc_method": "liao",
    },
}

USCS_CL = {
    "description": "Fine-grained soil — CL classification",
    "params": {
        "liquid_limit": 45.0,
        "plastic_limit": 25.0,
        "fines": 60.0,
    },
}

USCS_SW = {
    "description": "Well-graded sand — SW classification",
    "params": {
        "liquid_limit": 0.0,
        "plastic_limit": 0.0,
        "fines": 3.0,
        "sand": 80.0,
        "d_10": 0.1,
        "d_30": 0.8,
        "d_60": 3.0,
    },
}

AASHTO_A7 = {
    "description": "AASHTO A-7-6 classification",
    "params": {
        "liquid_limit": 45.0,
        "plastic_limit": 25.0,
        "fines": 60.0,
    },
}


# ============================================================================
# Ground Improvement Scenarios
# ============================================================================

WICK_DRAIN = {
    "description": "Wick drain consolidation",
    "params": {
        "spacing": 1.5,
        "ch": 3.0,
        "cv": 1.5,
        "Hdr": 6.0,
        "time": 0.5,
    },
}

AGGREGATE_PIER = {
    "description": "Aggregate pier improvement",
    "params": {
        "column_diameter": 0.6,
        "spacing": 2.0,
        "E_column": 100000.0,
        "E_soil": 5000.0,
    },
}

VIBRO_COMPACTION = {
    "description": "Vibro compaction feasibility",
    "params": {
        "fines_content": 8.0,
        "initial_N_spt": 8,
        "target_N_spt": 20,
    },
}


# ============================================================================
# Reliability Scenarios
# ============================================================================

FORM_RS = {
    "description": "FORM analysis: R-S with normal distributions",
    "params": {
        "variables": [
            {"name": "R", "dist": "normal", "mean": 200.0, "stdv": 20.0},
            {"name": "S", "dist": "normal", "mean": 100.0, "stdv": 15.0},
        ],
        "limit_state": "R - S",
    },
}


# ============================================================================
# Wave Equation Scenarios
# ============================================================================

BEARING_GRAPH = {
    "description": "Bearing graph for driven pile",
    "params": {
        "pile_length": 15.0,
        "pile_area": 0.01,
        "pile_E": 200e6,
        "pile_unit_weight": 78.5,
        "ram_weight": 50.0,
        "stroke": 1.0,
        "efficiency": 0.8,
        "cushion_stiffness": 1e6,
        "skin_fraction": 0.6,
    },
}


# ============================================================================
# Pile Group Scenarios
# ============================================================================

PILE_GROUP_3X3 = {
    "description": "3x3 pile group under vertical load",
    "params": {
        "n_rows": 3,
        "n_cols": 3,
        "spacing_x": 1.0,
        "spacing_y": 1.0,
        "Vz": 2700.0,
    },
}

GROUP_EFFICIENCY = {
    "description": "Group efficiency for 3x3 group",
    "params": {
        "n_rows": 3,
        "n_cols": 3,
        "pile_diameter": 0.3,
        "spacing": 0.9,
    },
}


# ============================================================================
# Downdrag Scenarios
# ============================================================================

DOWNDRAG_BASIC = {
    "description": "Pile downdrag with settling fill",
    "params": {
        "pile_length": 20.0,
        "pile_diameter": 0.356,
        "layers": [
            {
                "thickness": 5.0,
                "soil_type": "cohesive",
                "unit_weight": 17.0,
                "cu": 20.0,
                "settling": True,
            },
            {
                "thickness": 5.0,
                "soil_type": "cohesive",
                "unit_weight": 17.5,
                "cu": 40.0,
                "settling": True,
            },
            {
                "thickness": 15.0,
                "soil_type": "cohesionless",
                "unit_weight": 19.0,
                "phi": 35.0,
                "settling": False,
            },
        ],
        "Q_dead": 500.0,
        "gwt_depth": 2.0,
    },
}


# ============================================================================
# Cross-check: same problem for bearing_capacity vs geolysis
# ============================================================================

CROSS_CHECK_BEARING = {
    "description": "Same bearing capacity problem for cross-agent comparison",
    "bearing_capacity_params": {
        "width": 2.0,
        "length": 2.0,
        "depth": 1.5,
        "shape": "square",
        "cohesion": 0.0,
        "friction_angle": 30.0,
        "unit_weight": 18.0,
    },
    "geolysis_params": {
        "friction_angle": 30.0,
        "cohesion": 0.0,
        "moist_unit_wgt": 18.0,
        "depth": 1.5,
        "width": 2.0,
        "shape": "square",
        "ubc_method": "vesic",
    },
}
