"""
Groundhog Geotechnical Agent - Palantir Foundry AIP Agent Studio Version.

This file is formatted for Palantir Foundry's Code Repository application.
Register these three functions as tools in AIP Agent Studio:
  1. groundhog_agent        - Run a geotechnical calculation
  2. groundhog_list_methods - Browse available calculation methods
  3. groundhog_describe_method - Get detailed docs for a specific method

FOUNDRY SETUP:
  - Add 'groundhog' to your conda_recipe/meta.yaml dependencies
  - These functions accept and return JSON strings for maximum LLM compatibility
  - The LLM should call groundhog_list_methods first, then groundhog_describe_method
    for parameter details, then groundhog_agent to run the calculation
"""

import json
import math
import numpy as np
from functions.api import function

# --- Phase Relations ---
from groundhog.siteinvestigation.classification.phaserelations import (
    voidratio_porosity,
    porosity_voidratio,
    saturation_watercontent,
    bulkunitweight,
    dryunitweight_watercontent,
    voidratio_drydensity,
    bulkunitweight_dryunitweight,
    relative_density,
    voidratio_bulkunitweight,
    unitweight_watercontent_saturated,
    density_unitweight,
    unitweight_density,
    watercontent_voidratio,
    voidratio_watercontent,
)

# --- SPT Correlations ---
from groundhog.siteinvestigation.insitutests.spt_correlations import (
    overburdencorrection_spt_liaowhitman,
    spt_N60_correction,
    relativedensity_spt_kulhawymayne,
    undrainedshearstrength_spt_salgado,
    frictionangle_spt_kulhawymayne,
    relativedensityclass_spt_terzaghipeck,
    overburdencorrection_spt_ISO,
    frictionangle_spt_PHT,
    youngsmodulus_spt_AASHTO,
    undrainedshearstrengthclass_spt_terzaghipeck,
)

# --- CPT Correlations ---
from groundhog.siteinvestigation.insitutests.pcpt_correlations import (
    pcpt_normalisations,
    ic_soilclass_robertson,
    behaviourindex_pcpt_robertsonwride,
    gmax_sand_rixstokoe,
    gmax_clay_maynerix,
    relativedensity_ncsand_baldi,
    relativedensity_ocsand_baldi,
    relativedensity_sand_jamiolkowski,
    frictionangle_sand_kulhawymayne,
    undrainedshearstrength_clay_radlunne,
    ocr_cpt_lunne,
    sensitivity_frictionratio_lunne,
    unitweight_mayne,
    vs_ic_robertsoncabal,
    k0_sand_mayne,
    constrainedmodulus_pcpt_robertson,
)

# --- Bearing Capacity ---
from groundhog.shallowfoundations.capacity import (
    nq_frictionangle_sand,
    ngamma_frictionangle_vesic,
    ngamma_frictionangle_meyerhof,
    ngamma_frictionangle_davisbooker,
    verticalcapacity_undrained_api,
    verticalcapacity_drained_api,
    slidingcapacity_undrained_api,
    slidingcapacity_drained_api,
    effectivearea_rectangle_api,
    effectivearea_circle_api,
)

# --- Consolidation & Settlement ---
from groundhog.consolidation.dissipation.onedimensionalconsolidation import (
    consolidation_degree,
)
from groundhog.shallowfoundations.settlement import (
    primaryconsolidationsettlement_nc,
    primaryconsolidationsettlement_oc,
    consolidationsettlement_mv,
)
from groundhog.consolidation.groundwaterflow.pumpingtests import (
    hydraulicconductivity_unconfinedaquifer,
)

# --- Stress Distribution ---
from groundhog.shallowfoundations.stressdistribution import (
    stresses_pointload,
    stresses_stripload,
    stresses_circle,
    stresses_rectangle,
)

# --- Earth Pressure ---
from groundhog.excavations.basic import (
    earthpressurecoefficients_frictionangle,
    earthpressurecoefficients_poncelet,
    earthpressurecoefficients_rankine,
)

# --- Soil Classification ---
from groundhog.siteinvestigation.classification.categories import (
    relativedensity_categories,
    su_categories,
    uscs_categories,
    samplequality_voidratio_lunne,
)

# --- Deep Foundations ---
from groundhog.deepfoundations.axialcapacity.skinfriction import (
    API_unit_shaft_friction_sand_rp2geo,
    API_unit_shaft_friction_clay,
    unitskinfriction_sand_almhamre,
    unitskinfriction_clay_almhamre,
)
from groundhog.deepfoundations.axialcapacity.endbearing import (
    API_unit_end_bearing_clay,
    API_unit_end_bearing_sand_rp2geo,
    unitendbearing_sand_almhamre,
    unitendbearing_clay_almhamre,
)

# --- Soil Dynamics & Liquefaction ---
from groundhog.soildynamics.soilproperties import (
    modulusreduction_plasticity_ishibashi,
    gmax_shearwavevelocity,
    dampingratio_sandgravel_seed,
)
from groundhog.soildynamics.liquefaction import (
    cyclicstressratio_moss,
    liquefaction_robertsonfear,
    cyclicstressratio_youd,
)

# --- Soil Correlations ---
from groundhog.siteinvestigation.correlations.cohesionless import (
    gmax_sand_hardinblack,
    permeability_d10_hazen,
    hssmall_parameters_sand,
    stress_dilatancy_bolton,
)
from groundhog.siteinvestigation.correlations.cohesive import (
    compressionindex_watercontent_koppula,
    frictionangle_plasticityindex,
    cv_liquidlimit_usnavy,
    gmax_plasticityocr_andersen,
    k0_plasticity_kenney,
)
from groundhog.siteinvestigation.correlations.general import (
    k0_frictionangle_mesri,
)


# ---------------------------------------------------------------------------
# Method registry: maps method name -> groundhog function
# ---------------------------------------------------------------------------
METHOD_REGISTRY = {
    # Phase Relations
    "voidratio_from_porosity": voidratio_porosity,
    "porosity_from_voidratio": porosity_voidratio,
    "saturation_from_watercontent": saturation_watercontent,
    "bulk_unit_weight": bulkunitweight,
    "dry_unit_weight": dryunitweight_watercontent,
    "voidratio_from_dry_density": voidratio_drydensity,
    "bulk_unit_weight_from_dry": bulkunitweight_dryunitweight,
    "relative_density": relative_density,
    "voidratio_from_bulk_unit_weight": voidratio_bulkunitweight,
    "unit_weight_saturated": unitweight_watercontent_saturated,
    "density_from_unit_weight": density_unitweight,
    "unit_weight_from_density": unitweight_density,
    "watercontent_from_voidratio": watercontent_voidratio,
    "voidratio_from_watercontent": voidratio_watercontent,
    # SPT Correlations
    "spt_overburden_correction_liaowhitman": overburdencorrection_spt_liaowhitman,
    "spt_N60_correction": spt_N60_correction,
    "spt_relative_density_kulhawymayne": relativedensity_spt_kulhawymayne,
    "spt_undrained_shear_strength_salgado": undrainedshearstrength_spt_salgado,
    "spt_friction_angle_kulhawymayne": frictionangle_spt_kulhawymayne,
    "spt_relative_density_class": relativedensityclass_spt_terzaghipeck,
    "spt_overburden_correction_iso": overburdencorrection_spt_ISO,
    "spt_friction_angle_pht": frictionangle_spt_PHT,
    "spt_youngs_modulus_aashto": youngsmodulus_spt_AASHTO,
    "spt_consistency_class": undrainedshearstrengthclass_spt_terzaghipeck,
    # CPT Correlations
    "cpt_normalisations": pcpt_normalisations,
    "cpt_soil_class_robertson": ic_soilclass_robertson,
    "cpt_behaviour_index": behaviourindex_pcpt_robertsonwride,
    "cpt_gmax_sand": gmax_sand_rixstokoe,
    "cpt_gmax_clay": gmax_clay_maynerix,
    "cpt_relative_density_nc_sand": relativedensity_ncsand_baldi,
    "cpt_relative_density_oc_sand": relativedensity_ocsand_baldi,
    "cpt_relative_density_jamiolkowski": relativedensity_sand_jamiolkowski,
    "cpt_friction_angle_sand": frictionangle_sand_kulhawymayne,
    "cpt_undrained_shear_strength": undrainedshearstrength_clay_radlunne,
    "cpt_ocr": ocr_cpt_lunne,
    "cpt_sensitivity": sensitivity_frictionratio_lunne,
    "cpt_unit_weight": unitweight_mayne,
    "cpt_shear_wave_velocity": vs_ic_robertsoncabal,
    "cpt_k0_sand": k0_sand_mayne,
    "cpt_constrained_modulus": constrainedmodulus_pcpt_robertson,
    # Bearing Capacity
    "bearing_capacity_nq": nq_frictionangle_sand,
    "bearing_capacity_ngamma_vesic": ngamma_frictionangle_vesic,
    "bearing_capacity_ngamma_meyerhof": ngamma_frictionangle_meyerhof,
    "bearing_capacity_ngamma_davisbooker": ngamma_frictionangle_davisbooker,
    "bearing_capacity_undrained_api": verticalcapacity_undrained_api,
    "bearing_capacity_drained_api": verticalcapacity_drained_api,
    "sliding_capacity_undrained_api": slidingcapacity_undrained_api,
    "sliding_capacity_drained_api": slidingcapacity_drained_api,
    "effective_area_rectangle": effectivearea_rectangle_api,
    "effective_area_circle": effectivearea_circle_api,
    # Consolidation & Settlement
    "consolidation_degree": consolidation_degree,
    "primary_consolidation_settlement_nc": primaryconsolidationsettlement_nc,
    "primary_consolidation_settlement_oc": primaryconsolidationsettlement_oc,
    "consolidation_settlement_mv": consolidationsettlement_mv,
    "hydraulic_conductivity_unconfined": hydraulicconductivity_unconfinedaquifer,
    # Stress Distribution
    "stress_pointload": stresses_pointload,
    "stress_stripload": stresses_stripload,
    "stress_circle": stresses_circle,
    "stress_rectangle": stresses_rectangle,
    # Earth Pressure
    "earth_pressure_basic": earthpressurecoefficients_frictionangle,
    "earth_pressure_poncelet": earthpressurecoefficients_poncelet,
    "earth_pressure_rankine": earthpressurecoefficients_rankine,
    # Soil Classification
    "relative_density_category": relativedensity_categories,
    "su_category": su_categories,
    "uscs_description": uscs_categories,
    "sample_quality_lunne": samplequality_voidratio_lunne,
    # Deep Foundations
    "pile_shaft_friction_api_sand": API_unit_shaft_friction_sand_rp2geo,
    "pile_shaft_friction_api_clay": API_unit_shaft_friction_clay,
    "pile_shaft_friction_almhamre_sand": unitskinfriction_sand_almhamre,
    "pile_shaft_friction_almhamre_clay": unitskinfriction_clay_almhamre,
    "pile_end_bearing_api_clay": API_unit_end_bearing_clay,
    "pile_end_bearing_api_sand": API_unit_end_bearing_sand_rp2geo,
    "pile_end_bearing_almhamre_sand": unitendbearing_sand_almhamre,
    "pile_end_bearing_almhamre_clay": unitendbearing_clay_almhamre,
    # Soil Dynamics & Liquefaction
    "modulus_reduction_ishibashi": modulusreduction_plasticity_ishibashi,
    "gmax_from_shear_wave_velocity": gmax_shearwavevelocity,
    "damping_ratio_seed": dampingratio_sandgravel_seed,
    "cyclic_stress_ratio_moss": cyclicstressratio_moss,
    "cyclic_stress_ratio_youd": cyclicstressratio_youd,
    "liquefaction_robertson_fear": liquefaction_robertsonfear,
    # Soil Correlations
    "gmax_sand_hardin_black": gmax_sand_hardinblack,
    "permeability_hazen": permeability_d10_hazen,
    "hssmall_parameters_sand": hssmall_parameters_sand,
    "stress_dilatancy_bolton": stress_dilatancy_bolton,
    "compression_index_koppula": compressionindex_watercontent_koppula,
    "friction_angle_from_pi": frictionangle_plasticityindex,
    "cv_from_liquid_limit": cv_liquidlimit_usnavy,
    "gmax_clay_andersen": gmax_plasticityocr_andersen,
    "k0_from_plasticity": k0_plasticity_kenney,
    "k0_from_friction_angle": k0_frictionangle_mesri,
}


# ---------------------------------------------------------------------------
# Method metadata: brief descriptions + detailed info for LLM discovery
# ---------------------------------------------------------------------------
METHOD_INFO = {
    # Phase Relations
    "voidratio_from_porosity": {
        "category": "Phase Relations",
        "brief": "Convert porosity to void ratio.",
        "description": "Converts porosity (n) to void ratio (e) using e = n / (1 - n).",
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "porosity": {"type": "float", "required": True, "range": "0 to 1",
                         "description": "Porosity (n), ratio of void volume to total volume."},
        },
        "returns": {"voidratio [-]": "Void ratio (e)."},
    },
    "porosity_from_voidratio": {
        "category": "Phase Relations",
        "brief": "Convert void ratio to porosity.",
        "description": "Converts void ratio (e) to porosity (n) using n = e / (1 + e).",
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "voidratio": {"type": "float", "required": True, "range": "0 to 5",
                          "description": "Void ratio (e)."},
        },
        "returns": {"porosity [-]": "Porosity (n)."},
    },
    "saturation_from_watercontent": {
        "category": "Phase Relations",
        "brief": "Calculate degree of saturation from water content and void ratio.",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 4"},
            "voidratio": {"type": "float", "required": True, "range": "0 to 4"},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65},
        },
        "returns": {"saturation [-]": "Degree of saturation (0 to 1)."},
    },
    "bulk_unit_weight": {
        "category": "Phase Relations",
        "brief": "Calculate bulk and effective unit weight.",
        "parameters": {
            "saturation": {"type": "float", "required": True, "range": "0 to 1"},
            "voidratio": {"type": "float", "required": True, "range": "0 to 4"},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65},
            "unitweight_water": {"type": "float", "required": False, "default": 10.0},
        },
        "returns": {"bulk unit weight [kN/m3]": "Total bulk unit weight.",
                    "effective unit weight [kN/m3]": "Effective/submerged unit weight."},
    },
    "dry_unit_weight": {
        "category": "Phase Relations", "brief": "Calculate dry unit weight from water content and bulk unit weight.",
        "parameters": {
            "watercontent": {"type": "float", "required": True, "range": "0 to 4"},
            "bulkunitweight": {"type": "float", "required": True, "range": "10 to 25"},
        },
        "returns": {"dry unit weight [kN/m3]": "Dry unit weight."},
    },
    "voidratio_from_dry_density": {
        "category": "Phase Relations", "brief": "Calculate void ratio from dry density.",
        "parameters": {
            "dry_density": {"type": "float", "required": True, "range": "1000 to 2000"},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65},
        },
        "returns": {"Void ratio [-]": "Void ratio."},
    },
    "bulk_unit_weight_from_dry": {
        "category": "Phase Relations", "brief": "Calculate bulk unit weight from dry unit weight and water content.",
        "parameters": {
            "dryunitweight": {"type": "float", "required": True, "range": "1 to 15"},
            "watercontent": {"type": "float", "required": True, "range": "0 to 4"},
        },
        "returns": {"bulk unit weight [kN/m3]": "Bulk unit weight.",
                    "effective unit weight [kN/m3]": "Effective unit weight."},
    },
    "relative_density": {
        "category": "Phase Relations", "brief": "Calculate relative density from void ratios.",
        "parameters": {
            "void_ratio": {"type": "float", "required": True, "range": "0 to 5"},
            "e_min": {"type": "float", "required": True, "range": "0 to 5"},
            "e_max": {"type": "float", "required": True, "range": "0 to 5"},
        },
        "returns": {"Dr [-]": "Relative density (0 to 1)."},
    },
    "voidratio_from_bulk_unit_weight": {
        "category": "Phase Relations", "brief": "Back-calculate void ratio from bulk unit weight.",
        "parameters": {
            "bulkunitweight": {"type": "float", "required": True, "range": "10 to 25"},
            "saturation": {"type": "float", "required": False, "default": 1.0},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65},
        },
        "returns": {"e [-]": "Void ratio.", "w [-]": "Water content."},
    },
    "unit_weight_saturated": {
        "category": "Phase Relations", "brief": "Calculate saturated unit weight from water content.",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 2"},
        },
        "returns": {"gamma [kN/m3]": "Saturated unit weight."},
    },
    "density_from_unit_weight": {
        "category": "Phase Relations", "brief": "Convert unit weight (kN/m3) to density (kg/m3).",
        "parameters": {"gamma": {"type": "float", "required": True, "range": "0 to 30"}},
        "returns": {"Density [kg/m3]": "Mass density."},
    },
    "unit_weight_from_density": {
        "category": "Phase Relations", "brief": "Convert density (kg/m3) to unit weight (kN/m3).",
        "parameters": {"density": {"type": "float", "required": True, "range": "0 to 3000"}},
        "returns": {"Unit weight [kN/m3]": "Unit weight."},
    },
    "watercontent_from_voidratio": {
        "category": "Phase Relations", "brief": "Calculate water content from void ratio.",
        "parameters": {"voidratio": {"type": "float", "required": True}},
        "returns": {"Water content [-]": "Water content (decimal).", "Water content [%]": "Water content (percent)."},
    },
    "voidratio_from_watercontent": {
        "category": "Phase Relations", "brief": "Calculate void ratio from water content.",
        "parameters": {"water_content": {"type": "float", "required": True, "range": "0 to 2"}},
        "returns": {"Void ratio [-]": "Void ratio."},
    },
    # SPT Correlations
    "spt_overburden_correction_liaowhitman": {
        "category": "SPT Correlations", "brief": "Correct SPT N for overburden pressure (Liao & Whitman 1986).",
        "parameters": {
            "N": {"type": "float", "required": True}, "sigma_vo_eff": {"type": "float", "required": True},
        },
        "returns": {"CN [-]": "Correction factor.", "N1 [-]": "Corrected N value."},
    },
    "spt_N60_correction": {
        "category": "SPT Correlations", "brief": "Correct raw SPT N to N60 (60% energy efficiency).",
        "parameters": {
            "N": {"type": "float", "required": True},
            "borehole_diameter": {"type": "float", "required": True, "range": "60 to 200"},
            "rod_length": {"type": "float", "required": True},
            "country": {"type": "str", "required": True, "options": ["Japan", "United States", "Argentina", "China", "Other"]},
            "hammertype": {"type": "str", "required": True, "options": ["Donut", "Safety"]},
            "hammerrelease": {"type": "str", "required": True, "options": ["Free fall", "Rope and pulley"]},
        },
        "returns": {"N60 [-]": "Corrected N60.", "eta_H [%]": "Hammer efficiency."},
    },
    "spt_relative_density_kulhawymayne": {
        "category": "SPT Correlations", "brief": "Estimate relative density from corrected SPT (Kulhawy & Mayne 1990).",
        "parameters": {
            "N1_60": {"type": "float", "required": True, "range": "0 to 100"},
            "d_50": {"type": "float", "required": True, "range": "0.002 to 20"},
        },
        "returns": {"Dr [-]": "Relative density (0 to 1).", "Dr [pct]": "Relative density (%)."},
    },
    "spt_undrained_shear_strength_salgado": {
        "category": "SPT Correlations", "brief": "Estimate Su from plasticity index and N60 (Salgado 2008).",
        "parameters": {
            "pi": {"type": "float", "required": True, "range": "15 to 60"},
            "N_60": {"type": "float", "required": True, "range": "0 to 100"},
        },
        "returns": {"alpha_prime [-]": "Alpha factor.", "Su [kPa]": "Undrained shear strength."},
    },
    "spt_friction_angle_kulhawymayne": {
        "category": "SPT Correlations", "brief": "Estimate friction angle from SPT N (Kulhawy & Mayne 1990).",
        "parameters": {
            "N": {"type": "float", "required": True, "range": "0 to 60"},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 1000"},
        },
        "returns": {"Phi [deg]": "Friction angle in degrees."},
    },
    "spt_relative_density_class": {
        "category": "SPT Correlations", "brief": "Classify relative density from SPT N (Terzaghi & Peck 1967).",
        "parameters": {"N": {"type": "float", "required": True, "range": "0 to 60"}},
        "returns": {"Dr class": "Classification string."},
    },
    "spt_overburden_correction_iso": {
        "category": "SPT Correlations", "brief": "Correct SPT N for overburden per ISO 22476-3.",
        "parameters": {
            "N": {"type": "int", "required": True, "range": "0 to 60"},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "25 to 400"},
        },
        "returns": {"CN [-]": "Correction factor.", "N1 [-]": "Corrected N value."},
    },
    "spt_friction_angle_pht": {
        "category": "SPT Correlations", "brief": "Estimate friction angle from (N1)60 (PHT 1974).",
        "parameters": {"N1_60": {"type": "float", "required": True, "range": "0 to 60"}},
        "returns": {"Phi [deg]": "Friction angle in degrees."},
    },
    "spt_youngs_modulus_aashto": {
        "category": "SPT Correlations", "brief": "Estimate Young's modulus from (N1)60 by soil type (AASHTO 1997).",
        "parameters": {
            "N1_60": {"type": "float", "required": True, "range": "0 to 60"},
            "soiltype": {"type": "str", "required": True, "options": ["Silts", "Clean sands", "Coarse sands", "Gravels"]},
        },
        "returns": {"Es [MPa]": "Young's modulus."},
    },
    "spt_consistency_class": {
        "category": "SPT Correlations", "brief": "Classify cohesive soil consistency from SPT N (Terzaghi & Peck 1967).",
        "parameters": {"N": {"type": "float", "required": True, "range": "0 to 60"}},
        "returns": {"Consistency class": "Classification string.", "qu min [kPa]": "Lower qu bound.", "qu max [kPa]": "Upper qu bound."},
    },
    # CPT Correlations
    "cpt_normalisations": {
        "category": "CPT Correlations", "brief": "Normalize raw CPT data and classify soil type (Robertson).",
        "parameters": {
            "measured_qc": {"type": "float", "required": True, "range": "0 to 150", "description": "Cone resistance in MPa."},
            "measured_fs": {"type": "float", "required": True, "range": "0 to 10", "description": "Sleeve friction in MPa."},
            "measured_u2": {"type": "float", "required": True, "range": "-10 to 10", "description": "Pore pressure at shoulder in MPa."},
            "sigma_vo_tot": {"type": "float", "required": True, "description": "Total vertical stress in kPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective vertical stress in kPa."},
            "depth": {"type": "float", "required": True, "description": "Depth below surface in m."},
            "cone_area_ratio": {"type": "float", "required": True, "range": "0 to 1", "description": "Cone area ratio."},
        },
        "returns": {"qt [MPa]": "Corrected cone resistance.", "Ic [-]": "Soil behaviour type index.", "Ic class": "Soil type."},
    },
    "cpt_soil_class_robertson": {
        "category": "CPT Correlations", "brief": "Classify soil from Ic (Robertson chart).",
        "parameters": {"ic": {"type": "float", "required": True, "range": "1 to 5"}},
        "returns": {"Soil type number [-]": "Zone number.", "Soil type": "Soil type description."},
    },
    "cpt_behaviour_index": {
        "category": "CPT Correlations", "brief": "Calculate Ic from corrected CPT data (Robertson & Wride 1998).",
        "parameters": {
            "qt": {"type": "float", "required": True, "description": "Corrected cone resistance in MPa."},
            "fs": {"type": "float", "required": True, "description": "Sleeve friction in MPa."},
            "sigma_vo": {"type": "float", "required": True, "description": "Total vertical stress in kPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective vertical stress in kPa."},
        },
        "returns": {"Ic [-]": "Soil behaviour type index.", "Ic class": "Soil type description."},
    },
    "cpt_gmax_sand": {
        "category": "CPT Correlations", "brief": "Estimate Gmax for sand from CPT (Rix & Stokoe 1991).",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120", "description": "Cone tip resistance in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective stress in kPa."},
        },
        "returns": {"Gmax [kPa]": "Small-strain shear modulus."},
    },
    "cpt_gmax_clay": {
        "category": "CPT Correlations", "brief": "Estimate Gmax for clay from CPT (Mayne & Rix 1993).",
        "parameters": {"qc": {"type": "float", "required": True, "range": "0 to 120", "description": "Cone tip resistance in MPa."}},
        "returns": {"Gmax [kPa]": "Small-strain shear modulus."},
    },
    "cpt_relative_density_nc_sand": {
        "category": "CPT Correlations", "brief": "Estimate Dr for NC sand from CPT (Baldi et al 1986).",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120"},
            "sigma_vo_eff": {"type": "float", "required": True},
        },
        "returns": {"Dr [-]": "Relative density (0 to 1)."},
    },
    "cpt_relative_density_oc_sand": {
        "category": "CPT Correlations", "brief": "Estimate Dr for OC sand from CPT (Baldi et al 1986).",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120"},
            "sigma_vo_eff": {"type": "float", "required": True},
            "k0": {"type": "float", "required": True, "range": "0.3 to 5"},
        },
        "returns": {"Dr [-]": "Relative density (0 to 1)."},
    },
    "cpt_relative_density_jamiolkowski": {
        "category": "CPT Correlations", "brief": "Estimate Dr for sand from CPT (Jamiolkowski et al 2003).",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120"},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "50 to 400"},
            "k0": {"type": "float", "required": True, "range": "0.4 to 1.5"},
        },
        "returns": {"Dr dry [-]": "Dr for dry sand.", "Dr sat [-]": "Dr for saturated sand."},
    },
    "cpt_friction_angle_sand": {
        "category": "CPT Correlations", "brief": "Estimate friction angle for sand from CPT (Kulhawy & Mayne 1990).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120", "description": "Total cone resistance in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective stress in kPa."},
        },
        "returns": {"Phi [deg]": "Effective friction angle."},
    },
    "cpt_undrained_shear_strength": {
        "category": "CPT Correlations", "brief": "Estimate Su from CPT (Rad & Lunne 1988).",
        "parameters": {
            "qnet": {"type": "float", "required": True, "range": "0 to 120", "description": "Net cone resistance in MPa."},
            "Nk": {"type": "float", "required": True, "range": "8 to 30", "description": "Cone factor."},
        },
        "returns": {"Su [kPa]": "Undrained shear strength."},
    },
    "cpt_ocr": {
        "category": "CPT Correlations", "brief": "Estimate OCR for clay from CPT (Lunne et al 1997).",
        "parameters": {
            "Qt": {"type": "float", "required": True, "range": "2 to 34"},
            "Bq": {"type": "float", "required": False, "description": "Pore pressure ratio (optional)."},
        },
        "returns": {"OCR_Qt_BE [-]": "Best estimate OCR from Qt.", "OCR_Bq_BE [-]": "Best estimate OCR from Bq."},
    },
    "cpt_sensitivity": {
        "category": "CPT Correlations", "brief": "Estimate clay sensitivity from CPT (Rad & Lunne 1986).",
        "parameters": {"Rf": {"type": "float", "required": True, "range": "0.5 to 2.2"}},
        "returns": {"St BE [-]": "Best estimate sensitivity."},
    },
    "cpt_unit_weight": {
        "category": "CPT Correlations", "brief": "Estimate unit weight from CPT (Mayne et al 2010).",
        "parameters": {
            "ft": {"type": "float", "required": True, "range": "0 to 10", "description": "Sleeve friction in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 500"},
        },
        "returns": {"gamma [kN/m3]": "Total unit weight."},
    },
    "cpt_shear_wave_velocity": {
        "category": "CPT Correlations", "brief": "Estimate Vs and Gmax from CPT (Robertson & Cabal 2015).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 100", "description": "Total cone resistance in MPa."},
            "ic": {"type": "float", "required": True, "range": "1 to 4", "description": "Soil behaviour type index."},
            "sigma_vo": {"type": "float", "required": True, "description": "Total vertical stress in kPa."},
        },
        "returns": {"Vs [m/s]": "Shear wave velocity.", "Gmax [kPa]": "Small-strain shear modulus."},
    },
    "cpt_k0_sand": {
        "category": "CPT Correlations", "brief": "Estimate K0 for sand from CPT (Mayne 2007).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 100"},
            "sigma_vo_eff": {"type": "float", "required": True},
            "ocr": {"type": "float", "required": True, "range": "1 to 20"},
        },
        "returns": {"K0 CPT [-]": "K0 from CPT.", "K0 conventional [-]": "K0 conventional."},
    },
    "cpt_constrained_modulus": {
        "category": "CPT Correlations", "brief": "Estimate constrained modulus from CPT (Robertson 2009).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 100"},
            "ic": {"type": "float", "required": True, "range": "1 to 5"},
            "sigma_vo": {"type": "float", "required": True},
            "sigma_vo_eff": {"type": "float", "required": True},
        },
        "returns": {"M [kPa]": "Constrained modulus.", "mv [1/kPa]": "Coefficient of compressibility."},
    },
    # Bearing Capacity
    "bearing_capacity_nq": {
        "category": "Bearing Capacity", "brief": "Calculate Nq from friction angle.",
        "parameters": {"friction_angle": {"type": "float", "required": True, "range": "20 to 50"}},
        "returns": {"Nq [-]": "Bearing capacity factor Nq."},
    },
    "bearing_capacity_ngamma_vesic": {
        "category": "Bearing Capacity", "brief": "Calculate Ngamma (Vesic 1973).",
        "parameters": {"friction_angle": {"type": "float", "required": True, "range": "20 to 50"}},
        "returns": {"Ngamma [-]": "Bearing capacity factor Ngamma."},
    },
    "bearing_capacity_ngamma_meyerhof": {
        "category": "Bearing Capacity", "brief": "Calculate Ngamma (Meyerhof 1976, more conservative).",
        "parameters": {"friction_angle": {"type": "float", "required": True, "range": "20 to 50"}},
        "returns": {"Ngamma [-]": "Bearing capacity factor Ngamma."},
    },
    "bearing_capacity_ngamma_davisbooker": {
        "category": "Bearing Capacity", "brief": "Calculate Ngamma with roughness (Davis & Booker 1971).",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "range": "20 to 50"},
            "roughness_factor": {"type": "float", "required": True, "range": "0 to 1"},
        },
        "returns": {"Ngamma [-]": "Ngamma for given roughness."},
    },
    "bearing_capacity_undrained_api": {
        "category": "Bearing Capacity", "brief": "Undrained vertical bearing capacity (API RP 2GEO).",
        "parameters": {
            "effective_length": {"type": "float", "required": True, "description": "Effective length L' in m."},
            "effective_width": {"type": "float", "required": True, "description": "Effective width B' in m."},
            "su_base": {"type": "float", "required": True, "description": "Undrained shear strength at base in kPa."},
            "base_depth": {"type": "float", "required": False, "default": 0.0, "description": "Foundation depth in m."},
            "skirted": {"type": "bool", "required": False, "default": True},
        },
        "returns": {"qu [kPa]": "Net bearing pressure.", "vertical_capacity [kN]": "Vertical capacity."},
    },
    "bearing_capacity_drained_api": {
        "category": "Bearing Capacity", "brief": "Drained vertical bearing capacity (API RP 2GEO).",
        "parameters": {
            "vertical_effective_stress": {"type": "float", "required": True, "description": "Effective stress at base in kPa."},
            "effective_friction_angle": {"type": "float", "required": True, "range": "20 to 50"},
            "effective_unit_weight": {"type": "float", "required": True, "range": "3 to 12"},
            "effective_length": {"type": "float", "required": True},
            "effective_width": {"type": "float", "required": True},
        },
        "returns": {"qu [kPa]": "Net bearing pressure.", "vertical_capacity [kN]": "Vertical capacity."},
    },
    "sliding_capacity_undrained_api": {
        "category": "Bearing Capacity", "brief": "Undrained sliding capacity (API RP 2GEO).",
        "parameters": {
            "su_base": {"type": "float", "required": True, "description": "Su at base in kPa."},
            "foundation_area": {"type": "float", "required": True, "description": "Foundation area in m2."},
        },
        "returns": {"sliding_capacity [kN]": "Total sliding capacity."},
    },
    "sliding_capacity_drained_api": {
        "category": "Bearing Capacity", "brief": "Drained sliding capacity (API RP 2GEO).",
        "parameters": {
            "vertical_load": {"type": "float", "required": True, "description": "Vertical load in kN."},
            "effective_friction_angle": {"type": "float", "required": True, "range": "20 to 50"},
            "effective_unit_weight": {"type": "float", "required": True, "range": "3 to 12"},
        },
        "returns": {"sliding_capacity [kN]": "Total sliding capacity."},
    },
    "effective_area_rectangle": {
        "category": "Bearing Capacity", "brief": "Effective area for eccentric rectangular foundation (API RP 2GEO).",
        "parameters": {
            "length": {"type": "float", "required": True, "description": "Foundation length L in m."},
            "width": {"type": "float", "required": True, "description": "Foundation width B in m."},
            "eccentricity_length": {"type": "float", "required": False, "description": "Eccentricity in length dir in m."},
            "eccentricity_width": {"type": "float", "required": False, "description": "Eccentricity in width dir in m."},
        },
        "returns": {"effective_area [m2]": "Reduced area.", "effective_length [m]": "L'.", "effective_width [m]": "B'."},
    },
    "effective_area_circle": {
        "category": "Bearing Capacity", "brief": "Effective area for eccentric circular foundation (API RP 2GEO).",
        "parameters": {
            "foundation_radius": {"type": "float", "required": True, "description": "Foundation radius in m."},
            "eccentricity": {"type": "float", "required": False, "description": "Eccentricity in m."},
        },
        "returns": {"effective_area [m2]": "Reduced area.", "effective_length [m]": "L'.", "effective_width [m]": "B'."},
    },
    # Consolidation & Settlement
    "consolidation_degree": {
        "category": "Consolidation & Settlement",
        "brief": "Calculate degree of consolidation for given time and soil properties.",
        "parameters": {
            "time": {"type": "float", "required": True, "range": ">= 0", "description": "Time [s]."},
            "cv": {"type": "float", "required": True, "range": "0.1 to 1000", "description": "Coefficient of consolidation [m2/yr]."},
            "drainage_length": {"type": "float", "required": True, "range": ">= 0", "description": "Drainage length Hdr [m]."},
            "distribution": {"type": "string", "required": False, "default": "uniform", "description": "'uniform' or 'triangular'."},
        },
        "returns": {"U [pct]": "Average degree of consolidation [%].", "Tv [-]": "Time factor."},
    },
    "primary_consolidation_settlement_nc": {
        "category": "Consolidation & Settlement",
        "brief": "Primary consolidation settlement for normally consolidated clay.",
        "parameters": {
            "initial_height": {"type": "float", "required": True, "description": "Layer thickness H0 [m]."},
            "initial_voidratio": {"type": "float", "required": True, "range": "0.1 to 5.0", "description": "Initial void ratio e0."},
            "initial_effective_stress": {"type": "float", "required": True, "description": "Initial vertical effective stress [kPa]."},
            "effective_stress_increase": {"type": "float", "required": True, "description": "Stress increase [kPa]."},
            "compression_index": {"type": "float", "required": True, "range": "0.1 to 0.8", "description": "Compression index Cc."},
        },
        "returns": {"delta z [m]": "Settlement.", "delta e [-]": "Void ratio decrease.", "e final [-]": "Final void ratio."},
    },
    "primary_consolidation_settlement_oc": {
        "category": "Consolidation & Settlement",
        "brief": "Primary consolidation settlement for overconsolidated clay.",
        "parameters": {
            "initial_height": {"type": "float", "required": True, "description": "Layer thickness H0 [m]."},
            "initial_voidratio": {"type": "float", "required": True, "range": "0.1 to 5.0"},
            "initial_effective_stress": {"type": "float", "required": True, "description": "Initial stress [kPa]."},
            "preconsolidation_pressure": {"type": "float", "required": True, "description": "Preconsolidation pressure [kPa]."},
            "effective_stress_increase": {"type": "float", "required": True, "description": "Stress increase [kPa]."},
            "compression_index": {"type": "float", "required": True, "range": "0.1 to 0.8"},
            "recompression_index": {"type": "float", "required": True, "range": "0.015 to 0.35"},
        },
        "returns": {"delta z [m]": "Settlement.", "delta e [-]": "Void ratio decrease.", "e final [-]": "Final void ratio."},
    },
    "consolidation_settlement_mv": {
        "category": "Consolidation & Settlement",
        "brief": "Consolidation settlement using compressibility mv.",
        "parameters": {
            "initial_height": {"type": "float", "required": True, "description": "Layer thickness H0 [m]."},
            "effective_stress_increase": {"type": "float", "required": True, "description": "Stress increase [kPa]."},
            "compressibility": {"type": "float", "required": True, "range": "1e-4 to 10", "description": "mv [1/kPa]."},
        },
        "returns": {"delta z [m]": "Settlement.", "delta epsilon [-]": "Change in strain."},
    },
    "hydraulic_conductivity_unconfined": {
        "category": "Consolidation & Settlement",
        "brief": "Hydraulic conductivity from unconfined aquifer pumping test.",
        "parameters": {
            "radius_1": {"type": "float", "required": True, "description": "Radial distance to first standpipe [m]."},
            "radius_2": {"type": "float", "required": True, "description": "Radial distance to second standpipe [m]."},
            "piezometric_height_1": {"type": "float", "required": True, "description": "Height in first standpipe [m]."},
            "piezometric_height_2": {"type": "float", "required": True, "description": "Height in second standpipe [m]."},
            "flowrate": {"type": "float", "required": True, "description": "Flowrate [m3/s]."},
        },
        "returns": {"hydraulic_conductivity [m/s]": "Hydraulic conductivity k."},
    },
    # Stress Distribution
    "stress_pointload": {
        "category": "Stress Distribution",
        "brief": "Boussinesq stress distribution under a point load.",
        "parameters": {
            "pointload": {"type": "float", "required": True, "description": "Point load Q [kN]."},
            "z": {"type": "float", "required": True, "description": "Vertical distance from surface [m]."},
            "r": {"type": "float", "required": True, "description": "Radial distance from load [m]."},
            "poissonsratio": {"type": "float", "required": True, "range": "0 to 0.5"},
        },
        "returns": {"delta sigma z [kPa]": "Vertical stress.", "delta sigma r [kPa]": "Radial stress.",
                    "delta sigma theta [kPa]": "Tangential stress.", "delta tau rz [kPa]": "Shear stress."},
    },
    "stress_stripload": {
        "category": "Stress Distribution",
        "brief": "Stress distribution under a strip load.",
        "parameters": {
            "z": {"type": "float", "required": True, "description": "Vertical distance [m]."},
            "x": {"type": "float", "required": True, "description": "Horizontal offset from strip corner [m]."},
            "width": {"type": "float", "required": True, "description": "Strip width B [m]."},
            "imposedstress": {"type": "float", "required": True, "description": "Applied stress qs [kPa]."},
            "triangular": {"type": "bool", "required": False, "default": False, "description": "True for triangular loading."},
        },
        "returns": {"delta sigma z [kPa]": "Vertical stress.", "delta sigma x [kPa]": "Horizontal stress.",
                    "delta tau zx [kPa]": "Shear stress."},
    },
    "stress_circle": {
        "category": "Stress Distribution",
        "brief": "Stress distribution below center of circular foundation.",
        "parameters": {
            "z": {"type": "float", "required": True, "description": "Depth below base [m]."},
            "footing_radius": {"type": "float", "required": True, "description": "Foundation radius [m]."},
            "imposedstress": {"type": "float", "required": True, "description": "Applied stress qs [kPa]."},
            "poissonsratio": {"type": "float", "required": True, "range": "0 to 0.5"},
        },
        "returns": {"delta sigma z [kPa]": "Vertical stress.", "delta sigma r [kPa]": "Radial stress."},
    },
    "stress_rectangle": {
        "category": "Stress Distribution",
        "brief": "Stress distribution below corner of rectangular loaded area.",
        "parameters": {
            "imposedstress": {"type": "float", "required": True, "description": "Applied stress qs [kPa]."},
            "length": {"type": "float", "required": True, "description": "Longest edge L [m]."},
            "width": {"type": "float", "required": True, "description": "Shortest edge B [m]."},
            "z": {"type": "float", "required": True, "description": "Depth below footing [m]."},
        },
        "returns": {"delta sigma z [kPa]": "Vertical stress.", "delta sigma x [kPa]": "Horizontal stress (width).",
                    "delta sigma y [kPa]": "Horizontal stress (length).", "delta tau zx [kPa]": "Shear stress."},
    },
    # Earth Pressure
    "earth_pressure_basic": {
        "category": "Earth Pressure",
        "brief": "Active and passive earth pressure coefficients from friction angle.",
        "parameters": {
            "phi_eff": {"type": "float", "required": True, "range": "20 to 50", "description": "Effective friction angle [deg]."},
        },
        "returns": {"Ka [-]": "Active coefficient.", "Kp [-]": "Passive coefficient.",
                    "theta_a [radians]": "Active slip angle.", "theta_p [radians]": "Passive slip angle."},
    },
    "earth_pressure_poncelet": {
        "category": "Earth Pressure",
        "brief": "Earth pressure coefficients with wall friction and inclination (Poncelet/Coulomb).",
        "parameters": {
            "phi_eff": {"type": "float", "required": True, "range": "20 to 50"},
            "interface_friction_angle": {"type": "float", "required": True, "range": "15 to 40", "description": "Wall-soil friction [deg]."},
            "wall_angle": {"type": "float", "required": True, "range": "0 to 70", "description": "Wall angle to vertical [deg]."},
            "top_angle": {"type": "float", "required": True, "range": "0 to 70", "description": "Ground slope angle [deg]."},
        },
        "returns": {"KaC [-]": "Poncelet active coefficient.", "KpC [-]": "Poncelet passive coefficient."},
    },
    "earth_pressure_rankine": {
        "category": "Earth Pressure",
        "brief": "Rankine earth pressure coefficients for inclined wall with sloping ground.",
        "parameters": {
            "phi_eff": {"type": "float", "required": True, "range": "20 to 50"},
            "wall_angle": {"type": "float", "required": True, "range": "0 to 70", "description": "Wall angle to vertical [deg]."},
            "top_angle": {"type": "float", "required": True, "range": "0 to 70", "description": "Ground slope angle [deg]."},
        },
        "returns": {"KaR [-]": "Rankine active coefficient.", "KpR [-]": "Rankine passive coefficient."},
    },
    # Soil Classification
    "relative_density_category": {
        "category": "Soil Classification",
        "brief": "Classify relative density (Very loose to Very dense).",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "range": "0 to 1", "description": "Relative density Dr (fraction)."},
        },
        "returns": {"Relative density": "Category string."},
    },
    "su_category": {
        "category": "Soil Classification",
        "brief": "Classify undrained shear strength (BS 5930 or ASTM D-2488).",
        "parameters": {
            "undrained_shear_strength": {"type": "float", "required": True, "range": "0 to 1000", "description": "Su [kPa]."},
            "standard": {"type": "string", "required": False, "default": "BS 5930:2015", "description": "'BS 5930:2015' or 'ASTM D-2488'."},
        },
        "returns": {"strength class": "Strength classification."},
    },
    "uscs_description": {
        "category": "Soil Classification",
        "brief": "Get USCS soil type description from symbol.",
        "parameters": {
            "symbol": {"type": "string", "required": True, "description": "USCS symbol (e.g. CL, SW, SM)."},
        },
        "returns": {"Soil type": "Verbose description."},
    },
    "sample_quality_lunne": {
        "category": "Soil Classification",
        "brief": "Assess sample quality using void ratio change (Lunne et al.).",
        "parameters": {
            "voidratio": {"type": "float", "required": True, "range": "0.3 to 3", "description": "Initial void ratio e0."},
            "voidratio_change": {"type": "float", "required": True, "range": "-1 to 1", "description": "Change in void ratio delta_e."},
            "ocr": {"type": "float", "required": True, "range": "1 to 4", "description": "Overconsolidation ratio."},
        },
        "returns": {"delta e/e0 [-]": "Ratio for classification.", "Quality category": "Quality category."},
    },
    # Deep Foundations
    "pile_shaft_friction_api_sand": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in sand (API RP 2GEO beta method).",
        "parameters": {
            "api_relativedensity": {"type": "string", "required": True, "description": "'Medium dense', 'Dense', or 'Very dense'."},
            "api_soildescription": {"type": "string", "required": True, "description": "'Sand' or 'Sand-silt'."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Vertical effective stress [kPa]."},
        },
        "returns": {"f_s_comp_out [kPa]": "Compression shaft friction (outside).", "f_s_lim [kPa]": "Limiting shaft friction."},
    },
    "pile_shaft_friction_api_clay": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in clay (API RP 2GEO alpha method).",
        "parameters": {
            "undrained_shear_strength": {"type": "float", "required": True, "range": "0 to 400", "description": "Su [kPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Vertical effective stress [kPa]."},
        },
        "returns": {"f_s_comp_out [kPa]": "Compression shaft friction.", "alpha [-]": "Alpha adhesion factor."},
    },
    "pile_shaft_friction_almhamre_sand": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in sand with friction fatigue (Alm & Hamre).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120", "description": "Total cone resistance [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Vertical effective stress [kPa]."},
            "interface_friction_angle": {"type": "float", "required": True, "range": "10 to 50", "description": "Interface friction [deg]."},
            "depth": {"type": "float", "required": True, "description": "Calculation depth [m]."},
            "embedded_length": {"type": "float", "required": True, "description": "Pile tip depth [m]."},
        },
        "returns": {"f_s_comp_out [kPa]": "Outside compression shaft friction.", "f_s_initial [kPa]": "Initial unit skin friction."},
    },
    "pile_shaft_friction_almhamre_clay": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in clay with friction fatigue (Alm & Hamre).",
        "parameters": {
            "depth": {"type": "float", "required": True, "description": "Calculation depth [m]."},
            "embedded_length": {"type": "float", "required": True, "description": "Pile tip depth [m]."},
            "qt": {"type": "float", "required": True, "range": "0 to 120", "description": "Total cone resistance [MPa]."},
            "fs": {"type": "float", "required": True, "description": "CPT sleeve friction [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Vertical effective stress [kPa]."},
        },
        "returns": {"f_s_comp_out [kPa]": "Outside compression shaft friction.", "f_s_initial [kPa]": "Initial unit skin friction."},
    },
    "pile_end_bearing_api_clay": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in clay (API RP 2GEO).",
        "parameters": {
            "undrained_shear_strength": {"type": "float", "required": True, "range": "0 to 400", "description": "Su at pile tip [kPa]."},
        },
        "returns": {"q_b_coring [kPa]": "Unit end bearing (coring).", "q_b_plugged [kPa]": "Unit end bearing (plugged)."},
    },
    "pile_end_bearing_api_sand": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in sand (API RP 2GEO).",
        "parameters": {
            "api_relativedensity": {"type": "string", "required": True, "description": "'Medium dense', 'Dense', or 'Very dense'."},
            "api_soildescription": {"type": "string", "required": True, "description": "'Sand' or 'Sand-silt'."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective stress at tip [kPa]."},
        },
        "returns": {"q_b_coring [kPa]": "Unit end bearing (coring).", "q_b_plugged [kPa]": "Unit end bearing (plugged)."},
    },
    "pile_end_bearing_almhamre_sand": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in sand (Alm & Hamre CPT-based).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120", "description": "Total cone resistance [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective stress [kPa]."},
        },
        "returns": {"q_b_coring [kPa]": "Unit end bearing (coring).", "q_b_plugged [kPa]": "Unit end bearing (plugged)."},
    },
    "pile_end_bearing_almhamre_clay": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in clay (Alm & Hamre CPT-based).",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120", "description": "Total cone resistance [MPa]."},
        },
        "returns": {"q_b_coring [kPa]": "Unit end bearing (coring).", "q_b_plugged [kPa]": "Unit end bearing (plugged)."},
    },
    # Soil Dynamics & Liquefaction
    "modulus_reduction_ishibashi": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "G/Gmax modulus reduction curve and damping ratio (Ishibashi & Zhang).",
        "parameters": {
            "strain": {"type": "float", "required": True, "range": "0 to 10", "description": "Shear strain [%]."},
            "pi": {"type": "float", "required": True, "range": "0 to 200", "description": "Plasticity index PI [%]. Use 0 for sand."},
            "sigma_m_eff": {"type": "float", "required": True, "range": "0 to 400", "description": "Mean effective stress [kPa]."},
        },
        "returns": {"G/Gmax [-]": "Modulus reduction ratio.", "dampingratio [pct]": "Damping ratio [%]."},
    },
    "gmax_from_shear_wave_velocity": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Calculate Gmax from shear wave velocity and unit weight.",
        "parameters": {
            "Vs": {"type": "float", "required": True, "range": "0 to 600", "description": "Shear wave velocity [m/s]."},
            "gamma": {"type": "float", "required": True, "range": "12 to 22", "description": "Bulk unit weight [kN/m3]."},
        },
        "returns": {"rho [kg/m3]": "Density.", "Gmax [kPa]": "Small-strain shear modulus."},
    },
    "damping_ratio_seed": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Damping ratio for sand and gravel (Seed et al. 1986).",
        "parameters": {
            "cyclic_shear_strain": {"type": "float", "required": True, "range": "0.0001 to 1.0", "description": "Cyclic shear strain [%]."},
        },
        "returns": {"D LE [pct]": "Low estimate.", "D BE [pct]": "Best estimate.", "D HE [pct]": "High estimate."},
    },
    "cyclic_stress_ratio_moss": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Cyclic stress ratio CSR for liquefaction (Moss/Cetin formulation).",
        "parameters": {
            "sigma_vo": {"type": "float", "required": True, "description": "Total vertical stress [kPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective vertical stress [kPa]."},
            "magnitude": {"type": "float", "required": True, "range": "5.5 to 8.5", "description": "Earthquake magnitude Mw."},
            "acceleration": {"type": "float", "required": True, "description": "Max horizontal acceleration [m/s2]."},
            "depth": {"type": "float", "required": True, "description": "Depth [m]."},
        },
        "returns": {"CSR [-]": "Uncorrected CSR.", "CSR* [-]": "CSR adjusted to Mw=7.5.", "DWF [-]": "Duration weighting factor.", "rd [-]": "Depth reduction factor."},
    },
    "cyclic_stress_ratio_youd": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Cyclic stress ratio CSR (Youd et al. 2001 formulation).",
        "parameters": {
            "acceleration": {"type": "float", "required": True, "description": "Max horizontal acceleration [m/s2]."},
            "sigma_vo": {"type": "float", "required": True, "description": "Total vertical stress [kPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective vertical stress [kPa]."},
            "depth": {"type": "float", "required": True, "range": "0 to 23", "description": "Depth [m]."},
            "magnitude": {"type": "float", "required": True, "range": "0 to 8.5", "description": "Earthquake magnitude."},
        },
        "returns": {"CSR [-]": "Uncorrected CSR.", "CSR* [-]": "CSR adjusted to Mw=7.5.", "MSF [-]": "Magnitude scaling factor.", "rd [-]": "Depth reduction factor."},
    },
    "liquefaction_robertson_fear": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Liquefaction triggering from CPT (Robertson & Fear 1995).",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120", "description": "Cone tip resistance [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "description": "Effective stress [kPa]."},
            "CSR": {"type": "float", "required": True, "range": "0.073 to 0.49", "description": "Cyclic stress ratio."},
        },
        "returns": {"qc1 [-]": "Normalised cone resistance.", "liquefaction": "True if liquefaction predicted."},
    },
    # Soil Correlations
    "gmax_sand_hardin_black": {
        "category": "Soil Correlations",
        "brief": "Small-strain shear modulus for sand (Hardin & Black).",
        "parameters": {
            "sigma_m0": {"type": "float", "required": True, "range": "0 to 500", "description": "Mean effective stress p' [kPa]."},
            "void_ratio": {"type": "float", "required": True, "range": "0 to 4", "description": "Void ratio e0."},
        },
        "returns": {"Gmax [kPa]": "Small-strain shear modulus."},
    },
    "permeability_hazen": {
        "category": "Soil Correlations",
        "brief": "Permeability from grain size using Hazen correlation.",
        "parameters": {
            "grain_size": {"type": "float", "required": True, "range": "0.01 to 2.0", "description": "D10 grain size [mm]."},
        },
        "returns": {"k [m/s]": "Permeability."},
    },
    "hssmall_parameters_sand": {
        "category": "Soil Correlations",
        "brief": "HS Small constitutive model parameters for sand (PLAXIS).",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "range": "10 to 100", "description": "Relative density Dr [%]."},
        },
        "returns": {"E50_ref [kPa]": "Reference secant stiffness.", "G0_ref [kPa]": "Small-strain shear modulus.",
                    "phi_eff [deg]": "Effective friction angle.", "psi [deg]": "Dilation angle."},
    },
    "stress_dilatancy_bolton": {
        "category": "Soil Correlations",
        "brief": "Stress-dilatancy relationships for sand (Bolton 1986).",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "range": "0.1 to 1.0", "description": "Relative density Dr (fraction)."},
            "p_eff": {"type": "float", "required": True, "range": "20 to 10000", "description": "Effective pressure p' [kPa]."},
        },
        "returns": {"Ir [-]": "Relative dilatancy index.", "phi_max - phi_cs [deg]": "Peak-CS friction angle difference.",
                    "Dilation angle [deg]": "Dilation angle."},
    },
    "compression_index_koppula": {
        "category": "Soil Correlations",
        "brief": "Compression and recompression indices from water content (Koppula).",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 4", "description": "Natural water content (fraction)."},
        },
        "returns": {"Cc [-]": "Compression index.", "Cr [-]": "Recompression index."},
    },
    "friction_angle_from_pi": {
        "category": "Soil Correlations",
        "brief": "Drained friction angle of clay from plasticity index.",
        "parameters": {
            "plasticity_index": {"type": "float", "required": True, "range": "5 to 1000", "description": "Plasticity index PI [%]."},
        },
        "returns": {"Effective friction angle [deg]": "Drained friction angle."},
    },
    "cv_from_liquid_limit": {
        "category": "Soil Correlations",
        "brief": "Coefficient of consolidation from liquid limit (US Navy).",
        "parameters": {
            "liquid_limit": {"type": "float", "required": True, "range": "20 to 160", "description": "Liquid limit LL [%]."},
            "trend": {"type": "string", "required": False, "default": "NC", "description": "'Remoulded', 'NC', or 'OC'."},
        },
        "returns": {"cv [m2/yr]": "Coefficient of consolidation."},
    },
    "gmax_clay_andersen": {
        "category": "Soil Correlations",
        "brief": "Small-strain shear modulus for clay from PI, OCR, and stress (Andersen).",
        "parameters": {
            "pi": {"type": "float", "required": True, "range": "0 to 160", "description": "Plasticity index PI [%]."},
            "ocr": {"type": "float", "required": True, "range": "1 to 40", "description": "OCR."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 1000", "description": "Vertical effective stress [kPa]."},
        },
        "returns": {"Gmax [kPa]": "Small-strain shear modulus."},
    },
    "k0_from_plasticity": {
        "category": "Soil Correlations",
        "brief": "K0 from plasticity index and OCR (Kenney/Alpan).",
        "parameters": {
            "pi": {"type": "float", "required": True, "range": "5 to 80", "description": "Plasticity index PI [%]."},
            "ocr": {"type": "float", "required": False, "default": 1, "range": "1 to 30", "description": "OCR."},
        },
        "returns": {"K0 NC [-]": "K0 for NC.", "K0 [-]": "K0 for given OCR."},
    },
    "k0_from_friction_angle": {
        "category": "Soil Correlations",
        "brief": "K0 from critical-state friction angle and OCR (Mesri & Hayat).",
        "parameters": {
            "phi_cs": {"type": "float", "required": True, "range": "15 to 45", "description": "Critical state friction angle [deg]."},
            "ocr": {"type": "float", "required": False, "default": 1, "range": "1 to 30", "description": "OCR."},
        },
        "returns": {"K0 [-]": "Coefficient of lateral earth pressure at rest."},
    },
}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _clean_value(v):
    """Convert numpy/float types to plain Python types for JSON serialization."""
    if v is None:
        return None
    if isinstance(v, float) and math.isnan(v):
        return None
    if isinstance(v, (np.floating, np.integer)):
        return float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def _clean_result(result: dict) -> dict:
    """Clean all values in a groundhog result dict for JSON-safe output."""
    return {k: _clean_value(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# Foundry functions - register these three as tools in AIP Agent Studio
# ---------------------------------------------------------------------------

@function
def groundhog_agent(method: str, parameters_json: str) -> str:
    """
    Geotechnical engineering calculator using the groundhog library.

    Call this function with a method name and a JSON string of parameters.
    First use groundhog_list_methods() to see available methods, then
    groundhog_describe_method() for parameter details.

    90 methods across 11 categories: Phase Relations (14), SPT Correlations (10),
    CPT Correlations (16), Bearing Capacity (10), Consolidation & Settlement (5),
    Stress Distribution (4), Earth Pressure (3), Soil Classification (4),
    Deep Foundations (8), Soil Dynamics & Liquefaction (6), Soil Correlations (10).

    Parameters:
        method: The calculation method name (e.g. "bulk_unit_weight", "cpt_friction_angle_sand").
        parameters_json: JSON string of parameters (e.g. '{"porosity": 0.4}').

    Returns:
        JSON string with calculation results or an error message.
    """
    # Parse parameters JSON
    try:
        parameters = json.loads(parameters_json)
    except (json.JSONDecodeError, TypeError) as e:
        return json.dumps({"error": f"Invalid parameters_json: {str(e)}"})

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available methods: {available}"})

    func = METHOD_REGISTRY[method]

    try:
        raw_result = func(**parameters)
        return json.dumps(_clean_result(raw_result), default=str)
    except Exception as e:
        return json.dumps({"error": f"{type(e).__name__}: {str(e)}"})


@function
def groundhog_list_methods(category: str = "") -> str:
    """
    Lists all available geotechnical calculation methods.
    Use this to discover what calculations are available before calling groundhog_agent.

    Parameters:
        category: Optional filter. Options: "Phase Relations", "SPT Correlations",
            "CPT Correlations", "Bearing Capacity", "Consolidation & Settlement",
            "Stress Distribution", "Earth Pressure", "Soil Classification",
            "Deep Foundations", "Soil Dynamics & Liquefaction", "Soil Correlations".
            Leave empty for all methods.

    Returns:
        JSON string with method names and brief descriptions grouped by category.
    """
    result = {}
    for method_name, info in METHOD_INFO.items():
        if category and info["category"].lower() != category.lower():
            continue
        cat = info["category"]
        if cat not in result:
            result[cat] = {}
        result[cat][method_name] = info["brief"]
    if not result:
        return json.dumps({"error": f"No methods found for category '{category}'. "
                                     f"Available categories: Phase Relations, SPT Correlations, "
                                     f"CPT Correlations, Bearing Capacity, Consolidation & Settlement, "
                                     f"Stress Distribution, Earth Pressure, Soil Classification, "
                                     f"Deep Foundations, Soil Dynamics & Liquefaction, Soil Correlations"})
    return json.dumps(result)


@function
def groundhog_describe_method(method: str) -> str:
    """
    Returns detailed documentation for a specific groundhog agent method.
    Use this to understand what parameters a method needs before calling groundhog_agent.

    Parameters:
        method: The method name (e.g. "bulk_unit_weight", "cpt_normalisations").

    Returns:
        JSON string with: category, brief, description, parameters (types, ranges, defaults),
        and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return json.dumps({"error": f"Unknown method '{method}'. Available methods: {available}"})
    return json.dumps(METHOD_INFO[method], default=str)
