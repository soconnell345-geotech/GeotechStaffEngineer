"""
Groundhog Geotechnical Agent - Dispatcher function for Palantir Foundry AIP Agent Studio.

Single entry point that routes to groundhog library functions based on a method name
and a dictionary of parameters. Designed to be registered as one tool in Foundry.
"""

import math
import numpy as np

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
        "description": (
            "Converts porosity (n) to void ratio (e) using e = n / (1 - n). "
            "Porosity is the ratio of void volume to total volume; void ratio is the "
            "ratio of void volume to solids volume."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "porosity": {"type": "float", "required": True, "range": "0 to 1",
                         "description": "Porosity (n), ratio of void volume to total volume."},
        },
        "returns": {
            "voidratio [-]": "Void ratio (e), ratio of void volume to solids volume.",
        },
    },
    "porosity_from_voidratio": {
        "category": "Phase Relations",
        "brief": "Convert void ratio to porosity.",
        "description": (
            "Converts void ratio (e) to porosity (n) using n = e / (1 + e)."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "voidratio": {"type": "float", "required": True, "range": "0 to 5",
                          "description": "Void ratio (e)."},
        },
        "returns": {
            "porosity [-]": "Porosity (n).",
        },
    },
    "saturation_from_watercontent": {
        "category": "Phase Relations",
        "brief": "Calculate degree of saturation from water content and void ratio.",
        "description": (
            "Calculates saturation S = w * Gs / e. Saturation is the ratio of "
            "water volume to void volume. Useful for determining whether soil is "
            "fully saturated or partially saturated."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 4",
                              "description": "Water content (w), ratio of weight of water to weight of solids."},
            "voidratio": {"type": "float", "required": True, "range": "0 to 4",
                          "description": "Void ratio (e)."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity of soil grains (Gs)."},
        },
        "returns": {
            "saturation [-]": "Degree of saturation (S), 0 = dry, 1 = fully saturated.",
        },
    },
    "bulk_unit_weight": {
        "category": "Phase Relations",
        "brief": "Calculate bulk and effective unit weight from saturation, void ratio, and specific gravity.",
        "description": (
            "Calculates gamma = ((Gs + S*e) / (1+e)) * gamma_w. "
            "Returns both bulk unit weight and effective (submerged) unit weight. "
            "Effective unit weight = bulk unit weight minus unit weight of water."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "saturation": {"type": "float", "required": True, "range": "0 to 1",
                           "description": "Degree of saturation (S)."},
            "voidratio": {"type": "float", "required": True, "range": "0 to 4",
                          "description": "Void ratio (e)."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity (Gs)."},
            "unitweight_water": {"type": "float", "required": False, "default": 10.0,
                                 "description": "Unit weight of water in kN/m3."},
        },
        "returns": {
            "bulk unit weight [kN/m3]": "Total bulk unit weight (gamma).",
            "effective unit weight [kN/m3]": "Effective/submerged unit weight (gamma').",
        },
    },
    "dry_unit_weight": {
        "category": "Phase Relations",
        "brief": "Calculate dry unit weight from water content and bulk unit weight.",
        "description": (
            "Calculates gamma_d = gamma / (1 + w). Dry unit weight is the ratio of "
            "weight of solids to total volume."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "watercontent": {"type": "float", "required": True, "range": "0 to 4",
                             "description": "Water content (w) as decimal."},
            "bulkunitweight": {"type": "float", "required": True, "range": "10 to 25",
                               "description": "Bulk unit weight in kN/m3."},
        },
        "returns": {
            "dry unit weight [kN/m3]": "Dry unit weight (gamma_d).",
        },
    },
    "voidratio_from_dry_density": {
        "category": "Phase Relations",
        "brief": "Calculate void ratio from dry density and specific gravity.",
        "description": (
            "Calculates e = Gs * (rho_w / rho_d) - 1. "
            "Note: this function uses density (kg/m3), not unit weight (kN/m3)."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "dry_density": {"type": "float", "required": True, "range": "1000 to 2000",
                            "description": "Dry density in kg/m3."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity (Gs)."},
            "water_density": {"type": "float", "required": False, "default": 1000.0,
                              "description": "Water density in kg/m3."},
        },
        "returns": {
            "Void ratio [-]": "Void ratio (e).",
        },
    },
    "bulk_unit_weight_from_dry": {
        "category": "Phase Relations",
        "brief": "Calculate bulk and effective unit weight from dry unit weight and water content.",
        "description": (
            "Calculates gamma = (1 + w) * gamma_d. Inverse of dry_unit_weight calculation."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "dryunitweight": {"type": "float", "required": True, "range": "1 to 15",
                              "description": "Dry unit weight in kN/m3."},
            "watercontent": {"type": "float", "required": True, "range": "0 to 4",
                             "description": "Water content (w) as decimal."},
            "unitweight_water": {"type": "float", "required": False, "default": 10.0,
                                 "description": "Unit weight of water in kN/m3."},
        },
        "returns": {
            "bulk unit weight [kN/m3]": "Bulk unit weight (gamma).",
            "effective unit weight [kN/m3]": "Effective unit weight (gamma').",
        },
    },
    "relative_density": {
        "category": "Phase Relations",
        "brief": "Calculate relative density of cohesionless soil from void ratios.",
        "description": (
            "Calculates Dr = (e - e_min) / (e_max - e_min). "
            "Relative density indicates how dense a granular soil is compared to "
            "its loosest and densest possible states. Dr=0 is loosest, Dr=1 is densest."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "void_ratio": {"type": "float", "required": True, "range": "0 to 5",
                           "description": "Current in-situ void ratio (e)."},
            "e_min": {"type": "float", "required": True, "range": "0 to 5",
                      "description": "Void ratio at densest state (e_min)."},
            "e_max": {"type": "float", "required": True, "range": "0 to 5",
                      "description": "Void ratio at loosest state (e_max)."},
        },
        "returns": {
            "Dr [-]": "Relative density as a fraction (0 to 1).",
        },
    },
    "voidratio_from_bulk_unit_weight": {
        "category": "Phase Relations",
        "brief": "Back-calculate void ratio and water content from bulk unit weight.",
        "description": (
            "Derives void ratio e = (gamma_w * Gs - gamma) / (gamma - S * gamma_w) "
            "and water content w = S * e / Gs from bulk unit weight. "
            "Defaults to fully saturated soil (S=1)."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "bulkunitweight": {"type": "float", "required": True, "range": "10 to 25",
                               "description": "Bulk unit weight in kN/m3."},
            "saturation": {"type": "float", "required": False, "default": 1.0,
                           "description": "Degree of saturation (S). Default: 1.0 (fully saturated)."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity (Gs)."},
            "unitweight_water": {"type": "float", "required": False, "default": 10.0,
                                 "description": "Unit weight of water in kN/m3."},
        },
        "returns": {
            "e [-]": "Void ratio.",
            "w [-]": "Water content as decimal.",
        },
    },
    "unit_weight_saturated": {
        "category": "Phase Relations",
        "brief": "Calculate bulk unit weight of a fully saturated soil from water content.",
        "description": (
            "For a saturated soil, calculates gamma = (Gs*(1+w) / (1+w*Gs)) * gamma_w. "
            "Useful when only water content and specific gravity are known."
        ),
        "reference": "UGent in-house practice.",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 2",
                              "description": "Water content (w) as decimal."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity (Gs)."},
            "gamma_w": {"type": "float", "required": False, "default": 10.0,
                        "description": "Unit weight of water in kN/m3."},
        },
        "returns": {
            "gamma [kN/m3]": "Bulk unit weight of saturated soil.",
        },
    },
    "density_from_unit_weight": {
        "category": "Phase Relations",
        "brief": "Convert unit weight (kN/m3) to density (kg/m3).",
        "description": "Converts gamma (kN/m3) to rho (kg/m3) using rho = 1000 * gamma / g.",
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "gamma": {"type": "float", "required": True, "range": "0 to 30",
                      "description": "Unit weight in kN/m3."},
            "g": {"type": "float", "required": False, "default": 9.81,
                  "description": "Gravitational acceleration in m/s2."},
        },
        "returns": {
            "Density [kg/m3]": "Mass density.",
        },
    },
    "unit_weight_from_density": {
        "category": "Phase Relations",
        "brief": "Convert density (kg/m3) to unit weight (kN/m3).",
        "description": "Converts rho (kg/m3) to gamma (kN/m3) using gamma = rho * g / 1000.",
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "density": {"type": "float", "required": True, "range": "0 to 3000",
                        "description": "Density in kg/m3."},
            "g": {"type": "float", "required": False, "default": 9.81,
                  "description": "Gravitational acceleration in m/s2."},
        },
        "returns": {
            "Unit weight [kN/m3]": "Unit weight.",
        },
    },
    "watercontent_from_voidratio": {
        "category": "Phase Relations",
        "brief": "Calculate water content from void ratio (assumes saturated by default).",
        "description": (
            "Calculates w = S * e / Gs. By default assumes full saturation (S=1). "
            "Returns water content as both a decimal and a percentage."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "voidratio": {"type": "float", "required": True, "range": ">= 0",
                          "description": "Void ratio (e)."},
            "saturation": {"type": "float", "required": False, "default": 1.0,
                           "description": "Degree of saturation (S)."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity (Gs)."},
        },
        "returns": {
            "Water content [-]": "Water content as decimal.",
            "Water content [%]": "Water content as percentage.",
        },
    },
    "voidratio_from_watercontent": {
        "category": "Phase Relations",
        "brief": "Calculate void ratio from water content (assumes saturated by default).",
        "description": (
            "Calculates e = w * Gs / S. By default assumes full saturation (S=1)."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 2",
                              "description": "Water content (w) as decimal."},
            "saturation": {"type": "float", "required": False, "default": 1.0,
                           "description": "Degree of saturation (S)."},
            "specific_gravity": {"type": "float", "required": False, "default": 2.65,
                                 "description": "Specific gravity (Gs)."},
        },
        "returns": {
            "Void ratio [-]": "Void ratio (e).",
        },
    },
    # SPT Correlations
    "spt_overburden_correction_liaowhitman": {
        "category": "SPT Correlations",
        "brief": "Correct SPT N for overburden pressure (Liao & Whitman 1986).",
        "description": (
            "Applies overburden correction CN = sqrt(Pa / sigma'vo) to normalize SPT N "
            "to a standard effective overburden pressure of 100 kPa. N1 = CN * N. "
            "For non-granular soils, CN is taken as 1.0."
        ),
        "reference": "Liao & Whitman (1986). Overburden correction factors for SPT in sand.",
        "parameters": {
            "N": {"type": "float", "required": True, "range": ">= 0",
                  "description": "Field SPT N value (or N60 to get (N1)60)."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "> 0",
                             "description": "Effective vertical overburden stress in kPa."},
            "granular": {"type": "bool", "required": False, "default": True,
                         "description": "True for granular soil, False for cohesive (CN=1)."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
        },
        "returns": {
            "CN [-]": "Overburden correction factor.",
            "N1 [-]": "SPT N corrected to 100 kPa overburden.",
        },
    },
    "spt_N60_correction": {
        "category": "SPT Correlations",
        "brief": "Correct raw SPT N to N60 (60% energy efficiency).",
        "description": (
            "Corrects field SPT N number to N60 accounting for hammer efficiency (eta_H), "
            "borehole diameter (eta_B), sampler type (eta_S), and rod length (eta_R). "
            "N60 = N * eta_H * eta_B * eta_S * eta_R / 60. "
            "Default correction factors come from Seed et al (1985) and Skempton (1986) "
            "lookup tables based on country, hammer type, release mechanism, etc. "
            "Overrides can be specified for any correction factor."
        ),
        "reference": "Ameratunga et al (2016). Correlations of Soil and Rock Properties.",
        "parameters": {
            "N": {"type": "float", "required": True, "range": ">= 0",
                  "description": "Field SPT N value."},
            "borehole_diameter": {"type": "float", "required": True, "range": "60 to 200",
                                  "description": "Borehole diameter in mm."},
            "rod_length": {"type": "float", "required": True, "range": ">= 0",
                           "description": "Rod length in m."},
            "country": {"type": "str", "required": True,
                        "options": ["Japan", "United States", "Argentina", "China", "Other"],
                        "description": "Country where SPT was performed."},
            "hammertype": {"type": "str", "required": True, "options": ["Donut", "Safety"],
                           "description": "Type of hammer."},
            "hammerrelease": {"type": "str", "required": True,
                              "options": ["Free fall", "Rope and pulley"],
                              "description": "Hammer release mechanism."},
            "samplertype": {"type": "str", "required": False,
                            "default": "Standard sampler",
                            "options": ["Standard sampler", "With liner for dense sand and clay",
                                        "With liner for loose sand"],
                            "description": "Sampler type."},
            "eta_H": {"type": "float", "required": False,
                      "description": "Override hammer efficiency correction (%)."},
            "eta_B": {"type": "float", "required": False,
                      "description": "Override borehole diameter correction."},
            "eta_S": {"type": "float", "required": False,
                      "description": "Override sampler type correction."},
            "eta_R": {"type": "float", "required": False,
                      "description": "Override rod length correction."},
        },
        "returns": {
            "N60 [-]": "SPT N corrected to 60% energy efficiency.",
            "eta_H [%]": "Hammer efficiency correction as percentage.",
            "eta_H [-]": "Hammer efficiency correction as decimal.",
            "eta_B [-]": "Borehole diameter correction.",
            "eta_S [-]": "Sampler type correction.",
            "eta_R [-]": "Rod length correction.",
        },
    },
    "spt_relative_density_kulhawymayne": {
        "category": "SPT Correlations",
        "brief": "Estimate relative density from corrected SPT (N1)60 (Kulhawy & Mayne 1990).",
        "description": (
            "Dr = sqrt((N1)60 / (60 + 25*log10(d50)) / (CA * COCR)). "
            "Originally proposed for non-aged, normally consolidated sands. "
            "Correction factors for ageing (CA) and overconsolidation (COCR) can be applied. "
            "Requires energy- and stress-corrected (N1)60 as input."
        ),
        "reference": "Kulhawy & Mayne (1990). Manual on estimating soil properties.",
        "parameters": {
            "N1_60": {"type": "float", "required": True, "range": "0 to 100",
                      "description": "SPT number corrected for overburden and energy."},
            "d_50": {"type": "float", "required": True, "range": "0.002 to 20",
                     "description": "Median grain size (D50) in mm."},
            "time_since_deposition": {"type": "float", "required": False, "default": 1.0,
                                      "description": "Age of deposit in years (for CA correction)."},
            "ocr": {"type": "float", "required": False, "default": 1.0,
                    "description": "Overconsolidation ratio (for COCR correction)."},
        },
        "returns": {
            "Dr [-]": "Relative density as fraction.",
            "Dr [pct]": "Relative density as percentage.",
            "C_A [-]": "Ageing correction factor.",
            "C_OCR [-]": "Overconsolidation correction factor.",
        },
    },
    "spt_undrained_shear_strength_salgado": {
        "category": "SPT Correlations",
        "brief": "Estimate undrained shear strength from plasticity index and N60 (Salgado 2008).",
        "description": (
            "Su = alpha' * N60 * Pa, where alpha' depends on plasticity index (PI). "
            "Applicable to cohesive soils. alpha' ranges from 0.068 (PI=15%) to 0.043 (PI=60%)."
        ),
        "reference": "Salgado (2008). The engineering of foundations.",
        "parameters": {
            "pi": {"type": "float", "required": True, "range": "15 to 60",
                   "description": "Plasticity index (PI) in percent."},
            "N_60": {"type": "float", "required": True, "range": "0 to 100",
                     "description": "SPT N60 value."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
        },
        "returns": {
            "alpha_prime [-]": "Correlation factor based on PI.",
            "Su [kPa]": "Estimated undrained shear strength.",
        },
    },
    "spt_friction_angle_kulhawymayne": {
        "category": "SPT Correlations",
        "brief": "Estimate friction angle from SPT N and effective overburden (Kulhawy & Mayne 1990).",
        "description": (
            "phi = arctan((N / (12.2 + 20.3 * sigma'vo/Pa))^0.34). "
            "For cohesionless soils. Uses uncorrected N value (not N60 or (N1)60)."
        ),
        "reference": "Kulhawy & Mayne (1990). Manual on estimating soil properties.",
        "parameters": {
            "N": {"type": "float", "required": True, "range": "0 to 60",
                  "description": "SPT N value (uncorrected or N60)."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 1000",
                             "description": "Effective vertical overburden stress in kPa."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
        },
        "returns": {
            "Phi [deg]": "Internal friction angle in degrees.",
        },
    },
    "spt_relative_density_class": {
        "category": "SPT Correlations",
        "brief": "Classify relative density from uncorrected SPT N (Terzaghi & Peck 1967).",
        "description": (
            "For cohesionless soils. N<=4: Very loose, 4-10: Loose, 10-30: Medium dense, "
            "30-50: Dense, >50: Very dense."
        ),
        "reference": "Terzaghi & Peck (1967). Soil mechanics in engineering practice.",
        "parameters": {
            "N": {"type": "float", "required": True, "range": "0 to 60",
                  "description": "Uncorrected SPT N value."},
        },
        "returns": {
            "Dr class": "Relative density classification string.",
        },
    },
    "spt_overburden_correction_iso": {
        "category": "SPT Correlations",
        "brief": "Correct SPT N for overburden pressure per ISO 22476-3.",
        "description": (
            "CN = sqrt(98 / sigma'vo). Similar to Liao & Whitman but uses 98 kPa "
            "reference pressure. CN should be limited to 2.0 (preferably < 1.5). "
            "Lower limit on sigma'vo of 25 kPa is enforced."
        ),
        "reference": "BS EN ISO 22476-3.",
        "parameters": {
            "N": {"type": "int", "required": True, "range": "0 to 60",
                  "description": "SPT N value or N60."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "25 to 400",
                             "description": "Effective vertical overburden stress in kPa."},
            "granular": {"type": "bool", "required": False, "default": True,
                         "description": "True for granular, False for cohesive (CN=1)."},
        },
        "returns": {
            "CN [-]": "Overburden correction factor.",
            "N1 [-]": "Corrected N value.",
        },
    },
    "spt_friction_angle_pht": {
        "category": "SPT Correlations",
        "brief": "Estimate friction angle from (N1)60 (Peck, Hanson & Thornburn 1974).",
        "description": (
            "phi = 27.1 + 0.3*(N1)60 - 0.00054*(N1)60^2. "
            "Simple quadratic correlation for cohesionless soils."
        ),
        "reference": "Peck, Hanson & Thornburn (1974). Foundation Engineering.",
        "parameters": {
            "N1_60": {"type": "float", "required": True, "range": "0 to 60",
                      "description": "Corrected SPT (N1)60 value."},
        },
        "returns": {
            "Phi [deg]": "Friction angle in degrees.",
        },
    },
    "spt_youngs_modulus_aashto": {
        "category": "SPT Correlations",
        "brief": "Estimate Young's modulus from (N1)60 by soil type (AASHTO 1997).",
        "description": (
            "Es = multiplier * (N1)60 in MPa. "
            "Multiplier depends on soil type: Silts=0.4, Clean sands=0.7, "
            "Coarse sands=1.0, Gravels=1.1."
        ),
        "reference": "AASHTO (1997) LRFD Bridge Design Specifications.",
        "parameters": {
            "N1_60": {"type": "float", "required": True, "range": "0 to 60",
                      "description": "Corrected SPT (N1)60 value."},
            "soiltype": {"type": "str", "required": True,
                         "options": ["Silts", "Clean sands", "Coarse sands", "Gravels"],
                         "description": "Soil type category."},
        },
        "returns": {
            "Es [MPa]": "Estimated Young's modulus.",
        },
    },
    "spt_consistency_class": {
        "category": "SPT Correlations",
        "brief": "Classify consistency of cohesive soil from uncorrected SPT N (Terzaghi & Peck 1967).",
        "description": (
            "For cohesive soils. N<=2: Very soft (qu<25 kPa), 2-4: Soft (25-50), "
            "4-8: Medium (50-100), 8-15: Stiff (100-200), 15-30: Very stiff (200-400), "
            ">30: Hard (>400 kPa). qu is unconfined compressive strength."
        ),
        "reference": "Terzaghi & Peck (1967). Soil mechanics in engineering practice.",
        "parameters": {
            "N": {"type": "float", "required": True, "range": "0 to 60",
                  "description": "Uncorrected SPT N value."},
        },
        "returns": {
            "Consistency class": "Consistency classification string.",
            "qu min [kPa]": "Lower bound of unconfined compressive strength range.",
            "qu max [kPa]": "Upper bound of unconfined compressive strength range.",
        },
    },
    # CPT Correlations
    "cpt_normalisations": {
        "category": "CPT Correlations",
        "brief": "Normalize and correct raw CPT data and classify soil behavior type (Robertson).",
        "description": (
            "Carries out the necessary normalisation and correction on PCPT data to allow "
            "calculation of derived parameters and soil type classification. Corrects cone "
            "resistance for unequal area effect, calculates friction ratio, pore pressure ratio, "
            "normalised cone resistance (Qt, Qtn), normalised friction ratio (Fr), net cone "
            "resistance (qnet), soil behaviour type index (Ic), and Robertson classification. "
            "For downhole tests, a start depth can be specified."
        ),
        "reference": "Lunne, Robertson & Powell (1997). Cone penetration testing in geotechnical practice.",
        "parameters": {
            "measured_qc": {"type": "float", "required": True, "range": "0 to 150",
                            "description": "Measured cone resistance in MPa."},
            "measured_fs": {"type": "float", "required": True, "range": "0 to 10",
                            "description": "Measured sleeve friction in MPa."},
            "measured_u2": {"type": "float", "required": True, "range": "-10 to 10",
                            "description": "Pore pressure measured at the shoulder in MPa."},
            "sigma_vo_tot": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Total vertical stress in kPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Effective vertical stress in kPa."},
            "depth": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Depth below surface (or watertable for onshore) in m."},
            "cone_area_ratio": {"type": "float", "required": True, "range": "0 to 1",
                                "description": "Ratio between cone rod area and maximum cone area (a)."},
            "start_depth": {"type": "float", "required": False, "default": 0.0,
                            "description": "Start depth for downhole tests in m. Leave 0 for surface tests."},
            "unitweight_water": {"type": "float", "required": False, "default": 10.25,
                                 "description": "Unit weight of water in kN/m3 (default is seawater)."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa for normalisation."},
        },
        "returns": {
            "qt [MPa]": "Total cone resistance corrected for unequal area effect.",
            "qc [MPa]": "Cone resistance corrected for downhole effect.",
            "u2 [MPa]": "Pore pressure at the shoulder corrected for downhole effect.",
            "Delta u2 [MPa]": "Excess pore pressure above hydrostatic.",
            "Rf [pct]": "Friction ratio as a percentage.",
            "Bq [-]": "Pore pressure ratio.",
            "Qt [-]": "Normalised cone resistance.",
            "Fr [-]": "Normalised friction ratio.",
            "qnet [MPa]": "Net cone resistance.",
            "exponent_zhang [-]": "Stress exponent n (Zhang et al).",
            "Qtn [-]": "Stress-normalised cone resistance.",
            "Fr [%]": "Normalised friction ratio (percent).",
            "Ic [-]": "Soil behaviour type index.",
            "Ic class number [-]": "Robertson chart soil type number.",
            "Ic class": "Robertson chart soil type description.",
        },
    },
    "cpt_soil_class_robertson": {
        "category": "CPT Correlations",
        "brief": "Classify soil from soil behaviour type index Ic (Robertson chart).",
        "description": (
            "Provides soil type classification according to the soil behaviour type index Ic "
            "by Robertson and Wride. Ic < 1.31: Gravelly sand to sand (zone 7), "
            "1.31-2.05: Clean sands to silty sands (zone 6), 2.05-2.6: Silty sand to sandy silt "
            "(zone 5), 2.6-2.95: Clayey silt to silty clay (zone 4), 2.95-3.6: Clay to silty clay "
            "(zone 3), > 3.6: Organic soils-peats (zone 2)."
        ),
        "reference": "Fugro guidance on PCPT interpretation; Robertson & Wride (1998).",
        "parameters": {
            "ic": {"type": "float", "required": True, "range": "1 to 5",
                   "description": "Soil behaviour type index Ic."},
        },
        "returns": {
            "Soil type number [-]": "Zone number on the Robertson chart.",
            "Soil type": "Soil type description.",
        },
    },
    "cpt_behaviour_index": {
        "category": "CPT Correlations",
        "brief": "Calculate soil behaviour type index Ic from corrected CPT data (Robertson & Wride 1998).",
        "description": (
            "Calculates the soil behaviour type index Ic iteratively according to Robertson and Wride "
            "(1998). Ic below 2.5 indicates generally cohesionless coarse-grained soils; above 2.7 "
            "indicates cohesive fine-grained soils. Between 2.5 and 2.7, partially drained behaviour "
            "is expected. Uses Zhang et al exponent n for stress normalisation."
        ),
        "reference": "Robertson & Wride (1998); Fugro guidance on PCPT interpretation.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Corrected cone resistance qt in MPa."},
            "fs": {"type": "float", "required": True, "range": ">= 0",
                   "description": "Sleeve friction in MPa."},
            "sigma_vo": {"type": "float", "required": True, "range": ">= 0",
                         "description": "Total vertical stress in kPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Effective vertical stress in kPa."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
        },
        "returns": {
            "exponent_zhang [-]": "Stress exponent n (Zhang et al).",
            "Qtn [-]": "Normalised cone resistance.",
            "Fr [%]": "Normalised friction ratio.",
            "Ic [-]": "Soil behaviour type index.",
            "Ic class number [-]": "Robertson chart soil type number.",
            "Ic class": "Robertson chart soil type description.",
        },
    },
    "cpt_gmax_sand": {
        "category": "CPT Correlations",
        "brief": "Estimate small-strain shear modulus Gmax for sand from CPT (Rix & Stokoe 1991).",
        "description": (
            "Calculates the small-strain shear modulus for uncemented silica sand based on "
            "cone resistance and vertical effective stress. Gmax = 1634 * qc^0.25 * sigma'vo^0.375. "
            "Based on calibration chamber tests compared to PCPT, S-PCPT, and cross-hole tests "
            "(Baldi et al 1989). Note: qc is in MPa but is internally converted to kPa."
        ),
        "reference": "Rix & Stokoe (1991). Correlation of initial tangent modulus and cone penetration resistance.",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Cone tip resistance in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress in kPa."},
        },
        "returns": {
            "Gmax [kPa]": "Small-strain shear modulus.",
        },
    },
    "cpt_gmax_clay": {
        "category": "CPT Correlations",
        "brief": "Estimate small-strain shear modulus Gmax for clay from CPT (Mayne & Rix 1993).",
        "description": (
            "Determines Gmax from cone tip resistance for clay soils. Gmax = 2.78 * qc^1.335. "
            "Based on 481 data sets from 31 sites worldwide. Gmax ranged from about 0.7 to 800 MPa. "
            "Note: qc is in MPa but is internally converted to kPa."
        ),
        "reference": "Mayne & Rix (1993). Gmax-qc relationships for clays.",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Cone tip resistance in MPa."},
        },
        "returns": {
            "Gmax [kPa]": "Small-strain shear modulus.",
        },
    },
    "cpt_relative_density_nc_sand": {
        "category": "CPT Correlations",
        "brief": "Estimate relative density for NC sand from CPT (Baldi et al 1986).",
        "description": (
            "Calculates relative density for normally consolidated sand based on calibration "
            "chamber tests on silica sand. Dr = (1/C2) * ln(qc / (C0 * sigma'vo^C1)). "
            "The correlation is an approximation and the sand at site should be compared to "
            "the calibration chamber sands. Sensitive to compressibility and horizontal stress."
        ),
        "reference": "Baldi et al (1986). Calibration chamber tests on silica sand.",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Cone tip resistance in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress in kPa."},
        },
        "returns": {
            "Dr [-]": "Relative density as a fraction (0 to 1).",
        },
    },
    "cpt_relative_density_oc_sand": {
        "category": "CPT Correlations",
        "brief": "Estimate relative density for OC sand from CPT (Baldi et al 1986).",
        "description": (
            "Calculates relative density for overconsolidated sand based on calibration "
            "chamber tests on silica sand. Uses mean effective stress sigma'm = (sigma'vo + 2*K0*sigma'vo)/3. "
            "Requires an estimate of coefficient of lateral earth pressure K0. "
            "Sensitive to compressibility and horizontal stress."
        ),
        "reference": "Baldi et al (1986). Calibration chamber tests on silica sand.",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Cone tip resistance in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress in kPa."},
            "k0": {"type": "float", "required": True, "range": "0.3 to 5",
                   "description": "Coefficient of lateral earth pressure K0."},
        },
        "returns": {
            "Dr [-]": "Relative density as a fraction (0 to 1).",
        },
    },
    "cpt_relative_density_jamiolkowski": {
        "category": "CPT Correlations",
        "brief": "Estimate relative density for sand from CPT (Jamiolkowski et al 2003).",
        "description": (
            "Calculates relative density for dry and saturated sand based on calibration chamber tests. "
            "Uses mean effective stress and atmospheric pressure for normalisation. "
            "Calibrated for vertical effective stress 50-400 kPa and K0 0.4-1.5. "
            "Results outside this range should be assessed with care. "
            "Saturated correction can increase Dr by up to 10%."
        ),
        "reference": "Jamiolkowski, Lo Presti & Manassero (2003). Evaluation of relative density from CPT and DMT.",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Cone tip resistance in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "50 to 400",
                             "description": "Vertical effective stress in kPa."},
            "k0": {"type": "float", "required": True, "range": "0.4 to 1.5",
                   "description": "Coefficient of lateral earth pressure K0."},
        },
        "returns": {
            "Dr dry [-]": "Relative density for dry sand (0 to 1).",
            "Dr sat [-]": "Relative density for saturated sand (0 to 1).",
        },
    },
    "cpt_friction_angle_sand": {
        "category": "CPT Correlations",
        "brief": "Estimate friction angle for sand from CPT (Kulhawy & Mayne 1990).",
        "description": (
            "Determines the friction angle for sand based on calibration chamber tests. "
            "phi' = 17.6 + 11.0 * log10(qt/Pa / sqrt(sigma'vo/Pa)). "
            "Uses total cone resistance qt (corrected for area ratio)."
        ),
        "reference": "Kulhawy & Mayne (1990). Manual on estimating soil properties for foundation design.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Total cone resistance qt in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress in kPa."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
        },
        "returns": {
            "Phi [deg]": "Effective friction angle in degrees.",
        },
    },
    "cpt_undrained_shear_strength": {
        "category": "CPT Correlations",
        "brief": "Estimate undrained shear strength of clay from net CPT resistance (Rad & Lunne 1988).",
        "description": (
            "Calculates Su = qnet / Nk (in kPa). The cone factor Nk (typically 8-30) should "
            "be calibrated against high-quality lab tests (e.g. CIU triaxial). "
            "qnet is the net cone resistance (qt - sigma_vo) in MPa."
        ),
        "reference": "Rad & Lunne (1988). Direct correlations between piezocone test results and undrained shear strength of clay.",
        "parameters": {
            "qnet": {"type": "float", "required": True, "range": "0 to 120",
                     "description": "Net cone resistance (qt - sigma_vo) in MPa."},
            "Nk": {"type": "float", "required": True, "range": "8 to 30",
                   "description": "Empirical cone factor. Must be calibrated to site-specific lab tests."},
        },
        "returns": {
            "Su [kPa]": "Undrained shear strength.",
        },
    },
    "cpt_ocr": {
        "category": "CPT Correlations",
        "brief": "Estimate OCR for clay from normalized CPT (Lunne et al 1997).",
        "description": (
            "Estimates overconsolidation ratio for clay from normalised cone resistance Qt "
            "and/or pore pressure ratio Bq. Returns low, best, and high estimates for each. "
            "Based on high-quality undisturbed samples tested by NGI. "
            "If only Qt is provided, Bq-based estimates return NaN and vice versa."
        ),
        "reference": "Lunne, Robertson & Powell (1997). Cone penetration testing in geotechnical practice.",
        "parameters": {
            "Qt": {"type": "float", "required": True, "range": "2 to 34",
                   "description": "Normalised cone resistance Qt = (qt - sigma_vo) / sigma'vo."},
            "Bq": {"type": "float", "required": False, "default": "NaN",
                   "range": "0 to 1.4",
                   "description": "Pore pressure ratio Bq. Optional; if omitted, Bq-based OCR is NaN."},
        },
        "returns": {
            "OCR_Qt_LE [-]": "Low estimate OCR based on Qt.",
            "OCR_Qt_BE [-]": "Best estimate OCR based on Qt.",
            "OCR_Qt_HE [-]": "High estimate OCR based on Qt.",
            "OCR_Bq_LE [-]": "Low estimate OCR based on Bq.",
            "OCR_Bq_BE [-]": "Best estimate OCR based on Bq.",
            "OCR_Bq_HE [-]": "High estimate OCR based on Bq.",
        },
    },
    "cpt_sensitivity": {
        "category": "CPT Correlations",
        "brief": "Estimate clay sensitivity from CPT friction ratio (Rad & Lunne 1986).",
        "description": (
            "Calculates the sensitivity of clay from the friction ratio Rf = ft/qt (%). "
            "Based on measurements on Norwegian clays. Returns low, best, and high estimates. "
            "Ideally uses corrected total sleeve friction for calculating Rf."
        ),
        "reference": "Rad & Lunne (1986); Lunne, Robertson & Powell (1997).",
        "parameters": {
            "Rf": {"type": "float", "required": True, "range": "0.5 to 2.2",
                   "description": "Friction ratio in percent (Rf = fs/qt * 100)."},
        },
        "returns": {
            "St LE [-]": "Low estimate of sensitivity.",
            "St BE [-]": "Best estimate of sensitivity.",
            "St HE [-]": "High estimate of sensitivity.",
        },
    },
    "cpt_unit_weight": {
        "category": "CPT Correlations",
        "brief": "Estimate unit weight from CPT sleeve friction (Mayne et al 2010).",
        "description": (
            "Estimates total unit weight for sand, clay, and silt from CPT measurements. "
            "gamma = 1.95 * gamma_w * (sigma'vo/Pa)^0.06 * (ft/Pa)^0.06. "
            "Uses corrected total sleeve friction ft. Error band of +/-2 kN/m3. "
            "Does not apply to cemented soils."
        ),
        "reference": "Mayne, Peuchen & Bouwmeester (2010). Soil unit weight estimation from CPTs.",
        "parameters": {
            "ft": {"type": "float", "required": True, "range": "0 to 10",
                   "description": "Total sleeve friction in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 500",
                             "description": "Vertical effective stress in kPa."},
            "unitweight_water": {"type": "float", "required": False, "default": 10.25,
                                 "description": "Unit weight of water in kN/m3."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
        },
        "returns": {
            "gamma [kN/m3]": "Total unit weight.",
        },
    },
    "cpt_shear_wave_velocity": {
        "category": "CPT Correlations",
        "brief": "Estimate shear wave velocity and Gmax from CPT (Robertson & Cabal 2015).",
        "description": (
            "Calculates shear wave velocity Vs from total cone resistance qt and soil behaviour "
            "type index Ic. alpha_vs = 10^(0.55*Ic + 1.68), Vs = (alpha_vs * (qt - sigma_vo)/Pa)^0.5. "
            "Also computes Gmax = rho * Vs^2. Based on uncemented Holocene to Pleistocene age soils. "
            "Vs is sensitive to age and cementation."
        ),
        "reference": "Robertson & Cabal (2015). Guide to Cone Penetration Testing, 6th edition.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 100",
                   "description": "Total cone resistance qt in MPa."},
            "ic": {"type": "float", "required": True, "range": "1 to 4",
                   "description": "Soil behaviour type index Ic (Robertson & Wride)."},
            "sigma_vo": {"type": "float", "required": True, "range": "0 to 800",
                         "description": "Total vertical stress in kPa."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "description": "Atmospheric pressure in kPa."},
            "gamma": {"type": "float", "required": False, "default": 19.0,
                      "range": "12 to 22",
                      "description": "Bulk unit weight in kN/m3 (for Gmax calculation)."},
            "g": {"type": "float", "required": False, "default": 9.81,
                  "description": "Gravitational acceleration in m/s2."},
        },
        "returns": {
            "alpha_vs [-]": "Coefficient capturing soil behaviour influence on Vs.",
            "Vs [m/s]": "Shear wave velocity.",
            "Gmax [kPa]": "Small-strain shear modulus.",
        },
    },
    "cpt_k0_sand": {
        "category": "CPT Correlations",
        "brief": "Estimate K0 for sand from CPT (Mayne 2007).",
        "description": (
            "Calculates lateral earth pressure coefficient at rest from calibration chamber tests "
            "on clean sands. K0_CPT = 0.192 * (qt/Pa)^0.22 * (Pa/sigma'vo)^0.31 * OCR^0.27. "
            "Also computes conventional K0 = (1-sin(phi')) * OCR^sin(phi') and passive Kp for comparison."
        ),
        "reference": "Mayne (2007). NCHRP Synthesis 368: Cone Penetration Testing.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 100",
                   "description": "Total cone resistance qt in MPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress in kPa."},
            "ocr": {"type": "float", "required": True, "range": "1 to 20",
                    "description": "Overconsolidation ratio."},
            "atmospheric_pressure": {"type": "float", "required": False, "default": 100.0,
                                     "range": "90 to 110",
                                     "description": "Atmospheric pressure in kPa."},
            "friction_angle": {"type": "float", "required": False, "default": 32.0,
                               "range": "25 to 45",
                               "description": "Effective friction angle of the sand in degrees."},
        },
        "returns": {
            "K0 CPT [-]": "K0 derived from CPT correlation.",
            "K0 conventional [-]": "K0 from conventional (1-sin(phi'))*OCR^sin(phi') equation.",
            "Kp [-]": "Passive earth pressure coefficient (limiting K0 value).",
        },
    },
    "cpt_constrained_modulus": {
        "category": "CPT Correlations",
        "brief": "Estimate constrained modulus from CPT (Robertson 2009).",
        "description": (
            "Calculates the one-dimensional constrained modulus M = alphaM * (qt - sigma_vo). "
            "When Ic > 2.2 (fine-grained): alphaM = Qt (capped at 14). "
            "When Ic <= 2.2 (coarse-grained): alphaM = 0.0188 * 10^(0.55*Ic + 1.68). "
            "Also returns coefficient of volumetric compressibility mv = 1/M."
        ),
        "reference": "Robertson & Cabal (2022). CPT guide, 7th edition.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 100",
                   "description": "Corrected cone tip resistance qt in MPa."},
            "ic": {"type": "float", "required": True, "range": "1 to 5",
                   "description": "Soil behaviour type index Ic."},
            "sigma_vo": {"type": "float", "required": True, "range": "0 to 2000",
                         "description": "Total vertical stress in kPa."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 1000",
                             "description": "Vertical effective stress in kPa."},
            "coefficient1": {"type": "float", "required": False, "default": 0.0188,
                             "description": "First calibration coefficient."},
            "coefficient2": {"type": "float", "required": False, "default": 0.55,
                             "description": "Second calibration coefficient."},
            "coefficient3": {"type": "float", "required": False, "default": 1.68,
                             "description": "Third calibration coefficient."},
            "qt_pivot": {"type": "float", "required": False, "default": 14,
                         "description": "Qt threshold above which alphaM is capped (for Ic > 2.2)."},
        },
        "returns": {
            "alphaM [-]": "Multiplier on net cone resistance.",
            "M [kPa]": "Constrained modulus for one-dimensional compression.",
            "mv [1/kPa]": "Coefficient of volumetric compressibility.",
        },
    },
    # Bearing Capacity
    "bearing_capacity_nq": {
        "category": "Bearing Capacity",
        "brief": "Calculate bearing capacity factor Nq from friction angle.",
        "description": (
            "Calculates the bearing capacity factor Nq = exp(pi * tan(phi')) * tan^2(45 + phi'/2). "
            "Nq is used in the drained bearing capacity equation for the overburden pressure term."
        ),
        "reference": "Budhu (2011). Introduction to soil mechanics and foundations.",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "range": "20 to 50",
                               "description": "Peak effective friction angle in degrees."},
        },
        "returns": {
            "Nq [-]": "Bearing capacity factor Nq.",
        },
    },
    "bearing_capacity_ngamma_vesic": {
        "category": "Bearing Capacity",
        "brief": "Calculate bearing capacity factor Ngamma (Vesic 1973).",
        "description": (
            "Calculates Ngamma = 2 * (Nq + 1) * tan(phi'). "
            "This is the most commonly used expression. Note that alternative "
            "formulations (Meyerhof, Davis & Booker) are also available."
        ),
        "reference": "Budhu (2011). Introduction to soil mechanics and foundations; Vesic (1973).",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "range": "20 to 50",
                               "description": "Peak drained friction angle in degrees."},
        },
        "returns": {
            "Ngamma [-]": "Bearing capacity factor Ngamma.",
        },
    },
    "bearing_capacity_ngamma_meyerhof": {
        "category": "Bearing Capacity",
        "brief": "Calculate bearing capacity factor Ngamma (Meyerhof 1976, more conservative).",
        "description": (
            "Calculates Ngamma = (Nq - 1) * tan(1.4 * phi'). "
            "This formulation is more conservative compared to Vesic."
        ),
        "reference": "Budhu (2011). Introduction to soil mechanics and foundations; Meyerhof (1976).",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "range": "20 to 50",
                               "description": "Peak drained friction angle in degrees."},
            "frictionangle_multiplier": {"type": "float", "required": False, "default": 1.4,
                                         "description": "Multiplier on friction angle (default 1.4)."},
        },
        "returns": {
            "Ngamma [-]": "Bearing capacity factor Ngamma.",
        },
    },
    "bearing_capacity_ngamma_davisbooker": {
        "category": "Bearing Capacity",
        "brief": "Calculate bearing capacity factor Ngamma accounting for footing roughness (Davis & Booker 1971).",
        "description": (
            "Calculates Ngamma based on a refined plasticity method that accounts for footing roughness. "
            "Interpolates between smooth (Ngamma = 0.0663 * exp(9.3*phi')) and rough "
            "(Ngamma = 0.1054 * exp(9.6*phi')) footing values. This method is preferred in principle."
        ),
        "reference": "Budhu (2011). Introduction to soil mechanics and foundations; Davis & Booker (1971).",
        "parameters": {
            "friction_angle": {"type": "float", "required": True, "range": "20 to 50",
                               "description": "Peak drained friction angle in degrees."},
            "roughness_factor": {"type": "float", "required": True, "range": "0 to 1",
                                 "description": "Footing roughness factor (0=smooth, 1=rough)."},
        },
        "returns": {
            "Ngamma [-]": "Bearing capacity factor Ngamma for given roughness.",
            "Ngamma_smooth [-]": "Ngamma for fully smooth footing.",
            "Ngamma_rough [-]": "Ngamma for fully rough footing.",
        },
    },
    "bearing_capacity_undrained_api": {
        "category": "Bearing Capacity",
        "brief": "Calculate undrained vertical bearing capacity for shallow foundation (API RP 2GEO).",
        "description": (
            "Calculates undrained vertical bearing capacity for a shallow foundation on clay "
            "with constant or linearly increasing undrained shear strength. "
            "qu = su * Nc * Kc (constant su) or qu = F * (suo * Nc + kappa*B'/4) * Kc (linearly increasing). "
            "Kc = 1 + sc + dc - ic - bc - gc accounts for shape, depth, load inclination, "
            "foundation inclination, and ground surface inclination. "
            "For skirted foundations, base overburden is excluded. "
            "For non-skirted embedded foundations, base_sigma_v should be specified."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "effective_length": {"type": "float", "required": True, "range": ">= 0",
                                 "description": "Effective length of the foundation L' in m."},
            "effective_width": {"type": "float", "required": True, "range": ">= 0",
                                "description": "Minimum effective lateral dimension B' in m."},
            "su_base": {"type": "float", "required": True, "range": ">= 0",
                        "description": "Undrained shear strength at foundation base level Suo in kPa."},
            "su_increase": {"type": "float", "required": False, "default": 0.0,
                            "range": ">= 0",
                            "description": "Linear increase in undrained shear strength kappa in kPa/m."},
            "su_above_base": {"type": "float", "required": False, "default": "NaN",
                              "range": ">= 0",
                              "description": "Average undrained shear strength above base level in kPa. Required for linearly increasing su."},
            "base_depth": {"type": "float", "required": False, "default": 0.0,
                           "range": ">= 0",
                           "description": "Depth to the foundation base D in m."},
            "skirted": {"type": "bool", "required": False, "default": True,
                        "description": "True for skirted foundation, False for base-embedded without skirts."},
            "base_sigma_v": {"type": "float", "required": False, "default": 0.0,
                             "range": ">= 0",
                             "description": "Vertical total stress at base level in kPa. Only used for non-skirted."},
            "roughness": {"type": "float", "required": False, "default": 0.67,
                          "range": "0 to 1",
                          "description": "Foundation roughness (0=smooth, 1=rough). Used for F factor."},
            "horizontal_load": {"type": "float", "required": False, "default": 0.0,
                                "range": ">= 0",
                                "description": "Horizontal load on effective area of foundation H' in kN."},
            "foundation_inclination": {"type": "float", "required": False, "default": 0.0,
                                       "range": "-90 to 90",
                                       "description": "Foundation inclination nu in degrees."},
            "ground_surface_inclination": {"type": "float", "required": False, "default": 0.0,
                                           "range": "-90 to 90",
                                           "description": "Ground surface inclination beta in degrees."},
            "bearing_capacity_factor": {"type": "float", "required": False, "default": 5.14,
                                        "range": "3 to 12",
                                        "description": "Bearing capacity factor Nc (default 5.14)."},
            "factor_f_override": {"type": "float", "required": False, "default": "NaN",
                                  "range": "0 to 2",
                                  "description": "Direct specification of factor F. Overrides calculation from roughness."},
        },
        "returns": {
            "qu [kPa]": "Net bearing pressure.",
            "vertical_capacity [kN]": "Vertical bearing capacity.",
            "Su2 [kPa]": "Equivalent undrained shear strength (for linearly increasing su).",
            "K_c [-]": "Combined correction factor.",
            "s_c [-]": "Shape factor.",
            "d_c [-]": "Depth factor.",
            "i_c [-]": "Load inclination factor.",
            "b_c [-]": "Foundation inclination factor.",
            "g_c [-]": "Ground surface inclination factor.",
            "F [-]": "Correction factor for shear strength increase with depth.",
        },
    },
    "bearing_capacity_drained_api": {
        "category": "Bearing Capacity",
        "brief": "Calculate drained vertical bearing capacity for shallow foundation (API RP 2GEO).",
        "description": (
            "Calculates drained vertical bearing capacity for a shallow foundation in sand. "
            "qu = p'o * (Nq-1) * Kq + 0.5 * gamma' * B' * Ngamma * Kgamma (skirted). "
            "For non-skirted embedded foundations, Nq replaces (Nq-1). "
            "Kq and Kgamma are combined correction factors for shape, depth, load inclination, "
            "foundation inclination, and ground surface inclination."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "vertical_effective_stress": {"type": "float", "required": True, "range": ">= 0",
                                          "description": "Vertical effective stress at foundation base p'o in kPa."},
            "effective_friction_angle": {"type": "float", "required": True, "range": "20 to 50",
                                         "description": "Effective friction angle phi' in degrees."},
            "effective_unit_weight": {"type": "float", "required": True, "range": "3 to 12",
                                      "description": "Effective unit weight gamma' at foundation base in kN/m3."},
            "effective_length": {"type": "float", "required": True, "range": ">= 0",
                                 "description": "Effective length L' in m."},
            "effective_width": {"type": "float", "required": True, "range": ">= 0",
                                "description": "Minimum effective lateral dimension B' in m."},
            "base_depth": {"type": "float", "required": False, "default": 0.0,
                           "range": ">= 0",
                           "description": "Depth of foundation base D in m."},
            "skirted": {"type": "bool", "required": False, "default": True,
                        "description": "True for skirted, False for base-embedded without skirts."},
            "load_inclination": {"type": "float", "required": False, "default": 0.0,
                                 "range": ">= 0",
                                 "description": "Load inclination angle in degrees (arctan(H/V))."},
            "foundation_inclination": {"type": "float", "required": False, "default": 0.0,
                                       "range": "-90 to 90",
                                       "description": "Foundation inclination nu in degrees."},
            "ground_surface_inclination": {"type": "float", "required": False, "default": 0.0,
                                           "range": "-90 to 90",
                                           "description": "Ground surface inclination beta in degrees."},
        },
        "returns": {
            "qu [kPa]": "Net bearing pressure.",
            "vertical_capacity [kN]": "Vertical bearing capacity.",
            "N_q [-]": "Bearing capacity factor Nq.",
            "N_gamma [-]": "Bearing capacity factor Ngamma.",
            "K_q [-]": "Combined correction factor for Nq term.",
            "K_gamma [-]": "Combined correction factor for Ngamma term.",
            "s_q [-]": "Shape factor for Nq.",
            "s_gamma [-]": "Shape factor for Ngamma.",
            "d_q [-]": "Depth factor for Nq.",
            "d_gamma [-]": "Depth factor for Ngamma.",
            "i_q [-]": "Inclination factor for Nq.",
            "i_gamma [-]": "Inclination factor for Ngamma.",
            "b_q [-]": "Foundation inclination factor for Nq.",
            "b_gamma [-]": "Foundation inclination factor for Ngamma.",
            "g_q [-]": "Ground inclination factor for Nq.",
            "g_gamma [-]": "Ground inclination factor for Ngamma.",
        },
    },
    "sliding_capacity_undrained_api": {
        "category": "Bearing Capacity",
        "brief": "Calculate undrained sliding capacity for shallow foundation (API RP 2GEO).",
        "description": (
            "Calculates undrained sliding capacity for a shallow foundation on clay. "
            "Hd = Suo * A (base resistance). Delta_H = Kru * Su_ave * Ah (skirt resistance). "
            "Total sliding capacity = Hd + Delta_H. "
            "Kru = 4 recommended for full contact; reduce to 2 if active soil resistance cannot be relied upon."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "su_base": {"type": "float", "required": True, "range": ">= 0",
                        "description": "Undrained shear strength at foundation base level Suo in kPa."},
            "foundation_area": {"type": "float", "required": True, "range": ">= 0",
                                "description": "Actual foundation area (not effective area) A in m2."},
            "su_above_base": {"type": "float", "required": False, "default": 0.0,
                              "range": ">= 0",
                              "description": "Average undrained shear strength along skirt depth Su_ave in kPa."},
            "embedded_section_area": {"type": "float", "required": False, "default": 0.0,
                                      "range": ">= 0",
                                      "description": "Embedded vertical cross-sectional area Ah in m2."},
            "soil_reaction_coefficient": {"type": "float", "required": False, "default": 4.0,
                                          "range": "1 to 6",
                                          "description": "Soil reaction coefficient Kru (4 for full contact)."},
        },
        "returns": {
            "sliding_capacity [kN]": "Total sliding capacity (base + skirt).",
            "base_resistance [kN]": "Sliding resistance on the foundation base.",
            "skirt_resistance [kN]": "Sliding resistance from skirt passive/active resistance.",
        },
    },
    "sliding_capacity_drained_api": {
        "category": "Bearing Capacity",
        "brief": "Calculate drained sliding capacity for shallow foundation (API RP 2GEO).",
        "description": (
            "Calculates drained sliding capacity. Base resistance Hd = V * tan(phi'). "
            "Skirt resistance Delta_H = 0.5 * Krd * gamma' * Db * Ah. "
            "Krd = Kp - 1/Kp where Kp = tan^2(45 + phi'/2). "
            "Total sliding capacity = Hd + Delta_H."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "vertical_load": {"type": "float", "required": True, "range": ">= 0",
                              "description": "Actual vertical load Q during relevant loading condition in kN."},
            "effective_friction_angle": {"type": "float", "required": True, "range": "20 to 50",
                                         "description": "Effective friction angle phi' in degrees."},
            "effective_unit_weight": {"type": "float", "required": True, "range": "3 to 12",
                                      "description": "Effective unit weight gamma' in kN/m3."},
            "embedded_section_area": {"type": "float", "required": False, "default": 0.0,
                                      "range": ">= 0",
                                      "description": "Embedded vertical cross-sectional area Ah in m2."},
            "depth_to_base": {"type": "float", "required": False, "default": 0.0,
                              "range": ">= 0",
                              "description": "Depth below seafloor to base level Db in m."},
            "reaction_factor_override": {"type": "float", "required": False, "default": "NaN",
                                         "range": ">= 0",
                                         "description": "Override for drained horizontal reaction factor Krd."},
        },
        "returns": {
            "sliding_capacity [kN]": "Total sliding capacity (base + skirt).",
            "base_capacity [kN]": "Sliding resistance from base friction.",
            "skirt_capacity [kN]": "Sliding resistance from skirt passive/active resistance.",
            "K_rd [-]": "Drained horizontal reaction factor.",
            "K_p [-]": "Passive earth pressure coefficient.",
        },
    },
    "effective_area_rectangle": {
        "category": "Bearing Capacity",
        "brief": "Calculate effective area of rectangular foundation for eccentric loading (API RP 2GEO).",
        "description": (
            "Calculates the reduced area of a rectangular footing to account for load eccentricity. "
            "L' = L - 2*e1, B' = B - 2*e2, A' = L' * B'. "
            "Eccentricities can be specified either from moments (e = M/V) or directly. "
            "For undrained foundations, V can include weight of soil plug inside skirts."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "length": {"type": "float", "required": True, "range": ">= 0",
                       "description": "Longest foundation dimension L in m."},
            "width": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Shortest foundation dimension B in m."},
            "vertical_load": {"type": "float", "required": False, "default": "NaN",
                              "range": ">= 0.001",
                              "description": "Actual vertical load Q in kN. Required if specifying moments."},
            "moment_length": {"type": "float", "required": False, "default": "NaN",
                              "range": ">= 0",
                              "description": "Overturning moment along length M1 in kNm."},
            "moment_width": {"type": "float", "required": False, "default": "NaN",
                             "range": ">= 0",
                             "description": "Overturning moment along width M2 in kNm."},
            "eccentricity_length": {"type": "float", "required": False, "default": "NaN",
                                    "range": ">= 0",
                                    "description": "Direct eccentricity in length direction e1 in m."},
            "eccentricity_width": {"type": "float", "required": False, "default": "NaN",
                                   "range": ">= 0",
                                   "description": "Direct eccentricity in width direction e2 in m."},
        },
        "returns": {
            "effective_area [m2]": "Reduced effective area A'.",
            "effective_length [m]": "Effective length L'.",
            "effective_width [m]": "Effective width B'.",
            "eccentricity_length [m]": "Eccentricity in the length direction.",
            "eccentricity_width [m]": "Eccentricity in the width direction.",
        },
    },
    "effective_area_circle": {
        "category": "Bearing Capacity",
        "brief": "Calculate effective area of circular foundation for eccentric loading (API RP 2GEO).",
        "description": (
            "Calculates the reduced area for a circular foundation to account for load eccentricity. "
            "Uses the exact geometric solution with parameter s and equivalent effective length L' and "
            "width B'. Eccentricity can be specified through an overturning moment (e = M/V) or directly."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "foundation_radius": {"type": "float", "required": True, "range": ">= 0.01",
                                  "description": "Radius of the circular foundation R in m."},
            "vertical_load": {"type": "float", "required": False, "default": "NaN",
                              "range": ">= 0.01",
                              "description": "Actual vertical load Q in kN. Required if specifying moment."},
            "overturning_moment": {"type": "float", "required": False, "default": "NaN",
                                   "range": ">= 0",
                                   "description": "Overturning moment M in kNm."},
            "eccentricity": {"type": "float", "required": False, "default": "NaN",
                             "range": ">= 0",
                             "description": "Direct eccentricity e in m."},
        },
        "returns": {
            "effective_area [m2]": "Reduced effective area A'.",
            "effective_length [m]": "Equivalent effective length L'.",
            "effective_width [m]": "Equivalent effective width B'.",
            "s [m2]": "Geometric parameter s.",
            "eccentricity [m]": "Eccentricity used for the calculation.",
        },
    },
    # ===================== Consolidation & Settlement =====================
    "consolidation_degree": {
        "category": "Consolidation & Settlement",
        "brief": "Calculate degree of consolidation for given time and soil properties.",
        "description": (
            "Returns the average degree of consolidation for a given time and initial "
            "distribution of excess pore pressure. Solutions are interpolated from published "
            "Terzaghi 1-D consolidation solutions."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.",
        "parameters": {
            "time": {"type": "float", "required": True, "range": ">= 0",
                     "description": "Time at which to compute degree of consolidation [s]."},
            "cv": {"type": "float", "required": True, "range": "0.1 to 1000",
                   "description": "Coefficient of consolidation [m2/yr]."},
            "drainage_length": {"type": "float", "required": True, "range": ">= 0",
                                "description": "Drainage length Hdr [m]."},
            "distribution": {"type": "string", "required": False, "default": "uniform",
                             "description": "Shape of initial excess pore pressure distribution. Options: 'uniform', 'triangular'."},
        },
        "returns": {
            "U [pct]": "Average degree of consolidation [%].",
            "Tv [-]": "Time factor Tv.",
        },
    },
    "primary_consolidation_settlement_nc": {
        "category": "Consolidation & Settlement",
        "brief": "Primary consolidation settlement for normally consolidated clay.",
        "description": (
            "Calculates primary consolidation settlement for normally consolidated fine-grained soil "
            "using the compression index method. Settlement = H0/(1+e0) * Cc * log10((sigma'v0 + delta_sigma'v) / sigma'v0)."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "initial_height": {"type": "float", "required": True, "range": ">= 0",
                               "description": "Initial layer thickness H0 [m]."},
            "initial_voidratio": {"type": "float", "required": True, "range": "0.1 to 5.0",
                                  "description": "Initial void ratio e0."},
            "initial_effective_stress": {"type": "float", "required": True, "range": ">= 0",
                                         "description": "Initial vertical effective stress sigma'v0 [kPa]."},
            "effective_stress_increase": {"type": "float", "required": True, "range": ">= 0",
                                          "description": "Increase in vertical effective stress [kPa]."},
            "compression_index": {"type": "float", "required": True, "range": "0.1 to 0.8",
                                  "description": "Compression index Cc (base-10 log)."},
            "e_min": {"type": "float", "required": False, "default": "0.3",
                      "description": "Minimum void ratio below which no further consolidation occurs."},
        },
        "returns": {
            "delta z [m]": "Primary consolidation settlement [m].",
            "delta e [-]": "Decrease in void ratio.",
            "e final [-]": "Final void ratio after consolidation.",
        },
    },
    "primary_consolidation_settlement_oc": {
        "category": "Consolidation & Settlement",
        "brief": "Primary consolidation settlement for overconsolidated clay.",
        "description": (
            "Calculates primary consolidation settlement for overconsolidated clay using compression "
            "and recompression indices. Handles both cases: stresses remaining below or exceeding "
            "the preconsolidation pressure."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "initial_height": {"type": "float", "required": True, "range": ">= 0",
                               "description": "Initial layer thickness H0 [m]."},
            "initial_voidratio": {"type": "float", "required": True, "range": "0.1 to 5.0",
                                  "description": "Initial void ratio e0."},
            "initial_effective_stress": {"type": "float", "required": True, "range": ">= 0",
                                         "description": "Initial vertical effective stress [kPa]."},
            "preconsolidation_pressure": {"type": "float", "required": True, "range": ">= 0",
                                          "description": "Preconsolidation pressure pc' [kPa]."},
            "effective_stress_increase": {"type": "float", "required": True, "range": ">= 0",
                                          "description": "Increase in vertical effective stress [kPa]."},
            "compression_index": {"type": "float", "required": True, "range": "0.1 to 0.8",
                                  "description": "Compression index Cc."},
            "recompression_index": {"type": "float", "required": True, "range": "0.015 to 0.35",
                                    "description": "Recompression index Cr."},
            "e_min": {"type": "float", "required": False, "default": "0.3",
                      "description": "Minimum void ratio below which no further consolidation occurs."},
        },
        "returns": {
            "delta z [m]": "Primary consolidation settlement [m].",
            "delta e [-]": "Decrease in void ratio.",
            "e final [-]": "Final void ratio after consolidation.",
        },
    },
    "consolidation_settlement_mv": {
        "category": "Consolidation & Settlement",
        "brief": "Consolidation settlement using compressibility mv.",
        "description": (
            "Calculates consolidation settlement using the coefficient of volume compressibility mv "
            "(inverse of constrained modulus M). Note mv is stress-dependent."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "initial_height": {"type": "float", "required": True, "range": ">= 0",
                               "description": "Initial layer thickness H0 [m]."},
            "effective_stress_increase": {"type": "float", "required": True, "range": ">= 0",
                                          "description": "Increase in vertical effective stress [kPa]."},
            "compressibility": {"type": "float", "required": True, "range": "1e-4 to 10",
                                "description": "Coefficient of volume compressibility mv [1/kPa]."},
        },
        "returns": {
            "delta z [m]": "Consolidation settlement [m].",
            "delta epsilon [-]": "Change in strain.",
        },
    },
    "hydraulic_conductivity_unconfined": {
        "category": "Consolidation & Settlement",
        "brief": "Hydraulic conductivity from unconfined aquifer pumping test.",
        "description": (
            "Calculates hydraulic conductivity from two standpipes near a pumping well in an "
            "unconfined, non-leaking aquifer using Dupuit's formula."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.",
        "parameters": {
            "radius_1": {"type": "float", "required": True, "range": ">= 0",
                         "description": "Radial distance to first standpipe [m]."},
            "radius_2": {"type": "float", "required": True, "range": ">= 0",
                         "description": "Radial distance to second standpipe [m]."},
            "piezometric_height_1": {"type": "float", "required": True, "range": ">= 0",
                                     "description": "Piezometric height in first standpipe [m]."},
            "piezometric_height_2": {"type": "float", "required": True, "range": ">= 0",
                                     "description": "Piezometric height in second standpipe [m]."},
            "flowrate": {"type": "float", "required": True, "range": ">= 0",
                         "description": "Flowrate extracted from pumping well [m3/s]."},
        },
        "returns": {
            "hydraulic_conductivity [m/s]": "Hydraulic conductivity k [m/s].",
        },
    },
    # ===================== Stress Distribution =====================
    "stress_pointload": {
        "category": "Stress Distribution",
        "brief": "Boussinesq stress distribution under a point load.",
        "description": (
            "Calculates vertical, radial, tangential stresses and shear stress at a point "
            "below a surface point load using the Boussinesq (1885) solution."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "pointload": {"type": "float", "required": True, "range": "any",
                          "description": "Magnitude of the point load Q [kN]."},
            "z": {"type": "float", "required": True, "range": "any",
                  "description": "Vertical distance from surface [m]."},
            "r": {"type": "float", "required": True, "range": "any",
                  "description": "Radial distance from load [m]."},
            "poissonsratio": {"type": "float", "required": True, "range": "0 to 0.5",
                              "description": "Poisson's ratio of the soil."},
        },
        "returns": {
            "delta sigma z [kPa]": "Vertical stress increase.",
            "delta sigma r [kPa]": "Radial stress increase.",
            "delta sigma theta [kPa]": "Tangential stress increase.",
            "delta tau rz [kPa]": "Shear stress increase in rz plane.",
        },
    },
    "stress_stripload": {
        "category": "Stress Distribution",
        "brief": "Stress distribution under a strip load (uniform or triangular).",
        "description": (
            "Calculates stress redistribution at a point in the subsoil due to a strip load "
            "of given width applied at the surface. Supports uniform and triangular loading."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "z": {"type": "float", "required": True, "range": ">= 0",
                  "description": "Vertical distance from surface [m]."},
            "x": {"type": "float", "required": True, "range": "any",
                  "description": "Horizontal offset from leftmost corner of strip [m]."},
            "width": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Width of the strip footing B [m]."},
            "imposedstress": {"type": "float", "required": True, "range": "any",
                              "description": "Maximum imposed stress qs [kPa]."},
            "triangular": {"type": "bool", "required": False, "default": "False",
                           "description": "If True, use triangular load pattern instead of uniform."},
        },
        "returns": {
            "delta sigma z [kPa]": "Vertical stress increase.",
            "delta sigma x [kPa]": "Horizontal stress increase.",
            "delta tau zx [kPa]": "Shear stress increase.",
        },
    },
    "stress_circle": {
        "category": "Stress Distribution",
        "brief": "Stress distribution below center of a circular foundation.",
        "description": (
            "Calculates vertical and radial stress increases below the center of a uniformly "
            "loaded circular foundation."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "z": {"type": "float", "required": True, "range": ">= 0",
                  "description": "Depth below foundation base [m]."},
            "footing_radius": {"type": "float", "required": True, "range": ">= 0",
                               "description": "Radius of circular foundation r0 [m]."},
            "imposedstress": {"type": "float", "required": True, "range": "any",
                              "description": "Applied uniform stress qs [kPa]."},
            "poissonsratio": {"type": "float", "required": True, "range": "0 to 0.5",
                              "description": "Poisson's ratio of the soil."},
        },
        "returns": {
            "delta sigma z [kPa]": "Vertical stress increase.",
            "delta sigma r [kPa]": "Radial stress increase.",
        },
    },
    "stress_rectangle": {
        "category": "Stress Distribution",
        "brief": "Stress distribution below corner of a rectangular loaded area.",
        "description": (
            "Calculates stresses under the corner of a uniformly loaded rectangular area. "
            "Use superposition for stresses under other points."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundation engineering.",
        "parameters": {
            "imposedstress": {"type": "float", "required": True, "range": "any",
                              "description": "Applied uniform stress qs [kPa]."},
            "length": {"type": "float", "required": True, "range": ">= 0",
                       "description": "Longest edge of rectangle L [m]."},
            "width": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Shortest edge of rectangle B [m]."},
            "z": {"type": "float", "required": True, "range": ">= 0",
                  "description": "Depth below the footing [m]."},
        },
        "returns": {
            "delta sigma z [kPa]": "Vertical stress increase.",
            "delta sigma x [kPa]": "Horizontal stress increase (width direction).",
            "delta sigma y [kPa]": "Horizontal stress increase (length direction).",
            "delta tau zx [kPa]": "Shear stress increase in zx plane.",
        },
    },
    # ===================== Earth Pressure =====================
    "earth_pressure_basic": {
        "category": "Earth Pressure",
        "brief": "Active and passive earth pressure coefficients from friction angle (Mohr circle).",
        "description": (
            "Calculates Ka and Kp from effective friction angle using Mohr's circle construction. "
            "No wall friction or inclination. Also returns slip plane angles."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.",
        "parameters": {
            "phi_eff": {"type": "float", "required": True, "range": "20 to 50",
                        "description": "Effective friction angle [deg]."},
        },
        "returns": {
            "Ka [-]": "Active earth pressure coefficient.",
            "Kp [-]": "Passive earth pressure coefficient.",
            "theta_a [radians]": "Active slip plane angle with horizontal.",
            "theta_p [radians]": "Passive slip plane angle with horizontal.",
        },
    },
    "earth_pressure_poncelet": {
        "category": "Earth Pressure",
        "brief": "Earth pressure coefficients with wall friction and inclination (Poncelet/Coulomb).",
        "description": (
            "Calculates active and passive earth pressure coefficients for a retaining wall "
            "with interface friction, wall inclination, and sloping ground using Poncelet's "
            "formulation of Coulomb's theory."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.",
        "parameters": {
            "phi_eff": {"type": "float", "required": True, "range": "20 to 50",
                        "description": "Effective friction angle of the soil [deg]."},
            "interface_friction_angle": {"type": "float", "required": True, "range": "15 to 40",
                                         "description": "Wall-soil interface friction angle delta [deg]."},
            "wall_angle": {"type": "float", "required": True, "range": "0 to 70",
                           "description": "Wall angle to vertical eta [deg]."},
            "top_angle": {"type": "float", "required": True, "range": "0 to 70",
                          "description": "Ground slope angle beta [deg]."},
        },
        "returns": {
            "KaC [-]": "Poncelet's active earth pressure coefficient.",
            "KpC [-]": "Poncelet's passive earth pressure coefficient.",
        },
    },
    "earth_pressure_rankine": {
        "category": "Earth Pressure",
        "brief": "Rankine earth pressure coefficients for inclined wall with sloping ground.",
        "description": (
            "Calculates Rankine active and passive earth pressure coefficients for an inclined wall "
            "with sloping ground. Also returns slip plane angles and force inclinations. "
            "Wall friction is not considered."
        ),
        "reference": "Budhu (2011). Soil mechanics and foundations. John Wiley and Sons.",
        "parameters": {
            "phi_eff": {"type": "float", "required": True, "range": "20 to 50",
                        "description": "Effective friction angle [deg]."},
            "wall_angle": {"type": "float", "required": True, "range": "0 to 70",
                           "description": "Wall angle to vertical eta [deg]."},
            "top_angle": {"type": "float", "required": True, "range": "0 to 70",
                          "description": "Ground slope angle beta [deg]."},
        },
        "returns": {
            "KaR [-]": "Rankine active earth pressure coefficient.",
            "KpR [-]": "Rankine passive earth pressure coefficient.",
            "omega_a [-]": "Helper variable for active pressure.",
            "omega_p [-]": "Helper variable for passive pressure.",
            "theta_a [radians]": "Active slip plane angle.",
            "theta_p [radians]": "Passive slip plane angle.",
            "ksi_a [radians]": "Active resultant inclination to wall normal.",
            "ksi_p [radians]": "Passive resultant inclination to wall normal.",
        },
    },
    # ===================== Soil Classification =====================
    "relative_density_category": {
        "category": "Soil Classification",
        "brief": "Classify relative density (Very loose to Very dense).",
        "description": (
            "Categorizes relative density of cohesionless soil: Very loose (0-0.15), "
            "Loose (0.15-0.35), Medium dense (0.35-0.65), Dense (0.65-0.85), Very dense (0.85-1.0)."
        ),
        "reference": "API RP2 GEO.",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "range": "0 to 1",
                                 "description": "Relative density Dr as a fraction (0 to 1)."},
        },
        "returns": {
            "Relative density": "Descriptive category string.",
        },
    },
    "su_category": {
        "category": "Soil Classification",
        "brief": "Classify undrained shear strength (BS 5930 or ASTM D-2488).",
        "description": (
            "Classifies undrained shear strength into categories. Default is BS 5930:2015 "
            "(Extremely low to Extremely high). ASTM D-2488 also available (Very soft to Very hard)."
        ),
        "reference": "BS 5930:2015, ASTM D-2488.",
        "parameters": {
            "undrained_shear_strength": {"type": "float", "required": True, "range": "0 to 1000",
                                          "description": "Undrained shear strength Su [kPa]."},
            "standard": {"type": "string", "required": False, "default": "BS 5930:2015",
                         "description": "Classification standard. Options: 'BS 5930:2015', 'ASTM D-2488'."},
        },
        "returns": {
            "strength class": "Strength classification string.",
        },
    },
    "uscs_description": {
        "category": "Soil Classification",
        "brief": "Get USCS soil type description from symbol.",
        "description": (
            "Returns the verbose soil type description for a given USCS two-character symbol "
            "(e.g. 'CL' -> 'Inorganic clays of low to medium plasticity...')."
        ),
        "reference": "Unified Soil Classification System (USCS).",
        "parameters": {
            "symbol": {"type": "string", "required": True,
                       "description": "USCS symbol. Options: GW, GP, GM, GC, SW, SP, SM, SC, ML, CL, OL, MH, CH, OH."},
        },
        "returns": {
            "Soil type": "Verbose description of the soil type.",
        },
    },
    "sample_quality_lunne": {
        "category": "Soil Classification",
        "brief": "Assess sample quality using void ratio change (Lunne et al.).",
        "description": (
            "Determines sample quality for clays based on the change in void ratio when "
            "reconsolidating to in-situ stress. Categories: Very good to excellent, Good to fair, "
            "Poor, Very poor. Classification depends on OCR range."
        ),
        "reference": "Lunne et al. (2008). Effects of sample disturbance on consolidation behaviour.",
        "parameters": {
            "voidratio": {"type": "float", "required": True, "range": "0.3 to 3",
                          "description": "Initial void ratio e0."},
            "voidratio_change": {"type": "float", "required": True, "range": "-1 to 1",
                                 "description": "Change in void ratio delta_e when reconsolidating."},
            "ocr": {"type": "float", "required": True, "range": "1 to 4",
                    "description": "Overconsolidation ratio OCR."},
        },
        "returns": {
            "delta e/e0 [-]": "Ratio used for classification.",
            "Quality category": "Quality category string.",
        },
    },
    # ===================== Deep Foundations =====================
    "pile_shaft_friction_api_sand": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in sand (API RP 2GEO beta method).",
        "description": (
            "Calculates unit skin friction for piles in sand using the beta method from API RP 2GEO. "
            "Beta values are tabulated based on relative density and soil description."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "api_relativedensity": {"type": "string", "required": True,
                                     "description": "Options: 'Medium dense', 'Dense', 'Very dense'."},
            "api_soildescription": {"type": "string", "required": True,
                                     "description": "Options: 'Sand', 'Sand-silt'."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress [kPa]."},
            "tension_modifier": {"type": "float", "required": False, "default": "1.0",
                                 "range": "0 to 1",
                                 "description": "Tension reduction factor."},
        },
        "returns": {
            "f_s_comp_out [kPa]": "Compression shaft friction (outside).",
            "f_s_comp_in [kPa]": "Compression shaft friction (inside).",
            "f_s_tens_out [kPa]": "Tension shaft friction (outside).",
            "f_s_tens_in [kPa]": "Tension shaft friction (inside).",
            "f_s_lim [kPa]": "Limiting shaft friction.",
            "beta [-]": "Beta coefficient.",
        },
    },
    "pile_shaft_friction_api_clay": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in clay (API RP 2GEO alpha method).",
        "description": (
            "Calculates unit skin friction for piles in clay using the alpha method from API RP 2GEO. "
            "Alpha depends on the ratio of undrained shear strength to vertical effective stress."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "undrained_shear_strength": {"type": "float", "required": True, "range": "0 to 400",
                                          "description": "Undrained shear strength Su [kPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress [kPa]."},
        },
        "returns": {
            "f_s_comp_out [kPa]": "Compression shaft friction (outside).",
            "f_s_comp_in [kPa]": "Compression shaft friction (inside).",
            "f_s_tens_out [kPa]": "Tension shaft friction (outside).",
            "f_s_tens_in [kPa]": "Tension shaft friction (inside).",
            "psi [-]": "Su / sigma'vo ratio.",
            "alpha [-]": "Alpha adhesion factor.",
        },
    },
    "pile_shaft_friction_almhamre_sand": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in sand with friction fatigue (Alm & Hamre).",
        "description": (
            "Calculates unit skin friction in sand including friction fatigue effects, calibrated "
            "against North Sea jacket piles. 50% inside / 50% outside split by default."
        ),
        "reference": "Alm & Hamre (2001). Soil model for pile driveability predictions based on CPT.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Total cone resistance [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress [kPa]."},
            "interface_friction_angle": {"type": "float", "required": True, "range": "10 to 50",
                                         "description": "Interface friction angle delta [deg]."},
            "depth": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Depth at which to calculate [m]."},
            "embedded_length": {"type": "float", "required": True, "range": ">= 0",
                                "description": "Pile tip depth below mudline [m]."},
        },
        "returns": {
            "f_s_comp_out [kPa]": "Outside compression shaft friction.",
            "f_s_comp_in [kPa]": "Inside compression shaft friction.",
            "f_s_tens_out [kPa]": "Outside tension shaft friction.",
            "f_s_tens_in [kPa]": "Inside tension shaft friction.",
            "f_s_initial [kPa]": "Initial unit skin friction.",
            "f_s_res [kPa]": "Residual unit skin friction.",
        },
    },
    "pile_shaft_friction_almhamre_clay": {
        "category": "Deep Foundations",
        "brief": "Unit shaft friction in clay with friction fatigue (Alm & Hamre).",
        "description": (
            "Calculates unit skin friction in clay including friction fatigue, calibrated "
            "against North Sea jacket piles. 100% inside / 100% outside by default."
        ),
        "reference": "Alm & Hamre (2001). Soil model for pile driveability predictions based on CPT.",
        "parameters": {
            "depth": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Depth at which to calculate [m]."},
            "embedded_length": {"type": "float", "required": True, "range": ">= 0",
                                "description": "Pile tip depth below mudline [m]."},
            "qt": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Total cone resistance [MPa]."},
            "fs": {"type": "float", "required": True, "range": ">= 0",
                   "description": "CPT sleeve friction [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress [kPa]."},
        },
        "returns": {
            "f_s_comp_out [kPa]": "Outside compression shaft friction.",
            "f_s_comp_in [kPa]": "Inside compression shaft friction.",
            "f_s_tens_out [kPa]": "Outside tension shaft friction.",
            "f_s_tens_in [kPa]": "Inside tension shaft friction.",
            "f_s_initial [kPa]": "Initial unit skin friction.",
            "f_s_res [kPa]": "Residual unit skin friction.",
        },
    },
    "pile_end_bearing_api_clay": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in clay (API RP 2GEO).",
        "description": (
            "Calculates unit end bearing in clay as q = Nc * Su. Default Nc = 9."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "undrained_shear_strength": {"type": "float", "required": True, "range": "0 to 400",
                                          "description": "Undrained shear strength at pile tip [kPa]."},
            "N_c": {"type": "float", "required": False, "default": "9.0", "range": "7 to 12",
                    "description": "Bearing capacity factor Nc."},
        },
        "returns": {
            "q_b_coring [kPa]": "Unit end bearing (coring).",
            "q_b_plugged [kPa]": "Unit end bearing (plugged).",
            "plugged": "Plugged status.",
            "internal_friction": "Whether internal friction is considered.",
        },
    },
    "pile_end_bearing_api_sand": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in sand (API RP 2GEO).",
        "description": (
            "Calculates unit end bearing in sand as q = Nq * sigma'vo (API RP 2GEO). "
            "Nq values are tabulated by relative density and soil description."
        ),
        "reference": "API RP 2GEO (2011). Geotechnical and Foundation Design Considerations.",
        "parameters": {
            "api_relativedensity": {"type": "string", "required": True,
                                     "description": "Options: 'Medium dense', 'Dense', 'Very dense'."},
            "api_soildescription": {"type": "string", "required": True,
                                     "description": "Options: 'Sand', 'Sand-silt'."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress at pile tip [kPa]."},
        },
        "returns": {
            "q_b_coring [kPa]": "Unit end bearing (coring).",
            "q_b_plugged [kPa]": "Unit end bearing (plugged).",
            "plugged": "Plugged status.",
            "internal_friction": "Whether internal friction is considered.",
            "q_b_lim [kPa]": "Limiting end bearing.",
            "Nq [-]": "Bearing capacity factor Nq.",
        },
    },
    "pile_end_bearing_almhamre_sand": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in sand (Alm & Hamre CPT-based).",
        "description": (
            "Calculates unit end bearing in sand from CPT data: qb = 0.15 * qt * (qt/sigma'vo)^0.2."
        ),
        "reference": "Alm & Hamre (2001). Soil model for pile driveability predictions based on CPT.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Total cone resistance [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress [kPa]."},
        },
        "returns": {
            "q_b_coring [kPa]": "Unit end bearing (coring).",
            "q_b_plugged [kPa]": "Unit end bearing (plugged).",
            "plugged []": "Plugged status (False for driven piles).",
            "internal_friction []": "Internal friction considered.",
        },
    },
    "pile_end_bearing_almhamre_clay": {
        "category": "Deep Foundations",
        "brief": "Unit end bearing in clay (Alm & Hamre CPT-based).",
        "description": (
            "Calculates unit end bearing in clay from CPT data: qb = 0.6 * qt."
        ),
        "reference": "Alm & Hamre (2001). Soil model for pile driveability predictions based on CPT.",
        "parameters": {
            "qt": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Total cone resistance [MPa]."},
        },
        "returns": {
            "q_b_coring [kPa]": "Unit end bearing (coring).",
            "q_b_plugged [kPa]": "Unit end bearing (plugged, 0 for clay).",
            "plugged []": "Plugged status.",
            "internal_friction []": "Internal friction considered.",
        },
    },
    # ===================== Soil Dynamics & Liquefaction =====================
    "modulus_reduction_ishibashi": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "G/Gmax modulus reduction curve and damping ratio (Ishibashi & Zhang).",
        "description": (
            "Calculates modulus reduction ratio G/Gmax as a function of shear strain, plasticity "
            "index, and mean effective stress. Also calculates damping ratio. Use PI=0 for "
            "cohesionless soils."
        ),
        "reference": "Ishibashi & Zhang (1993). Unified dynamic shear moduli and damping ratios.",
        "parameters": {
            "strain": {"type": "float", "required": True, "range": "0 to 10",
                       "description": "Shear strain amplitude [%]."},
            "pi": {"type": "float", "required": True, "range": "0 to 200",
                   "description": "Plasticity index PI [%]. Use 0 for sand."},
            "sigma_m_eff": {"type": "float", "required": True, "range": "0 to 400",
                            "description": "Mean effective stress [kPa]."},
        },
        "returns": {
            "G/Gmax [-]": "Modulus reduction ratio.",
            "K [-]": "K factor in the equation.",
            "m [-]": "Exponent m.",
            "n [-]": "Factor n for plasticity.",
            "dampingratio [pct]": "Damping ratio [%].",
        },
    },
    "gmax_from_shear_wave_velocity": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Calculate Gmax from shear wave velocity and unit weight.",
        "description": (
            "Calculates the small-strain shear modulus Gmax from shear wave velocity Vs "
            "and bulk unit weight using Gmax = rho * Vs^2."
        ),
        "reference": "Robertson & Cabal (2015). Guide to Cone Penetration Testing.",
        "parameters": {
            "Vs": {"type": "float", "required": True, "range": "0 to 600",
                   "description": "Shear wave velocity [m/s]."},
            "gamma": {"type": "float", "required": True, "range": "12 to 22",
                      "description": "Bulk unit weight [kN/m3]."},
        },
        "returns": {
            "rho [kg/m3]": "Density of the material.",
            "Gmax [kPa]": "Small-strain shear modulus.",
        },
    },
    "damping_ratio_seed": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Damping ratio for sand and gravel (Seed et al. 1986).",
        "description": (
            "Returns low, best, and high estimates of damping ratio for sand and gravel "
            "as a function of cyclic shear strain, based on compiled test data."
        ),
        "reference": "Seed et al. (1986). Moduli and damping factors for dynamic analyses.",
        "parameters": {
            "cyclic_shear_strain": {"type": "float", "required": True, "range": "0.0001 to 1.0",
                                    "description": "Cyclic shear strain [%]."},
        },
        "returns": {
            "D LE [pct]": "Low estimate damping ratio [%].",
            "D BE [pct]": "Best estimate damping ratio [%].",
            "D HE [pct]": "High estimate damping ratio [%].",
        },
    },
    "cyclic_stress_ratio_moss": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Cyclic stress ratio CSR for liquefaction (Moss/Cetin formulation).",
        "description": (
            "Calculates equivalent uniform cyclic stress ratio CSR using Seed & Idriss simplified "
            "method with depth reduction factor and duration weighting factor from Cetin et al (2004). "
            "CSR* is adjusted to magnitude 7.5 event."
        ),
        "reference": "Moss et al. (2006). CPT-Based Assessment of Seismic Soil Liquefaction Potential.",
        "parameters": {
            "sigma_vo": {"type": "float", "required": True, "range": ">= 0",
                         "description": "Total vertical stress [kPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Effective vertical stress [kPa]."},
            "magnitude": {"type": "float", "required": True, "range": "5.5 to 8.5",
                          "description": "Earthquake magnitude Mw."},
            "acceleration": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Maximum horizontal ground acceleration [m/s2]."},
            "depth": {"type": "float", "required": True, "range": ">= 0",
                      "description": "Depth at which CSR is calculated [m]."},
        },
        "returns": {
            "CSR [-]": "Uncorrected cyclic stress ratio.",
            "CSR* [-]": "CSR adjusted to Mw=7.5.",
            "DWF [-]": "Duration weighting factor.",
            "rd [-]": "Depth reduction factor.",
        },
    },
    "cyclic_stress_ratio_youd": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Cyclic stress ratio CSR (Youd et al. 2001 formulation).",
        "description": (
            "Calculates CSR adjusted to magnitude 7.5 using Seed & Idriss simplified equation "
            "with Youd et al. (2001) recommendations for rd and MSF. Valid for depths up to 23m."
        ),
        "reference": "Youd et al. (2001). Liquefaction resistance of soils.",
        "parameters": {
            "acceleration": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Maximum horizontal ground acceleration [m/s2]."},
            "sigma_vo": {"type": "float", "required": True, "range": ">= 0",
                         "description": "Total vertical stress [kPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Effective vertical stress [kPa]."},
            "depth": {"type": "float", "required": True, "range": "0 to 23",
                      "description": "Depth [m]. Max 23m for rd formula."},
            "magnitude": {"type": "float", "required": True, "range": "0 to 8.5",
                          "description": "Earthquake magnitude M."},
        },
        "returns": {
            "CSR [-]": "Uncorrected cyclic stress ratio.",
            "CSR* [-]": "CSR adjusted to Mw=7.5.",
            "MSF [-]": "Magnitude scaling factor.",
            "rd [-]": "Depth reduction factor.",
        },
    },
    "liquefaction_robertson_fear": {
        "category": "Soil Dynamics & Liquefaction",
        "brief": "Liquefaction triggering assessment from CPT (Robertson & Fear 1995).",
        "description": (
            "Determines if cyclic liquefaction can be triggered based on normalised cone tip "
            "resistance and CSR. For clean sands only (no fines correction). "
            "Use CSR* for magnitudes other than 7.5."
        ),
        "reference": "Robertson & Fear (1995). Application of CPT to evaluate liquefaction potential.",
        "parameters": {
            "qc": {"type": "float", "required": True, "range": "0 to 120",
                   "description": "Cone tip resistance [MPa]."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": ">= 0",
                             "description": "Vertical effective stress [kPa]."},
            "CSR": {"type": "float", "required": True, "range": "0.073 to 0.49",
                    "description": "Cyclic stress ratio (use CSR* for non-7.5 events)."},
        },
        "returns": {
            "qc1 [-]": "Normalised cone resistance.",
            "qc1 liquefaction [-]": "qc1 at liquefaction threshold for given CSR.",
            "qc liquefaction [MPa]": "Cone resistance at liquefaction threshold.",
            "liquefaction": "True if liquefaction is predicted.",
        },
    },
    # ===================== Soil Correlations =====================
    "gmax_sand_hardin_black": {
        "category": "Soil Correlations",
        "brief": "Small-strain shear modulus for sand (Hardin & Black).",
        "description": (
            "Calculates Gmax from void ratio and mean effective stress using the Hardin & Black (1968) "
            "correlation. Default calibration from PISA project for dense marine sand."
        ),
        "reference": "Hardin & Black (1968). Vibration modulus of normally consolidated clay.",
        "parameters": {
            "sigma_m0": {"type": "float", "required": True, "range": "0 to 500",
                         "description": "Mean effective stress p' [kPa]."},
            "void_ratio": {"type": "float", "required": True, "range": "0 to 4",
                           "description": "In-situ void ratio e0."},
        },
        "returns": {
            "Gmax [kPa]": "Small-strain shear modulus.",
        },
    },
    "permeability_hazen": {
        "category": "Soil Correlations",
        "brief": "Permeability from grain size using Hazen correlation.",
        "description": (
            "Estimates permeability of granular soil from the D10 grain size using k = C * D10^2."
        ),
        "reference": "Terzaghi, Peck & Mesri (1996). Soil mechanics in engineering practice.",
        "parameters": {
            "grain_size": {"type": "float", "required": True, "range": "0.01 to 2.0",
                           "description": "D10 grain size (10th percentile) [mm]."},
            "coefficient_C": {"type": "float", "required": False, "default": "0.01",
                              "description": "Calibration coefficient C."},
        },
        "returns": {
            "k [m/s]": "Permeability [m/s].",
        },
    },
    "hssmall_parameters_sand": {
        "category": "Soil Correlations",
        "brief": "HS Small constitutive model parameters for sand (PLAXIS).",
        "description": (
            "Calculates all HS Small model parameters for PLAXIS as a function of relative "
            "density. Calibrated against Toyoura, Ham River, Hostun and Ticino sand."
        ),
        "reference": "Brinkgreve et al. (2010). Validation of empirical formulas for sands.",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "range": "10 to 100",
                                 "description": "Relative density Dr [%]."},
        },
        "returns": {
            "gamma_unsat [kN/m3]": "Unsaturated unit weight.",
            "gamma_sat [kN/m3]": "Saturated unit weight.",
            "E50_ref [kPa]": "Reference secant stiffness.",
            "Eoed_ref [kPa]": "Reference oedometric stiffness.",
            "Eur_ref [kPa]": "Reference unloading-reloading stiffness.",
            "G0_ref [kPa]": "Reference small-strain shear modulus.",
            "m [-]": "Stiffness exponent.",
            "gamma_07 [-]": "Strain at 70% Gmax reduction.",
            "phi_eff [deg]": "Effective friction angle.",
            "psi [deg]": "Dilation angle.",
            "Rf [-]": "Failure ratio.",
        },
    },
    "stress_dilatancy_bolton": {
        "category": "Soil Correlations",
        "brief": "Stress-dilatancy relationships for sand (Bolton 1986).",
        "description": (
            "Calculates relative dilatancy index, difference between peak and critical-state "
            "friction angle, dilation angle and strain ratio for plane strain or triaxial conditions."
        ),
        "reference": "Bolton (1986). The strength and dilatancy of sands.",
        "parameters": {
            "relative_density": {"type": "float", "required": True, "range": "0.1 to 1.0",
                                 "description": "Relative density Dr (fraction)."},
            "p_eff": {"type": "float", "required": True, "range": "20 to 10000",
                      "description": "Effective pressure p' [kPa]."},
            "Q": {"type": "float", "required": False, "default": "10", "range": "5 to 10",
                  "description": "Calibration factor Q (10 for quartz/feldspar)."},
            "R": {"type": "float", "required": False, "default": "1",
                  "description": "Calibration factor R."},
            "stress_condition": {"type": "string", "required": False, "default": "triaxial strain",
                                 "description": "Options: 'triaxial strain', 'plane strain'."},
        },
        "returns": {
            "Ir [-]": "Relative dilatancy index.",
            "phi_max - phi_cs [deg]": "Peak minus critical-state friction angle.",
            "Dilation angle [deg]": "Dilation angle.",
            "-depsilon_v/depsilon_1__max [-]": "Max volumetric to principal strain ratio.",
        },
    },
    "compression_index_koppula": {
        "category": "Soil Correlations",
        "brief": "Compression and recompression indices from water content (Koppula).",
        "description": (
            "Estimates Cc from natural water content (Cc = wn) and Cr using a user-defined "
            "Cc/Cr ratio (typically 5 to 10)."
        ),
        "reference": "Koppula (1981). Statistical evaluation of compression index.",
        "parameters": {
            "water_content": {"type": "float", "required": True, "range": "0 to 4",
                              "description": "Natural water content wn (fraction, e.g. 0.5 for 50%)."},
            "cc_cr_ratio": {"type": "float", "required": False, "default": "7.5", "range": "5 to 10",
                            "description": "Ratio Cc/Cr."},
        },
        "returns": {
            "Cc [-]": "Compression index.",
            "Cr [-]": "Recompression index.",
        },
    },
    "friction_angle_from_pi": {
        "category": "Soil Correlations",
        "brief": "Drained friction angle of clay from plasticity index.",
        "description": (
            "Estimates drained friction angle of clay based on plasticity index using a spline "
            "fit to compiled data on soft to stiff clays."
        ),
        "reference": "Terzaghi, Peck & Mesri (1996). Soil mechanics in engineering practice.",
        "parameters": {
            "plasticity_index": {"type": "float", "required": True, "range": "5 to 1000",
                                 "description": "Plasticity index PI [%]."},
        },
        "returns": {
            "Effective friction angle [deg]": "Drained friction angle of the clay.",
        },
    },
    "cv_from_liquid_limit": {
        "category": "Soil Correlations",
        "brief": "Coefficient of consolidation from liquid limit (US Navy).",
        "description": (
            "Estimates cv from liquid limit using US Navy (1982) correlations. Three trends: "
            "Remoulded (upper bound), NC (normally consolidated), OC (overconsolidated, lower bound)."
        ),
        "reference": "U.S. Navy (1982). Soil mechanics design manual 7.1.",
        "parameters": {
            "liquid_limit": {"type": "float", "required": True, "range": "20 to 160",
                             "description": "Liquid limit LL [%]."},
            "trend": {"type": "string", "required": False, "default": "NC",
                      "description": "Options: 'Remoulded', 'NC', 'OC'."},
        },
        "returns": {
            "cv [m2/yr]": "Coefficient of consolidation.",
        },
    },
    "gmax_clay_andersen": {
        "category": "Soil Correlations",
        "brief": "Small-strain shear modulus for clay from PI, OCR, and stress (Andersen).",
        "description": (
            "Calculates Gmax for cohesive soils based on plasticity index, OCR, and "
            "vertical effective stress. Calibrated on shear wave velocity tests."
        ),
        "reference": "Andersen (2015). Cyclic soil parameters for offshore foundation design.",
        "parameters": {
            "pi": {"type": "float", "required": True, "range": "0 to 160",
                   "description": "Plasticity index PI [%]."},
            "ocr": {"type": "float", "required": True, "range": "1 to 40",
                    "description": "Overconsolidation ratio OCR."},
            "sigma_vo_eff": {"type": "float", "required": True, "range": "0 to 1000",
                             "description": "Vertical effective stress [kPa]."},
        },
        "returns": {
            "sigma_0_ref [kPa]": "Reference stress.",
            "Gmax [kPa]": "Small-strain shear modulus.",
        },
    },
    "k0_from_plasticity": {
        "category": "Soil Correlations",
        "brief": "K0 from plasticity index and OCR (Kenney/Alpan).",
        "description": (
            "Calculates K0 for NC and OC clay using Kenney (1959) correlation with "
            "plasticity index, modified by Alpan (1967) for overconsolidation."
        ),
        "reference": "Alpan (1967). The empirical evaluation of K0 and K0R.",
        "parameters": {
            "pi": {"type": "float", "required": True, "range": "5 to 80",
                   "description": "Plasticity index PI [%]."},
            "ocr": {"type": "float", "required": False, "default": "1", "range": "1 to 30",
                    "description": "Overconsolidation ratio OCR."},
        },
        "returns": {
            "K0 NC [-]": "K0 for normally consolidated conditions.",
            "K0 [-]": "K0 for given OCR.",
        },
    },
    "k0_from_friction_angle": {
        "category": "Soil Correlations",
        "brief": "K0 from critical-state friction angle and OCR (Mesri & Hayat).",
        "description": (
            "Calculates K0 using Jaky's equation modified for overconsolidation: "
            "K0 = (1 - sin(phi'cs)) * OCR^(sin(phi'cs)). Works for sands and clays."
        ),
        "reference": "Mesri & Hayat (1993). The coefficient of earth pressure at rest.",
        "parameters": {
            "phi_cs": {"type": "float", "required": True, "range": "15 to 45",
                       "description": "Critical state friction angle [deg]."},
            "ocr": {"type": "float", "required": False, "default": "1", "range": "1 to 30",
                    "description": "Overconsolidation ratio OCR."},
        },
        "returns": {
            "K0 [-]": "Coefficient of lateral earth pressure at rest.",
        },
    },
}


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


def groundhog_agent(method: str, parameters: dict) -> dict:
    """
    Geotechnical engineering calculator using the groundhog library.
    Call this function with a method name and a dictionary of parameters.

    Returns a dictionary with the calculation results, or an error message.

    ========================== PHASE RELATIONS ==========================

    METHOD: "voidratio_from_porosity"
        Converts porosity to void ratio.
        Parameters:
            - porosity (float): Porosity, ratio of void volume to total volume. Range: 0 to 1.
        Returns: voidratio [-]

    METHOD: "porosity_from_voidratio"
        Converts void ratio to porosity.
        Parameters:
            - voidratio (float): Void ratio, ratio of void volume to solids volume. Range: 0 to 5.
        Returns: porosity [-]

    METHOD: "saturation_from_watercontent"
        Calculates saturation from water content, void ratio, and specific gravity.
        Parameters:
            - water_content (float): Ratio of weight of water to weight of solids. Range: 0 to 4.
            - voidratio (float): Void ratio. Range: 0 to 4.
            - specific_gravity (float, optional): Specific gravity of soil grains. Default: 2.65.
        Returns: saturation [-]

    METHOD: "bulk_unit_weight"
        Calculates bulk and effective unit weight from saturation, void ratio, and specific gravity.
        Parameters:
            - saturation (float): Degree of saturation. Range: 0 to 1.
            - voidratio (float): Void ratio. Range: 0 to 4.
            - specific_gravity (float, optional): Specific gravity. Default: 2.65.
            - unitweight_water (float, optional): Unit weight of water in kN/m3. Default: 10.0.
        Returns: bulk unit weight [kN/m3], effective unit weight [kN/m3]

    METHOD: "dry_unit_weight"
        Calculates dry unit weight from water content and bulk unit weight.
        Parameters:
            - watercontent (float): Water content (decimal). Range: 0 to 4.
            - bulkunitweight (float): Bulk unit weight in kN/m3. Range: 10 to 25.
        Returns: dry unit weight [kN/m3]

    METHOD: "voidratio_from_dry_density"
        Calculates void ratio from dry density and specific gravity.
        Parameters:
            - dry_density (float): Dry density in kg/m3. Range: 1000 to 2000.
            - specific_gravity (float, optional): Specific gravity. Default: 2.65.
            - water_density (float, optional): Water density in kg/m3. Default: 1000.
        Returns: Void ratio [-]

    METHOD: "bulk_unit_weight_from_dry"
        Calculates bulk and effective unit weight from dry unit weight and water content.
        Parameters:
            - dryunitweight (float): Dry unit weight in kN/m3. Range: 1 to 15.
            - watercontent (float): Water content (decimal). Range: 0 to 4.
            - unitweight_water (float, optional): Unit weight of water in kN/m3. Default: 10.0.
        Returns: bulk unit weight [kN/m3], effective unit weight [kN/m3]

    METHOD: "relative_density"
        Calculates relative density from void ratio and min/max void ratios.
        Parameters:
            - void_ratio (float): Current void ratio. Range: 0 to 5.
            - e_min (float): Void ratio at minimum density. Range: 0 to 5.
            - e_max (float): Void ratio at maximum density. Range: 0 to 5.
        Returns: Dr [-]

    METHOD: "voidratio_from_bulk_unit_weight"
        Calculates void ratio and water content from bulk unit weight.
        Parameters:
            - bulkunitweight (float): Bulk unit weight in kN/m3. Range: 10 to 25.
            - saturation (float, optional): Degree of saturation. Default: 1.0 (saturated).
            - specific_gravity (float, optional): Specific gravity. Default: 2.65.
            - unitweight_water (float, optional): Unit weight of water in kN/m3. Default: 10.0.
        Returns: e [-], w [-]

    METHOD: "unit_weight_saturated"
        Calculates bulk unit weight for a saturated soil from water content.
        Parameters:
            - water_content (float): Water content (decimal). Range: 0 to 2.
            - specific_gravity (float, optional): Specific gravity. Default: 2.65.
            - gamma_w (float, optional): Unit weight of water in kN/m3. Default: 10.0.
        Returns: gamma [kN/m3]

    METHOD: "density_from_unit_weight"
        Converts unit weight (kN/m3) to density (kg/m3).
        Parameters:
            - gamma (float): Unit weight in kN/m3. Range: 0 to 30.
            - g (float, optional): Gravitational acceleration in m/s2. Default: 9.81.
        Returns: Density [kg/m3]

    METHOD: "unit_weight_from_density"
        Converts density (kg/m3) to unit weight (kN/m3).
        Parameters:
            - density (float): Density in kg/m3. Range: 0 to 3000.
            - g (float, optional): Gravitational acceleration in m/s2. Default: 9.81.
        Returns: Unit weight [kN/m3]

    METHOD: "watercontent_from_voidratio"
        Calculates water content from void ratio (assumes full saturation by default).
        Parameters:
            - voidratio (float): Void ratio. Range: >= 0.
            - saturation (float, optional): Degree of saturation. Default: 1.0.
            - specific_gravity (float, optional): Specific gravity. Default: 2.65.
        Returns: Water content [-], Water content [%]

    METHOD: "voidratio_from_watercontent"
        Calculates void ratio from water content (assumes full saturation by default).
        Parameters:
            - water_content (float): Water content (decimal). Range: 0 to 2.
            - saturation (float, optional): Degree of saturation. Default: 1.0.
            - specific_gravity (float, optional): Specific gravity. Default: 2.65.
        Returns: Void ratio [-]

    ========================= SPT CORRELATIONS ==========================

    METHOD: "spt_overburden_correction_liaowhitman"
        Corrects SPT N value for overburden pressure (Liao & Whitman 1986).
        Parameters:
            - N (float): Field SPT N value or N60. Range: >= 0.
            - sigma_vo_eff (float): Effective overburden pressure in kPa. Range: >= 0.
            - granular (bool, optional): Whether soil is granular. Default: True.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: CN [-], N1 [-]

    METHOD: "spt_N60_correction"
        Corrects raw SPT N to N60 (60% energy efficiency) accounting for hammer,
        borehole diameter, sampler type, and rod length.
        Parameters:
            - N (float): Field SPT N value. Range: >= 0.
            - borehole_diameter (float): Borehole diameter in mm. Range: 60 to 200.
            - rod_length (float): Rod length in m. Range: >= 0.
            - country (str): "Japan", "United States", "Argentina", "China", or "Other".
            - hammertype (str): "Donut" or "Safety".
            - hammerrelease (str): "Free fall" or "Rope and pulley".
            - samplertype (str, optional): "Standard sampler", "With liner for dense sand and clay",
              or "With liner for loose sand". Default: "Standard sampler".
            - eta_H (float, optional): Override for hammer efficiency correction.
            - eta_B (float, optional): Override for borehole diameter correction.
            - eta_S (float, optional): Override for sampler type correction.
            - eta_R (float, optional): Override for rod length correction.
        Returns: N60 [-], eta_H [%], eta_H [-], eta_B [-], eta_S [-], eta_R [-]

    METHOD: "spt_relative_density_kulhawymayne"
        Estimates relative density from corrected SPT (N1)60 (Kulhawy & Mayne 1990).
        Parameters:
            - N1_60 (float): Corrected SPT number. Range: 0 to 100.
            - d_50 (float): Median grain size in mm. Range: 0.002 to 20.
            - time_since_deposition (float, optional): Years since deposition. Default: 1.
            - ocr (float, optional): Overconsolidation ratio. Default: 1.
        Returns: Dr [-], Dr [pct], C_A [-], C_OCR [-]

    METHOD: "spt_undrained_shear_strength_salgado"
        Calculates undrained shear strength from plasticity index and N60 (Salgado 2008).
        Parameters:
            - pi (float): Plasticity index in %. Range: 15 to 60.
            - N_60 (float): SPT N60 value. Range: 0 to 100.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: alpha_prime [-], Su [kPa]

    METHOD: "spt_friction_angle_kulhawymayne"
        Estimates friction angle from SPT N value and effective overburden (Kulhawy & Mayne 1990).
        Parameters:
            - N (float): SPT N value. Range: 0 to 60.
            - sigma_vo_eff (float): Effective overburden pressure in kPa. Range: 0 to 1000.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: Phi [deg]

    METHOD: "spt_relative_density_class"
        Classifies relative density from uncorrected SPT N (Terzaghi & Peck 1967).
        For cohesionless soils.
        Parameters:
            - N (float): Uncorrected SPT N value. Range: 0 to 60.
        Returns: Dr class (str: "Very loose", "Loose", "Medium dense", "Dense", "Very dense")

    METHOD: "spt_overburden_correction_iso"
        Corrects SPT N for overburden pressure per ISO 22476-3.
        Parameters:
            - N (int): SPT N value or N60. Range: 0 to 60.
            - sigma_vo_eff (float): Effective overburden in kPa. Range: 25 to 400.
            - granular (bool, optional): Whether soil is granular. Default: True.
        Returns: CN [-], N1 [-]

    METHOD: "spt_friction_angle_pht"
        Estimates friction angle from (N1)60 (Peck, Hanson & Thornburn 1974).
        Parameters:
            - N1_60 (float): Corrected SPT value. Range: 0 to 60.
        Returns: Phi [deg]

    METHOD: "spt_youngs_modulus_aashto"
        Estimates Young's modulus from (N1)60 for different soil types (AASHTO 1997).
        Parameters:
            - N1_60 (float): Corrected SPT value. Range: 0 to 60.
            - soiltype (str): "Silts", "Clean sands", "Coarse sands", or "Gravels".
        Returns: Es [MPa]

    METHOD: "spt_consistency_class"
        Classifies consistency of cohesive soil from uncorrected SPT N (Terzaghi & Peck 1967).
        Parameters:
            - N (float): Uncorrected SPT N value. Range: 0 to 60.
        Returns: Consistency class (str), qu min [kPa], qu max [kPa]

    ========================= CPT CORRELATIONS ==========================

    METHOD: "cpt_normalisations"
        Normalize and correct raw CPT data and classify soil behavior type (Robertson).
        Parameters:
            - measured_qc (float): Measured cone resistance in MPa.
            - measured_fs (float): Measured sleeve friction in MPa.
            - measured_u2 (float): Pore pressure at shoulder in MPa.
            - sigma_vo_tot (float): Total vertical stress in kPa.
            - sigma_vo_eff (float): Effective vertical stress in kPa.
            - depth (float): Depth below surface in m.
            - cone_area_ratio (float): Cone area ratio (0 to 1).
            - start_depth (float, optional): Start depth for downhole tests in m. Default: 0.
            - unitweight_water (float, optional): Unit weight of water in kN/m3. Default: 10.25.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: qt, qc, u2, Delta u2, Rf, Bq, Qt, Fr, qnet, Qtn, Ic, Ic class

    METHOD: "cpt_soil_class_robertson"
        Classify soil from soil behaviour type index Ic (Robertson chart).
        Parameters:
            - ic (float): Soil behaviour type index Ic. Range: 1 to 5.
        Returns: Soil type number [-], Soil type

    METHOD: "cpt_behaviour_index"
        Calculate soil behaviour type index Ic from corrected CPT data (Robertson & Wride 1998).
        Parameters:
            - qt (float): Corrected cone resistance in MPa. Range: 0 to 120.
            - fs (float): Sleeve friction in MPa.
            - sigma_vo (float): Total vertical stress in kPa.
            - sigma_vo_eff (float): Effective vertical stress in kPa.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: exponent_zhang, Qtn, Fr, Ic, Ic class number, Ic class

    METHOD: "cpt_gmax_sand"
        Estimate small-strain shear modulus Gmax for sand from CPT (Rix & Stokoe 1991).
        Parameters:
            - qc (float): Cone tip resistance in MPa. Range: 0 to 120.
            - sigma_vo_eff (float): Vertical effective stress in kPa.
        Returns: Gmax [kPa]

    METHOD: "cpt_gmax_clay"
        Estimate small-strain shear modulus Gmax for clay from CPT (Mayne & Rix 1993).
        Parameters:
            - qc (float): Cone tip resistance in MPa. Range: 0 to 120.
        Returns: Gmax [kPa]

    METHOD: "cpt_relative_density_nc_sand"
        Estimate relative density for NC sand from CPT (Baldi et al 1986).
        Parameters:
            - qc (float): Cone tip resistance in MPa. Range: 0 to 120.
            - sigma_vo_eff (float): Vertical effective stress in kPa.
        Returns: Dr [-]

    METHOD: "cpt_relative_density_oc_sand"
        Estimate relative density for OC sand from CPT (Baldi et al 1986).
        Parameters:
            - qc (float): Cone tip resistance in MPa. Range: 0 to 120.
            - sigma_vo_eff (float): Vertical effective stress in kPa.
            - k0 (float): Coefficient of lateral earth pressure. Range: 0.3 to 5.
        Returns: Dr [-]

    METHOD: "cpt_relative_density_jamiolkowski"
        Estimate relative density for sand from CPT (Jamiolkowski et al 2003).
        Parameters:
            - qc (float): Cone tip resistance in MPa. Range: 0 to 120.
            - sigma_vo_eff (float): Vertical effective stress in kPa. Range: 50 to 400.
            - k0 (float): Coefficient of lateral earth pressure. Range: 0.4 to 1.5.
        Returns: Dr dry [-], Dr sat [-]

    METHOD: "cpt_friction_angle_sand"
        Estimate friction angle for sand from CPT (Kulhawy & Mayne 1990).
        Parameters:
            - qt (float): Total cone resistance in MPa. Range: 0 to 120.
            - sigma_vo_eff (float): Vertical effective stress in kPa.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: Phi [deg]

    METHOD: "cpt_undrained_shear_strength"
        Estimate undrained shear strength of clay from net CPT resistance (Rad & Lunne 1988).
        Parameters:
            - qnet (float): Net cone resistance in MPa. Range: 0 to 120.
            - Nk (float): Empirical cone factor. Range: 8 to 30.
        Returns: Su [kPa]

    METHOD: "cpt_ocr"
        Estimate OCR for clay from normalized CPT (Lunne et al 1997).
        Parameters:
            - Qt (float): Normalised cone resistance. Range: 2 to 34.
            - Bq (float, optional): Pore pressure ratio. Range: 0 to 1.4.
        Returns: OCR low/best/high estimates from Qt and Bq

    METHOD: "cpt_sensitivity"
        Estimate clay sensitivity from CPT friction ratio (Rad & Lunne 1986).
        Parameters:
            - Rf (float): Friction ratio in percent. Range: 0.5 to 2.2.
        Returns: St LE [-], St BE [-], St HE [-]

    METHOD: "cpt_unit_weight"
        Estimate unit weight from CPT sleeve friction (Mayne et al 2010).
        Parameters:
            - ft (float): Total sleeve friction in MPa. Range: 0 to 10.
            - sigma_vo_eff (float): Vertical effective stress in kPa. Range: 0 to 500.
            - unitweight_water (float, optional): Unit weight of water in kN/m3. Default: 10.25.
            - atmospheric_pressure (float, optional): Atmospheric pressure in kPa. Default: 100.
        Returns: gamma [kN/m3]

    METHOD: "cpt_shear_wave_velocity"
        Estimate shear wave velocity and Gmax from CPT (Robertson & Cabal 2015).
        Parameters:
            - qt (float): Total cone resistance in MPa. Range: 0 to 100.
            - ic (float): Soil behaviour type index Ic. Range: 1 to 4.
            - sigma_vo (float): Total vertical stress in kPa. Range: 0 to 800.
            - atmospheric_pressure (float, optional): Default: 100.
            - gamma (float, optional): Bulk unit weight in kN/m3. Default: 19.
        Returns: alpha_vs [-], Vs [m/s], Gmax [kPa]

    METHOD: "cpt_k0_sand"
        Estimate K0 for sand from CPT (Mayne 2007).
        Parameters:
            - qt (float): Total cone resistance in MPa. Range: 0 to 100.
            - sigma_vo_eff (float): Vertical effective stress in kPa.
            - ocr (float): Overconsolidation ratio. Range: 1 to 20.
            - atmospheric_pressure (float, optional): Default: 100.
            - friction_angle (float, optional): Friction angle in degrees. Default: 32.
        Returns: K0 CPT [-], K0 conventional [-], Kp [-]

    METHOD: "cpt_constrained_modulus"
        Estimate constrained modulus from CPT (Robertson 2009).
        Parameters:
            - qt (float): Corrected cone tip resistance in MPa. Range: 0 to 100.
            - ic (float): Soil behaviour type index Ic. Range: 1 to 5.
            - sigma_vo (float): Total vertical stress in kPa. Range: 0 to 2000.
            - sigma_vo_eff (float): Vertical effective stress in kPa. Range: 0 to 1000.
        Returns: alphaM [-], M [kPa], mv [1/kPa]

    ======================== BEARING CAPACITY ===========================

    METHOD: "bearing_capacity_nq"
        Calculate bearing capacity factor Nq from friction angle.
        Parameters:
            - friction_angle (float): Peak effective friction angle in degrees. Range: 20 to 50.
        Returns: Nq [-]

    METHOD: "bearing_capacity_ngamma_vesic"
        Calculate bearing capacity factor Ngamma (Vesic 1973).
        Parameters:
            - friction_angle (float): Peak drained friction angle in degrees. Range: 20 to 50.
        Returns: Ngamma [-]

    METHOD: "bearing_capacity_ngamma_meyerhof"
        Calculate bearing capacity factor Ngamma (Meyerhof 1976, more conservative).
        Parameters:
            - friction_angle (float): Peak drained friction angle in degrees. Range: 20 to 50.
            - frictionangle_multiplier (float, optional): Multiplier on phi. Default: 1.4.
        Returns: Ngamma [-]

    METHOD: "bearing_capacity_ngamma_davisbooker"
        Calculate Ngamma accounting for footing roughness (Davis & Booker 1971).
        Parameters:
            - friction_angle (float): Peak drained friction angle in degrees. Range: 20 to 50.
            - roughness_factor (float): Roughness (0=smooth, 1=rough). Range: 0 to 1.
        Returns: Ngamma [-], Ngamma_smooth [-], Ngamma_rough [-]

    METHOD: "bearing_capacity_undrained_api"
        Undrained vertical bearing capacity for shallow foundation (API RP 2GEO).
        Parameters:
            - effective_length (float): Effective length L' in m.
            - effective_width (float): Effective width B' in m.
            - su_base (float): Undrained shear strength at base in kPa.
            - su_increase (float, optional): Linear Su increase in kPa/m. Default: 0.
            - su_above_base (float, optional): Average Su above base in kPa.
            - base_depth (float, optional): Depth to foundation base in m. Default: 0.
            - skirted (bool, optional): Whether foundation is skirted. Default: True.
            - base_sigma_v (float, optional): Total vertical stress at base in kPa. Default: 0.
            - roughness (float, optional): Foundation roughness (0-1). Default: 0.67.
            - horizontal_load (float, optional): Horizontal load on effective area in kN. Default: 0.
            - foundation_inclination (float, optional): Foundation inclination in degrees. Default: 0.
            - ground_surface_inclination (float, optional): Ground inclination in degrees. Default: 0.
            - bearing_capacity_factor (float, optional): Nc factor. Default: 5.14.
            - factor_f_override (float, optional): Direct F factor specification.
        Returns: qu [kPa], vertical_capacity [kN], K_c, s_c, d_c, i_c, b_c, g_c, F

    METHOD: "bearing_capacity_drained_api"
        Drained vertical bearing capacity for shallow foundation (API RP 2GEO).
        Parameters:
            - vertical_effective_stress (float): Effective stress at base p'o in kPa.
            - effective_friction_angle (float): Friction angle phi' in degrees. Range: 20 to 50.
            - effective_unit_weight (float): Effective unit weight gamma' in kN/m3. Range: 3 to 12.
            - effective_length (float): Effective length L' in m.
            - effective_width (float): Effective width B' in m.
            - base_depth (float, optional): Foundation depth D in m. Default: 0.
            - skirted (bool, optional): Whether foundation is skirted. Default: True.
            - load_inclination (float, optional): Load inclination in degrees. Default: 0.
            - foundation_inclination (float, optional): Foundation inclination in degrees. Default: 0.
            - ground_surface_inclination (float, optional): Ground inclination in degrees. Default: 0.
        Returns: qu [kPa], vertical_capacity [kN], N_q, N_gamma, K_q, K_gamma, all shape/depth/inclination factors

    METHOD: "sliding_capacity_undrained_api"
        Undrained sliding capacity for shallow foundation (API RP 2GEO).
        Parameters:
            - su_base (float): Undrained shear strength at base in kPa.
            - foundation_area (float): Actual foundation area in m2.
            - su_above_base (float, optional): Average Su along skirt depth in kPa. Default: 0.
            - embedded_section_area (float, optional): Embedded vertical section area in m2. Default: 0.
            - soil_reaction_coefficient (float, optional): Kru coefficient (1-6). Default: 4.
        Returns: sliding_capacity [kN], base_resistance [kN], skirt_resistance [kN]

    METHOD: "sliding_capacity_drained_api"
        Drained sliding capacity for shallow foundation (API RP 2GEO).
        Parameters:
            - vertical_load (float): Vertical load Q in kN.
            - effective_friction_angle (float): Friction angle phi' in degrees. Range: 20 to 50.
            - effective_unit_weight (float): Effective unit weight in kN/m3. Range: 3 to 12.
            - embedded_section_area (float, optional): Embedded vertical section area in m2. Default: 0.
            - depth_to_base (float, optional): Depth to base Db in m. Default: 0.
            - reaction_factor_override (float, optional): Override for Krd factor.
        Returns: sliding_capacity [kN], base_capacity [kN], skirt_capacity [kN], K_rd, K_p

    METHOD: "effective_area_rectangle"
        Effective area of rectangular foundation for eccentric loading (API RP 2GEO).
        Parameters:
            - length (float): Longest foundation dimension L in m.
            - width (float): Shortest foundation dimension B in m.
            - EITHER (vertical_load + moment_length + moment_width) OR (eccentricity_length + eccentricity_width)
        Returns: effective_area [m2], effective_length [m], effective_width [m], eccentricities

    METHOD: "effective_area_circle"
        Effective area of circular foundation for eccentric loading (API RP 2GEO).
        Parameters:
            - foundation_radius (float): Foundation radius R in m.
            - EITHER (vertical_load + overturning_moment) OR eccentricity
        Returns: effective_area [m2], effective_length [m], effective_width [m], s [m2], eccentricity [m]

    ===================== CONSOLIDATION & SETTLEMENT =====================

    METHOD: "consolidation_degree"
        Degree of consolidation for given time and soil properties.
        Parameters:
            - time (float): Time [s].
            - cv (float): Coefficient of consolidation [m2/yr]. Range: 0.1 to 1000.
            - drainage_length (float): Drainage length Hdr [m].
            - distribution (str, optional): 'uniform' or 'triangular'. Default: 'uniform'.
        Returns: U [pct], Tv [-]

    METHOD: "primary_consolidation_settlement_nc"
        Primary consolidation settlement for normally consolidated clay.
        Parameters:
            - initial_height (float): Layer thickness H0 [m].
            - initial_voidratio (float): Initial void ratio e0. Range: 0.1 to 5.0.
            - initial_effective_stress (float): Initial effective stress [kPa].
            - effective_stress_increase (float): Stress increase [kPa].
            - compression_index (float): Cc. Range: 0.1 to 0.8.
            - e_min (float, optional): Min void ratio. Default: 0.3.
        Returns: delta z [m], delta e [-], e final [-]

    METHOD: "primary_consolidation_settlement_oc"
        Primary consolidation settlement for overconsolidated clay.
        Parameters:
            - initial_height, initial_voidratio, initial_effective_stress, preconsolidation_pressure,
              effective_stress_increase, compression_index, recompression_index (all float, required)
            - e_min (float, optional): Default: 0.3.
        Returns: delta z [m], delta e [-], e final [-]

    METHOD: "consolidation_settlement_mv"
        Consolidation settlement using compressibility mv.
        Parameters:
            - initial_height (float): Layer thickness [m].
            - effective_stress_increase (float): Stress increase [kPa].
            - compressibility (float): mv [1/kPa]. Range: 1e-4 to 10.
        Returns: delta z [m], delta epsilon [-]

    METHOD: "hydraulic_conductivity_unconfined"
        Hydraulic conductivity from unconfined aquifer pumping test.
        Parameters:
            - radius_1, radius_2 (float): Radial distances to standpipes [m].
            - piezometric_height_1, piezometric_height_2 (float): Piezometric heights [m].
            - flowrate (float): Flowrate from pumping well [m3/s].
        Returns: hydraulic_conductivity [m/s]

    ======================== STRESS DISTRIBUTION ========================

    METHOD: "stress_pointload"
        Boussinesq stress distribution under a point load.
        Parameters:
            - pointload (float): Point load Q [kN].
            - z (float): Depth [m].
            - r (float): Radial distance [m].
            - poissonsratio (float): Poisson's ratio. Range: 0 to 0.5.
        Returns: delta sigma z/r/theta [kPa], delta tau rz [kPa]

    METHOD: "stress_stripload"
        Stress distribution under a strip load.
        Parameters:
            - z (float): Depth [m]. x (float): Horizontal offset [m].
            - width (float): Strip width B [m].
            - imposedstress (float): Applied stress [kPa].
            - triangular (bool, optional): Triangular load? Default: False.
        Returns: delta sigma z/x [kPa], delta tau zx [kPa]

    METHOD: "stress_circle"
        Stress below center of circular foundation.
        Parameters:
            - z (float): Depth [m]. footing_radius (float): Radius r0 [m].
            - imposedstress (float): Applied stress [kPa].
            - poissonsratio (float): Range: 0 to 0.5.
        Returns: delta sigma z [kPa], delta sigma r [kPa]

    METHOD: "stress_rectangle"
        Stress below corner of rectangular loaded area.
        Parameters:
            - imposedstress (float): Applied stress [kPa].
            - length (float): L [m]. width (float): B [m].
            - z (float): Depth [m].
        Returns: delta sigma z/x/y [kPa], delta tau zx [kPa]

    ========================= EARTH PRESSURE =========================

    METHOD: "earth_pressure_basic"
        Ka and Kp from friction angle (Mohr circle, no wall friction).
        Parameters: phi_eff (float): Friction angle [deg]. Range: 20 to 50.
        Returns: Ka [-], Kp [-], theta_a [rad], theta_p [rad]

    METHOD: "earth_pressure_poncelet"
        Earth pressure with wall friction and inclination (Poncelet/Coulomb).
        Parameters:
            - phi_eff (float): Friction angle [deg]. Range: 20 to 50.
            - interface_friction_angle (float): Wall friction [deg]. Range: 15 to 40.
            - wall_angle (float): Wall inclination [deg]. Range: 0 to 70.
            - top_angle (float): Ground slope [deg]. Range: 0 to 70.
        Returns: KaC [-], KpC [-]

    METHOD: "earth_pressure_rankine"
        Rankine earth pressure for inclined wall with sloping ground.
        Parameters:
            - phi_eff (float): Friction angle [deg]. Range: 20 to 50.
            - wall_angle (float): Wall inclination [deg]. Range: 0 to 70.
            - top_angle (float): Ground slope [deg]. Range: 0 to 70.
        Returns: KaR [-], KpR [-], omega_a/p [-], theta_a/p [rad], ksi_a/p [rad]

    ======================== SOIL CLASSIFICATION ========================

    METHOD: "relative_density_category"
        Classify relative density (Very loose to Very dense).
        Parameters: relative_density (float): Dr as fraction (0 to 1).
        Returns: Relative density (string)

    METHOD: "su_category"
        Classify undrained shear strength.
        Parameters:
            - undrained_shear_strength (float): Su [kPa]. Range: 0 to 1000.
            - standard (str, optional): 'BS 5930:2015' or 'ASTM D-2488'. Default: 'BS 5930:2015'.
        Returns: strength class (string)

    METHOD: "uscs_description"
        Get USCS soil type description from symbol.
        Parameters: symbol (str): e.g. 'CL', 'SP', 'CH', etc.
        Returns: Soil type (string)

    METHOD: "sample_quality_lunne"
        Assess sample quality from void ratio change (Lunne et al.).
        Parameters:
            - voidratio (float): e0. Range: 0.3 to 3.
            - voidratio_change (float): delta_e. Range: -1 to 1.
            - ocr (float): OCR. Range: 1 to 4.
        Returns: delta e/e0 [-], Quality category (string)

    ======================== DEEP FOUNDATIONS ========================

    METHOD: "pile_shaft_friction_api_sand"
        Unit shaft friction in sand (API RP 2GEO beta method).
        Parameters:
            - api_relativedensity (str): 'Medium dense', 'Dense', or 'Very dense'.
            - api_soildescription (str): 'Sand' or 'Sand-silt'.
            - sigma_vo_eff (float): Vertical effective stress [kPa].
        Returns: f_s_comp_out/in [kPa], f_s_tens_out/in [kPa], f_s_lim [kPa], beta [-]

    METHOD: "pile_shaft_friction_api_clay"
        Unit shaft friction in clay (API RP 2GEO alpha method).
        Parameters:
            - undrained_shear_strength (float): Su [kPa]. Range: 0 to 400.
            - sigma_vo_eff (float): Vertical effective stress [kPa].
        Returns: f_s_comp_out/in [kPa], f_s_tens_out/in [kPa], psi [-], alpha [-]

    METHOD: "pile_shaft_friction_almhamre_sand"
        Unit shaft friction in sand with friction fatigue (Alm & Hamre).
        Parameters:
            - qt (float): Cone resistance [MPa]. sigma_vo_eff (float): [kPa].
            - interface_friction_angle (float): [deg]. depth (float): [m].
            - embedded_length (float): Pile tip depth [m].
        Returns: f_s_comp_out/in [kPa], f_s_tens_out/in [kPa], f_s_initial [kPa], f_s_res [kPa]

    METHOD: "pile_shaft_friction_almhamre_clay"
        Unit shaft friction in clay with friction fatigue (Alm & Hamre).
        Parameters:
            - depth (float): [m]. embedded_length (float): [m].
            - qt (float): [MPa]. fs (float): Sleeve friction [MPa].
            - sigma_vo_eff (float): [kPa].
        Returns: f_s_comp_out/in [kPa], f_s_tens_out/in [kPa], f_s_initial [kPa], f_s_res [kPa]

    METHOD: "pile_end_bearing_api_clay"
        Unit end bearing in clay (API RP 2GEO): q = Nc * Su.
        Parameters:
            - undrained_shear_strength (float): Su at pile tip [kPa]. Range: 0 to 400.
            - N_c (float, optional): Bearing capacity factor. Default: 9.0.
        Returns: q_b_coring [kPa], q_b_plugged [kPa]

    METHOD: "pile_end_bearing_api_sand"
        Unit end bearing in sand (API RP 2GEO): q = Nq * sigma'vo.
        Parameters:
            - api_relativedensity (str): 'Medium dense', 'Dense', or 'Very dense'.
            - api_soildescription (str): 'Sand' or 'Sand-silt'.
            - sigma_vo_eff (float): [kPa].
        Returns: q_b_coring [kPa], q_b_plugged [kPa], q_b_lim [kPa], Nq [-]

    METHOD: "pile_end_bearing_almhamre_sand"
        Unit end bearing in sand (Alm & Hamre CPT-based).
        Parameters: qt (float): [MPa]. sigma_vo_eff (float): [kPa].
        Returns: q_b_coring [kPa], q_b_plugged [kPa]

    METHOD: "pile_end_bearing_almhamre_clay"
        Unit end bearing in clay (Alm & Hamre): qb = 0.6 * qt.
        Parameters: qt (float): Cone resistance [MPa].
        Returns: q_b_coring [kPa], q_b_plugged [kPa]

    =================== SOIL DYNAMICS & LIQUEFACTION ===================

    METHOD: "modulus_reduction_ishibashi"
        G/Gmax and damping ratio (Ishibashi & Zhang 1993).
        Parameters:
            - strain (float): Shear strain [%]. Range: 0 to 10.
            - pi (float): Plasticity index [%]. Range: 0 to 200.
            - sigma_m_eff (float): Mean effective stress [kPa]. Range: 0 to 400.
        Returns: G/Gmax [-], K [-], m [-], n [-], dampingratio [pct]

    METHOD: "gmax_from_shear_wave_velocity"
        Gmax from shear wave velocity and unit weight.
        Parameters:
            - Vs (float): Shear wave velocity [m/s]. Range: 0 to 600.
            - gamma (float): Bulk unit weight [kN/m3]. Range: 12 to 22.
        Returns: rho [kg/m3], Gmax [kPa]

    METHOD: "damping_ratio_seed"
        Damping ratio for sand/gravel (Seed et al. 1986).
        Parameters: cyclic_shear_strain (float): [%]. Range: 0.0001 to 1.0.
        Returns: D LE [pct], D BE [pct], D HE [pct]

    METHOD: "cyclic_stress_ratio_moss"
        CSR for liquefaction assessment (Moss/Cetin formulation).
        Parameters:
            - sigma_vo (float): Total vertical stress [kPa].
            - sigma_vo_eff (float): Effective vertical stress [kPa].
            - magnitude (float): Earthquake magnitude. Range: 5.5 to 8.5.
            - acceleration (float): Max ground acceleration [m/s2].
            - depth (float): Depth [m].
        Returns: CSR [-], CSR* [-], DWF [-], rd [-]

    METHOD: "cyclic_stress_ratio_youd"
        CSR using Youd et al. (2001) formulation.
        Parameters:
            - acceleration (float): Max ground acceleration [m/s2].
            - sigma_vo (float): Total vertical stress [kPa].
            - sigma_vo_eff (float): Effective vertical stress [kPa].
            - depth (float): [m]. Range: 0 to 23.
            - magnitude (float): Earthquake magnitude. Range: 0 to 8.5.
        Returns: CSR [-], CSR* [-], MSF [-], rd [-]

    METHOD: "liquefaction_robertson_fear"
        Liquefaction triggering from CPT (Robertson & Fear 1995).
        Parameters:
            - qc (float): Cone tip resistance [MPa]. Range: 0 to 120.
            - sigma_vo_eff (float): Effective vertical stress [kPa].
            - CSR (float): Cyclic stress ratio. Range: 0.073 to 0.49.
        Returns: qc1 [-], qc1 liquefaction [-], qc liquefaction [MPa], liquefaction (bool)

    ======================== SOIL CORRELATIONS ========================

    METHOD: "gmax_sand_hardin_black"
        Small-strain shear modulus for sand (Hardin & Black).
        Parameters:
            - sigma_m0 (float): Mean effective stress [kPa]. Range: 0 to 500.
            - void_ratio (float): In-situ void ratio. Range: 0 to 4.
        Returns: Gmax [kPa]

    METHOD: "permeability_hazen"
        Permeability from grain size (Hazen): k = C * D10^2.
        Parameters: grain_size (float): D10 [mm]. Range: 0.01 to 2.0.
        Returns: k [m/s]

    METHOD: "hssmall_parameters_sand"
        HS Small model parameters for sand (PLAXIS).
        Parameters: relative_density (float): Dr [%]. Range: 10 to 100.
        Returns: gamma_unsat/sat, E50/Eoed/Eur_ref, G0_ref, m, gamma_07, phi_eff, psi, Rf

    METHOD: "stress_dilatancy_bolton"
        Stress-dilatancy (Bolton 1986).
        Parameters:
            - relative_density (float): Dr (fraction). Range: 0.1 to 1.0.
            - p_eff (float): Effective pressure [kPa]. Range: 20 to 10000.
            - stress_condition (str, optional): 'triaxial strain' or 'plane strain'. Default: 'triaxial strain'.
        Returns: Ir [-], phi_max - phi_cs [deg], Dilation angle [deg]

    METHOD: "compression_index_koppula"
        Compression index from water content (Koppula).
        Parameters:
            - water_content (float): Natural water content (fraction). Range: 0 to 4.
            - cc_cr_ratio (float, optional): Cc/Cr ratio. Default: 7.5.
        Returns: Cc [-], Cr [-]

    METHOD: "friction_angle_from_pi"
        Drained friction angle of clay from plasticity index.
        Parameters: plasticity_index (float): PI [%]. Range: 5 to 1000.
        Returns: Effective friction angle [deg]

    METHOD: "cv_from_liquid_limit"
        Coefficient of consolidation from liquid limit (US Navy).
        Parameters:
            - liquid_limit (float): LL [%]. Range: 20 to 160.
            - trend (str, optional): 'Remoulded', 'NC', or 'OC'. Default: 'NC'.
        Returns: cv [m2/yr]

    METHOD: "gmax_clay_andersen"
        Small-strain shear modulus for clay (Andersen 2015).
        Parameters:
            - pi (float): Plasticity index [%]. Range: 0 to 160.
            - ocr (float): OCR. Range: 1 to 40.
            - sigma_vo_eff (float): Effective vertical stress [kPa]. Range: 0 to 1000.
        Returns: sigma_0_ref [kPa], Gmax [kPa]

    METHOD: "k0_from_plasticity"
        K0 from plasticity index and OCR (Kenney/Alpan).
        Parameters:
            - pi (float): Plasticity index [%]. Range: 5 to 80.
            - ocr (float, optional): OCR. Default: 1. Range: 1 to 30.
        Returns: K0 NC [-], K0 [-]

    METHOD: "k0_from_friction_angle"
        K0 from critical-state friction angle and OCR (Mesri & Hayat).
        Parameters:
            - phi_cs (float): Critical state friction angle [deg]. Range: 15 to 45.
            - ocr (float, optional): OCR. Default: 1. Range: 1 to 30.
        Returns: K0 [-]
    """

    if method not in METHOD_REGISTRY:
        available = ", ".join(sorted(METHOD_REGISTRY.keys()))
        return {
            "error": f"Unknown method '{method}'. Available methods: {available}"
        }

    func = METHOD_REGISTRY[method]

    try:
        raw_result = func(**parameters)
        return _clean_result(raw_result)
    except Exception as e:
        return {"error": f"{type(e).__name__}: {str(e)}"}


def groundhog_list_methods(category: str = "") -> dict:
    """
    Lists all available geotechnical calculation methods in the groundhog agent.
    Use this to discover what calculations are available before calling groundhog_agent.

    Optionally filter by category. If no category is provided, all methods are returned.

    Parameters:
        category (str, optional): Filter by category. Options: "Phase Relations",
            "SPT Correlations", "CPT Correlations", "Bearing Capacity".
            Leave empty or omit to list all methods.

    Returns a dictionary with method names as keys and brief descriptions as values,
    grouped by category.
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
        return {"error": f"No methods found for category '{category}'. "
                         f"Available categories: Phase Relations, SPT Correlations, "
                         f"CPT Correlations, Bearing Capacity, Consolidation & Settlement, "
                         f"Stress Distribution, Earth Pressure, Soil Classification, "
                         f"Deep Foundations, Soil Dynamics & Liquefaction, Soil Correlations"}
    return result


def groundhog_describe_method(method: str) -> dict:
    """
    Returns detailed documentation for a specific groundhog agent method.
    Use this to understand exactly what a method does, what parameters it needs,
    what it returns, and what reference it is based on before calling groundhog_agent.

    Parameters:
        method (str): The method name, e.g. "bulk_unit_weight" or "cpt_friction_angle_sand".

    Returns a dictionary with: category, description, reference, parameters (with types,
    ranges, defaults), and return values.
    """
    if method not in METHOD_INFO:
        available = ", ".join(sorted(METHOD_INFO.keys()))
        return {"error": f"Unknown method '{method}'. Available methods: {available}"}
    return METHOD_INFO[method]
