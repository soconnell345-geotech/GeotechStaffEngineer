"""
Test script for groundhog_agent dispatcher.
Runs one test call per method to verify all 90 methods work correctly.
"""

import json
from groundhog_agent import groundhog_agent, groundhog_list_methods, groundhog_describe_method


def test(method, parameters, label=""):
    result = groundhog_agent(method, parameters)
    status = "FAIL" if "error" in result else "OK"
    print(f"[{status}] {method}" + (f" ({label})" if label else ""))
    for k, v in result.items():
        if isinstance(v, float):
            print(f"       {k}: {v:.4f}")
        else:
            print(f"       {k}: {v}")
    print()
    return result


print("=" * 70)
print("PHASE RELATIONS (14 methods)")
print("=" * 70)

test("voidratio_from_porosity", {"porosity": 0.4})
test("porosity_from_voidratio", {"voidratio": 0.67})
test("saturation_from_watercontent", {"water_content": 0.25, "voidratio": 0.7})
test("bulk_unit_weight", {"saturation": 1.0, "voidratio": 0.65})
test("dry_unit_weight", {"watercontent": 0.2, "bulkunitweight": 20.0})
test("voidratio_from_dry_density", {"dry_density": 1600.0})
test("bulk_unit_weight_from_dry", {"dryunitweight": 14.0, "watercontent": 0.15})
test("relative_density", {"void_ratio": 0.55, "e_min": 0.4, "e_max": 0.9})
test("voidratio_from_bulk_unit_weight", {"bulkunitweight": 20.0})
test("unit_weight_saturated", {"water_content": 0.3})
test("density_from_unit_weight", {"gamma": 20.0})
test("unit_weight_from_density", {"density": 2000.0})
test("watercontent_from_voidratio", {"voidratio": 0.7})
test("voidratio_from_watercontent", {"water_content": 0.3})

print("=" * 70)
print("SPT CORRELATIONS (10 methods)")
print("=" * 70)

test("spt_overburden_correction_liaowhitman", {"N": 15, "sigma_vo_eff": 150.0})
test("spt_N60_correction", {
    "N": 20,
    "borehole_diameter": 100.0,
    "rod_length": 12.0,
    "country": "United States",
    "hammertype": "Safety",
    "hammerrelease": "Rope and pulley",
})
test("spt_relative_density_kulhawymayne", {"N1_60": 25.0, "d_50": 0.5})
test("spt_undrained_shear_strength_salgado", {"pi": 30.0, "N_60": 10.0})
test("spt_friction_angle_kulhawymayne", {"N": 20.0, "sigma_vo_eff": 100.0})
test("spt_relative_density_class", {"N": 25.0})
test("spt_overburden_correction_iso", {"N": 15, "sigma_vo_eff": 100.0})
test("spt_friction_angle_pht", {"N1_60": 20.0})
test("spt_youngs_modulus_aashto", {"N1_60": 30.0, "soiltype": "Clean sands"})
test("spt_consistency_class", {"N": 12.0})

print("=" * 70)
print("CPT CORRELATIONS (16 methods)")
print("=" * 70)

# Full CPT normalisation (all 7 required parameters)
test("cpt_normalisations", {
    "measured_qc": 5.0,
    "measured_fs": 0.05,
    "measured_u2": 0.2,
    "sigma_vo_tot": 100.0,
    "sigma_vo_eff": 60.0,
    "depth": 5.0,
    "cone_area_ratio": 0.8,
})

test("cpt_soil_class_robertson", {"ic": 2.5})

# Behaviour index needs qt, fs, sigma_vo, sigma_vo_eff
test("cpt_behaviour_index", {
    "qt": 5.0,
    "fs": 0.05,
    "sigma_vo": 100.0,
    "sigma_vo_eff": 60.0,
})

test("cpt_gmax_sand", {"qc": 10.0, "sigma_vo_eff": 100.0})

# Clay Gmax: parameter is qc (not qt), in MPa
test("cpt_gmax_clay", {"qc": 5.0})

test("cpt_relative_density_nc_sand", {"qc": 15.0, "sigma_vo_eff": 100.0})

# OC sand needs k0
test("cpt_relative_density_oc_sand", {"qc": 15.0, "sigma_vo_eff": 100.0, "k0": 0.8})

# Jamiolkowski needs sigma_vo_eff in 50-400 range and k0 in 0.4-1.5
test("cpt_relative_density_jamiolkowski", {"qc": 15.0, "sigma_vo_eff": 150.0, "k0": 0.8})

test("cpt_friction_angle_sand", {"qt": 10.0, "sigma_vo_eff": 100.0})

# Undrained shear strength needs qnet (net cone resistance) and Nk
test("cpt_undrained_shear_strength", {"qnet": 0.5, "Nk": 15.0})

test("cpt_ocr", {"Qt": 5.0})

test("cpt_sensitivity", {"Rf": 1.0})

test("cpt_unit_weight", {"ft": 0.05, "sigma_vo_eff": 100.0})

test("cpt_shear_wave_velocity", {"qt": 10.0, "ic": 2.0, "sigma_vo": 150.0})

test("cpt_k0_sand", {"qt": 10.0, "sigma_vo_eff": 100.0, "ocr": 1.5})

test("cpt_constrained_modulus", {
    "qt": 10.0, "ic": 2.0, "sigma_vo": 150.0, "sigma_vo_eff": 100.0,
})

print("=" * 70)
print("BEARING CAPACITY (10 methods)")
print("=" * 70)

test("bearing_capacity_nq", {"friction_angle": 35.0})
test("bearing_capacity_ngamma_vesic", {"friction_angle": 35.0})
test("bearing_capacity_ngamma_meyerhof", {"friction_angle": 35.0})

# Davis & Booker needs roughness_factor
test("bearing_capacity_ngamma_davisbooker", {
    "friction_angle": 35.0, "roughness_factor": 0.5,
})

# Undrained bearing capacity: effective_length, effective_width, su_base
test("bearing_capacity_undrained_api", {
    "effective_length": 10.0,
    "effective_width": 10.0,
    "su_base": 50.0,
    "base_depth": 2.0,
})

# Drained bearing capacity: vertical_effective_stress, effective_friction_angle,
# effective_unit_weight, effective_length, effective_width
test("bearing_capacity_drained_api", {
    "vertical_effective_stress": 100.0,
    "effective_friction_angle": 35.0,
    "effective_unit_weight": 9.0,
    "effective_length": 10.0,
    "effective_width": 10.0,
})

# Undrained sliding: su_base, foundation_area
test("sliding_capacity_undrained_api", {
    "su_base": 50.0,
    "foundation_area": 100.0,
})

# Drained sliding: vertical_load, effective_friction_angle, effective_unit_weight
test("sliding_capacity_drained_api", {
    "vertical_load": 500.0,
    "effective_friction_angle": 35.0,
    "effective_unit_weight": 9.0,
})

# Effective area rectangle with direct eccentricities
test("effective_area_rectangle", {
    "length": 10.0,
    "width": 5.0,
    "eccentricity_length": 0.5,
    "eccentricity_width": 0.3,
})

# Effective area circle with eccentricity
test("effective_area_circle", {
    "foundation_radius": 5.0,
    "eccentricity": 1.0,
})

print("=" * 70)
print("CONSOLIDATION & SETTLEMENT (5 methods)")
print("=" * 70)

test("consolidation_degree", {
    "time": 100000, "cv": 1.0, "drainage_length": 5.0,
})
test("primary_consolidation_settlement_nc", {
    "initial_height": 3.0, "initial_voidratio": 1.2,
    "initial_effective_stress": 100.0, "effective_stress_increase": 50.0,
    "compression_index": 0.4,
})
test("primary_consolidation_settlement_oc", {
    "initial_height": 3.0, "initial_voidratio": 1.0,
    "initial_effective_stress": 80.0, "preconsolidation_pressure": 200.0,
    "effective_stress_increase": 50.0, "compression_index": 0.3,
    "recompression_index": 0.05,
})
test("consolidation_settlement_mv", {
    "initial_height": 5.0, "effective_stress_increase": 80.0,
    "compressibility": 0.001,
})
test("hydraulic_conductivity_unconfined", {
    "radius_1": 3.0, "radius_2": 10.0,
    "piezometric_height_1": 4.0, "piezometric_height_2": 5.0,
    "flowrate": 0.001,
})

print("=" * 70)
print("STRESS DISTRIBUTION (4 methods)")
print("=" * 70)

test("stress_pointload", {
    "pointload": 100.0, "z": 2.0, "r": 1.0, "poissonsratio": 0.3,
})
test("stress_stripload", {
    "z": 3.0, "x": 2.0, "width": 4.0, "imposedstress": 100.0,
})
test("stress_circle", {
    "z": 2.0, "footing_radius": 3.0, "imposedstress": 100.0,
    "poissonsratio": 0.3,
})
test("stress_rectangle", {
    "imposedstress": 100.0, "length": 6.0, "width": 3.0, "z": 2.0,
})

print("=" * 70)
print("EARTH PRESSURE (3 methods)")
print("=" * 70)

test("earth_pressure_basic", {"phi_eff": 30.0})
test("earth_pressure_poncelet", {
    "phi_eff": 30.0, "interface_friction_angle": 20.0,
    "wall_angle": 5.0, "top_angle": 10.0,
})
test("earth_pressure_rankine", {
    "phi_eff": 30.0, "wall_angle": 5.0, "top_angle": 10.0,
})

print("=" * 70)
print("SOIL CLASSIFICATION (4 methods)")
print("=" * 70)

test("relative_density_category", {"relative_density": 0.5})
test("su_category", {"undrained_shear_strength": 60.0})
test("uscs_description", {"symbol": "CL"})
test("sample_quality_lunne", {
    "voidratio": 1.2, "voidratio_change": 0.05, "ocr": 1.5,
})

print("=" * 70)
print("DEEP FOUNDATIONS (8 methods)")
print("=" * 70)

test("pile_shaft_friction_api_sand", {
    "api_relativedensity": "Dense", "api_soildescription": "Sand",
    "sigma_vo_eff": 150.0,
})
test("pile_shaft_friction_api_clay", {
    "undrained_shear_strength": 80.0, "sigma_vo_eff": 100.0,
})
test("pile_shaft_friction_almhamre_sand", {
    "qt": 15.0, "sigma_vo_eff": 100.0, "interface_friction_angle": 28.0,
    "depth": 10.0, "embedded_length": 30.0,
})
test("pile_shaft_friction_almhamre_clay", {
    "depth": 10.0, "embedded_length": 30.0, "qt": 2.0,
    "fs": 0.02, "sigma_vo_eff": 80.0,
})
test("pile_end_bearing_api_clay", {
    "undrained_shear_strength": 100.0,
})
test("pile_end_bearing_api_sand", {
    "api_relativedensity": "Dense", "api_soildescription": "Sand",
    "sigma_vo_eff": 200.0,
})
test("pile_end_bearing_almhamre_sand", {
    "qt": 20.0, "sigma_vo_eff": 150.0,
})
test("pile_end_bearing_almhamre_clay", {
    "qt": 3.0,
})

print("=" * 70)
print("SOIL DYNAMICS & LIQUEFACTION (6 methods)")
print("=" * 70)

test("modulus_reduction_ishibashi", {
    "strain": 0.1, "pi": 30.0, "sigma_m_eff": 100.0,
})
test("gmax_from_shear_wave_velocity", {
    "Vs": 250.0, "gamma": 19.0,
})
test("damping_ratio_seed", {
    "cyclic_shear_strain": 0.01,
})
test("cyclic_stress_ratio_moss", {
    "sigma_vo": 150.0, "sigma_vo_eff": 100.0,
    "magnitude": 7.0, "acceleration": 2.0, "depth": 8.0,
})
test("cyclic_stress_ratio_youd", {
    "acceleration": 2.0, "sigma_vo": 150.0, "sigma_vo_eff": 100.0,
    "depth": 8.0, "magnitude": 7.0,
})
test("liquefaction_robertson_fear", {
    "qc": 5.0, "sigma_vo_eff": 100.0, "CSR": 0.15,
})

print("=" * 70)
print("SOIL CORRELATIONS (10 methods)")
print("=" * 70)

test("gmax_sand_hardin_black", {
    "sigma_m0": 100.0, "void_ratio": 0.6,
})
test("permeability_hazen", {"grain_size": 0.2})
test("hssmall_parameters_sand", {"relative_density": 60.0})
test("stress_dilatancy_bolton", {
    "relative_density": 0.7, "p_eff": 200.0,
})
test("compression_index_koppula", {"water_content": 0.5})
test("friction_angle_from_pi", {"plasticity_index": 30.0})
test("cv_from_liquid_limit", {"liquid_limit": 50.0})
test("gmax_clay_andersen", {
    "pi": 30.0, "ocr": 2.0, "sigma_vo_eff": 100.0,
})
test("k0_from_plasticity", {"pi": 30.0, "ocr": 2})
test("k0_from_friction_angle", {"phi_cs": 30.0, "ocr": 2})

print("=" * 70)
print("ERROR HANDLING")
print("=" * 70)

test("nonexistent_method", {"x": 1}, "bad method name")
test("voidratio_from_porosity", {"porosity": 5.0}, "out-of-range input")

print("=" * 70)
print("HELPER: groundhog_list_methods()")
print("=" * 70)

print("\n--- All methods ---")
all_methods = groundhog_list_methods()
for cat, methods in all_methods.items():
    print(f"\n  {cat} ({len(methods)} methods):")
    for name, brief in methods.items():
        print(f"    {name}: {brief}")

print("\n--- Filtered to CPT only ---")
cpt_only = groundhog_list_methods("CPT Correlations")
for cat, methods in cpt_only.items():
    print(f"\n  {cat}:")
    for name, brief in methods.items():
        print(f"    {name}: {brief}")

print("\n--- Bad category ---")
print(groundhog_list_methods("Nonexistent"))

print("\n" + "=" * 70)
print("HELPER: groundhog_describe_method()")
print("=" * 70)

print("\n--- Detail for cpt_normalisations ---")
print(json.dumps(groundhog_describe_method("cpt_normalisations"), indent=2))

print("\n--- Detail for bearing_capacity_undrained_api ---")
print(json.dumps(groundhog_describe_method("bearing_capacity_undrained_api"), indent=2))

print("\n--- Bad method ---")
print(groundhog_describe_method("nonexistent"))
