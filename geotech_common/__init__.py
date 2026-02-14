"""
Shared utilities for geotechnical engineering modules.

Provides unit conversions, soil property correlations, water utilities,
and parameter validation used across all geotech modules.
"""

from geotech_common.units import (
    m_to_ft, ft_to_m, m_to_in, in_to_m, mm_to_in, in_to_mm,
    kN_to_kips, kips_to_kN, kN_to_lbs, lbs_to_kN,
    kPa_to_ksf, ksf_to_kPa, kPa_to_psi, psi_to_kPa,
    kPa_to_tsf, tsf_to_kPa,
    kNm3_to_pcf, pcf_to_kNm3,
)
from geotech_common.water import GAMMA_W, pore_pressure
from geotech_common.soil_properties import (
    spt_to_phi, spt_to_cu, spt_to_relative_density,
    phi_to_Ka, phi_to_Kp, phi_to_K0,
)
from geotech_common.soil_profile import (
    SoilLayer, GroundwaterCondition, SoilProfile, SoilProfileBuilder,
)
from geotech_common.engineering_checks import (
    check_bearing_capacity, check_settlement, check_pile_capacity,
    check_lateral_pile, check_sheet_pile, check_wave_equation,
    check_pile_group, check_foundation_selection, check_parameter_consistency,
)

__all__ = [
    # Units
    'm_to_ft', 'ft_to_m', 'm_to_in', 'in_to_m', 'mm_to_in', 'in_to_mm',
    'kN_to_kips', 'kips_to_kN', 'kN_to_lbs', 'lbs_to_kN',
    'kPa_to_ksf', 'ksf_to_kPa', 'kPa_to_psi', 'psi_to_kPa',
    'kPa_to_tsf', 'tsf_to_kPa',
    'kNm3_to_pcf', 'pcf_to_kNm3',
    # Water
    'GAMMA_W', 'pore_pressure',
    # Soil properties
    'spt_to_phi', 'spt_to_cu', 'spt_to_relative_density',
    'phi_to_Ka', 'phi_to_Kp', 'phi_to_K0',
    # Soil profile
    'SoilLayer', 'GroundwaterCondition', 'SoilProfile', 'SoilProfileBuilder',
    # Engineering checks
    'check_bearing_capacity', 'check_settlement', 'check_pile_capacity',
    'check_lateral_pile', 'check_sheet_pile', 'check_wave_equation',
    'check_pile_group', 'check_foundation_selection', 'check_parameter_consistency',
]
