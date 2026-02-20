"""
Tier 2: Multi-function engineering workflow tests.

Tests realistic engineering workflows where the LLM chains multiple
Foundry agent calls to solve a complete design problem.

Each test simulates the sequence of calls an LLM would make,
passing outputs from one agent as inputs to another.
"""

import pytest

from foundry_test_harness.harness import FoundryAgentHarness, AgentError
from foundry_test_harness import scenarios as S

from bearing_capacity_agent_foundry import bearing_capacity_agent
from settlement_agent_foundry import settlement_agent
from axial_pile_agent_foundry import axial_pile_agent
from drilled_shaft_agent_foundry import drilled_shaft_agent
from seismic_geotech_agent_foundry import seismic_geotech_agent
from retaining_walls_agent_foundry import retaining_walls_agent
from slope_stability_agent_foundry import slope_stability_agent
from geolysis_agent_foundry import geolysis_agent
from ground_improvement_agent_foundry import ground_improvement_agent
from pile_group_agent_foundry import pile_group_agent
from wave_equation_agent_foundry import wave_equation_agent

H = FoundryAgentHarness()


# ============================================================================
# Workflow 1: Shallow Foundation Design
# ============================================================================

class TestFoundationDesignWorkflow:
    """
    Typical workflow:
    1. Classify soil -> understand soil type
    2. Correct SPT -> get design N-value
    3. Compute bearing capacity -> check adequacy
    4. Check settlement -> verify serviceability
    """

    def test_sand_foundation_workflow(self):
        """Complete shallow foundation design on sand.

        Workflow: classify_uscs -> correct_spt -> bearing_capacity -> settlement
        """
        # Step 1: Classify the soil
        r_class = H.call(geolysis_agent, "classify_uscs", {
            "liquid_limit": 0.0,
            "plastic_limit": 0.0,
            "fines": 5.0,
            "sand": 75.0,
            "d_10": 0.15,
            "d_30": 0.5,
            "d_60": 2.0,
        })
        assert "S" in r_class["symbol"]  # Should be some kind of sand

        # Step 2: Correct SPT N-value
        r_spt = H.call(geolysis_agent, "correct_spt", {
            "recorded_spt_n_value": 20,
            "eop": 100.0,
            "hammer_type": "safety",
            "opc_method": "liao",
        })
        n1_60 = r_spt["n1_60"]
        assert n1_60 > 0

        # Step 3: Compute bearing capacity
        r_bc = H.call(bearing_capacity_agent, "bearing_capacity_analysis", {
            "width": 2.0,
            "length": 2.0,
            "depth": 1.5,
            "shape": "square",
            "friction_angle": 32.0,
            "cohesion": 0.0,
            "unit_weight": 18.0,
        })
        q_allowable = r_bc["q_allowable_kPa"]
        assert q_allowable > 50

        # Step 4: Check settlement using q_allowable as applied load
        r_sett = H.call(settlement_agent, "elastic_settlement", {
            "q_net": q_allowable,
            "B": 2.0,
            "Es": 15000.0,
            "nu": 0.3,
        })
        assert r_sett["immediate_settlement_mm"] > 0
        assert r_sett["immediate_settlement_mm"] < 100

    def test_clay_foundation_workflow(self):
        """Foundation design on clay: undrained bearing capacity + consolidation.

        Workflow: classify -> undrained BC (phi=0) -> consolidation settlement
        """
        # Step 1: Classify clay
        r_class = H.call(geolysis_agent, "classify_uscs", {
            "liquid_limit": 50.0,
            "plastic_limit": 25.0,
            "fines": 85.0,
        })
        assert "C" in r_class["symbol"]  # Should be CL or CH

        # Step 2: Undrained bearing capacity
        r_bc = H.call(bearing_capacity_agent, "bearing_capacity_analysis", {
            "width": 3.0,
            "length": 3.0,
            "depth": 1.5,
            "shape": "square",
            "cohesion": 75.0,
            "friction_angle": 0.0,
            "unit_weight": 17.5,
        })
        q_allow = r_bc["q_allowable_kPa"]
        assert q_allow > 50

        # Step 3: Consolidation settlement
        r_sett = H.call(settlement_agent, "consolidation_settlement", {
            "layers": [
                {
                    "thickness": 4.0,
                    "depth_to_center": 5.0,
                    "e0": 1.0,
                    "Cc": 0.3,
                    "Cr": 0.06,
                    "sigma_v0": 75.0,
                }
            ],
            "delta_sigma": min(q_allow, 100.0),
        })
        assert r_sett["consolidation_settlement_mm"] > 0

    def test_bearing_then_settlement_check(self):
        """Get q_allowable, then verify settlement is acceptable."""
        r_bc = H.call(bearing_capacity_agent, "bearing_capacity_analysis", {
            "width": 2.0,
            "depth": 1.0,
            "shape": "strip",
            "friction_angle": 28.0,
            "unit_weight": 17.0,
        })
        q_allow = r_bc["q_allowable_kPa"]

        r_sett = H.call(settlement_agent, "elastic_settlement", {
            "q_net": q_allow,
            "B": 2.0,
            "Es": 8000.0,
            "nu": 0.3,
        })
        assert r_sett["immediate_settlement_mm"] > 0

    def test_spt_to_bearing_capacity(self):
        """SPT N -> corrected N1_60 -> allowable bearing from SPT."""
        r_spt = H.call(geolysis_agent, "correct_spt", {
            "recorded_spt_n_value": 30,
            "eop": 150.0,
            "opc_method": "liao",
        })

        r_bc = H.call(geolysis_agent, "allowable_bc_spt", {
            "corrected_spt_n_value": r_spt["n1_60"],
            "width": 2.5,
            "depth": 1.5,
        })
        assert r_bc["bearing_capacity_kpa"] > 0


# ============================================================================
# Workflow 2: Seismic Design
# ============================================================================

class TestSeismicDesignWorkflow:
    """
    Typical workflow:
    1. Site classification -> get Fa, Fv
    2. Seismic earth pressure -> get KAE
    3. Retaining wall design with seismic loads
    """

    def test_seismic_retaining_wall(self):
        """Site class -> seismic coefficient -> retaining wall design."""
        # Step 1: Site classification
        r_site = H.call(seismic_geotech_agent, "site_classification", {
            "vs30": 250.0,
            "Ss": 1.0,
            "S1": 0.4,
        })
        assert r_site["site_class"] == "D"

        # Step 2: Seismic earth pressure coefficient
        pga = r_site.get("Fpga", 1.0) * 0.4
        kh = 0.5 * min(pga, 0.5)

        r_mo = H.call(seismic_geotech_agent, "seismic_earth_pressure", {
            "phi": 32.0,
            "kh": kh,
            "delta": 21.0,
        })
        assert r_mo["KAE"] > 0.3

        # Step 3: Design retaining wall (static check)
        r_wall = H.call(retaining_walls_agent, "cantilever_wall", {
            "wall_height": 5.0,
            "gamma_backfill": 18.0,
            "phi_backfill": 32.0,
            "base_width": 4.0,
            "toe_length": 1.0,
            "stem_thickness_base": 0.5,
            "base_thickness": 0.5,
            "phi_foundation": 32.0,
        })
        assert r_wall["FOS_sliding"] > 1.0

    def test_site_class_to_design_spectrum(self):
        """Site class -> Fa/Fv -> SDS/SD1 (ASCE 7 design spectrum)."""
        r = H.call(seismic_geotech_agent, "site_classification", {
            "vs30": 300.0,
            "Ss": 0.8,
            "S1": 0.3,
        })
        Fa = r.get("Fa", r.get("Fpga", 1.0))
        assert Fa > 0
        assert r["site_class"] in ["A", "B", "BC", "C", "CD", "D", "DE", "E"]

    def test_liquefaction_to_residual_strength(self):
        """Evaluate liquefaction -> get residual strength for liquefied layers."""
        # Step 1: Check if layer liquefies
        r_liq = H.call(seismic_geotech_agent, "liquefaction_evaluation", {
            "depths": [5.0],
            "N160": [8.0],
            "FC": [5.0],
            "gamma": [18.0],
            "amax_g": 0.3,
            "gwt_depth": 2.0,
        })
        assert r_liq["min_FOS_liq"] < 1.0, "Loose sand at 0.3g should liquefy"

        # Step 2: Get residual strength
        r_sr = H.call(seismic_geotech_agent, "residual_strength", {
            "N160cs": 8.0,
        })
        assert r_sr["Sr_kPa"] > 0
        assert r_sr["Sr_kPa"] < 50


# ============================================================================
# Workflow 3: Deep Foundation Design
# ============================================================================

class TestDeepFoundationWorkflow:
    """
    Typical workflow:
    1. Single pile capacity -> determine design load
    2. Group efficiency -> effective group capacity
    3. Pile group load distribution with moment
    """

    def test_pile_to_group(self):
        """Single pile capacity -> group efficiency -> group capacity."""
        # Step 1: Single pile capacity
        r_pile = H.call(axial_pile_agent, "axial_pile_capacity",
                        S.DRIVEN_PILE_SAND["params"])
        q_single = r_pile["Q_allowable_kN"]
        assert q_single > 0

        # Step 2: Group efficiency for 3x3 group
        r_eff = H.call(pile_group_agent, "group_efficiency", {
            "n_rows": 3,
            "n_cols": 3,
            "pile_diameter": 0.356,
            "spacing": 1.07,
        })
        eta = r_eff["converse_labarre_Eg"]
        assert 0 < eta <= 1.0

        # Step 3: Group capacity = n * Q_single * eta
        n_piles = 9
        q_group = n_piles * q_single * eta
        assert q_group > q_single

    def test_pile_group_load_distribution(self):
        """Pile group with moment: corner piles should take more load."""
        r = H.call(pile_group_agent, "pile_group_simple", {
            "n_rows": 3,
            "n_cols": 3,
            "spacing_x": 1.0,
            "spacing_y": 1.0,
            "Vz": 2700.0,
            "Mx": 500.0,
        })
        # pile_forces is a list of dicts with 'axial_kN'
        loads = [p["axial_kN"] for p in r["pile_forces"]]
        assert len(loads) == 9
        # With Mx, some piles should have higher loads than others
        assert max(loads) > min(loads)

    def test_drilled_shaft_lrfd(self):
        """Drilled shaft capacity -> LRFD resistance factors."""
        r_cap = H.call(drilled_shaft_agent, "drilled_shaft_capacity",
                       S.DRILLED_SHAFT_CLAY["params"])
        assert r_cap["Q_ultimate_kN"] > 0

        r_lrfd = H.call(drilled_shaft_agent, "lrfd_capacity", {
            **S.DRILLED_SHAFT_CLAY["params"],
            "tip_soil_type": "cohesive",
        })
        assert r_lrfd.get("Q_factored_kN", r_lrfd.get("phi_side", 0)) >= 0


# ============================================================================
# Workflow 4: Slope + Ground Improvement
# ============================================================================

class TestSlopeImprovementWorkflow:
    """
    Typical workflow:
    1. Slope stability -> FOS too low
    2. Ground improvement feasibility -> select method
    3. Design improvement -> verify consolidation time
    """

    def test_slope_to_improvement_feasibility(self):
        """Low FOS slope -> check ground improvement feasibility."""
        # Step 1: Analyze slope
        r_slope = H.call(slope_stability_agent, "analyze_slope", {
            "xc": 15.0,
            "yc": 18.0,
            "radius": 13.0,
            "surface_points": [[0, 10], [10, 10], [20, 5], [30, 5]],
            "soil_layers": [
                {
                    "name": "Soft Clay",
                    "top_elevation": 10.0,
                    "bottom_elevation": 0.0,
                    "gamma": 17.0,
                    "phi": 5.0,
                    "c_prime": 10.0,
                }
            ],
        })
        assert r_slope["FOS"] > 0

        # Step 2: Check improvement feasibility
        r_feas = H.call(ground_improvement_agent, "feasibility", {
            "soil_type": "soft_clay",
            "cu_kPa": 15.0,
            "thickness_m": 10.0,
        })
        # Should return some recommendations
        assert isinstance(r_feas, dict)
        assert len(r_feas) > 0

    def test_settlement_to_wick_drains(self):
        """Excessive consolidation settlement -> design wick drains."""
        # Step 1: Compute settlement (expect large value)
        r_sett = H.call(settlement_agent, "consolidation_settlement", {
            "layers": [
                {
                    "thickness": 6.0,
                    "depth_to_center": 5.0,
                    "e0": 1.5,
                    "Cc": 0.5,
                    "Cr": 0.1,
                    "sigma_v0": 50.0,
                }
            ],
            "delta_sigma": 80.0,
        })
        settlement_mm = r_sett["consolidation_settlement_mm"]
        assert settlement_mm > 50

        # Step 2: Design wick drains to accelerate consolidation
        r_drain = H.call(ground_improvement_agent, "wick_drains", {
            "spacing": 1.5,
            "ch": 3.0,
            "cv": 1.0,
            "Hdr": 6.0,
            "time": 0.5,
        })
        assert r_drain["U_total_percent"] > 50


# ============================================================================
# Workflow 5: Wave Equation + Pile
# ============================================================================

class TestWaveEquationWorkflow:
    """
    Typical workflow:
    1. Axial pile capacity -> required ultimate capacity
    2. Bearing graph -> blow count at design capacity
    """

    def test_pile_then_bearing_graph(self):
        """Design pile capacity -> generate bearing graph."""
        # Step 1: Get required pile capacity
        r_pile = H.call(axial_pile_agent, "axial_pile_capacity",
                        S.DRIVEN_PILE_SAND["params"])
        q_ult = r_pile["Q_ultimate_kN"]
        assert q_ult > 0

        # Step 2: Generate bearing graph
        r_bg = H.call(wave_equation_agent, "bearing_graph", {
            "pile_length": 15.0,
            "pile_area": 0.01,
            "ram_weight": 50.0,
            "stroke": 1.0,
            "efficiency": 0.8,
            "cushion_stiffness": 1e6,
            "skin_fraction": 0.6,
            "R_min": max(200, q_ult * 0.5),
            "R_max": q_ult * 1.5,
            "R_step": 200,
        })
        assert len(r_bg["R_values_kN"]) > 0
        assert len(r_bg["blow_counts_per_m"]) > 0

    def test_drivability_study(self):
        """Drivability: blow count and stresses at multiple depths."""
        r = H.call(wave_equation_agent, "drivability", {
            "pile_area": 0.01,
            "depths": [5.0, 10.0, 15.0],
            "R_at_depth": [300.0, 600.0, 1000.0],
            "ram_weight": 50.0,
            "stroke": 1.0,
            "cushion_stiffness": 1e6,
        })
        # Drivability returns 'points' list and 'can_drive' bool
        assert r["can_drive"] is True or r["can_drive"] is False
        assert "points" in r
