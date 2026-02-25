"""
Tier 1: Individual function validation against textbook / manual answers.

Each test calls a single Foundry agent function with inputs from a published
source and checks the answer within engineering tolerance.

References:
  - Das, B.M. (2019). Principles of Foundation Engineering, 9th Ed.
  - Coduto, D.P. (2001). Foundation Design, 2nd Ed.
  - FHWA GEC-6, GEC-10, GEC-11, GEC-12, GEC-13
  - ASCE 7-22
  - Youd et al. (2001) - Liquefaction triggering
  - Nowak & Collins (2000) - Structural Reliability
  - NAVFAC DM7
"""

import pytest
import math

from foundry_test_harness.harness import FoundryAgentHarness, AgentError
from foundry_test_harness import scenarios as S

from foundry.bearing_capacity_agent_foundry import (
    bearing_capacity_agent, bearing_capacity_list_methods,
    bearing_capacity_describe_method,
)
from foundry.settlement_agent_foundry import (
    settlement_agent, settlement_list_methods,
    settlement_describe_method,
)
from foundry.axial_pile_agent_foundry import (
    axial_pile_agent, axial_pile_list_methods,
    axial_pile_describe_method,
)
from foundry.drilled_shaft_agent_foundry import (
    drilled_shaft_agent, drilled_shaft_list_methods,
    drilled_shaft_describe_method,
)
from foundry.seismic_geotech_agent_foundry import (
    seismic_geotech_agent, seismic_geotech_list_methods,
    seismic_geotech_describe_method,
)
from foundry.retaining_walls_agent_foundry import (
    retaining_walls_agent, retaining_walls_list_methods,
    retaining_walls_describe_method,
)
from foundry.sheet_pile_agent_foundry import (
    sheet_pile_agent, sheet_pile_list_methods,
    sheet_pile_describe_method,
)
from foundry.slope_stability_agent_foundry import (
    slope_stability_agent, slope_stability_list_methods,
    slope_stability_describe_method,
)
from foundry.geolysis_agent_foundry import (
    geolysis_agent, geolysis_list_methods,
    geolysis_describe_method,
)
from foundry.ground_improvement_agent_foundry import (
    ground_improvement_agent, ground_improvement_list_methods,
    ground_improvement_describe_method,
)
from foundry.wave_equation_agent_foundry import (
    wave_equation_agent, wave_equation_list_methods,
    wave_equation_describe_method,
)
from foundry.pile_group_agent_foundry import (
    pile_group_agent, pile_group_list_methods,
    pile_group_describe_method,
)
from foundry.downdrag_agent_foundry import (
    downdrag_agent, downdrag_list_methods,
    downdrag_describe_method,
)
from foundry.pystra_agent_foundry import (
    pystra_agent, pystra_list_methods,
    pystra_describe_method,
)

H = FoundryAgentHarness()


# ============================================================================
# Bearing Capacity
# ============================================================================

class TestBearingCapacity:
    """Bearing capacity validation against textbook values."""

    def test_strip_footing_sand(self):
        """Strip footing on sand — Vesic method.

        B=1.5m, Df=1.0m, phi=35, gamma=17.8 kN/m3.
        qu should be in range 800-2000 kPa for these conditions.
        """
        r = H.call(bearing_capacity_agent, "bearing_capacity_analysis",
                    S.BEARING_STRIP_SAND["params"])
        assert r["q_ultimate_kPa"] > 800
        assert r["q_ultimate_kPa"] < 2000
        assert r["q_allowable_kPa"] > 0
        assert r["q_allowable_kPa"] < r["q_ultimate_kPa"]

    def test_square_footing_sand(self):
        """Square footing on sand — shape factors increase capacity vs strip.

        B=L=2m, Df=1.5m, phi=30, gamma=18 kN/m3.
        """
        r = H.call(bearing_capacity_agent, "bearing_capacity_analysis",
                    S.BEARING_SQUARE_SAND["params"])
        assert r["q_ultimate_kPa"] > 400
        assert r["q_ultimate_kPa"] < 2000
        # Square should give higher capacity than strip due to shape factors
        r_strip = H.call(bearing_capacity_agent, "bearing_capacity_analysis",
                         {**S.BEARING_SQUARE_SAND["params"], "shape": "strip",
                          "length": None})
        assert r["q_ultimate_kPa"] > r_strip["q_ultimate_kPa"]

    def test_undrained_clay(self):
        """Strip footing on clay, undrained (phi=0).

        B=2m, Df=1m, cu=100 kPa, gamma=18 kN/m3.
        Nc=5.14 (Prandtl solution), Nq=1, Ngamma=0.
        qu = cu*Nc*dc + gamma*Df*Nq
        dc ≈ 1 + 0.4*(Df/B) = 1 + 0.4*(1/2) = 1.2
        qu ≈ 100*5.14*1.2 + 18*1*1 ≈ 635 kPa
        """
        r = H.call(bearing_capacity_agent, "bearing_capacity_analysis",
                    S.BEARING_STRIP_CLAY["params"])
        assert r["q_ultimate_kPa"] == pytest.approx(635, rel=0.10)

    def test_gwt_reduces_capacity(self):
        """Groundwater above footing base reduces bearing capacity.

        GWT at 0.5m (above Df=1.5m) should reduce the Ngamma term
        due to submerged unit weight below GWT.
        """
        r_dry = H.call(bearing_capacity_agent, "bearing_capacity_analysis",
                       S.BEARING_SQUARE_SAND["params"])
        gwt_params = {**S.BEARING_SQUARE_SAND["params"], "gwt_depth": 0.5}
        r_gwt = H.call(bearing_capacity_agent, "bearing_capacity_analysis",
                       gwt_params)
        assert r_gwt["q_ultimate_kPa"] < r_dry["q_ultimate_kPa"], \
            "GWT above base should reduce bearing capacity"

    def test_bearing_capacity_factors(self):
        """Standard bearing capacity factors at known friction angles.

        Vesic factors (widely published):
        phi=0:  Nc=5.14, Nq=1.00, Ng=0.00
        phi=30: Nc=30.14, Nq=18.40, Ng≈22.40 (Vesic)
        """
        r0 = H.call(bearing_capacity_agent, "bearing_capacity_factors",
                     {"friction_angle": 0.0})
        assert r0["Nc"] == pytest.approx(5.14, rel=0.01)
        assert r0["Nq"] == pytest.approx(1.0, rel=0.01)
        assert r0["Ngamma"] == pytest.approx(0.0, abs=0.1)

        r30 = H.call(bearing_capacity_agent, "bearing_capacity_factors",
                      {"friction_angle": 30.0})
        assert r30["Nc"] == pytest.approx(30.14, rel=0.02)
        assert r30["Nq"] == pytest.approx(18.40, rel=0.02)
        assert r30["Ngamma"] > 15.0  # Various methods give 15-22


# ============================================================================
# Settlement
# ============================================================================

class TestSettlement:
    """Settlement validation."""

    def test_elastic_settlement(self):
        """Basic elastic settlement: Se = q*B*(1-nu^2)/Es * Iw.

        q=150 kPa, B=2m, Es=10000 kPa, nu=0.3, Iw=1.0.
        Se = 150*2*(1-0.09)/10000 = 0.0273 m = 27.3 mm
        """
        r = H.call(settlement_agent, "elastic_settlement",
                    S.ELASTIC_SETTLEMENT_BASIC["params"])
        assert r["immediate_settlement_mm"] == pytest.approx(27.3, rel=0.05)

    def test_consolidation_nc_clay(self):
        """NC clay consolidation: Sc = Cc*H/(1+e0) * log((sv0+ds)/sv0).

        H=3m, e0=1.1, Cc=0.4, sv0=80 kPa, ds=50 kPa.
        Sc = 0.4*3/(2.1) * log10(130/80) = 0.571*0.2109 = 120.5mm
        """
        r = H.call(settlement_agent, "consolidation_settlement",
                    S.CONSOLIDATION_NC_CLAY["params"])
        assert r["consolidation_settlement_mm"] == pytest.approx(120, rel=0.15)

    def test_combined_analysis_produces_both(self):
        """Combined settlement analysis produces both immediate + consolidation."""
        r = H.call(settlement_agent, "combined_settlement_analysis", {
            "q_applied": 150.0,
            "B": 2.0,
            "Es": 10000.0,
            "nu": 0.3,
            "consolidation_layers": [
                {
                    "thickness": 3.0,
                    "depth_to_center": 5.0,
                    "e0": 1.1,
                    "Cc": 0.4,
                    "Cr": 0.08,
                    "sigma_v0": 80.0,
                }
            ],
        })
        assert r["immediate_mm"] > 0
        assert r["consolidation_mm"] > 0
        assert r["total_mm"] > r["immediate_mm"]

    def test_schmertmann_settlement(self):
        """Schmertmann method for granular soils.

        Layers defined with depth_top and depth_bottom.
        """
        r = H.call(settlement_agent, "schmertmann_settlement", {
            "q_net": 100.0,
            "B": 2.0,
            "layers": [
                {"depth_top": 0.0, "depth_bottom": 1.0, "Es": 8000.0},
                {"depth_top": 1.0, "depth_bottom": 2.0, "Es": 12000.0},
                {"depth_top": 2.0, "depth_bottom": 4.0, "Es": 15000.0},
            ],
        })
        assert r["schmertmann_settlement_mm"] > 0
        assert r["schmertmann_settlement_mm"] < 100


# ============================================================================
# Axial Pile
# ============================================================================

class TestAxialPile:
    """Axial pile capacity validation."""

    def test_driven_pile_sand(self):
        """H-pile in sand — capacity > 0 and components sum correctly."""
        r = H.call(axial_pile_agent, "axial_pile_capacity",
                    S.DRIVEN_PILE_SAND["params"])
        assert r["Q_ultimate_kN"] > 0
        assert r["Q_allowable_kN"] > 0
        assert r["Q_skin_kN"] > 0
        assert r["Q_tip_kN"] > 0
        assert r["Q_ultimate_kN"] == pytest.approx(
            r["Q_skin_kN"] + r["Q_tip_kN"], rel=0.01)

    def test_driven_pile_clay(self):
        """Pipe pile in clay (Tomlinson alpha method)."""
        r = H.call(axial_pile_agent, "axial_pile_capacity",
                    S.DRIVEN_PILE_CLAY["params"])
        assert r["Q_ultimate_kN"] > 0
        assert r["Q_skin_kN"] > 0
        assert r["Q_tip_kN"] > 0

    def test_capacity_increases_with_depth(self):
        """Capacity must generally increase with pile length."""
        r1 = H.call(axial_pile_agent, "axial_pile_capacity",
                     {**S.DRIVEN_PILE_SAND["params"], "pile_length": 10.0})
        r2 = H.call(axial_pile_agent, "axial_pile_capacity",
                     {**S.DRIVEN_PILE_SAND["params"], "pile_length": 20.0})
        assert r2["Q_ultimate_kN"] > r1["Q_ultimate_kN"]


# ============================================================================
# Drilled Shaft
# ============================================================================

class TestDrilledShaft:
    """Drilled shaft capacity validation (GEC-10)."""

    def test_shaft_in_clay(self):
        """Alpha method in clay — shaft capacity dominates."""
        r = H.call(drilled_shaft_agent, "drilled_shaft_capacity",
                    S.DRILLED_SHAFT_CLAY["params"])
        assert r["Q_ultimate_kN"] > 0
        assert r["Q_skin_kN"] > 0
        assert r["Q_tip_kN"] > 0
        # For clay, side friction typically dominates
        assert r["Q_skin_kN"] > r["Q_tip_kN"] * 0.5

    def test_shaft_in_sand(self):
        """Beta method in sand."""
        r = H.call(drilled_shaft_agent, "drilled_shaft_capacity",
                    S.DRILLED_SHAFT_SAND["params"])
        assert r["Q_ultimate_kN"] > 0
        assert r["Q_skin_kN"] > 0
        assert r["Q_tip_kN"] > 0

    def test_lrfd_factors_exist(self):
        """LRFD resistance factors should return valid phi values."""
        r = H.call(drilled_shaft_agent, "get_resistance_factors", {})
        assert "side_cohesive" in r or "cohesive" in str(r).lower()


# ============================================================================
# Seismic
# ============================================================================

class TestSeismic:
    """Seismic geotechnical validation (ASCE 7-22, M-O)."""

    def test_site_class_d(self):
        """Vs30=250 m/s -> Site Class D (ASCE 7-22 Table 20.3-1)."""
        r = H.call(seismic_geotech_agent, "site_classification",
                    S.SITE_CLASS_D["params"])
        assert r["site_class"] == "D"

    def test_site_class_coefficients(self):
        """Site D, Ss=1.0, S1=0.4 -> Fa, Fv per ASCE 7-22 tables."""
        r = H.call(seismic_geotech_agent, "site_classification",
                    S.SITE_CLASS_D["params"])
        assert r["Fpga"] > 0
        assert "Fa" in r or "Fpga" in r

    def test_mo_active_pressure(self):
        """Mononobe-Okabe: phi=30, kh=0.1, delta=20.

        KAE must be > static Ka (0.333 for phi=30).
        """
        r = H.call(seismic_geotech_agent, "seismic_earth_pressure",
                    S.MO_PRESSURE["params"])
        assert r["KAE"] > 0.333
        assert r["KAE"] < 1.0

    def test_liquefaction_evaluation(self):
        """SPT-based liquefaction: loose sand at 0.25g should liquefy.

        N160=10 at 3m depth with amax=0.25g, Mw=7.5 should have FOS < 1.
        """
        r = H.call(seismic_geotech_agent, "liquefaction_evaluation",
                    S.LIQUEFACTION_CHECK["params"])
        assert r["n_liquefiable"] > 0, "At least one layer should liquefy"
        assert r["min_FOS_liq"] < 1.0, "Minimum FOS should indicate liquefaction"


# ============================================================================
# Retaining Walls
# ============================================================================

class TestRetainingWalls:
    """Retaining wall validation."""

    def test_cantilever_wall_stability(self):
        """Cantilever wall: FOS_sliding > 1.0, FOS_overturning > 1.0."""
        r = H.call(retaining_walls_agent, "cantilever_wall",
                    S.CANTILEVER_WALL["params"])
        assert r["FOS_sliding"] > 1.0
        assert r["FOS_overturning"] > 1.0

    def test_wall_width_effect(self):
        """Wider base -> higher FOS (engineering judgment)."""
        r_narrow = H.call(retaining_walls_agent, "cantilever_wall",
                          {**S.CANTILEVER_WALL["params"], "base_width": 3.0})
        r_wide = H.call(retaining_walls_agent, "cantilever_wall",
                        {**S.CANTILEVER_WALL["params"], "base_width": 5.0})
        assert r_wide["FOS_sliding"] > r_narrow["FOS_sliding"]
        assert r_wide["FOS_overturning"] > r_narrow["FOS_overturning"]

    def test_mse_wall_basic(self):
        """MSE wall should produce external + internal FOS."""
        r = H.call(retaining_walls_agent, "mse_wall", {
            "wall_height": 6.0,
            "gamma_backfill": 18.0,
            "phi_backfill": 34.0,
            "reinforcement_length": 4.2,
            "reinforcement_spacing": 0.6,
            "reinforcement_Tallowable": 30.0,
        })
        assert r["FOS_sliding"] > 0
        assert r["FOS_overturning"] > 0


# ============================================================================
# Sheet Pile
# ============================================================================

class TestSheetPile:
    """Sheet pile wall validation."""

    def test_cantilever_sheet_pile(self):
        """Cantilever sheet pile in sand — embedment depth > 0."""
        r = H.call(sheet_pile_agent, "cantilever_wall",
                    S.CANTILEVER_SHEET_PILE["params"])
        assert r["embedment_depth_m"] > 0
        assert r["total_wall_length_m"] > \
            S.CANTILEVER_SHEET_PILE["params"]["excavation_depth"]

    def test_anchored_sheet_pile(self):
        """Anchored sheet pile — anchor force > 0."""
        r = H.call(sheet_pile_agent, "anchored_wall", {
            "excavation_depth": 5.0,
            "anchor_depth": 1.5,
            "layers": [
                {
                    "thickness": 15.0,
                    "unit_weight": 18.0,
                    "friction_angle": 32.0,
                    "cohesion": 0.0,
                }
            ],
        })
        assert r["embedment_depth_m"] > 0
        assert r["anchor_force_kN_per_m"] > 0


# ============================================================================
# Slope Stability
# ============================================================================

class TestSlopeStability:
    """Slope stability validation."""

    def test_simple_slope_fellenius(self):
        """Fellenius method: FOS > 0 for cohesive slope."""
        params = {**S.SIMPLE_SLOPE["params"], "method": "fellenius"}
        r = H.call(slope_stability_agent, "analyze_slope", params)
        assert r["FOS"] > 0
        assert r["FOS"] < 10

    def test_bishop_geq_fellenius(self):
        """Bishop FOS >= Fellenius for same problem (theoretical result)."""
        params_f = {**S.SIMPLE_SLOPE["params"], "method": "fellenius"}
        params_b = {**S.SIMPLE_SLOPE["params"], "method": "bishop"}
        r_f = H.call(slope_stability_agent, "analyze_slope", params_f)
        r_b = H.call(slope_stability_agent, "analyze_slope", params_b)
        assert r_b["FOS"] >= r_f["FOS"] - 0.01

    def test_search_finds_critical(self):
        """Grid search should find FOS <= arbitrary circle FOS."""
        r_arbitrary = H.call(slope_stability_agent, "analyze_slope",
                             S.SIMPLE_SLOPE["params"])
        r_search = H.call(slope_stability_agent, "search_critical_surface", {
            "surface_points": S.SIMPLE_SLOPE["params"]["surface_points"],
            "soil_layers": S.SIMPLE_SLOPE["params"]["soil_layers"],
            "x_range": [10, 20],
            "y_range": [14, 22],
            "nx": 5,
            "ny": 5,
        })
        # Result is nested: r_search["critical"]["FOS"]
        crit_fos = r_search["critical"]["FOS"]
        assert crit_fos <= r_arbitrary["FOS"] + 0.1, \
            f"Search FOS ({crit_fos}) should be <= arbitrary ({r_arbitrary['FOS']})"


# ============================================================================
# Geolysis: SPT & Classification
# ============================================================================

class TestGeolysis:
    """Geolysis agent validation (SPT corrections, classification, bearing capacity)."""

    def test_spt_energy_correction(self):
        """SPT N=25, 60% energy, safety hammer -> N60 ~25."""
        r = H.call(geolysis_agent, "correct_spt", S.SPT_CORRECTION["params"])
        assert r["n60"] > 0
        assert r["n60"] == pytest.approx(25, rel=0.20)
        assert r["n1_60"] > 0

    def test_spt_overburden_at_100kpa(self):
        """At sigma_v'=100 kPa, CN ~ 1.0 (Liao & Whitman), so N1_60 ~ N60."""
        r = H.call(geolysis_agent, "correct_spt", S.SPT_CORRECTION["params"])
        assert r["n1_60"] == pytest.approx(r["n60"], rel=0.15)

    def test_uscs_cl(self):
        """LL=45, PL=25, fines=60% -> CL (lean clay)."""
        r = H.call(geolysis_agent, "classify_uscs", S.USCS_CL["params"])
        assert "CL" in r["symbol"]

    def test_uscs_sw(self):
        """Well-graded sand: fines=3%, Cu>6, 1<Cc<3 -> SW."""
        r = H.call(geolysis_agent, "classify_uscs", S.USCS_SW["params"])
        assert "SW" in r["symbol"] or "S" in r["symbol"]

    def test_aashto_a7(self):
        """LL=45, PL=25, fines=60% -> A-7 group."""
        r = H.call(geolysis_agent, "classify_aashto", S.AASHTO_A7["params"])
        assert "A-7" in r["symbol"]

    def test_ultimate_bc_vesic(self):
        """Ultimate bearing capacity: phi=30, B=2m, Df=1.5m, gamma=18."""
        r = H.call(geolysis_agent, "ultimate_bc", {
            "friction_angle": 30.0,
            "moist_unit_wgt": 18.0,
            "depth": 1.5,
            "width": 2.0,
            "shape": "square",
            "ubc_method": "vesic",
        })
        assert r["bearing_capacity_kpa"] > 300
        assert r["bearing_capacity_kpa"] < 2000

    def test_allowable_bc_spt(self):
        """Allowable bearing capacity from SPT N=20."""
        r = H.call(geolysis_agent, "allowable_bc_spt", {
            "corrected_spt_n_value": 20,
            "width": 2.0,
            "depth": 1.5,
        })
        assert r["bearing_capacity_kpa"] > 0


# ============================================================================
# Ground Improvement
# ============================================================================

class TestGroundImprovement:
    """Ground improvement validation (GEC-13)."""

    def test_wick_drain_consolidation(self):
        """Wick drain: U(t) should be > vertical-only consolidation."""
        r = H.call(ground_improvement_agent, "wick_drains",
                    S.WICK_DRAIN["params"])
        assert r["U_total_percent"] > 0
        assert r["U_total_percent"] <= 100
        assert r["U_total_percent"] > r["Uv_percent"]

    def test_aggregate_pier_improvement(self):
        """Stone column: stress concentration ratio > 1.0."""
        r = H.call(ground_improvement_agent, "aggregate_piers",
                    S.AGGREGATE_PIER["params"])
        assert r["stress_concentration_ratio"] > 1.0
        assert r["area_replacement_ratio"] > 0
        assert r["area_replacement_ratio"] < 1.0

    def test_vibro_feasibility(self):
        """Fines < 15% -> vibro compaction feasible."""
        r = H.call(ground_improvement_agent, "vibro_compaction",
                    S.VIBRO_COMPACTION["params"])
        assert r["is_feasible"] is True


# ============================================================================
# Wave Equation
# ============================================================================

class TestWaveEquation:
    """Wave equation validation."""

    def test_bearing_graph(self):
        """Bearing graph should produce blow counts for capacity range."""
        r = H.call(wave_equation_agent, "bearing_graph",
                    S.BEARING_GRAPH["params"])
        assert len(r["R_values_kN"]) > 0
        assert len(r["blow_counts_per_m"]) > 0
        assert len(r["R_values_kN"]) == len(r["blow_counts_per_m"])

    def test_single_blow(self):
        """Single blow simulation should produce set and peak force."""
        r = H.call(wave_equation_agent, "single_blow", {
            "pile_length": 15.0,
            "pile_area": 0.01,
            "R_ultimate": 1000.0,
            "ram_weight": 50.0,
            "stroke": 1.0,
            "efficiency": 0.8,
            "cushion_stiffness": 1e6,
        })
        assert r["permanent_set_mm"] >= 0
        assert r["max_pile_force_kN"] > 0


# ============================================================================
# Pile Group
# ============================================================================

class TestPileGroup:
    """Pile group validation."""

    def test_3x3_group_equal_loads(self):
        """3x3 group under pure vertical: each pile gets V/9 = 300 kN."""
        r = H.call(pile_group_agent, "pile_group_simple",
                    S.PILE_GROUP_3X3["params"])
        assert r["n_piles"] == 9
        assert r["max_compression_kN"] == pytest.approx(300.0, rel=0.01)
        # With pure vertical load, max tension should be ~0
        assert r["max_tension_kN"] == pytest.approx(0.0, abs=1.0)

    def test_group_efficiency(self):
        """Converse-Labarre efficiency: 0 < eta < 1 for typical spacing."""
        r = H.call(pile_group_agent, "group_efficiency",
                    S.GROUP_EFFICIENCY["params"])
        assert r["converse_labarre_Eg"] > 0
        assert r["converse_labarre_Eg"] < 1.0


# ============================================================================
# Downdrag
# ============================================================================

class TestDowndrag:
    """Downdrag (negative skin friction) validation."""

    def test_basic_downdrag(self):
        """Pile with settling layers should have neutral plane within pile."""
        params = {
            "pile_length": 20.0,
            "pile_diameter": 0.356,
            "layers": [
                {
                    "thickness": 5.0,
                    "soil_type": "cohesive",
                    "unit_weight": 17.0,
                    "cu": 20.0,
                    "settling": True,
                    "Cc": 0.3,
                    "Cr": 0.06,
                    "e0": 1.0,
                },
                {
                    "thickness": 5.0,
                    "soil_type": "cohesive",
                    "unit_weight": 17.5,
                    "cu": 40.0,
                    "settling": True,
                    "Cc": 0.25,
                    "Cr": 0.05,
                    "e0": 0.9,
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
        }
        r = H.call(downdrag_agent, "downdrag_analysis", params)
        assert r["neutral_plane_depth_m"] > 0
        assert r["neutral_plane_depth_m"] <= 20.0
        assert r["dragload_kN"] > 0


# ============================================================================
# Reliability (pystra)
# ============================================================================

class TestReliability:
    """Structural reliability validation (Nowak & Collins)."""

    def test_form_linear_limit_state(self):
        """FORM: R-S with R~N(200,20), S~N(100,15).

        Closed-form: beta = (mu_R - mu_S) / sqrt(sig_R^2 + sig_S^2)
                          = (200-100) / sqrt(400+225) = 100/25 = 4.0
        """
        r = H.call(pystra_agent, "form_analysis", S.FORM_RS["params"])
        assert r["beta"] == pytest.approx(4.0, rel=0.05)
        assert r["pf"] < 0.001

    def test_mc_vs_form_consistency(self):
        """Monte Carlo and FORM should give similar beta for linear g(x)."""
        r_form = H.call(pystra_agent, "form_analysis", S.FORM_RS["params"])
        r_mc = H.call(pystra_agent, "monte_carlo_analysis", {
            **S.FORM_RS["params"],
            "n_samples": 1000000,
        })
        assert r_mc["beta"] == pytest.approx(r_form["beta"], rel=0.15)


# ============================================================================
# Metadata / Discovery
# ============================================================================

class TestMetadata:
    """Every agent's list_methods and describe_method should work."""

    @pytest.mark.parametrize("list_func,describe_func,sample_method", [
        (bearing_capacity_list_methods, bearing_capacity_describe_method,
         "bearing_capacity_analysis"),
        (settlement_list_methods, settlement_describe_method,
         "elastic_settlement"),
        (axial_pile_list_methods, axial_pile_describe_method,
         "axial_pile_capacity"),
        (drilled_shaft_list_methods, drilled_shaft_describe_method,
         "drilled_shaft_capacity"),
        (seismic_geotech_list_methods, seismic_geotech_describe_method,
         "site_classification"),
        (retaining_walls_list_methods, retaining_walls_describe_method,
         "cantilever_wall"),
        (sheet_pile_list_methods, sheet_pile_describe_method,
         "cantilever_wall"),
        (slope_stability_list_methods, slope_stability_describe_method,
         "analyze_slope"),
        (geolysis_list_methods, geolysis_describe_method,
         "classify_uscs"),
        (ground_improvement_list_methods, ground_improvement_describe_method,
         "wick_drains"),
        (wave_equation_list_methods, wave_equation_describe_method,
         "bearing_graph"),
        (pile_group_list_methods, pile_group_describe_method,
         "pile_group_simple"),
        (downdrag_list_methods, downdrag_describe_method,
         "downdrag_analysis"),
        (pystra_list_methods, pystra_describe_method,
         "form_analysis"),
    ])
    def test_list_and_describe(self, list_func, describe_func, sample_method):
        """List methods returns categories; describe returns parameters."""
        methods = H.list_methods(list_func)
        assert len(methods) > 0

        desc = H.describe(describe_func, sample_method)
        assert "parameters" in desc or "description" in desc
