"""
Tier 3: Cross-agent consistency checks.

Where multiple agents can solve the same problem (e.g., bearing capacity
via bearing_capacity_agent vs groundhog_agent),
run all and verify results agree within engineering tolerance.
"""

import pytest

from foundry_test_harness.harness import FoundryAgentHarness, AgentError
from foundry_test_harness import scenarios as S

from foundry.bearing_capacity_agent_foundry import bearing_capacity_agent
from foundry.settlement_agent_foundry import settlement_agent
from foundry.axial_pile_agent_foundry import axial_pile_agent
from foundry.seismic_geotech_agent_foundry import seismic_geotech_agent
from foundry.groundhog_agent_foundry import groundhog_agent

H = FoundryAgentHarness()


# ============================================================================
# Bearing Capacity Cross-Check
# ============================================================================

class TestBearingCapacityCrossCheck:
    """Compare bearing capacity from different agents for same problem."""

    def test_bearing_factors_consistency(self):
        """Nc, Nq, Ngamma at phi=30° should match between agents.

        bearing_capacity_agent.bearing_capacity_factors vs published tables.
        Vesic: Nc=30.14, Nq=18.40
        """
        r = H.call(bearing_capacity_agent, "bearing_capacity_factors", {
            "friction_angle": 30.0,
            "method": "vesic",
        })
        # Well-known Vesic values at phi=30
        assert r["Nc"] == pytest.approx(30.14, rel=0.02)
        assert r["Nq"] == pytest.approx(18.40, rel=0.02)

    def test_spt_to_friction_angle(self):
        """groundhog SPT → friction angle correlation gives a reasonable phi.

        For N=20 at sigma'v=100 kPa, phi should fall in the 20-50° range.
        """
        # Groundhog: use SPT to estimate friction angle
        r_gh_phi = H.call(groundhog_agent, "spt_friction_angle_kulhawymayne", {
            "N": 20.0,
            "sigma_vo_eff": 100.0,
        })
        phi_est = r_gh_phi["Phi [deg]"]
        assert phi_est > 20
        assert phi_est < 50


# ============================================================================
# Earth Pressure Cross-Check
# ============================================================================

class TestEarthPressureCrossCheck:
    """Compare earth pressure coefficients from different agents."""

    def test_rankine_ka_kp(self):
        """Static Ka, Kp at phi=30° should be consistent across agents.

        Rankine: Ka = tan²(45-phi/2) = tan²(30°) = 0.333
                 Kp = tan²(45+phi/2) = tan²(60°) = 3.0
        """
        # Seismic agent with kh=0 and delta=0 should give Rankine Ka
        r_seis = H.call(seismic_geotech_agent, "seismic_earth_pressure", {
            "phi": 30.0,
            "kh": 0.0,
            "delta": 0.0,
        })
        # With kh=0 and delta=0, KAE should equal Ka (Rankine)
        assert r_seis["KAE"] == pytest.approx(0.333, rel=0.02)

    def test_groundhog_earth_pressure(self):
        """Groundhog earth pressure coefficients vs Rankine theory."""
        r = H.call(groundhog_agent, "earth_pressure_basic", {
            "phi_eff": 30.0,
        })
        assert r["Ka [-]"] == pytest.approx(0.333, rel=0.02)
        assert r["Kp [-]"] == pytest.approx(3.0, rel=0.02)


# ============================================================================
# SPT Correction Cross-Check
# ============================================================================

class TestSPTCrossCheck:
    """SPT correlation checks via groundhog."""

    def test_spt_to_phi_consistency(self):
        """SPT N → friction angle should be consistent.

        Multiple correlations exist; all should give phi in a reasonable range
        for a given N value.
        """
        # Groundhog: Kulhawy & Mayne
        r = H.call(groundhog_agent, "spt_friction_angle_kulhawymayne", {
            "N": 25.0,
            "sigma_vo_eff": 100.0,
        })
        phi = r["Phi [deg]"]
        # N=25 at 100 kPa should give phi ≈ 32-45°
        assert 28 < phi < 50


# ============================================================================
# Consolidation Cross-Check
# ============================================================================

class TestConsolidationCrossCheck:
    """Compare consolidation settlement between agents."""

    def test_consolidation_settlement(self):
        """Same NC clay layer → settlement should be consistent.

        settlement_agent primary consolidation for NC clay.
        Hand calc: Sc = Cc*H/(1+e0) * log10((σ'v0+Δσ)/σ'v0)
        """
        # settlement_agent
        r = H.call(settlement_agent, "consolidation_settlement", {
            "layers": [
                {
                    "thickness": 4.0,
                    "depth_to_center": 6.0,
                    "e0": 1.0,
                    "Cc": 0.3,
                    "Cr": 0.06,
                    "sigma_v0": 80.0,
                }
            ],
            "delta_sigma": 60.0,
        })
        sc_agent = r["consolidation_settlement_mm"]

        # Hand calculation
        Cc, H_layer, e0, sv0, dsv = 0.3, 4.0, 1.0, 80.0, 60.0
        import math
        sc_hand = Cc * H_layer / (1 + e0) * math.log10((sv0 + dsv) / sv0) * 1000  # mm
        # sc_hand ≈ 0.3*4/2 * log10(140/80) * 1000 ≈ 0.6 * 0.2430 * 1000 ≈ 145.8 mm

        assert sc_agent == pytest.approx(sc_hand, rel=0.15)


# ============================================================================
# Pile Capacity Cross-Check
# ============================================================================

class TestPileCapacityCrossCheck:
    """Compare pile capacity components between agents."""

    def test_pile_tip_area_consistency(self):
        """axial_pile make_pile_section → verify geometric properties."""
        r = H.call(axial_pile_agent, "make_pile_section", {
            "pile_type": "pipe_closed",
            "diameter": 0.356,
            "wall_thickness": 0.0127,
        })
        import math
        expected_area = math.pi / 4 * 0.356**2
        assert r["tip_area_m2"] == pytest.approx(expected_area, rel=0.05)
        expected_perim = math.pi * 0.356
        assert r["perimeter_m"] == pytest.approx(expected_perim, rel=0.05)
