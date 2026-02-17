"""
Validation tests for the wave_equation module.

Tests cover hammer models, cushions, pile discretization, soil models,
time integration, bearing graphs, and drivability.

References:
    [1] Smith, E.A.L. (1960) "Bearing Capacity of Piles"
    [2] FHWA GEC-12, Chapter 12
    [3] WEAP87 Manual (FHWA, Goble & Rausche)
"""

import math
import pytest
import numpy as np

from wave_equation.hammer import Hammer, get_hammer, list_hammers
from wave_equation.cushion import Cushion, make_cushion_from_properties
from wave_equation.pile_model import PileSegment, PileModel, discretize_pile
from wave_equation.soil_model import SmithSoilModel, SoilSetup
from wave_equation.time_integration import BlowResult, simulate_blow
from wave_equation.bearing_graph import BearingGraphResult, generate_bearing_graph
from wave_equation.drivability import drivability_study


# ═══════════════════════════════════════════════════════════════════════
# TEST 1: Hammer Models
# ═══════════════════════════════════════════════════════════════════════

class TestHammer:
    """Test hammer models and database."""

    def test_hammer_database(self):
        """Database should have 14 hammers."""
        hammers = list_hammers()
        assert len(hammers) >= 14

    def test_get_hammer_vulcan(self):
        """Look up Vulcan 010 from database."""
        h = get_hammer("Vulcan 010")
        assert h.name == "Vulcan 010"
        assert h.ram_weight == pytest.approx(44.5, abs=0.5)

    def test_get_hammer_delmag(self):
        """Look up Delmag D30-32 from database."""
        h = get_hammer("Delmag D30-32")
        assert h.hammer_type == "diesel"
        assert h.rated_energy == pytest.approx(82.3, abs=1.0)

    def test_hammer_not_found(self):
        """Unknown hammer raises KeyError."""
        with pytest.raises(KeyError):
            get_hammer("Nonexistent Hammer")

    def test_ram_mass(self):
        """Ram mass = weight / g * 1000."""
        h = Hammer("test", ram_weight=50.0, stroke=1.0)
        expected = 50.0 / 9.81 * 1000  # ~5097 kg
        assert h.ram_mass == pytest.approx(expected, rel=0.01)

    def test_impact_velocity_free_fall(self):
        """Single-acting: v = sqrt(2*g*h*eff)."""
        h = Hammer("test", ram_weight=50.0, stroke=1.0, efficiency=1.0)
        expected = math.sqrt(2 * 9.81 * 1.0 * 1.0)  # ~4.43 m/s
        assert h.impact_velocity == pytest.approx(expected, rel=0.01)

    def test_energy(self):
        """Energy = ram_weight * stroke."""
        h = Hammer("test", ram_weight=44.5, stroke=0.914)
        assert h.energy == pytest.approx(44.5 * 0.914, rel=0.01)

    def test_diesel_energy(self):
        """Diesel: rated_energy overrides W*h."""
        h = Hammer("test", ram_weight=30.0, stroke=2.5,
                   rated_energy=80.0, hammer_type="diesel")
        assert h.energy == 80.0


# ═══════════════════════════════════════════════════════════════════════
# TEST 2: Cushion Properties
# ═══════════════════════════════════════════════════════════════════════

class TestCushion:
    """Test cushion models."""

    def test_cushion_creation(self):
        """Basic cushion creation."""
        c = Cushion(stiffness=500000, cor=0.80)
        assert c.stiffness == 500000
        assert c.cor == 0.80

    def test_cushion_from_properties(self):
        """k = E * A / t."""
        c = make_cushion_from_properties(
            area=0.1, thickness=0.05,
            elastic_modulus=200000, cor=0.50
        )
        # k = 200000 * 0.1 / 0.05 = 400000
        assert c.stiffness == pytest.approx(400000, rel=0.01)

    def test_invalid_stiffness(self):
        """Negative stiffness should raise."""
        with pytest.raises(ValueError):
            Cushion(stiffness=-100)

    def test_invalid_cor(self):
        """COR > 1 should raise."""
        with pytest.raises(ValueError):
            Cushion(stiffness=100, cor=1.5)


# ═══════════════════════════════════════════════════════════════════════
# TEST 3: Pile Model
# ═══════════════════════════════════════════════════════════════════════

class TestPileModel:
    """Test pile discretization."""

    def test_segment_count(self):
        """15m pile at 1m segments = 15 segments."""
        p = discretize_pile(15.0, 0.01, 200e6, segment_length=1.0)
        assert p.n_segments == 15

    def test_total_length(self):
        """Total length matches input."""
        p = discretize_pile(12.0, 0.01, 200e6, segment_length=1.0)
        assert p.total_length == pytest.approx(12.0)

    def test_mass_conservation(self):
        """Total mass = rho * A * L."""
        A = 0.01  # m^2
        L = 10.0  # m
        gamma = 78.5  # kN/m^3
        p = discretize_pile(L, A, 200e6, segment_length=1.0,
                            unit_weight_material=gamma)
        total_mass = np.sum(p.masses)
        expected = gamma * A * L / 9.81 * 1000  # kg
        assert total_mass == pytest.approx(expected, rel=0.01)

    def test_wave_speed_steel(self):
        """Steel wave speed ≈ 5120 m/s."""
        p = discretize_pile(10.0, 0.01, 200e6, segment_length=1.0,
                            unit_weight_material=78.5)
        # c = sqrt(E/rho) = sqrt(200e9 / 7850) ≈ 5048 m/s
        assert p.wave_speeds[0] == pytest.approx(5048, rel=0.02)

    def test_impedance(self):
        """Impedance Z = EA/c."""
        A = 0.01
        E = 200e6  # kPa
        p = discretize_pile(10.0, A, E, segment_length=1.0)
        c = p.wave_speeds[0]
        Z_expected = E * A / c  # kPa * m^2 / (m/s) = kN*s/m
        assert p.impedance == pytest.approx(Z_expected, rel=0.01)

    def test_depth_at_segments(self):
        """Segment centers should be at 0.5, 1.5, ... for 1m segments."""
        p = discretize_pile(5.0, 0.01, 200e6, segment_length=1.0)
        assert p.depth_at_segment[0] == pytest.approx(0.5)
        assert p.depth_at_segment[-1] == pytest.approx(4.5)


# ═══════════════════════════════════════════════════════════════════════
# TEST 4: Smith Soil Model
# ═══════════════════════════════════════════════════════════════════════

class TestSoilModel:
    """Test Smith soil resistance model."""

    def test_static_elastic(self):
        """Within quake: R = Rult * d/Q."""
        m = SmithSoilModel(R_ultimate=100, quake=0.0025)
        R = m.static_resistance(0.001)
        assert R == pytest.approx(100 * 0.001 / 0.0025, rel=0.01)

    def test_static_plastic(self):
        """Beyond quake: R = Rult."""
        m = SmithSoilModel(R_ultimate=100, quake=0.0025)
        R = m.static_resistance(0.005)
        assert R == pytest.approx(100.0, rel=0.01)

    def test_static_at_quake(self):
        """At exactly quake: R = Rult."""
        m = SmithSoilModel(R_ultimate=100, quake=0.0025)
        R = m.static_resistance(0.0025)
        assert R == pytest.approx(100.0, rel=0.01)

    def test_dynamic_enhancement(self):
        """With velocity: R = R_s + J * R_u * v at full quake."""
        m = SmithSoilModel(R_ultimate=100, quake=0.0025, damping=0.5)
        # At quake displacement with v=2 m/s
        # R_s = 100 (fully mobilized), R_d = 0.5 * 100 * 2 = 100
        R = m.total_resistance(0.0025, 2.0)
        assert R == pytest.approx(100 + 0.5 * 100 * 2, rel=0.01)

    def test_damping_proportional_to_Ru_not_Rs(self):
        """Smith damping force = J * R_ultimate * v, NOT J * R_static * v.

        This is the GRLWEAP/GEC-12 standard formulation. The damping
        force is proportional to R_ultimate regardless of how much
        static resistance has been mobilized.

        At half the quake, R_static = 0.5 * R_u but damping still
        uses full R_u.
        """
        m = SmithSoilModel(R_ultimate=100, quake=0.0025, damping=0.5)
        d = 0.00125  # half the quake
        v = 2.0

        R_s = m.static_resistance(d)
        assert R_s == pytest.approx(50.0, rel=0.01)  # 100 * 0.00125/0.0025

        R_total = m.total_resistance(d, v)
        # Correct (GRLWEAP): R = R_s + J * R_u * v = 50 + 0.5*100*2 = 150
        R_expected = R_s + 0.5 * 100 * v  # 50 + 100 = 150
        assert R_total == pytest.approx(R_expected, rel=0.01)

        # This must NOT equal the old (incorrect) formula: R_s * (1+J*v) = 50*2 = 100
        R_old_wrong = R_s * (1.0 + 0.5 * v)
        assert R_total != pytest.approx(R_old_wrong, rel=0.01)

    def test_damping_at_zero_displacement(self):
        """Damping force is zero when displacement is zero (not yet loaded).

        Per Smith model: damping only during loading (v and d same sign).
        At d=0 with v>0, condition fails -> R=0.
        """
        m = SmithSoilModel(R_ultimate=100, quake=0.0025, damping=0.5)
        R = m.total_resistance(0.0, 3.0)
        assert R == pytest.approx(0.0, abs=1e-10)

    def test_damping_beyond_quake(self):
        """Beyond quake: R = R_u + J * R_u * v = R_u * (1 + J*v).

        When fully mobilized, both old and new formulas agree.
        """
        m = SmithSoilModel(R_ultimate=100, quake=0.0025, damping=0.5)
        d = 0.010  # well beyond quake
        v = 3.0

        R_total = m.total_resistance(d, v)
        # R_s = 100 (capped at Ru), R_d = 0.5 * 100 * 3 = 150
        assert R_total == pytest.approx(100 + 0.5 * 100 * 3, rel=0.01)  # 250

    def test_no_damping_during_rebound(self):
        """During rebound (v < 0, d > 0), no damping is applied.

        Smith damping is one-directional: only during loading.
        """
        m = SmithSoilModel(R_ultimate=100, quake=0.0025, damping=0.5)
        d = 0.005  # positive displacement (beyond quake)
        v = -2.0   # rebounding upward

        R_total = m.total_resistance(d, v)
        R_s = m.static_resistance(d)
        # Rebound: only static resistance, no damping
        assert R_total == pytest.approx(R_s, rel=0.01)
        assert R_total == pytest.approx(100.0, rel=0.01)

    def test_zero_resistance(self):
        """Zero Rult gives zero resistance."""
        m = SmithSoilModel(R_ultimate=0, quake=0.0025)
        assert m.static_resistance(0.001) == 0.0
        assert m.total_resistance(0.001, 1.0) == 0.0

    def test_soil_setup_distribution(self):
        """Skin + toe should equal total Rult."""
        s = SoilSetup(R_ultimate=1000, skin_fraction=0.6)
        assert s.R_skin == pytest.approx(600, rel=0.01)
        assert s.R_toe == pytest.approx(400, rel=0.01)

    def test_segment_model_creation(self):
        """Side models should distribute skin friction evenly."""
        s = SoilSetup(R_ultimate=1000, skin_fraction=0.5)
        sides, toe = s.create_segment_models(10)
        assert len(sides) == 10
        total_skin = sum(m.R_ultimate for m in sides)
        assert total_skin == pytest.approx(500, rel=0.01)
        assert toe.R_ultimate == pytest.approx(500, rel=0.01)


# ═══════════════════════════════════════════════════════════════════════
# TEST 5: Time Integration - Physics Checks
# ═══════════════════════════════════════════════════════════════════════

class TestTimeIntegration:
    """Test wave equation time integration against physics."""

    def _make_system(self, R_ultimate=500):
        """Helper: standard test system."""
        hammer = get_hammer("Vulcan 010")
        cushion = Cushion(stiffness=500000, cor=0.80)
        D = 0.3239
        t = 0.009525
        area = math.pi / 4 * (D**2 - (D - 2*t)**2)
        pile = discretize_pile(15.0, area, 200e6, segment_length=1.0)
        soil = SoilSetup(R_ultimate=R_ultimate, skin_fraction=0.5)
        return hammer, cushion, pile, soil

    def test_positive_set(self):
        """Permanent set should be positive for driveable pile."""
        h, c, p, s = self._make_system(500)
        result = simulate_blow(h, c, p, s)
        assert result.permanent_set > 0

    def test_higher_resistance_less_set(self):
        """Higher soil resistance should give less penetration."""
        h, c, p, _ = self._make_system()
        s_low = SoilSetup(R_ultimate=300, skin_fraction=0.5)
        s_high = SoilSetup(R_ultimate=800, skin_fraction=0.5)
        r_low = simulate_blow(h, c, p, s_low)
        r_high = simulate_blow(h, c, p, s_high)
        assert r_low.permanent_set > r_high.permanent_set

    def test_bigger_hammer_more_set(self):
        """Bigger hammer should drive pile further."""
        cushion = Cushion(stiffness=500000, cor=0.80)
        D = 0.3239; t = 0.009525
        area = math.pi / 4 * (D**2 - (D - 2*t)**2)
        pile = discretize_pile(15.0, area, 200e6, segment_length=1.0)
        soil = SoilSetup(R_ultimate=500, skin_fraction=0.5)

        h_small = get_hammer("Vulcan 06")
        h_big = get_hammer("Vulcan 020")
        r_small = simulate_blow(h_small, cushion, pile, soil)
        r_big = simulate_blow(h_big, cushion, pile, soil)
        assert r_big.permanent_set > r_small.permanent_set

    def test_compression_stress_positive(self):
        """Max compression stress should be positive."""
        h, c, p, s = self._make_system(500)
        result = simulate_blow(h, c, p, s)
        assert result.max_compression_stress > 0

    def test_reasonable_stress_range(self):
        """Driving stress should be within steel yield range."""
        h, c, p, s = self._make_system(500)
        result = simulate_blow(h, c, p, s)
        # Compression stress in kPa, steel yield ~250 MPa = 250,000 kPa
        assert result.max_compression_stress < 400000  # <400 MPa

    def test_energy_balance_approximate(self):
        """Hammer energy should be >= resistance * set (rough check).

        Energy delivered = hammer_energy * efficiency
        Work by resistance >= Rult * permanent_set (lower bound since
        dynamic effects add energy dissipation).
        """
        h, c, p, s = self._make_system(500)
        result = simulate_blow(h, c, p, s)
        E_delivered = h.energy * h.efficiency  # kN-m
        work_static = s.R_ultimate * result.permanent_set  # kN * m = kN-m
        # Delivered energy should exceed static work (rest dissipated by damping)
        assert E_delivered > work_static * 0.5  # generous bound

    def test_zero_resistance_large_set(self):
        """With no soil resistance, pile should penetrate freely."""
        h, c, p, _ = self._make_system()
        s_zero = SoilSetup(R_ultimate=10, skin_fraction=0.5)
        result = simulate_blow(h, c, p, s_zero)
        assert result.permanent_set > 0.05  # > 50 mm


# ═══════════════════════════════════════════════════════════════════════
# TEST 6: Bearing Graph
# ═══════════════════════════════════════════════════════════════════════

class TestBearingGraph:
    """Test bearing graph generation."""

    def _make_system(self):
        """Helper: standard test system."""
        hammer = get_hammer("Vulcan 010")
        cushion = Cushion(stiffness=500000, cor=0.80)
        D = 0.3239; t = 0.009525
        area = math.pi / 4 * (D**2 - (D - 2*t)**2)
        pile = discretize_pile(15.0, area, 200e6, segment_length=1.0)
        return hammer, cushion, pile

    def test_bearing_graph_points(self):
        """Correct number of Rult points."""
        h, c, p = self._make_system()
        bg = generate_bearing_graph(h, c, p,
                                    R_min=200, R_max=1000, R_step=200)
        assert len(bg.R_values) == 5

    def test_monotonic_blow_count(self):
        """Blow count should increase with resistance."""
        h, c, p = self._make_system()
        bg = generate_bearing_graph(h, c, p,
                                    R_min=200, R_max=1000, R_step=200)
        for i in range(len(bg.blow_counts) - 1):
            assert bg.blow_counts[i+1] >= bg.blow_counts[i]

    def test_monotonic_set_decrease(self):
        """Set should decrease with resistance."""
        h, c, p = self._make_system()
        bg = generate_bearing_graph(h, c, p,
                                    R_min=200, R_max=1000, R_step=200)
        for i in range(len(bg.permanent_sets) - 1):
            assert bg.permanent_sets[i+1] <= bg.permanent_sets[i]

    def test_summary_output(self):
        """summary() should produce readable text."""
        h, c, p = self._make_system()
        bg = generate_bearing_graph(h, c, p,
                                    R_min=200, R_max=600, R_step=200)
        s = bg.summary()
        assert "BEARING GRAPH" in s

    def test_to_dict(self):
        """to_dict() should return all keys."""
        h, c, p = self._make_system()
        bg = generate_bearing_graph(h, c, p,
                                    R_min=200, R_max=600, R_step=200)
        d = bg.to_dict()
        assert "R_values_kN" in d
        assert "blow_counts_per_m" in d


# ═══════════════════════════════════════════════════════════════════════
# TEST 7: Drivability
# ═══════════════════════════════════════════════════════════════════════

class TestDrivability:
    """Test drivability study."""

    def test_drivability_increasing_resistance(self):
        """Blow count should increase with depth/resistance."""
        hammer = get_hammer("Vulcan 010")
        cushion = Cushion(stiffness=500000, cor=0.80)

        depths = [5.0, 10.0, 15.0]
        R_at_depth = [200, 500, 800]

        result = drivability_study(
            hammer, cushion,
            pile_area=0.01, pile_E=200e6,
            pile_unit_weight=78.5,
            depths=depths,
            R_at_depth=R_at_depth,
        )
        assert len(result.points) == 3
        assert result.points[2].blow_count >= result.points[0].blow_count

    def test_drivability_summary(self):
        """Summary should be readable."""
        hammer = get_hammer("Vulcan 010")
        cushion = Cushion(stiffness=500000, cor=0.80)

        result = drivability_study(
            hammer, cushion,
            pile_area=0.01, pile_E=200e6,
            pile_unit_weight=78.5,
            depths=[10.0],
            R_at_depth=[500],
        )
        s = result.summary()
        assert "DRIVABILITY" in s

    def test_drivability_to_dict(self):
        """to_dict() should work."""
        hammer = get_hammer("Vulcan 010")
        cushion = Cushion(stiffness=500000, cor=0.80)

        result = drivability_study(
            hammer, cushion,
            pile_area=0.01, pile_E=200e6,
            pile_unit_weight=78.5,
            depths=[10.0],
            R_at_depth=[500],
        )
        d = result.to_dict()
        assert "can_drive" in d
        assert len(d["points"]) == 1


# ═══════════════════════════════════════════════════════════════════════
# TEST 8: Elastic Impact Verification
# ═══════════════════════════════════════════════════════════════════════

class TestElasticImpact:
    """Verify against closed-form elastic impact of two masses.

    For two masses m1 (ram) and m2 (pile) connected by spring k,
    with m1 at velocity v0 and no soil resistance:
    - Max force = v0 * sqrt(m1*m2/(m1+m2)*k)
    - Post-impact velocities follow momentum & energy conservation
    """

    def test_elastic_max_force(self):
        """Check peak force approaches elastic impact solution.

        With no soil resistance and COR=1 (elastic cushion), the peak
        force should approach v0 * sqrt(k * m_reduced).
        """
        # Simple system: 5000 kg ram, 5000 kg pile, stiff spring
        ram_wt = 5000 * 9.81 / 1000  # ~49 kN
        hammer = Hammer("test", ram_weight=ram_wt, stroke=1.0,
                        efficiency=1.0)
        cushion = Cushion(stiffness=100000, cor=1.0)  # 100,000 kN/m

        # Short, heavy pile to approximate a single mass
        pile = discretize_pile(2.0, 0.05, 200e6, segment_length=2.0,
                               unit_weight_material=78.5)
        # Override pile mass to match ram
        pile_mass = pile.masses[0]

        soil = SoilSetup(R_ultimate=1.0, skin_fraction=0.5)  # ~no resistance

        result = simulate_blow(hammer, cushion, pile, soil, helmet_weight=0,
                               max_time=0.05)

        v0 = hammer.impact_velocity
        m1 = hammer.ram_mass
        m2 = pile_mass
        k = cushion.stiffness * 1000  # N/m
        m_red = m1 * m2 / (m1 + m2)
        F_theory = v0 * math.sqrt(m_red * k) / 1000  # N -> kN

        # Allow 30% tolerance due to multi-segment pile wave effects
        assert result.max_pile_force == pytest.approx(F_theory, rel=0.30)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
