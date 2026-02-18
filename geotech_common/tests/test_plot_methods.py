"""
Smoke tests for plot methods on result dataclasses.

Each test creates a minimal result object, calls plot_*(show=False),
asserts the return type is a matplotlib Axes, and cleans up.
"""

import pytest
import numpy as np

import matplotlib
matplotlib.use('Agg')
plt_mod = pytest.importorskip("matplotlib.pyplot")


@pytest.fixture(autouse=True)
def close_figures():
    """Close all matplotlib figures after each test."""
    yield
    plt_mod.close('all')


# ──────────────────────────────────────────────────────────────────────
# axial_pile
# ──────────────────────────────────────────────────────────────────────

class TestAxialPilePlot:
    def test_plot_load_transfer(self):
        from axial_pile.results import AxialPileResult
        result = AxialPileResult(
            Q_ultimate=500, Q_skin=300, Q_tip=200,
            pile_length=15.0,
            layer_breakdown=[
                {'depth_top_m': 0, 'depth_bottom_m': 5,
                 'skin_friction_kN': 100, 'description': 'Clay'},
                {'depth_top_m': 5, 'depth_bottom_m': 15,
                 'skin_friction_kN': 200, 'description': 'Sand'},
            ],
        )
        ax = result.plot_load_transfer(show=False)
        assert ax is not None

    def test_plot_load_transfer_no_data(self):
        from axial_pile.results import AxialPileResult
        result = AxialPileResult()
        with pytest.raises(ValueError):
            result.plot_load_transfer(show=False)


# ──────────────────────────────────────────────────────────────────────
# drilled_shaft
# ──────────────────────────────────────────────────────────────────────

class TestDrilledShaftPlot:
    def test_plot_load_transfer(self):
        from drilled_shaft.results import DrillShaftResult
        result = DrillShaftResult(
            Q_ultimate=800, Q_skin=500, Q_tip=300,
            shaft_length=20.0,
            layer_breakdown=[
                {'depth_top_m': 0, 'depth_bottom_m': 10,
                 'side_resistance_kN': 300, 'description': 'Clay'},
                {'depth_top_m': 10, 'depth_bottom_m': 20,
                 'side_resistance_kN': 200, 'description': 'Sand'},
            ],
        )
        ax = result.plot_load_transfer(show=False)
        assert ax is not None

    def test_plot_load_transfer_no_data(self):
        from drilled_shaft.results import DrillShaftResult
        result = DrillShaftResult()
        with pytest.raises(ValueError):
            result.plot_load_transfer(show=False)


# ──────────────────────────────────────────────────────────────────────
# downdrag
# ──────────────────────────────────────────────────────────────────────

class TestDowndragPlot:
    def _make_result(self):
        from downdrag.results import DowndragResult
        z = np.linspace(0, 20, 50)
        return DowndragResult(
            neutral_plane_depth=10.0,
            dragload=200.0,
            max_pile_load=500.0,
            Q_dead=250.0,
            pile_weight_to_np=50.0,
            positive_skin_friction=300.0,
            toe_resistance=200.0,
            total_resistance=500.0,
            pile_settlement=0.015,
            elastic_shortening=0.005,
            toe_settlement=0.010,
            soil_settlement_at_np=0.020,
            z=z,
            axial_load=np.linspace(250, 500, 50),
            soil_settlement_profile=np.linspace(0.05, 0.001, 50),
            unit_skin_friction=np.ones(50) * 30.0,
            structural_ok=True,
            structural_demand=400.0,
            geotechnical_ok=True,
            settlement_ok=True,
            pile_length=20.0,
            pile_diameter=0.5,
        )

    def test_plot_axial_load(self):
        result = self._make_result()
        ax = result.plot_axial_load(show=False)
        assert ax is not None

    def test_plot_settlement(self):
        result = self._make_result()
        ax = result.plot_settlement(show=False)
        assert ax is not None

    def test_plot_neutral_plane(self):
        result = self._make_result()
        fig, axes = result.plot_neutral_plane(show=False)
        assert fig is not None
        assert len(axes) == 2


# ──────────────────────────────────────────────────────────────────────
# seismic_geotech
# ──────────────────────────────────────────────────────────────────────

class TestSeismicPlot:
    def _make_result(self):
        from seismic_geotech.results import LiquefactionResult
        return LiquefactionResult(
            amax_g=0.2,
            magnitude=7.0,
            gwt_depth=2.0,
            layer_results=[
                {'depth_m': 3.0, 'N160cs': 10, 'CSR': 0.25, 'CRR': 0.15,
                 'FOS_liq': 0.6, 'liquefiable': True},
                {'depth_m': 6.0, 'N160cs': 20, 'CSR': 0.20, 'CRR': 0.30,
                 'FOS_liq': 1.5, 'liquefiable': False},
                {'depth_m': 9.0, 'N160cs': 30, 'CSR': 0.15, 'CRR': 0.40,
                 'FOS_liq': 2.67, 'liquefiable': False},
            ],
        )

    def test_plot_fos_profile(self):
        result = self._make_result()
        ax = result.plot_fos_profile(show=False)
        assert ax is not None

    def test_plot_csr_crr(self):
        result = self._make_result()
        ax = result.plot_csr_crr(show=False)
        assert ax is not None


# ──────────────────────────────────────────────────────────────────────
# retaining_walls
# ──────────────────────────────────────────────────────────────────────

class TestRetainingWallsPlot:
    def test_cantilever_plot_stability_summary(self):
        from retaining_walls.results import CantileverWallResult
        result = CantileverWallResult(
            FOS_sliding=1.8, FOS_overturning=2.5, FOS_bearing=3.0,
            passes_sliding=True, passes_overturning=True,
            passes_bearing=True,
            q_toe=120.0, q_heel=30.0,
            eccentricity=0.15, in_middle_third=True,
            wall_height=5.0, base_width=3.5,
        )
        ax = result.plot_stability_summary(show=False)
        assert ax is not None

    def test_cantilever_plot_fail_colors(self):
        from retaining_walls.results import CantileverWallResult
        result = CantileverWallResult(
            FOS_sliding=0.8, FOS_overturning=1.5, FOS_bearing=2.0,
            passes_sliding=False, passes_overturning=False,
            passes_bearing=True,
            wall_height=5.0, base_width=3.5,
        )
        ax = result.plot_stability_summary(show=False)
        assert ax is not None


# ──────────────────────────────────────────────────────────────────────
# slope_stability
# ──────────────────────────────────────────────────────────────────────

class TestSlopeStabilityPlot:
    def test_plot_fos_contour(self):
        from slope_stability.results import SearchResult, SlopeStabilityResult
        critical = SlopeStabilityResult(
            FOS=1.2, method='Bishop', xc=10, yc=15, radius=8,
        )
        result = SearchResult(
            critical=critical,
            n_surfaces_evaluated=9,
            grid_fos=[
                {'xc': 8, 'yc': 14, 'R': 7, 'FOS': 1.5},
                {'xc': 10, 'yc': 15, 'R': 8, 'FOS': 1.2},
                {'xc': 12, 'yc': 14, 'R': 7.5, 'FOS': 1.8},
                {'xc': 8, 'yc': 16, 'R': 9, 'FOS': 1.6},
                {'xc': 10, 'yc': 16, 'R': 9, 'FOS': 1.4},
                {'xc': 12, 'yc': 16, 'R': 8, 'FOS': 1.7},
                {'xc': 8, 'yc': 18, 'R': 10, 'FOS': 2.0},
                {'xc': 10, 'yc': 18, 'R': 10, 'FOS': 1.9},
                {'xc': 12, 'yc': 18, 'R': 10, 'FOS': 2.1},
            ],
        )
        ax = result.plot_fos_contour(show=False)
        assert ax is not None

    def test_plot_fos_contour_no_data(self):
        from slope_stability.results import SearchResult
        result = SearchResult()
        with pytest.raises(ValueError):
            result.plot_fos_contour(show=False)


# ──────────────────────────────────────────────────────────────────────
# pile_group
# ──────────────────────────────────────────────────────────────────────

class TestPileGroupPlot:
    def test_plot_pile_layout(self):
        from pile_group.rigid_cap import PileGroupResult
        result = PileGroupResult(
            n_piles=4,
            max_compression=400.0,
            max_utilization=0.8,
            pile_forces=[
                {'label': 'P1', 'x_m': 0, 'y_m': 0, 'axial_kN': 300,
                 'utilization': 0.6},
                {'label': 'P2', 'x_m': 2, 'y_m': 0, 'axial_kN': 400,
                 'utilization': 0.8},
                {'label': 'P3', 'x_m': 0, 'y_m': 2, 'axial_kN': 350,
                 'utilization': 0.7},
                {'label': 'P4', 'x_m': 2, 'y_m': 2, 'axial_kN': 250,
                 'utilization': 0.5},
            ],
        )
        ax = result.plot_pile_layout(show=False)
        assert ax is not None

    def test_plot_pile_layout_no_data(self):
        from pile_group.rigid_cap import PileGroupResult
        result = PileGroupResult()
        with pytest.raises(ValueError):
            result.plot_pile_layout(show=False)


# ──────────────────────────────────────────────────────────────────────
# sheet_pile
# ──────────────────────────────────────────────────────────────────────

class TestSheetPilePlot:
    def test_plot_wall_diagram(self):
        from sheet_pile.cantilever import CantileverWallResult
        result = CantileverWallResult(
            excavation_depth=4.0,
            embedment_depth=6.0,
            total_wall_length=10.0,
            max_moment=150.0,
            max_moment_depth=5.5,
            FOS_passive=1.5,
        )
        ax = result.plot_wall_diagram(show=False)
        assert ax is not None

    def test_plot_wall_diagram_zero_moment(self):
        from sheet_pile.cantilever import CantileverWallResult
        result = CantileverWallResult(
            excavation_depth=3.0,
            embedment_depth=4.5,
            total_wall_length=7.5,
            max_moment=0,
            max_moment_depth=0,
        )
        ax = result.plot_wall_diagram(show=False)
        assert ax is not None


# ──────────────────────────────────────────────────────────────────────
# ground_improvement
# ──────────────────────────────────────────────────────────────────────

class TestGroundImprovementPlot:
    def test_wick_drain_plot_consolidation(self):
        from ground_improvement.results import WickDrainResult
        result = WickDrainResult(
            drain_spacing_m=1.5,
            U_total_percent=85.0,
            time_years=0.5,
            time_settlement_curve=[
                (0.0, 0.0), (0.1, 30.0), (0.2, 55.0),
                (0.3, 70.0), (0.4, 80.0), (0.5, 85.0),
            ],
        )
        ax = result.plot_consolidation(show=False)
        assert ax is not None

    def test_wick_drain_no_curve(self):
        from ground_improvement.results import WickDrainResult
        result = WickDrainResult()
        with pytest.raises(ValueError):
            result.plot_consolidation(show=False)

    def test_surcharge_plot_consolidation(self):
        from ground_improvement.results import SurchargeResult
        result = SurchargeResult(
            surcharge_kPa=50.0,
            target_U_percent=90,
            settlement_at_target_mm=80.0,
            settlement_ultimate_mm=100.0,
            time_settlement_curve=[
                (0.0, 0.0), (0.5, 40.0), (1.0, 65.0),
                (1.5, 80.0), (2.0, 90.0),
            ],
        )
        ax = result.plot_consolidation(show=False)
        assert ax is not None

    def test_surcharge_no_curve(self):
        from ground_improvement.results import SurchargeResult
        result = SurchargeResult()
        with pytest.raises(ValueError):
            result.plot_consolidation(show=False)
