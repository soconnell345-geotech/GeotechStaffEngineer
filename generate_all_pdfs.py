"""
Generate PDF calculation packages for all 13 analysis modules.

Creates a sample analysis for each module, runs it, and generates
a professional Mathcad-style PDF calc package via pdflatex.

Output: sample_pdfs/<module_name>.pdf
"""

import os
import sys
import time
import traceback
from pathlib import Path

# Ensure we're running from the project root
PROJECT_ROOT = Path(__file__).resolve().parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_DIR = PROJECT_ROOT / "sample_pdfs"
OUTPUT_DIR.mkdir(exist_ok=True)

# Common metadata
PROJECT_NAME = "Sample Project"
ENGINEER = "S. OConnell, PE"
COMPANY = "Geotech Associates"


def generate_bearing_capacity():
    """Bearing capacity: square footing on sand."""
    from bearing_capacity.footing import Footing
    from bearing_capacity.soil_profile import SoilLayer, BearingSoilProfile
    from bearing_capacity.capacity import BearingCapacityAnalysis
    from calc_package import generate_calc_package

    footing = Footing(width=2.0, length=2.0, depth=1.5, shape="square")
    layer1 = SoilLayer(cohesion=0.0, friction_angle=32.0, unit_weight=18.5)
    soil = BearingSoilProfile(layer1=layer1, gwt_depth=3.0)

    analysis = BearingCapacityAnalysis(
        footing=footing, soil=soil,
        factor_of_safety=3.0,
        ngamma_method="vesic",
        factor_method="vesic",
    )
    result = analysis.compute()

    pdf_path = generate_calc_package(
        module="bearing_capacity",
        result=result,
        analysis=analysis,
        project_name=PROJECT_NAME,
        project_number="BC-2026-001",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "bearing_capacity.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_lateral_pile():
    """Lateral pile: pipe pile in soft clay."""
    from lateral_pile import Pile, LateralPileAnalysis
    from lateral_pile.soil import SoilLayer
    from lateral_pile.py_curves import SoftClayMatlock

    from calc_package import generate_calc_package

    pile = Pile(length=15.0, diameter=0.61, thickness=0.0127, E=200e6)
    layers = [
        SoilLayer(
            top=0.0, bottom=15.0,
            py_model=SoftClayMatlock(c=50.0, gamma=17.0, eps50=0.02, J=0.5),
            description="Soft clay",
        ),
    ]

    analysis = LateralPileAnalysis(pile, layers)
    result = analysis.solve(Vt=100.0, Mt=0.0, Q=0.0, head_condition="free")

    pdf_path = generate_calc_package(
        module="lateral_pile",
        result=result,
        analysis=analysis,
        project_name=PROJECT_NAME,
        project_number="LP-2026-002",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "lateral_pile.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_slope_stability():
    """Slope stability: simple embankment with Bishop method."""
    from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
    from slope_stability.analysis import analyze_slope
    from calc_package import generate_calc_package

    soil_layers = [
        SlopeSoilLayer(
            name="Fill",
            top_elevation=10.0, bottom_elevation=0.0,
            gamma=19.0, phi=28.0, c_prime=5.0,
            analysis_mode="drained",
        ),
    ]
    surface_points = [
        (0.0, 5.0), (10.0, 5.0), (20.0, 10.0), (30.0, 10.0), (40.0, 10.0),
    ]

    geom = SlopeGeometry(
        surface_points=surface_points,
        soil_layers=soil_layers,
    )

    result = analyze_slope(
        geom, xc=20.0, yc=18.0, radius=13.0,
        method="bishop", n_slices=30, FOS_required=1.5,
        include_slice_data=True,
    )

    analysis_dict = {"geom": geom}
    pdf_path = generate_calc_package(
        module="slope_stability",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="SS-2026-003",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "slope_stability.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_settlement():
    """Settlement: consolidation of clay under footing load."""
    from settlement import SettlementAnalysis, ConsolidationLayer
    from calc_package import generate_calc_package

    consol_layers = [
        ConsolidationLayer(
            thickness=3.0, depth_to_center=3.0,
            e0=0.9, Cc=0.3, Cr=0.05,
            sigma_v0=50.0, sigma_p=60.0,
            description="Soft clay",
        ),
        ConsolidationLayer(
            thickness=3.0, depth_to_center=6.0,
            e0=0.8, Cc=0.25, Cr=0.04,
            sigma_v0=80.0, sigma_p=100.0,
            description="Medium clay",
        ),
    ]

    analysis = SettlementAnalysis(
        q_applied=120.0,
        q_overburden=30.0,
        B=3.0, L=3.0,
        footing_shape="square",
        stress_method="2:1",
        immediate_method="elastic",
        Es_immediate=15000.0,
        nu=0.3,
        consolidation_layers=consol_layers,
        cv=3.0,
        Hdr=3.0,
        drainage="double",
    )
    result = analysis.compute()

    pdf_path = generate_calc_package(
        module="settlement",
        result=result,
        analysis=analysis,
        project_name=PROJECT_NAME,
        project_number="SE-2026-004",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "settlement.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_axial_pile():
    """Axial pile: H-pile in layered sand/clay."""
    from axial_pile import AxialPileAnalysis, AxialSoilLayer, AxialSoilProfile, PileSection
    from calc_package import generate_calc_package

    pile = PileSection(
        name="HP 14x73",
        pile_type="h_pile",
        area=0.0138,       # m2
        perimeter=1.63,     # m
        tip_area=0.0903,    # m2 (box area)
        width=0.3607,       # m (flange width)
        depth=0.3521,       # m (depth)
    )

    layers = [
        AxialSoilLayer(
            soil_type="cohesionless",
            thickness=5.0, unit_weight=18.0,
            friction_angle=30.0, description="Medium sand",
        ),
        AxialSoilLayer(
            soil_type="cohesive",
            thickness=5.0, unit_weight=17.0,
            cohesion=60.0, description="Stiff clay",
        ),
        AxialSoilLayer(
            soil_type="cohesionless",
            thickness=10.0, unit_weight=19.5,
            friction_angle=35.0, description="Dense sand",
        ),
    ]
    soil = AxialSoilProfile(layers=layers, gwt_depth=2.0)

    analysis = AxialPileAnalysis(
        pile=pile, soil=soil,
        pile_length=18.0,
        factor_of_safety=2.5,
        method="auto",
    )
    result = analysis.compute()

    pdf_path = generate_calc_package(
        module="axial_pile",
        result=result,
        analysis=analysis,
        project_name=PROJECT_NAME,
        project_number="AP-2026-005",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "axial_pile.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_drilled_shaft():
    """Drilled shaft: cohesive soil with GEC-10 alpha method."""
    from drilled_shaft import DrillShaft, ShaftSoilLayer, ShaftSoilProfile, DrillShaftAnalysis
    from calc_package import generate_calc_package

    shaft = DrillShaft(
        diameter=1.2,
        length=20.0,
        concrete_fc=28000.0,
    )

    layers = [
        ShaftSoilLayer(
            soil_type="cohesive",
            thickness=8.0, unit_weight=17.5,
            cu=50.0, description="Soft to medium clay",
        ),
        ShaftSoilLayer(
            soil_type="cohesive",
            thickness=7.0, unit_weight=18.5,
            cu=100.0, description="Stiff clay",
        ),
        ShaftSoilLayer(
            soil_type="cohesionless",
            thickness=10.0, unit_weight=19.0,
            phi=35.0, N60=30, description="Dense sand",
        ),
    ]
    soil = ShaftSoilProfile(layers=layers, gwt_depth=3.0)

    analysis = DrillShaftAnalysis(
        shaft=shaft, soil=soil,
        factor_of_safety=2.5,
    )
    result = analysis.compute()

    pdf_path = generate_calc_package(
        module="drilled_shaft",
        result=result,
        analysis=analysis,
        project_name=PROJECT_NAME,
        project_number="DS-2026-006",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "drilled_shaft.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_downdrag():
    """Downdrag: pile through settling fill over dense sand."""
    from downdrag import DowndragAnalysis, DowndragSoilLayer, DowndragSoilProfile
    from calc_package import generate_calc_package

    layers = [
        DowndragSoilLayer(
            soil_type="cohesive",
            thickness=5.0, unit_weight=17.0,
            cu=25.0, settling=True, alpha=1.0,
            Cc=0.3, e0=1.0, sigma_p=40.0,
            description="Settling fill / soft clay",
        ),
        DowndragSoilLayer(
            soil_type="cohesive",
            thickness=5.0, unit_weight=18.0,
            cu=60.0, settling=False,
            description="Stiff clay (not settling)",
        ),
        DowndragSoilLayer(
            soil_type="cohesionless",
            thickness=10.0, unit_weight=19.5,
            phi=35.0, settling=False,
            description="Dense sand bearing layer",
        ),
    ]
    soil = DowndragSoilProfile(layers=layers, gwt_depth=1.0)

    analysis = DowndragAnalysis(
        pile_length=18.0,
        pile_diameter=0.61,
        pile_E=200e6,
        Q_dead=500.0,
        soil=soil,
        fill_thickness=2.0,
        fill_unit_weight=20.0,
        structural_capacity=2000.0,
    )
    result = analysis.compute()

    pdf_path = generate_calc_package(
        module="downdrag",
        result=result,
        analysis=analysis,
        project_name=PROJECT_NAME,
        project_number="DD-2026-007",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "downdrag.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_seismic_geotech():
    """Seismic geotech: site classification analysis."""
    from seismic_geotech import classify_site, site_coefficients
    from calc_package import generate_calc_package

    site_class = classify_site(vs30=270.0)
    result = site_coefficients(site_class, Ss=1.0, S1=0.4)

    analysis_dict = {
        "analysis_type": "site_classification",
        "vs30": 270.0,
        "Ss": 1.0,
        "S1": 0.4,
    }

    pdf_path = generate_calc_package(
        module="seismic_geotech",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="SG-2026-008",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "seismic_geotech.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_retaining_walls():
    """Retaining walls: cantilever wall with Rankine pressures."""
    from retaining_walls import CantileverWallGeometry, analyze_cantilever_wall
    from calc_package import generate_calc_package

    geom = CantileverWallGeometry(
        wall_height=5.0,
        stem_thickness_top=0.3,
        base_thickness=0.6,
        surcharge=10.0,
    )

    result = analyze_cantilever_wall(
        geom=geom,
        gamma_backfill=18.0,
        phi_backfill=30.0,
        c_backfill=0.0,
        phi_foundation=32.0,
        gamma_concrete=24.0,
        FOS_sliding=1.5,
        FOS_overturning=2.0,
        pressure_method="rankine",
    )

    analysis_dict = {
        "wall_type": "cantilever",
        "geom": geom,
        "gamma_backfill": 18.0,
        "phi_backfill": 30.0,
        "pressure_method": "rankine",
        "gamma_concrete": 24.0,
        "FOS_sliding": 1.5,
        "FOS_overturning": 2.0,
    }

    pdf_path = generate_calc_package(
        module="retaining_walls",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="RW-2026-009",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "retaining_walls.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_ground_improvement():
    """Ground improvement: wick drain analysis."""
    from ground_improvement import analyze_wick_drains
    from calc_package import generate_calc_package

    result = analyze_wick_drains(
        spacing=1.5,
        ch=3.0,
        cv=1.0,
        Hdr=8.0,
        time=0.5,
        dw=0.066,
        pattern="triangular",
        smear_ratio=2.0,
        kh_ks_ratio=2.0,
    )

    analysis_dict = {
        "method": "wick_drains",
        "spacing": 1.5,
        "ch": 3.0,
        "cv": 1.0,
        "Hdr": 8.0,
        "time": 0.5,
        "dw": 0.066,
        "pattern": "triangular",
        "smear_ratio": 2.0,
        "kh_ks_ratio": 2.0,
    }

    pdf_path = generate_calc_package(
        module="ground_improvement",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="GI-2026-010",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "ground_improvement.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_wave_equation():
    """Wave equation: bearing graph for Delmag D30-32."""
    from wave_equation import (
        get_hammer, make_cushion_from_properties,
        discretize_pile, generate_bearing_graph,
    )
    from calc_package import generate_calc_package

    hammer = get_hammer("Delmag D30-32")
    cushion = make_cushion_from_properties(
        area=0.0855,          # m2
        thickness=0.075,      # m
        elastic_modulus=3.5e9, # Pa (plywood)
        cor=0.8,
    )
    pile = discretize_pile(
        length=20.0,
        area=0.01,            # m2 (HP pile)
        elastic_modulus=200e9, # Pa (steel)
        segment_length=1.0,
        unit_weight_material=78.5,
    )

    result = generate_bearing_graph(
        hammer=hammer, cushion=cushion, pile=pile,
        skin_fraction=0.5,
        quake_side=0.0025, quake_toe=0.0025,
        damping_side=0.16, damping_toe=0.5,
        R_min=200.0, R_max=2000.0, R_step=200.0,
    )

    analysis_dict = {"hammer": hammer, "cushion": cushion, "pile": pile}

    pdf_path = generate_calc_package(
        module="wave_equation",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="WE-2026-011",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "wave_equation.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_pile_group():
    """Pile group: 3x3 rectangular layout under vertical + moment load."""
    from pile_group import create_rectangular_layout, GroupLoad, analyze_vertical_group_simple
    from calc_package import generate_calc_package

    piles = create_rectangular_layout(
        n_rows=3, n_cols=3,
        spacing_x=1.5, spacing_y=1.5,
    )
    load = GroupLoad(Vz=2700.0, Mx=200.0, My=150.0)

    result = analyze_vertical_group_simple(piles=piles, load=load)

    analysis_dict = {
        "piles": piles,
        "load": load,
        "pile_diameter": 0.3,
        "pile_spacing": 1.5,
    }

    pdf_path = generate_calc_package(
        module="pile_group",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="PG-2026-012",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "pile_group.pdf"),
        format="pdf",
    )
    return pdf_path


def generate_sheet_pile():
    """Sheet pile: cantilever wall in sand."""
    from sheet_pile import WallSoilLayer, analyze_cantilever
    from calc_package import generate_calc_package

    layers = [
        WallSoilLayer(
            thickness=5.0, unit_weight=18.0,
            friction_angle=30.0, cohesion=0.0,
            description="Medium sand",
        ),
        WallSoilLayer(
            thickness=10.0, unit_weight=19.0,
            friction_angle=33.0, cohesion=0.0,
            description="Dense sand",
        ),
    ]

    result = analyze_cantilever(
        soil_layers=layers,
        excavation_depth=3.5,
        surcharge=10.0,
        FOS_passive=1.5,
        pressure_method="rankine",
    )

    analysis_dict = {
        "wall_type": "cantilever",
        "excavation_depth": 3.5,
        "soil_layers": layers,
        "surcharge": 10.0,
        "FOS_passive": 1.5,
        "pressure_method": "rankine",
    }

    pdf_path = generate_calc_package(
        module="sheet_pile",
        result=result,
        analysis=analysis_dict,
        project_name=PROJECT_NAME,
        project_number="SP-2026-013",
        engineer=ENGINEER,
        company=COMPANY,
        output_path=str(OUTPUT_DIR / "sheet_pile.pdf"),
        format="pdf",
    )
    return pdf_path


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------

MODULES = [
    ("bearing_capacity", generate_bearing_capacity),
    ("lateral_pile", generate_lateral_pile),
    ("slope_stability", generate_slope_stability),
    ("settlement", generate_settlement),
    ("axial_pile", generate_axial_pile),
    ("drilled_shaft", generate_drilled_shaft),
    ("downdrag", generate_downdrag),
    ("seismic_geotech", generate_seismic_geotech),
    ("retaining_walls", generate_retaining_walls),
    ("ground_improvement", generate_ground_improvement),
    ("wave_equation", generate_wave_equation),
    ("pile_group", generate_pile_group),
    ("sheet_pile", generate_sheet_pile),
]


def main():
    print("=" * 70)
    print("  PDF Calc Package Generator - All 13 Modules")
    print(f"  Output directory: {OUTPUT_DIR}")
    print("=" * 70)

    successes = []
    failures = []

    for name, func in MODULES:
        print(f"\n--- [{len(successes) + len(failures) + 1}/13] {name} ---")
        t0 = time.time()
        try:
            pdf_path = func()
            elapsed = time.time() - t0
            size_kb = os.path.getsize(pdf_path) / 1024
            print(f"  OK  {pdf_path}  ({size_kb:.0f} KB, {elapsed:.1f}s)")
            successes.append((name, pdf_path, size_kb, elapsed))
        except Exception as exc:
            elapsed = time.time() - t0
            print(f"  FAIL  {name}: {exc}")
            traceback.print_exc()
            failures.append((name, str(exc), elapsed))

    # Summary
    print("\n" + "=" * 70)
    print("  SUMMARY")
    print("=" * 70)
    print(f"  Successes: {len(successes)} / 13")
    for name, path, size_kb, elapsed in successes:
        print(f"    [OK]   {name:25s}  {size_kb:6.0f} KB  {elapsed:5.1f}s")

    if failures:
        print(f"\n  Failures: {len(failures)} / 13")
        for name, err, elapsed in failures:
            print(f"    [FAIL] {name:25s}  {elapsed:5.1f}s  {err[:80]}")

    print("=" * 70)
    return len(failures) == 0


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
