#!/usr/bin/env python
"""Regenerate worked_examples.json by running representative calls through the
live dispatch layer, so the manual's worked examples quote real output from the
shipped code. Run this before build_manual.py when the analysis modules change.

Usage:  python gen_worked_examples.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
WORKTREE = HERE.parent.parent
sys.path.insert(0, str(WORKTREE))

from funhouse_agent.dispatch import call_agent  # noqa: E402

# (agent, method, parameters) — realistic, self-consistent inputs.
EXAMPLES = [
    ("bearing_capacity", "bearing_capacity_analysis", dict(
        width=2.0, length=10.0, depth=1.5, shape="rectangular", cohesion=10.0,
        friction_angle=30.0, unit_weight=18.0, gwt_depth=5.0, factor_of_safety=3.0)),
    ("settlement", "consolidation_settlement", dict(
        layers=[dict(thickness=4.0, depth_to_center=2.0, e0=0.8, Cc=0.25, Cr=0.03, sigma_v0=36.0)],
        delta_sigma=50.0)),
    ("axial_pile", "axial_pile_capacity", dict(
        pile_type="pipe_closed", diameter=0.4, wall_thickness=0.012, pile_length=15.0,
        layers=[dict(thickness=8.0, soil_type="cohesionless", unit_weight=18.0, friction_angle=32.0),
                dict(thickness=12.0, soil_type="cohesionless", unit_weight=19.0, friction_angle=36.0)],
        gwt_depth=5.0, factor_of_safety=2.5)),
    ("drilled_shaft", "drilled_shaft_capacity", dict(
        diameter=1.0, shaft_length=15.0,
        layers=[dict(thickness=8.0, soil_type="cohesive", unit_weight=18.0, cu=75.0),
                dict(thickness=10.0, soil_type="cohesionless", unit_weight=19.0, phi=34.0, N60=30)],
        gwt_depth=4.0)),
    ("lateral_pile", "lateral_pile_analysis", dict(
        pile_type="pipe", pile_diameter=0.61, pile_length=15.0,
        layers=[dict(top=0.0, bottom=15.0, model="SandReese", phi=34.0, gamma=19.0, k=24430.0)],
        Vt=200.0, head_condition="free")),
    ("pile_group", "pile_group_6dof", dict(
        n_rows=3, n_cols=3, spacing=1.2, axial_stiffness=200000.0, lateral_stiffness=20000.0,
        Vz=6000.0, Vx=300.0, Mx=800.0)),
    ("downdrag", "downdrag_analysis", dict(
        pile_length=18.0, pile_diameter=0.4,
        layers=[dict(thickness=6.0, soil_type="cohesive", unit_weight=17.0, cu=30.0, Cc=0.3, e0=1.0, settling=True, beta=0.3),
                dict(thickness=14.0, soil_type="cohesionless", unit_weight=19.0, phi=34.0, settling=False)],
        Q_dead=400.0, fill_thickness=2.0, fill_unit_weight=19.0, gwt_depth=3.0)),
    ("wave_equation", "bearing_graph", dict(
        pile_length=20.0, pile_area=0.012, ram_weight=45.0, stroke=1.5, efficiency=0.8,
        R_min=500.0, R_max=3000.0, R_step=500.0)),
    ("sheet_pile", "cantilever_wall", dict(
        excavation_depth=4.0,
        layers=[dict(thickness=15.0, unit_weight=18.0, friction_angle=32.0, cohesion=0.0)],
        surcharge=10.0, FOS_passive=1.5, embedment_increase=1.2)),
    ("soe", "braced_excavation", dict(
        excavation_depth=8.0,
        layers=[dict(thickness=20.0, unit_weight=18.0, friction_angle=30.0, cohesion=5.0)],
        supports=[dict(depth=2.0, support_type="strut", spacing=4.0),
                  dict(depth=5.0, support_type="strut", spacing=4.0)],
        surcharge=12.0, gwt_depth=10.0, excavation_width=12.0)),
    ("retaining_walls", "cantilever_wall", dict(
        wall_height=6.0, gamma_backfill=18.0, phi_backfill=32.0, surcharge=12.0, q_allowable=250.0)),
    ("ground_improvement", "aggregate_piers", dict(
        column_diameter=0.76, spacing=1.5, pattern="triangular", E_column=40000.0,
        E_soil=8000.0, S_unreinforced=100.0, q_unreinforced=150.0)),
    ("slope_stability", "search_critical_surface", dict(
        surface_points=[[0, 0], [10, 0], [25, 10], [45, 10]],
        soil_layers=[dict(name="clayey sand", top_elevation=10.0, bottom_elevation=-10.0,
                          gamma=19.0, phi=28.0, c_prime=12.0, analysis_mode="drained")],
        surface_type="entry_exit", method="bishop",
        x_entry_range=[10, 20], x_exit_range=[25, 40], nx=6, ny=6, n_slices=25)),
    ("fem2d", "fem2d_slope_srm", dict(
        surface_points=[[0, 10], [10, 10], [20, 0], [40, 0]],
        soil_layers=[dict(name="embankment", bottom_elevation=-10.0, E=1.0e5, nu=0.3,
                          c=10.0, phi=20.0, psi=0.0, gamma=20.0)],
        nx=40, ny=20, srf_tol=0.05, element_type="t6")),
    ("seismic_geotech", "site_classification", dict(vs30=350.0, Ss=1.2, S1=0.5)),
    ("seismic_geotech", "seismic_earth_pressure", dict(
        phi=30.0, kh=0.2, kv=0.0, delta=20.0, gamma=18.0, H=6.0, include_passive=True)),
    ("liquefaction", "liquefaction_analysis", dict(
        depth=[3.0, 6.0, 9.0, 12.0], N160=[12, 15, 10, 18], FC=[10, 15, 8, 20],
        gamma=[18.0, 18.0, 19.0, 19.0], gwt_depth=2.0, amax_g=0.3, m_w=7.0)),
    ("reliability", "monte_carlo", dict(
        variables={"R": {"mean": 500.0, "cov": 0.20, "dist": "lognormal"},
                   "S": {"mean": 300.0, "cov": 0.15, "dist": "normal"}},
        g_expression="R - S", convention="margin", n=50000, seed=42)),
]


def main():
    out = {}
    ok = 0
    for mod, meth, params in EXAMPLES:
        res = call_agent(mod, meth, params)
        out[f"{mod}.{meth}"] = {"params": params, "result": res}
        good = not (isinstance(res, dict) and "error" in res)
        ok += good
        print(f"[{'OK ' if good else 'ERR'}] {mod}.{meth}"
              + ("" if good else f"  -> {res.get('error')}"))
    (HERE / "worked_examples.json").write_text(
        json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nWrote worked_examples.json  ({ok}/{len(EXAMPLES)} succeeded)")


if __name__ == "__main__":
    main()
