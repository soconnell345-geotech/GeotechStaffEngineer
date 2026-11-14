"""GM-140 step 9: the rapid-drawdown rows of the account's section 11,
each stage beside print.

Run from the repository root with the tree to measure on the path:

    python module_work/gm140/step9_section11.py            # this tree
    PYTHONPATH=<a checkout of master> python module_work/gm140/step9_section11.py

It runs on the code before step 9 too (it reads a result's new fields only
where they exist), so the same script gives the "before" and the "after"
columns of `STEP9_CHECK.md`. Every value is in the units of the printed
problem's factor of safety (dimensionless); the inputs are converted from
ft, psf and pcf as the validation tests do.

The rows that are not rapid drawdown do not pass through
`slope_stability/rapid_drawdown.py`; their pinned tests are run unchanged
(see the CHECK file).
"""
import inspect
import math
import sys

import slope_stability
from slope_stability.geometry import SlopeGeometry, SlopeSoilLayer
from slope_stability import rapid_drawdown as rd

FT, PSF, PCF = 0.3048, 0.04788, 0.157087
NEW = "stage_solver" in inspect.signature(rd.rapid_drawdown_fos).parameters


def stages(r):
    s2 = getattr(r, "stage2_fos", None)
    s3 = getattr(r, "stage3_fos", None)
    f = lambda v: "-" if v is None else f"{v:.3f}"
    return f"FOS {f(r.FOS)} (stage 1 {f(r.stage1_fos)}, 2 {f(s2)}, 3 {f(s3)})"


def em_dam(face=((0, 0), (222, 74), (312, 110), (420, 110)), bottom=-20.0):
    return SlopeGeometry(
        surface_points=[(-100 * FT, 0.0)] + [(x * FT, z * FT) for x, z in face],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=bottom * FT,
            gamma=135 * PCF, phi=30.0, c_prime=0.0,
            R_c=1200 * PSF, R_phi=16.0)])


def slide_dam():
    """The section of the #95/#96 validation tests."""
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in
                        ((0, 0), (220, 73), (312, 110), (380, 110))],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=110 * FT, bottom_elevation=-1.0 * FT,
            gamma=135 * PCF, phi=30.0, c_prime=0.0,
            R_c=1200 * PSF, R_phi=16.0)])


CIRCLE = dict(xc=169.5 * FT, yc=210 * FT, radius=210 * FT)
EDGES = [350.2, 330.0, 312.0, 294.5, 267.0, 245.0, 222.0, 200.0, 169.5,
         135.0, 105.0]


def one(geom, pool, method, n=50, **kw):
    return rd.rapid_drawdown_fos(geom, pool * FT, 24 * FT, method=method,
                                 n_slices=n, **CIRCLE, **kw)


def dam_98():
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in
                        ((0, 0), (100, 40), (140, 60), (180, 60))],
        soil_layers=[
            SlopeSoilLayer(name="ClayeySiltySand", top_elevation=60 * FT,
                           bottom_elevation=51 * FT, gamma=128 * PCF,
                           c_prime=240 * PSF, phi=32.7, R_c=650 * PSF,
                           R_phi=13.0),
            SlopeSoilLayer(name="MicaceousSand", top_elevation=51 * FT,
                           bottom_elevation=30 * FT, gamma=123 * PCF,
                           c_prime=220 * PSF, phi=22.5, R_c=450 * PSF,
                           R_phi=11.0),
            SlopeSoilLayer(name="CretaceousClay", top_elevation=30 * FT,
                           bottom_elevation=17 * FT, gamma=124 * PCF,
                           c_prime=180 * PSF, phi=19.0, R_c=180 * PSF,
                           R_phi=13.0),
            SlopeSoilLayer(name="ClayeySandyGravel", top_elevation=17 * FT,
                           bottom_elevation=-2 * FT, gamma=125 * PCF,
                           c_prime=0.0, phi=40.0, R_phi=None)])


def search_98(method):
    return rd.search_rapid_drawdown(
        dam_98(), 47 * FT, 15 * FT, method=method, surface_type="circular",
        nx=6, ny=5, x_range=(30 * FT, 140 * FT), y_range=(60 * FT, 150 * FT),
        n_slices=20)


def pilarcitos(face):
    return SlopeGeometry(
        surface_points=[(x * FT, z * FT) for x, z in face],
        soil_layers=[SlopeSoilLayer(
            name="fill", top_elevation=78 * FT, bottom_elevation=-8 * FT,
            gamma=135 * PCF, phi=45.0, c_prime=0.0,
            R_c=60 * PSF, R_phi=23.0)])


def deep(geom):
    x0, x1 = geom.x_range
    w = x1 - x0
    return dict(x_entry_range=(x0, x0 + 0.4 * w),
                x_exit_range=(x0 + 0.6 * w, x1))


def search_pilarcitos_true(method, windows=True):
    g = pilarcitos(((0, 0), (116, 58), (176, 78), (260, 78)))
    return rd.search_rapid_drawdown(
        g, 72 * FT, 37 * FT, method=method, surface_type="circular",
        nx=7, ny=6, x_range=(10.0, 54.0), y_range=(24.5, 75.0), n_slices=20,
        **(deep(g) if windows else {}))


def search_pilarcitos_25(method, infinite_slope=True):
    run = 2.5 * 78.0
    g = pilarcitos(((0, 0), (run, 78.0), (run + 80, 78.0)))
    return rd.search_rapid_drawdown(
        g, 72 * FT, 37 * FT, method=method, surface_type="circular",
        nx=7, ny=6, x_range=(0.2 * run * FT, run * FT),
        y_range=(80 * FT, 90 * FT + 2 * 78.0 * FT), n_slices=20,
        infinite_slope=infinite_slope, **deep(g))


def main():
    print("tree:", slope_stability.__file__, "| step 9 code:", NEW)
    rows = []

    def row(name, printed, fn):
        try:
            got = fn()
        except Exception as exc:        # a row must not stop the table
            got = f"error: {type(exc).__name__}: {exc}"[:150]
        rows.append((name, got, printed))
        print(f"{name} | {got} | {printed}", flush=True)

    # The manual's example, EM G-4 (pool El. 103 to El. 24)
    for m in ("duncan_3stage", "lowe_karafiath", "corps_2stage"):
        row(f"EM example, {m}, Spencer, 50 slices",
            {"duncan_3stage": "2.20 / 1.52 / 1.44 (Bishop; pp. G-15 to G-20)",
             "lowe_karafiath": "stage 2: 1.52 (p. G-18)",
             "corps_2stage": "3.49 / 1.35 by Modified Swedish at 19.4 deg"
             }[m], lambda m=m: stages(one(em_dam(), 103, m)))
    if NEW:
        ex = dict(n=12, extra_boundaries=[x * FT for x in EDGES])
        row("EM example, duncan_3stage, Bishop, the manual's 12 slices",
            "2.20 / 1.52 / 1.44",
            lambda: stages(one(em_dam(), 103, "duncan_3stage",
                               stage_solver="bishop", **ex)))
        row("EM example, corps_2stage, force equilibrium at 19.4 deg, "
            "50 slices", "3.49 / 1.35 (pp. G-9 to G-15)",
            lambda: stages(one(em_dam(), 103, "corps_2stage",
                               stage_solver=("force", 19.4))))
        row("EM example, corps_2stage, force equilibrium at 19.4 deg, "
            "the manual's 12 slices", "3.49 / 1.35",
            lambda: stages(one(em_dam(), 103, "corps_2stage",
                               stage_solver=("force", 19.4), **ex)))
    # Slide2 #95 and #96 (the tests' section, pool El. 110 as Slide2 prints)
    row("B7 #95 (V-037), corps_2stage, pool 110", "Slide2 1.347",
        lambda: stages(one(slide_dam(), 110, "corps_2stage")))
    if NEW:
        row("B7 #95, corps_2stage, force equilibrium at 19.4 deg, pool 110",
            "Slide2 1.347",
            lambda: stages(one(slide_dam(), 110, "corps_2stage",
                               stage_solver=("force", 19.4))))
        row("B7 #95, corps_2stage, force equilibrium at 19.4 deg, pool 103",
            "Slide2 1.347; EM 1.35",
            lambda: stages(one(slide_dam(), 103, "corps_2stage",
                               stage_solver=("force", 19.4))))
    row("B7 #96 (V-038), duncan_3stage, pool 110", "Slide2 1.443",
        lambda: stages(one(slide_dam(), 110, "duncan_3stage")))
    row("B7 #96, duncan_3stage, pool 103", "Slide2 1.443; EM 1.44",
        lambda: stages(one(slide_dam(), 103, "duncan_3stage")))
    row("V-048 #95/#96 dam, lowe_karafiath, pool 110",
        "none printed for this dam (EM stage 2: 1.52 at pool 103)",
        lambda: stages(one(slide_dam(), 110, "lowe_karafiath")))
    # B15: the planar family on the #95/#96 section
    def b15():
        r = rd.search_rapid_drawdown(
            slide_dam(), 110 * FT, 24 * FT, method="corps_2stage",
            surface_type="circular", nx=4, ny=3,
            x_range=(140 * FT, 200 * FT), y_range=(190 * FT, 240 * FT),
            n_slices=20)
        c = r.search.candidates
        return (f"{r.FOS:.3f} ({r.search.governing_surface}); families "
                + ", ".join(f"{k} {v:.3f}" for k, v in sorted(c.items())))
    row("B15 V-041 #95/#96 section, Corps search", "infinite slope 0.772", b15)
    # #98, search on the recovered section
    for m, p in (("corps_2stage", "0.931"), ("duncan_3stage", "1.039"),
                 ("lowe_karafiath", "1.075")):
        row(f"B7 #98 (V-041/V-048), {m} search", p,
            lambda m=m: f"{search_98(m).FOS:.3f}")
    # Pilarcitos (V-052): the 2.5:1 section and the true section
    for m, p in (("corps_2stage", "0.823"), ("lowe_karafiath", "1.047"),
                 ("duncan_3stage", "1.043")):
        row(f"V-052 Pilarcitos 2.5:1, {m}, deep windows", p,
            lambda m=m: f"{search_pilarcitos_25(m).FOS:.3f}")
        row(f"V-052 Pilarcitos 2.5:1, {m}, deep windows, no planes", p,
            lambda m=m: f"{search_pilarcitos_25(m, False).FOS:.3f}")
        row(f"B13 / V-052b Pilarcitos true section, {m}, deep windows", p,
            lambda m=m: f"{search_pilarcitos_true(m).FOS:.3f}")
        row(f"V-052b Pilarcitos true section, {m}, automatic windows", p,
            lambda m=m: f"{search_pilarcitos_true(m, False).FOS:.3f}")
    print("\n| Benchmark | Module | Printed |\n|---|---|---|")
    for name, got, printed in rows:
        print(f"| {name} | {got} | {printed} |")


if __name__ == "__main__":
    sys.exit(main())
