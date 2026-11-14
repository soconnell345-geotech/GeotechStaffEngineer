"""GM-140 step 8 (supports as printed): the rows of the account's section 11
that carry supports, and the two printed benchmarks this step adds.

``step2_section11.py``, ``step3_section11.py`` and
``circle_ends_section11.py`` are run unchanged for their own rows (#85 with
its support in each method, active and passive; the whole unsupported set).
This script adds: #86 with its five tiebacks, V-040 (the micro-pile row),
V-043's force at three radii (the slice-boundary tie, GM-353), and the
check problems of Duncan, Wright and Brandon (2014) Figs. 8.1 and 8.4.

    PYTHONPATH=. python module_work/gm140/step8_section11.py           # this tree
    PYTHONPATH=<a master worktree> python <this file>                  # master

One line per measured value. It runs on both trees: what exists only after
the step is reported as "not available before step 8". Each geometry is
taken from the test module that pins it, or is the figure's as the brief
gives it (nothing else is re-entered).
"""
import sys
import warnings

warnings.simplefilter("ignore")
sys.path.insert(0, "validation_examples")

from slope_stability.analysis import (analyze_slope,           # noqa: E402
                                      search_critical_surface)
from slope_stability.geometry import (SlopeGeometry,           # noqa: E402
                                      SlopeSoilLayer)
from slope_stability.reinforcement import Geosynthetic         # noqa: E402
from slope_stability.slip_surface import CircularSlipSurface   # noqa: E402
from slope_stability.tests import test_search_reinforcement as tsr  # noqa: E402
import test_published_v040_e6 as v040e6                        # noqa: E402
import test_published_v040_slope as v040                       # noqa: E402
import test_published_v043_slope_reinforced as v043            # noqa: E402

FT, PCF, PSF, LB_FT = 0.3048, 0.157087, 0.04788026, 0.0145939


def f4(v):
    return "none" if v is None else f"{v:.4f}"


def slide86():
    for method in ("bishop", "spencer", "gle"):
        res = search_critical_surface(tsr._slide86(), method=method,
                                      x_entry_range=(0.0, 4.0),
                                      x_exit_range=(30.0, 75.0))
        c = res.critical
        print(f"#86 five tiebacks, {method}: FOS {f4(c.FOS)}, governs "
              f"{res.governing_surface}, supports crossed "
              f"{len(c.reinforcements or [])}")


def v040_rows():
    for m in ("bishop", "spencer"):
        print(f"V-040 no pile, {m}: {f4(v040._fos(m))}")
    for conv in ("active", "passive"):
        for m in ("bishop", "spencer", "fellenius"):
            try:
                v = v040e6._fos(v040e6._pile(conv), m)
                print(f"V-040 pile row, {conv}, {m}: {f4(v)}")
            except ValueError as exc:
                print(f"V-040 pile row, {conv}, {m}: no number "
                      f"({type(exc).__name__})")


def v043_force():
    xc, yc, r = v043._SAND_CIRC
    for dr in (-1e-8, 0.0, 1e-8):
        slip = CircularSlipSurface(xc=xc, yc=yc, radius=r + dr)
        lo, hi = 0.0, 300.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            fs = analyze_slope(v043._geom("sand", T=mid), slip_surface=slip,
                               method="spencer").FOS
            if fs < 1.35:
                lo = mid
            else:
                hi = mid
        print(f"V-043 force to reach 1.35 (Spencer), radius {dr:+.0e} m: "
              f"{0.5 * (lo + hi):.2f} kN/m")


def fig_8_1():
    h = 20 * FT
    for conv, printed in (("active", 2.19), ("passive", 1.72)):
        geom = SlopeGeometry(
            surface_points=[(-15, 0), (0, 0), (h / 2, h), (40, h)],
            soil_layers=[
                SlopeSoilLayer("clay", h, 0.0, 100 * PCF, cu=500 * PSF,
                               analysis_mode="undrained"),
                SlopeSoilLayer("firm", 0.0, -5.0, 100 * PCF, cu=5000.0,
                               analysis_mode="undrained")],
            geosynthetics=[Geosynthetic(elevation=10 * FT,
                                        T_allow=10000 * LB_FT,
                                        support_convention=conv)])
        best = None
        for i in range(0, 21):
            for j in range(0, 46):
                xc, r = 0.1 * i, h + 0.02 + 0.1 * j
                try:
                    res = analyze_slope(
                        geom, slip_surface=CircularSlipSurface(xc, r, r),
                        method="bishop", n_slices=40)
                except ValueError:
                    continue
                if res.FOS is not None and (best is None
                                            or res.FOS < best[0]):
                    best = (res.FOS, xc, r, res.x_entry, res.x_exit)
        print(f"Fig. 8.1, Bishop, tangent circles, {conv}: FOS "
              f"{f4(best[0])} (printed {printed}), circle xc {best[1]:.2f} "
              f"yc = R {best[2]:.3f} m (crest at {h:.3f}), x "
              f"{best[3]:.3f} to {best[4]:.3f}")


def fig_8_4():
    for conv, printed in (("active", 1.62), ("passive", 1.46)):
        g = SlopeGeometry(
            surface_points=[(0, 0), (30, 24), (75, 24)],
            soil_layers=[SlopeSoilLayer(name="fill", top_elevation=24.0,
                                        bottom_elevation=0.0, gamma=130.0,
                                        phi=37.0, c_prime=0.0)])
        try:
            g.geosynthetics = [
                Geosynthetic(elevation=float(z), T_allow=800.0,
                             x_start=z / 0.8, x_end=z / 0.8 + 20.0,
                             development_length=4.0, anchored_end="start",
                             support_convention=conv)
                for z in (0, 4, 8, 12, 16, 20)]
        except TypeError:
            print(f"Fig. 8.4, {conv}: not available before step 8 (a "
                  f"layer's force could not fall off toward its end)")
            continue
        for method in ("spencer", "bishop"):
            res = search_critical_surface(g, method=method,
                                          x_entry_range=(0.0, 4.0),
                                          x_exit_range=(30.0, 75.0))
            c = res.critical
            print(f"Fig. 8.4, {method}, {conv}: FOS {f4(c.FOS)} (printed, "
                  f"Spencer, {printed}), governs {res.governing_surface}, "
                  f"circle ({c.xc:.2f}, {c.yc:.2f}) R {c.radius:.2f}")


if __name__ == "__main__":
    slide86()
    v040_rows()
    v043_force()
    fig_8_1()
    fig_8_4()
