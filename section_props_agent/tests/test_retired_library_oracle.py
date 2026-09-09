"""Regression against the retired `sectionproperties` FE library.

The library's outputs were pinned to JSON before it was removed (it needs
`cytriangle`, a wheel the corporate proxy quarantines). Nothing of the
library is used here — the file is a frozen table of numbers the native
engine has to keep reproducing.

Where the native engine deliberately differs, the difference is declared in
``TOLERANCES``/``SKIP`` with the reason, so a future change that drifts
somewhere else still fails.
"""

import json
import pathlib

import pytest

from section_props_agent import analyze_polygon_section, analyze_section

ORACLE = (pathlib.Path(__file__).resolve().parents[2]
          / "module_work" / "structural_native" / "oracles.json")

pytestmark = pytest.mark.skipif(
    not ORACLE.exists(),
    reason="pinned oracle file is dev-tree only (not shipped in the wheel)")

#: default agreement demanded of every reported quantity
DEFAULT_TOL = 0.005

#: quantity -> tolerance, with the reason the native value is allowed to differ
TOLERANCES = {
    # The library meshed circles as 64-gons, so its circular areas and moments
    # are a fraction of a percent light; the native path is analytic.
    ("circle_400", "*"): 0.005,
    ("circle_50", "*"): 0.005,
    ("chs_400x12", "*"): 0.005,
    ("chs_100x5", "*"): 0.005,
    # J from the industry closed forms rather than an FE warping solve:
    # Darwish-Johnston/AISC for the I-sections, Bredt for the closed box.
    # These are the values steel tables quote.
    ("i_400x180_square", "j_mm4"): 0.02,
    ("i_400x180_r10", "j_mm4"): 0.04,
    ("i_310x165_r11", "j_mm4"): 0.04,
    ("rhs_300x200x10", "j_mm4"): 0.01,
    ("rhs_300x200x10_r15", "j_mm4"): 0.01,
    # Cw from the standard Iy*h0^2/4 rather than the FE solve
    ("i_400x180_r10", "gamma_mm6"): 0.02,
    ("i_310x165_r11", "gamma_mm6"): 0.02,
    ("i_400x180_square", "gamma_mm6"): 0.02,
}

#: (case, key) pairs that are zero by theory. A circular section does not
#: warp, so its warping constant is exactly zero; the library's small
#: non-zero values there are FE mesh noise (4e5 mm^6 against an I-section's
#: 2e11 on the same units).
ZERO_BY_THEORY = {
    ("circle_400", "gamma_mm6"), ("circle_50", "gamma_mm6"),
    ("chs_400x12", "gamma_mm6"), ("chs_100x5", "gamma_mm6"),
}

#: quantities the native engine deliberately does not report
SKIP = {
    # The warping constant is only defined for the open shapes where it is
    # used in design (the I-section) and is exactly zero for circular ones.
    # For a solid rectangle, a closed box or an arbitrary polygon there is no
    # defensible closed form, and design practice neglects warping restraint
    # there, so the engine returns None instead of inventing a number.
    "gamma_mm6": ("rect_300x500", "rect_100x200", "square_200",
                  "rhs_300x200x10", "rhs_300x200x10_r15",
                  "poly_L", "poly_triangle", "poly_rot_rect",
                  "poly_rect_equiv"),
}


def _load():
    return json.loads(ORACLE.read_text(encoding="utf-8"))


def _tol(case, key):
    if (case, key) in TOLERANCES:
        return TOLERANCES[(case, key)]
    if (case, "*") in TOLERANCES:
        return TOLERANCES[(case, "*")]
    return DEFAULT_TOL


def _check(case, native, expected):
    for key, want in expected.items():
        if isinstance(want, str):
            continue
        if case in SKIP.get(key, ()):
            assert key not in native, (
                f"{case}: {key} is documented as not reported")
            continue
        assert key in native, f"{case}: native result is missing {key}"
        got = native[key]
        if (case, key) in ZERO_BY_THEORY:
            assert got == 0.0
            continue
        # a product moment that is analytically zero comes back as round-off
        # in the library and as a true zero here
        if abs(want) < 1e-6 * max(abs(expected.get("ixx_mm4", 1.0)), 1.0):
            assert abs(got) < 1e-6 * max(abs(expected.get("ixx_mm4", 1.0)), 1.0)
            continue
        assert got == pytest.approx(want, rel=_tol(case, key)), (
            f"{case}/{key}: native {got:.8g} vs retired library {want:.8g}")


@pytest.mark.parametrize("case", sorted(_load()["section"]) if ORACLE.exists()
                         else [])
def test_parametric_shapes_match_the_retired_library(case):
    entry = _load()["section"][case]
    kwargs = dict(entry["inputs"])
    shape = kwargs.pop("shape")
    _check(case, analyze_section(shape, **kwargs).to_dict(), entry["outputs"])


@pytest.mark.parametrize("case", sorted(_load()["polygon"]) if ORACLE.exists()
                         else [])
def test_polygons_match_the_retired_library(case):
    entry = _load()["polygon"][case]
    result = analyze_polygon_section(entry["inputs"]["points"]).to_dict()
    _check(case, result, entry["outputs"])


def test_oracle_file_records_its_provenance():
    prov = _load()["_provenance"]
    assert "sectionproperties" in prov and "no library code was copied" in \
        prov["note"]
