"""Pin the CURRENT library-backed outputs as numerical oracles.

Run this while `sectionproperties` and `concreteproperties` are still
installed. The JSON it writes is the regression oracle for the native
re-implementation that replaces them (the same discipline used for the
groundhog removal in 5.11.2: the library's outputs are used only as
numerical test oracles, pinned before deletion -- no code is copied).

    python module_work/structural_native/pin_oracles.py

Writes: module_work/structural_native/oracles.json
"""

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from concrete_props_agent import analyze_rc_rectangle          # noqa: E402
from section_props_agent import (                              # noqa: E402
    analyze_polygon_section, analyze_section,
)

OUT = pathlib.Path(__file__).with_name("oracles.json")

# --------------------------------------------------------------- sections
SECTION_CASES = [
    ("rect_300x500", {"shape": "rectangle", "d": 500.0, "b": 300.0}),
    # the 2:1 rectangle behind test_torsion_constant (Roark J = 0.2287 b^3 d)
    ("rect_100x200", {"shape": "rectangle", "d": 200.0, "b": 100.0}),
    ("rect_no_warping", {"shape": "rectangle", "d": 200.0, "b": 100.0,
                         "warping": False}),
    ("square_200", {"shape": "rectangle", "d": 200.0, "b": 200.0}),
    ("circle_400", {"shape": "circle", "d": 400.0}),
    ("circle_50", {"shape": "circle", "d": 50.0}),
    ("chs_400x12", {"shape": "chs", "d": 400.0, "t": 12.0}),
    ("chs_100x5", {"shape": "chs", "d": 100.0, "t": 5.0}),
    ("rhs_300x200x10", {"shape": "rhs", "d": 300.0, "b": 200.0, "t": 10.0}),
    ("rhs_300x200x10_r15", {"shape": "rhs", "d": 300.0, "b": 200.0, "t": 10.0,
                            "r_out": 15.0}),
    ("i_400x180_square", {"shape": "i_section", "d": 400.0, "b": 180.0,
                          "t_f": 14.0, "t_w": 9.0}),
    ("i_400x180_r10", {"shape": "i_section", "d": 400.0, "b": 180.0,
                       "t_f": 14.0, "t_w": 9.0, "r": 10.0}),
    # a real rolled shape (310UB40-ish) for the AISC/Darwish-Johnston J check
    ("i_310x165_r11", {"shape": "i_section", "d": 310.0, "b": 165.0,
                       "t_f": 11.8, "t_w": 6.6, "r": 11.4}),
]

POLYGON_CASES = [
    # L-shape: the asymmetric case that exercises Ixy / principal axes / phi
    ("poly_L", [(0.0, 0.0), (200.0, 0.0), (200.0, 50.0),
                (50.0, 50.0), (50.0, 300.0), (0.0, 300.0)]),
    ("poly_triangle", [(0.0, 0.0), (300.0, 0.0), (0.0, 400.0)]),
    # rectangle rotated 30 deg -- pins the phi_deg sign convention
    ("poly_rot_rect", [(0.0, 0.0), (259.807621, 150.0),
                       (209.807621, 236.602540), (-50.0, 86.602540)]),
    ("poly_rect_equiv", [(0.0, 0.0), (300.0, 0.0), (300.0, 500.0),
                         (0.0, 500.0)]),
]

# --------------------------------------------------------------- RC cases
RC_CASES = [
    # the ACI hand-calc anchor: 300x550, 3-N28, f'c 32, fy 500, cover 48
    ("rc_anchor", {"b_mm": 300.0, "h_mm": 550.0, "fc_MPa": 32.0,
                   "fy_MPa": 500.0, "n_bot": 3, "dia_bot_mm": 28.0,
                   "cover_mm": 48.0}),
    ("rc_anchor_interaction", {"b_mm": 300.0, "h_mm": 550.0, "fc_MPa": 32.0,
                               "fy_MPa": 500.0, "n_bot": 3, "dia_bot_mm": 28.0,
                               "cover_mm": 48.0, "include_interaction": True,
                               "n_interaction_points": 24}),
    ("rc_doubly", {"b_mm": 300.0, "h_mm": 550.0, "fc_MPa": 32.0,
                   "fy_MPa": 500.0, "n_bot": 3, "dia_bot_mm": 28.0,
                   "cover_mm": 48.0, "n_top": 2, "dia_top_mm": 16.0}),
    # f'c 45 -> beta1 below 0.85 (ACI Table 22.2.2.4.3)
    ("rc_fc45", {"b_mm": 400.0, "h_mm": 600.0, "fc_MPa": 45.0,
                 "fy_MPa": 420.0, "n_bot": 4, "dia_bot_mm": 25.0,
                 "cover_mm": 40.0, "n_top": 2, "dia_top_mm": 20.0,
                 "include_interaction": True, "n_interaction_points": 16}),
    ("rc_small", {"b_mm": 250.0, "h_mm": 400.0, "fc_MPa": 21.0,
                  "fy_MPa": 420.0, "n_bot": 2, "dia_bot_mm": 16.0,
                  "cover_mm": 40.0}),
    # explicit Ec override (skips the ACI 4700*sqrt(f'c) default)
    ("rc_ec_override", {"b_mm": 300.0, "h_mm": 550.0, "fc_MPa": 32.0,
                        "fy_MPa": 500.0, "n_bot": 3, "dia_bot_mm": 28.0,
                        "cover_mm": 48.0, "ec_MPa": 30000.0}),
]


def main():
    oracles = {"section": {}, "polygon": {}, "rc": {}}

    for name, kwargs in SECTION_CASES:
        kw = dict(kwargs)
        shape = kw.pop("shape")
        print(f"  section  {name} ...", flush=True)
        oracles["section"][name] = {
            "inputs": dict(kwargs),
            "outputs": analyze_section(shape, **kw).to_dict(),
        }

    for name, pts in POLYGON_CASES:
        print(f"  polygon  {name} ...", flush=True)
        oracles["polygon"][name] = {
            "inputs": {"points": pts},
            "outputs": analyze_polygon_section(pts).to_dict(),
        }

    for name, kwargs in RC_CASES:
        print(f"  rc       {name} ...", flush=True)
        oracles["rc"][name] = {
            "inputs": dict(kwargs),
            "outputs": analyze_rc_rectangle(**kwargs).to_dict(),
        }

    import concreteproperties
    import sectionproperties
    oracles["_provenance"] = {
        "sectionproperties": sectionproperties.__version__
        if hasattr(sectionproperties, "__version__") else "unknown",
        "concreteproperties": concreteproperties.__version__
        if hasattr(concreteproperties, "__version__") else "unknown",
        "note": "Pinned before the libraries were removed (cytriangle is "
                "quarantined by the corporate proxy). Values are numerical "
                "oracles only; no library code was copied.",
    }

    OUT.write_text(json.dumps(oracles, indent=2, sort_keys=True),
                   encoding="utf-8")
    n = sum(len(v) for k, v in oracles.items() if not k.startswith("_"))
    print(f"\nwrote {OUT}  ({n} cases)")


if __name__ == "__main__":
    main()
