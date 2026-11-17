"""Draw GM-313's example sheets (made-up walls; not designs).

    python module_work/gm313/make_examples.py          # the committed set
    python module_work/gm313/make_examples.py --all    # and three more

The committed set (the repository is public and kept small): the made-up
fixture ``dxf_export/tests/fixtures/wall_section_example.json`` as
``wall_example.dxf``, ``.pdf`` and ``.png``, and the PNG alone of a real
GM-312 description, the "key_surcharge" wall of
``dxf_export/tests/fixtures/gm312_real_descriptions.json``, as
``wall_real_design.png``. ``--all`` also writes the low wall, the tall wall
with a long heel and the front-battered stem with a shear key (each as
its three files), which are not committed.
"""
import copy
import json
import os
import sys

from dxf_export.render import write_sheet
from dxf_export.wall_section import wall_section_drawing

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "..", "..", "dxf_export", "tests", "fixtures")
FIXTURE = os.path.join(FIXTURES, "wall_section_example.json")
REAL = os.path.join(FIXTURES, "gm312_real_descriptions.json")
COMMITTED = ("wall_example",)


def examples():
    with open(FIXTURE, encoding="utf-8") as handle:
        base = json.load(handle)
    out = {"wall_example": (base, {})}

    low = copy.deepcopy(base)
    low["title"] = "Low wall (made-up example)"
    low["geometry"].update(wall_height=1.8, base_width=1.4, toe_length=0.3,
                           heel_length=0.8, stem_thickness_top=0.25,
                           stem_thickness_base=0.3, base_thickness=0.3,
                           embedment_depth=0.6)
    low["bars"][0]["end"]["ldh_mm"] = 180.0
    out["wall_low"] = (low, {})

    tall = copy.deepcopy(base)
    tall["title"] = "Tall wall with a long heel (made-up example)"
    tall["geometry"].update(wall_height=7.5, base_width=5.4, toe_length=0.9,
                            heel_length=3.8, stem_thickness_top=0.35,
                            stem_thickness_base=0.7, base_thickness=0.7,
                            embedment_depth=1.2, backfill_slope_deg=10.0)
    out["wall_tall_long_heel"] = (tall, {"size": "tabloid"})

    key = copy.deepcopy(base)
    key["title"] = ("Front-battered stem with a shear key "
                    "(made-up example)")
    key["geometry"].update(wall_height=3.6, base_width=2.4, toe_length=0.6,
                           heel_length=1.3, stem_thickness_top=0.25,
                           stem_thickness_base=0.5, base_thickness=0.4,
                           stem_batter="front", embedment_depth=0.8,
                           has_shear_key=True, key_depth=0.4, key_width=0.4,
                           key_distance_from_toe=0.9)
    out["wall_front_batter_key"] = (key, {})
    return out


if __name__ == "__main__":
    every = "--all" in sys.argv[1:]
    for name, (description, sheet) in examples().items():
        if not every and name not in COMMITTED:
            continue
        drawing = wall_section_drawing(
            description, sheet=sheet,
            title_block={"project": "Made-up example", "drawn_by": "GSE",
                         "date": "2026-11-13", "drawing_number": name})
        result = write_sheet(drawing, os.path.join(HERE, name))
        print(name, result.to_dict()["scale"], result.counts,
              result.warnings)
    # a real GM-312 description, as the PNG alone
    from dxf_export.render import render_sheet
    with open(REAL, encoding="utf-8") as handle:
        real = json.load(handle)["key_surcharge"]
    drawing = wall_section_drawing(
        real, title_block={"project": "Made-up wall, GM-312 design",
                           "drawn_by": "GSE", "date": "2026-11-13",
                           "drawing_number": "wall_real_design"})
    files = render_sheet(drawing)
    with open(os.path.join(HERE, "wall_real_design.png"), "wb") as handle:
        handle.write(files["png"])
    print("wall_real_design", f"1:{files['info'].scale}",
          drawing.layout_warnings)
