"""Draw GM-315's four plot_data examples into this folder as
``plot_data_<name>.png``, and print each one's warnings, the full layout
check's findings and its smallest lettering as printed at the calc
package's width. Run from the repo root:
``python module_work/gm315/render_plot_data_examples.py``."""
import os
import sys

import funhouse_agent  # noqa: F401  (registers the app with the core)
from aec_agent_core.charts.data_plot import render_data_plot
from aec_agent_core.charts.figure_examples import example
from aec_agent_core.charts.layout_ink import parse_lettering, png_texts, \
    LETTERING_KEY

HERE = os.path.dirname(os.path.abspath(__file__))
NAMES = {"wall_section_pressures": "wall_pressures",
         "settlement_stress_panels": "settlement_stress_profile",
         "pile_design_panels": "pile_design_profile",
         "contour_top_of_rock": "contour_top_of_rock"}
PRINT_WIDTH_IN = 6.5        # the calc package's text width


def main() -> int:
    bad = 0
    for name, short in NAMES.items():
        params = dict(example(name)["params"])
        res = render_data_plot(params.pop("series", None), **params)
        path = os.path.join(HERE, f"plot_data_{short}.png")
        with open(path, "wb") as fh:
            fh.write(res.png_bytes)
        lettering = parse_lettering(png_texts(res.png_bytes)
                                    .get(LETTERING_KEY))
        print(name, res.width_px, res.height_px,
              "twin" if res.figure is not None else "no twin")
        printed = lettering["smallest_pt"] * min(
            1.0, PRINT_WIDTH_IN / lettering["saved_width_in"])
        print("  lettering:", lettering, f"prints at {printed:.2f} pt")
        bad += printed < 7.0
        for w in res.warnings:
            print("  warning:", w)
            bad += not w.startswith("no interactive chart")
    return bad


if __name__ == "__main__":
    sys.exit(main())
