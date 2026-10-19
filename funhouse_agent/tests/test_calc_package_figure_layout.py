"""A canned calc package's figures, drawn for the width they print at,
have no lettering on another's, under a line or cut off (BACKLOG GM-194).

Each figure is checked by the layout check the plotting box uses
(``geotech_common.layout_ink.layout_check``) as ``figure_to_base64`` saves
it, after it was fitted to the printed width. Before GM-194 these packages
carried 1 to 8 findings a figure: keys over curves, notes placed in data
units on lines and markers, contour values under flow arrows. The slope
packages are left out (their figures wait for GM-140).
"""

import io
import json
import os

import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")

PROBES = os.path.join(os.path.dirname(__file__), "method_probes.json")

#: probe name -> what failed before GM-194
CASES = [
    "axial_pile_package",          # soil names over the frame, key on tip
    "bearing_capacity_package",    # B on its arrow, D_f on the ground line
    "downdrag_package",            # NP note on the title (no load)
    "drilled_shaft_package",       # key over the bars and the tip line
    "pile_group_package",          # labels under their markers
    "retaining_wall_package",      # B on R, P_a on its outline, boxes
    "sheet_pile_package",          # the key on the shear curve
    "wave_equation_package",       # the key under the twin axis' lines
]


def _probe(name):
    with open(PROBES, encoding="utf-8") as fh:
        return dict(json.load(fh)["calc_package|" + name]["params"])


@pytest.fixture
def checked(monkeypatch):
    """Every figure a package saves as a PNG to memory (what
    ``figure_to_base64`` does), with the layout check's findings."""
    from matplotlib.figure import Figure

    from geotech_common.layout_ink import layout_check
    found = []
    real = Figure.savefig

    def spy(self, fname, *a, **kw):
        if kw.get("format") == "png" and isinstance(fname, io.BytesIO):
            r = layout_check(self, "x", dpi=kw.get("dpi", 150))
            found.append({key: r.get(key) for key in
                          ("text_overlaps", "text_on_a_line", "text_cut_off")
                          if r.get(key)})
        return real(self, fname, *a, **kw)

    monkeypatch.setattr(Figure, "savefig", spy)
    return found


@pytest.mark.parametrize("name", CASES)
def test_package_figures_have_clear_lettering(tmp_path, checked, name):
    from funhouse_agent.adapters import calc_package as adapter
    params = {**_probe(name), "format": "html",
              "output_path": str(tmp_path / "pkg.html")}
    result = adapter.METHOD_REGISTRY[name](params)
    assert result["status"] == "success", result.get("error")
    assert checked, "no figure was drawn"
    assert checked == [{}] * len(checked), checked


def test_seepage_head_values_clear_of_the_flow_arrows(checked):
    """Contour values ringed and over the quiver (they were under every
    arrow at the printed width)."""
    import numpy as np

    from calc_package.renderer import figure_to_base64
    from fem2d import analyze_seepage, generate_rect_mesh
    from fem2d.plotting import plot_seepage
    nodes, elements = generate_rect_mesh(0, 10, -4, 0, 12, 6)
    head_bcs = ([(int(i), 6.0) for i in np.where(np.abs(nodes[:, 0]) < 0.01)[0]]
                + [(int(i), 2.0)
                   for i in np.where(np.abs(nodes[:, 0] - 10) < 0.01)[0]])
    result = analyze_seepage(nodes, elements, k=1e-5, head_bcs=head_bcs)
    figure_to_base64(plot_seepage(result))
    assert checked == [{}], checked
