"""``calculation_html``: a package's calculation without the package
(BACKLOG GM-67, for the agent's ``render_tables``).

The package and the calculation print the same sections by the same
template part (``CALC_SECTIONS_TEMPLATE``); the agent-side tests
(``funhouse_agent/tests/test_render_tables.py``) compare the two for three
packages. These hold the rendering itself.
"""

import re

from calc_package import calculation_html, generate_calc_package
from calc_package.renderer import TABLES_SCOPE, tables_css
from calc_package.template import (
    CALC_PACKAGE_CSS, CALC_PACKAGE_TEMPLATE, CALC_SECTIONS_CSS,
    CALC_SECTIONS_TEMPLATE)


def _footing():
    from bearing_capacity.capacity import BearingCapacityAnalysis
    from bearing_capacity.footing import Footing
    from bearing_capacity.soil_profile import BearingSoilProfile, SoilLayer
    footing = Footing(width=2.0, depth=1.5, shape="square")
    soil = BearingSoilProfile(layer1=SoilLayer(friction_angle=30,
                                               unit_weight=18))
    analysis = BearingCapacityAnalysis(footing=footing, soil=soil)
    return analysis.compute(), analysis


def test_the_package_template_holds_the_sections_part():
    assert CALC_SECTIONS_TEMPLATE in CALC_PACKAGE_TEMPLATE
    assert CALC_SECTIONS_CSS in CALC_PACKAGE_CSS


def test_every_rule_is_scoped_and_sets_no_fill():
    css = tables_css()
    rules = re.findall(r"([^{}]+)\{([^{}]*)\}", css)
    assert len(rules) > 20
    for selectors, body in rules:
        for sel in selectors.split(","):
            assert sel.strip().startswith(f".{TABLES_SCOPE} "), sel
        fills = re.findall(r"background[\w-]*\s*:\s*([^;]+)", body)
        assert all(f.strip() == "transparent" for f in fills), (selectors,
                                                                 body)


def test_the_calculation_is_the_packages_without_its_frame():
    result, analysis = _footing()
    package = generate_calc_package("bearing_capacity", result, analysis)
    calc = calculation_html("bearing_capacity", result, analysis)
    assert "header-block" in package and "header-block" not in calc
    assert "<img" not in calc and "Figures" not in calc
    # the package numbers its h2 sections; the calculation's h3 are not
    pkg_titles = [t for t in re.findall(r"<h2>\d+\. (.*?)</h2>", package)
                  if t != "Figures"]
    calc_titles = re.findall(r"<h3>(.*?)</h3>", calc)
    assert calc_titles == pkg_titles + ["References"]
    assert f'<div class="{TABLES_SCOPE}">' in calc
    # every step's equation is the package's
    eq = re.compile(r'<div class="equation">(.*?)</div>', re.DOTALL)
    assert eq.findall(calc) == eq.findall(package) != []
