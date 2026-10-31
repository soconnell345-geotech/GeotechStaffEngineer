"""GEC-13's printed Priebe curve, Fig. 5-21 curve (4) (GM-221, GM-276).

GEC-13 (FHWA-NHI-16-027) Vol. I prints no Priebe formula. It prints the
Priebe curve, Fig. 5-21 curve (4) "PRIEBE phi = 37.5°" (p. 5-36), reads it
in Example 2 (Sec. 4.3.2, Fig. 5-27, p. 5-49) and says on p. 5-37 "For
preliminary estimates, the Priebe curve may be used to evaluate the upper
bound effectiveness and cost at various spacings." The tool's note quotes
that, with the page, and leaves the design value to the engineer.
The curve was traced off the rendered page on its grid lines and drawn
back over the figure; a read is good to about one line width, +/- 0.03.
Since GM-276 (owner, OWN-70: "just the printed curve. discard your
formula") the curve is the module's only Priebe basis.

Example 2 (pp. 5-48, 5-49): columns 3.0 ft at 5.7 ft, A/A_sc = (5.7)^2 /
(3.0)^2 = 3.6; "Using the Priebe curve from Figure 5-27" the text prints
"The settlement ratio is 2.7. Therefore, the settlement of the embankment
using stone columns as ground improvement is 10 inches", from 27 in.
without columns (p. 5-48). Fig. 5-27's arrows are drawn at A/A_sc 3.53
and settlement ratio 2.63, on the curve, and labelled "3.6" and "2.7". At (5.7 / 3)^2
= 3.61 the curve reads 2.55 on both printed copies (Fig. 5-21, p. 5-36;
Fig. 5-27, p. 5-49), so the module gives 27 / 2.55 = 10.6 in. where the
example prints 10 in.
"""

import pytest

from ground_improvement import aggregate_piers
from ground_improvement.settlement_ratio import (
    PRIEBE_CURVE_4, priebe_curve_settlement_ratio)

TOL = 0.03  # one line width on the printed figure


def _curve(area_ratio):
    return priebe_curve_settlement_ratio(1.0 / area_ratio)


#: Curve (4) read off the printed page between the table's nodes (GM-276
#: review): GEC-13 Fig. 5-21, p. 5-36, each read the line's centre in the
#: pixel column at that A/A_c, calibrated on the printed grid lines; Fig.
#: 5-27 (p. 5-49) gives the same reads within 0.02.
PAGE_READS_P5_36 = {
    1.90: 5.10, 1.95: 4.95, 2.06: 4.67, 2.15: 4.48, 2.40: 3.98, 2.60: 3.64,
    2.90: 3.22, 3.15: 2.94, 3.40: 2.71, 3.61: 2.54, 3.90: 2.34, 4.15: 2.21,
    4.40: 2.10, 4.75: 2.01, 5.25: 1.91, 5.75: 1.83, 6.25: 1.75, 6.75: 1.68,
    7.25: 1.61, 7.75: 1.54,
}


@pytest.mark.parametrize("area_ratio", sorted(PAGE_READS_P5_36))
def test_the_curve_as_read_off_the_figure(area_ratio):
    """Between its nodes the table follows the printed line within 0.02."""
    assert area_ratio not in PRIEBE_CURVE_4
    assert _curve(area_ratio) == pytest.approx(
        PAGE_READS_P5_36[area_ratio], abs=0.02)


def test_example_2_reads_the_curve_where_its_arrows_are_drawn():
    # the arrows of Fig. 5-27 as drawn (p. 5-49): A/A_sc 3.53, settlement ratio 2.63
    assert _curve(3.53) == pytest.approx(2.63, abs=TOL)


def test_example_2_at_its_own_area_ratio():
    """At the example's A/A_sc = (5.7 / 3.0)^2 = 3.61 the printed curve
    reads 2.55, and 27 in. (p. 5-48) becomes 27 / 2.55 = 10.6 in. GEC-13
    prints 2.7 and 10 in. (p. 5-49): its read, labelled on arrows drawn at
    A/A_sc 3.53, is 0.15 above the curve at 3.61, and 10 in. is 0.6 in.
    below the module's 10.6."""
    a = (5.7 / 3.0) ** 2
    assert a == pytest.approx(3.61, abs=0.005)
    n = _curve(a)
    assert n == pytest.approx(2.55, abs=TOL)
    assert 27.0 / n == pytest.approx(10.6, abs=0.15)
    printed_n, printed_s = 2.7, 10.0
    assert printed_n - n == pytest.approx(0.15, abs=TOL)
    assert 27.0 / n - printed_s == pytest.approx(0.6, abs=0.15)


def test_the_formula_is_gone():
    """GM-276: no Priebe formula and no 42.5 deg default in the module;
    the printed curve is the only basis."""
    assert not hasattr(aggregate_piers, "priebe_basic_improvement_factor")
    assert not hasattr(aggregate_piers, "PRIEBE_METHODS")
    assert "37.5" in aggregate_piers.PRIEBE_BASIS


def test_the_note_is_gec_13s_own_words():
    """GM-276 (owner, OWN-70: "print hard number, leave to engineer to
    decide with note"): the note gives GEC-13's upper-bound sentence and
    leaves the design value to the engineer, quoting only the page."""
    note = aggregate_piers.PRIEBE_NOTE
    assert "the upper bound, for preliminary estimates" in note
    assert "the engineer's to judge" in note
    for page, words in aggregate_piers.PRIEBE_GEC13_WORDS:
        assert f"p. {page}: '{words}'" in note


def _printed(text):
    """Page text with its line breaks and typographic quotes made plain."""
    for a, b in (("“", '"'), ("”", '"'), ("’", "'")):
        text = text.replace(a, b)
    return " ".join(text.split())


def test_the_quoted_words_are_on_the_printed_pages():
    """Every sentence the note quotes is on its GEC-13 page as printed
    (the owner: nothing from memory). Needs the GEC-13 PDF
    (GEOTECH_REFERENCES_DOCS or geotech-references/docs)."""
    pymupdf = pytest.importorskip("pymupdf")
    from funhouse_agent import worked_examples
    try:
        pdf = worked_examples.resolve_source_pdf(
            {"id": "GM-276", "source_doc": "GEC 13 Vol 1.pdf"})
    except FileNotFoundError:
        pytest.skip("GEC-13 Vol. 1 PDF not available")
    doc = pymupdf.open(str(pdf))
    try:
        pages = {}
        for page in doc:
            text = _printed(page.get_text())
            label = text.split(" ", 1)[0]
            pages.setdefault(label, text)
    finally:
        doc.close()
    for page, words in aggregate_piers.PRIEBE_GEC13_WORDS:
        assert words in pages[page], (page, words)


def test_between_the_reads_linear_in_a_s():
    a0, a1 = 1 / 3.0, 1 / 3.25
    mid = (a0 + a1) / 2
    assert priebe_curve_settlement_ratio(mid) == pytest.approx(
        (PRIEBE_CURVE_4[3.0] + PRIEBE_CURVE_4[3.25]) / 2)


@pytest.mark.parametrize("as_ratio", [0.1, 0.7])
def test_off_the_drawn_curve_is_refused(as_ratio):
    with pytest.raises(ValueError, match="curve is not extended"):
        priebe_curve_settlement_ratio(as_ratio)


def test_a_bad_area_ratio_is_refused():
    with pytest.raises(ValueError, match="replacement ratio"):
        priebe_curve_settlement_ratio(1.0)
