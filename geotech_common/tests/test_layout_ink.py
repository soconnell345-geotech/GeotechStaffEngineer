"""The layout check by drawn pixels (``geotech_common.layout_ink``), shared
by the plotting box and ``plot_data``'s notes (BACKLOG GM-108, GM-109).

The box's own classes of finding are tested through the box in
``funhouse_agent/tests/test_plot_box.py``; here, what moving it into the
app's process asked for: one check at a time, other threads' figures drawn
as usual, leader arrows counted as lines, and findings for chosen texts.
"""

import io
import threading
import time

import pytest

pytest.importorskip("matplotlib")

from geotech_common import layout_ink  # noqa: E402
from geotech_common.layout_ink import layout_check  # noqa: E402


def _figure(w=5, h=4, dpi=100):
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(w, h), dpi=dpi)
    FigureCanvasAgg(fig)
    ax = fig.subplots()
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    return fig, ax


def _crossed(out):
    return {words: what for words, what in out.get("text_on_a_line", [])}


# ---------------------------------------------------------------------------
# leader arrows are lines
# ---------------------------------------------------------------------------

def test_a_leader_through_other_lettering_is_found():
    """A leader arrow draws apart from its annotation's text, and was never
    measured: one crossing another label went unreported."""
    fig, ax = _figure()
    ax.text(5, 5, "Other label", ha="center", va="center", fontsize=10)
    ax.annotate("Note", xy=(9.5, 5), xytext=(0.5, 5), va="center",
                arrowprops={"arrowstyle": "-|>", "color": "black",
                            "lw": 1.2})
    crossed = _crossed(layout_check(fig, "x.png"))
    assert crossed.get("Other label") == "the leader of 'Note'", crossed
    assert "Note" not in crossed           # never its own lettering


def test_an_arrow_alone_is_a_line_too():
    from matplotlib.patches import FancyArrowPatch
    fig, ax = _figure()
    ax.text(5, 3, "Under an arrow", ha="center", va="center", fontsize=10)
    ax.text(5, 7, "Under an empty note", ha="center", va="center",
            fontsize=10)
    ax.add_patch(FancyArrowPatch((0.5, 3), (9.5, 3), arrowstyle="-|>",
                                 mutation_scale=10, lw=1.2, color="black"))
    ax.annotate("", xy=(9.5, 7), xytext=(0.5, 7),
                arrowprops={"arrowstyle": "-|>", "lw": 1.2})
    crossed = _crossed(layout_check(fig, "x.png"))
    assert crossed.get("Under an arrow") == "an arrow", crossed
    assert crossed.get("Under an empty note") == "an arrow", crossed


def test_a_leader_clear_of_lettering_says_nothing():
    fig, ax = _figure()
    ax.text(5, 8, "Well clear", ha="center", fontsize=10)
    ax.annotate("Note", xy=(9, 2), xytext=(1, 2), va="center",
                arrowprops={"arrowstyle": "-|>"})
    out = layout_check(fig, "x.png")
    assert not out.get("text_on_a_line"), out


# ---------------------------------------------------------------------------
# findings for chosen texts
# ---------------------------------------------------------------------------

def test_arrows_drawn_as_filled_shapes_are_lines_too():
    """BACKLOG GM-145: an arrow drawn as a filled shape (``ax.arrow`` with no
    edge, a quiver) was a fill under the lettering, and a fill under
    lettering is where it stands: the arrow through a label went
    unreported."""
    for draw in (
            lambda ax: ax.arrow(0.5, 4.6, 9, 0.8, width=0.08, lw=0,
                                color="black"),
            lambda ax: ax.quiver([0.5], [4.6], [9], [0.8], angles="xy",
                                 scale_units="xy", scale=1, width=0.006)):
        fig, ax = _figure()
        ax.text(5, 5, "Under an arrow", ha="center", va="center",
                fontsize=10)
        draw(ax)
        assert _crossed(layout_check(fig, "x.png")) == {
            "Under an arrow": "an arrow"}
        fig, ax = _figure()
        ax.text(5, 8, "Clear of it", ha="center", va="center", fontsize=10)
        draw(ax)
        assert not _crossed(layout_check(fig, "x.png"))


# ---------------------------------------------------------------------------
# hatching is judged as lines are (BACKLOG GM-145)
# ---------------------------------------------------------------------------

def _hatched(ax, **text_kw):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0, 0), 10, 5, facecolor="#eeeeee",
                           edgecolor="#555555", hatch="//", lw=0.8,
                           label="Soft clay"))
    ax.fill_between([0, 10], 6, 9, facecolor="none", edgecolor="#555555",
                    hatch="xx", lw=0)
    ax.text(5, 2.5, "CLAY: soft", ha="center", va="center", fontsize=10,
            **text_kw)
    ax.text(5, 7.5, "SAND: dense", ha="center", va="center", fontsize=10,
            **text_kw)


def test_hatching_under_lettering_is_found():
    """A hatched shape drawn under a label was a fill (where the label
    stands), so a hatch through the letters went unreported: a patch's and
    a collection's hatching are both found, named by the shape's label."""
    fig, ax = _figure()
    _hatched(ax)
    assert _crossed(layout_check(fig, "x.png")) == {
        "CLAY: soft": "the hatching of 'Soft clay'",
        "SAND: dense": "hatching"}


def test_a_box_or_a_halo_keeps_hatching_off_the_letters():
    from matplotlib import patheffects
    fig, ax = _figure()
    _hatched(ax, bbox={"facecolor": "white", "edgecolor": "none"})
    assert not _crossed(layout_check(fig, "x.png"))
    fig, ax = _figure()
    _hatched(ax, path_effects=[patheffects.withStroke(linewidth=3,
                                                      foreground="white")])
    assert not _crossed(layout_check(fig, "x.png"))


@pytest.mark.parametrize("halo", [True, False])
def test_a_halo_on_a_long_label_keeps_hatching_off_its_letters(halo):
    """Lettering with a halo is drawn as paths, the halo and the letters
    alike, and its letters were measured as plain lettering draws them:
    glyphs hinted to whole pixels (matplotlib 3.11's default), about half a
    pixel a letter wider at 200 dpi, so on a long label they drifted off
    the halo round them and the hatching at the halo's edge was counted
    (CI on 3.11). Without the halo the hatching still crosses the label."""
    import matplotlib
    from matplotlib import patheffects
    from matplotlib.patches import Rectangle
    rc = {"text.hinting": "default"}
    if tuple(int(p) for p in matplotlib.__version__.split(".")[:2]) < (3, 11):
        rc["text.hinting_factor"] = 1       # 3.11 has no factor
    words = "Soft clay zone, very soft to soft, grey"
    kw = ({"path_effects": [patheffects.withStroke(linewidth=2.5,
                                                   foreground="white")]}
          if halo else {})
    with matplotlib.rc_context(rc):
        fig, ax = _figure(w=6.5)
        ax.add_patch(Rectangle((0, 0), 10, 5, facecolor="none",
                               edgecolor="#555555", hatch="//", lw=0.8,
                               label="Soft clay"))
        ax.text(5, 2.5, words, ha="center", va="center", fontsize=10, **kw)
        found = _crossed(layout_check(fig, "x.png"))
    assert found == ({} if halo else {words: "the hatching of 'Soft clay'"})


def test_a_plain_fill_under_lettering_is_still_where_it_stands():
    """Only the hatching moved: a label on a tinted zone is not crossed."""
    from matplotlib.patches import Rectangle
    fig, ax = _figure()
    ax.add_patch(Rectangle((0, 0), 10, 5, facecolor="#c0d0e0",
                           edgecolor="none"))
    ax.text(5, 2.5, "On a tint", ha="center", va="center", fontsize=10)
    assert not _crossed(layout_check(fig, "x.png"))


def test_texts_keeps_the_findings_on_those_texts_by_index():
    fig, ax = _figure()
    ax.plot([0, 10], [5, 5], color="black", lw=1.5)
    a = ax.text(5, 5, "On the line", ha="center", va="center")
    b = ax.text(2, 8, "First", fontsize=10)
    c = ax.text(2.3, 8.05, "Second", fontsize=10)
    ax.text(5, 5.02, "Not chosen", ha="center", va="center")
    full = layout_check(fig, "x.png")
    out = layout_check(fig, "x.png", texts=[c, a])
    assert full.get("text_on_a_line") and full.get("text_overlaps")
    # 'On the line' (1) is crossed by the line, and overlaps 'Not chosen'
    lines = {k: what for k, what, _by in out.get("text_on_a_line", [])}
    assert 1 in lines and 0 not in lines, out
    met = {(k, words) for k, words, _other in out["text_overlaps"]}
    assert (0, "First") in met and (1, "Not chosen") in met, out
    others = {words: other for _k, words, other in out["text_overlaps"]}
    assert others["First"] is b
    # nothing on texts not chosen
    assert all(k in (0, 1) for k, *_ in out["text_overlaps"])


def test_the_budgets_are_the_callers():
    fig, ax = _figure()
    for k in range(20):
        ax.text(k % 10, k // 2, f"P{k}")
    out = layout_check(fig, "x.png", seconds=0.0)
    assert "too many texts to check in time (0 s" in out["unchecked"], out
    out = layout_check(fig, "x.png", mb=0)
    assert "unchecked" not in out or "in memory" in out["unchecked"]


# ---------------------------------------------------------------------------
# in the app's process: one check at a time, other threads untouched
# ---------------------------------------------------------------------------

def _busy():
    """A figure that takes the check a while: forty labels and a line."""
    fig, ax = _figure()
    ax.plot([0, 10], [0, 10], color="black")
    for k in range(40):
        ax.text(k % 10, k // 4, f"P{k}")
    return fig


def _red_line_pixels() -> int:
    """Save a figure with one thick red line; the red pixels in it."""
    import numpy as np
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    fig = Figure(figsize=(1.5, 1.5), dpi=50)
    FigureCanvasAgg(fig)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_axis_off()
    ax.plot([0, 1], [0.5, 0.5], color="red", lw=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    buf = io.BytesIO()
    fig.savefig(buf, format="rgba", dpi=50)
    px = np.frombuffer(buf.getvalue(), np.uint8).reshape(-1, 4)
    return int(((px[:, 0] > 200) & (px[:, 1] < 80) & (px[:, 2] < 80)).sum())


def _while_checking(work, n_checks=4):
    """Run ``work()`` over and over on this thread while another thread
    runs the check ``n_checks`` times; ``work``'s results and the
    checks'."""
    checks, errors, done = [], [], threading.Event()

    def checker():
        try:
            for _ in range(n_checks):
                checks.append(layout_check(_busy(), "x.png"))
        except Exception as exc:                           # noqa: BLE001
            errors.append(exc)
        finally:
            done.set()

    t = threading.Thread(target=checker)
    t.start()
    got = []
    try:
        while not done.is_set():
            got.append(work())
    finally:
        t.join(60)
    assert not errors, errors
    return got, checks


def test_a_figure_saved_on_another_thread_keeps_its_drawing():
    """Review of GM-109: the check swaps ``draw`` on matplotlib's classes
    for the whole process, and 300 of 300 figures another thread saved
    meanwhile lost their line."""
    alone = _red_line_pixels()
    assert alone > 100
    got, checks = _while_checking(_red_line_pixels)
    assert len(got) >= 5, "no save overlapped a check"
    assert set(got) == {alone}, sorted(set(got))
    first = checks[0]
    assert "unchecked" not in first, first
    assert all(c == first for c in checks)


def test_two_checks_at_once_each_get_their_own_answer():
    fig, ax = _figure()
    ax.plot([0, 10], [5, 5], color="black", lw=1.5)
    ax.text(5, 5, "On the line", ha="center", va="center")
    alone = layout_check(fig, "x.png")
    got, checks = _while_checking(lambda: layout_check(fig, "x.png"), 3)
    assert got and all(g == alone for g in got), got[:2]
    assert all(c == checks[0] for c in checks)


def test_matplotlib_is_left_as_it_was():
    from matplotlib.collections import Collection
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, Patch
    from matplotlib.text import Text
    kinds = (Text, Line2D, Patch, Collection, FancyArrowPatch)
    before = [vars(k)["draw"] for k in kinds]
    fig, ax = _figure()
    ax.annotate("Note", xy=(9, 5), xytext=(1, 5),
                arrowprops={"arrowstyle": "-|>"})
    layout_check(fig, "x.png")
    layout_check(fig, "x.png", seconds=0.0)
    assert [vars(k)["draw"] for k in kinds] == before
    assert not fig.artists
    assert not layout_ink._LOCK.locked()


def test_a_check_takes_little_time():
    fig, ax = _figure()
    ax.annotate("Note", xy=(9, 5), xytext=(1, 5),
                arrowprops={"arrowstyle": "-|>"})
    t = time.monotonic()
    layout_check(fig, "x.png", dpi=150)
    assert time.monotonic() - t < 2.0


# ---------------------------------------------------------------------------
# lettering as printed (GM-168)
# ---------------------------------------------------------------------------

def test_a_png_carries_its_lettering_and_reads_it_back():
    from calc_package.renderer import figure_to_base64
    import base64
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot([0, 1], [0, 1])
    ax.tick_params(labelsize=6.5)
    ax.set_xlabel("Depth (m)", fontsize=9)
    try:
        # as drawn: the record, not the fitting (GM-186), is tested here
        png = base64.b64decode(figure_to_base64(fig, dpi=120,
                                                print_width_in=None))
    finally:
        plt.close(fig)
    meta = layout_ink.parse_lettering(
        layout_ink.png_texts(png)[layout_ink.LETTERING_KEY])
    assert meta["smallest_pt"] == pytest.approx(6.5)
    # the width saved: the PNG's pixels over its dpi
    import struct
    width_px = struct.unpack(">I", png[16:20])[0]
    assert meta["saved_width_in"] == pytest.approx(width_px / 120, abs=0.01)
    assert "checked_width_in" not in meta


def test_a_tick_label_outside_the_view_is_not_lettering():
    fig, ax = _figure()
    ax.set_xticks([0, 5, 10, 50])
    ax.set_xticklabels(["0", "5", "10", "far"], fontsize=3)
    ax.set_xlim(0, 10)
    ax.tick_params(axis="x", labelsize=3)
    ax.get_xticklabels()[1].set_fontsize(9)
    sizes, _w = layout_ink.lettering_sizes(fig)
    texts = [t for t, _bb, _a in layout_ink.drawn_texts(
        fig, fig.canvas.get_renderer())]
    assert "far" not in texts
    assert min(sizes) == pytest.approx(3.0)


def test_png_text_helpers_leave_other_files_alone():
    assert layout_ink.png_with_text(b"not a png", "k", "v") == b"not a png"
    assert layout_ink.png_texts(b"GIF89a") == {}
    assert layout_ink.parse_lettering("something else") == {}
    text = layout_ink.lettering_text(6.123, 9.8765, 6.5)
    assert layout_ink.parse_lettering(text) == {
        "smallest_pt": 6.12, "saved_width_in": 9.877,
        "checked_width_in": 6.5}


# ---------------------------------------------------------------------------
# drawn for the width it prints at (BACKLOG GM-186)
# ---------------------------------------------------------------------------

def _pyplot_figure(w, h, fontsize=8.0, title=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(w, h))
    ax.plot([0, 10], [0, 5])
    ax.set_xlabel("Distance (m)", fontsize=fontsize)
    ax.set_ylabel("Settlement (mm)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    if title:
        ax.set_title(title, fontsize=fontsize)
    fig.tight_layout()
    return plt, fig, ax


def _printed_smallest(fig, width_in):
    sizes, saved_w = layout_ink.lettering_sizes(fig)
    return min(sizes) * width_in / saved_w, saved_w


def test_lettering_of_a_wide_figure_prints_at_7_pt():
    """Drawn 10 in wide with 8 pt lettering and printed 5 in wide it
    printed 4 pt; drawn for the print width it prints 7 pt or more."""
    plt, fig, _ax = _pyplot_figure(10, 5, fontsize=8)
    try:
        assert _printed_smallest(fig, 5.0)[0] < 4.5
        out = layout_ink.draw_for_print(fig, 5.0)
        printed, saved_w = _printed_smallest(fig, 5.0)
        assert out["drawn_in"] == [10.0, 5.0] and "kept" not in out
        assert out["widest_in"] == pytest.approx(5.0 * 8 / 7, abs=0.01)
        assert saved_w <= out["widest_in"]
        assert printed >= layout_ink.MIN_PRINT_PT
        w, h = fig.get_size_inches()
        assert w / h == pytest.approx(2.0, rel=0.01)       # shape kept
    finally:
        plt.close(fig)


def test_lettering_under_7_pt_is_raised_tick_labels_included():
    """Small lettering is raised to 7 pt, the tick labels through their
    axis: ticks a later draw makes keep the size."""
    plt, fig, ax = _pyplot_figure(4, 3, fontsize=5)
    ax.text(5, 2, "note", fontsize=6)
    try:
        out = layout_ink.draw_for_print(fig, 5.0)
        assert out["raised"] >= 3
        ax.set_xlim(0, 37)                     # new ticks at the next draw
        fig.canvas.draw()
        sizes = [t.get_fontsize() for _s, _bb, t in
                 layout_ink.drawn_texts(fig, fig.canvas.get_renderer())]
        assert min(sizes) >= layout_ink.MIN_PRINT_PT
    finally:
        plt.close(fig)


def test_lettering_of_a_figure_that_fits_is_left_alone():
    plt, fig, _ax = _pyplot_figure(4, 3, fontsize=10)
    try:
        out = layout_ink.draw_for_print(fig, 5.0)
        assert out["raised"] == 0 and "kept" not in out
        assert list(fig.get_size_inches()) == [4.0, 3.0]
    finally:
        plt.close(fig)


def test_lettering_far_outside_leaves_the_figure_as_drawn():
    """A note placed in data units far beyond the axes (a degenerate
    range): no smaller figure brings it in, so the figure, its lettering
    included, stays as drawn and says why."""
    plt, fig, ax = _pyplot_figure(6, 4, fontsize=8)
    far = ax.text(400, 2, "far note", fontsize=5)
    try:
        out = layout_ink.draw_for_print(fig, 5.0)
        assert "far outside" in out["kept"]
        assert list(fig.get_size_inches()) == [6.0, 4.0]
        assert far.get_fontsize() == 5
    finally:
        plt.close(fig)


def test_lettering_kept_wide_by_a_long_title_leaves_the_figure_as_drawn():
    """A title wider than the widest the figure may be saved at keeps it
    wide: the figure goes back as drawn, size and layout."""
    plt, fig, ax = _pyplot_figure(9.5, 5, fontsize=8)
    ax.set_title("A very long figure title that runs well past the page "
                 "width at its size", fontsize=16)
    fig.tight_layout()
    before = ax.get_position().bounds
    try:
        out = layout_ink.draw_for_print(fig, 5.0)
        assert "keeps it wide" in out["kept"]
        assert list(fig.get_size_inches()) == [9.5, 5.0]
        assert ax.get_position().bounds == pytest.approx(before, abs=1e-6)
    finally:
        plt.close(fig)


def test_lettering_of_a_figure_legend_keeps_clear_of_the_axes():
    """A legend of the figure's own under the panels keeps its band when
    the figure is laid out again (``tight_layout`` leaves it out)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(10, 5), sharey=True)
    for ax in axes:
        ax.plot([0, 1], [0, 1], label="Spans above the excavation")
        ax.plot([0, 1], [1, 0], label="Free earth body below the support")
        ax.set_xlabel("Pressure (kPa)", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="lower center",
               ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    try:
        layout_ink.draw_for_print(fig, 5.0)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        legend_top = fig.legends[0].get_window_extent(r).y1
        lowest = min(ax.get_tightbbox(r).y0 for ax in axes)
        assert legend_top <= lowest + 1
    finally:
        plt.close(fig)
