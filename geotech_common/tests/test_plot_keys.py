"""The shared placement helpers for figure lettering (BACKLOG GM-194):
``key_under_axes`` (one key under the axes, a note heading it) and
``halo`` (a ring round lettering that must sit on the drawing)."""

import pytest

pytest.importorskip("matplotlib")

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from geotech_common.plotting import halo, key_under_axes  # noqa: E402


def _box(artist, fig):
    fig.canvas.draw()
    return artist.get_window_extent(fig.canvas.get_renderer())


def test_the_key_goes_under_the_axes_and_the_axes_key_goes():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], label="Q(z)")
    ax.plot([0, 1], [1, 0], label="Pile tip")
    ax.legend(loc="lower right")
    key = key_under_axes(fig, ax)
    assert ax.get_legend() is None
    assert fig.legends == [key]
    assert [t.get_text() for t in key.get_texts()] == ["Q(z)", "Pile tip"]
    # under the axes, tick labels and axis title included, and on the figure
    box = _box(key, fig)
    under = ax.get_tightbbox(fig.canvas.get_renderer())
    assert box.y1 <= under.y0 + 1
    assert box.y0 >= 0
    plt.close(fig)


def test_a_note_heads_the_key_and_twins_share_one_key():
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1], label="R_ult")
    twin = ax.twinx()
    twin.plot([0, 1], [3, 2], label="Comp. Stress")
    twin.plot([0, 1], [2, 3], label="R_ult")        # one entry per label
    key = key_under_axes(fig, [ax, twin], note="FOS_slide = 1.28")
    assert key.get_title().get_text() == "FOS_slide = 1.28"
    assert [t.get_text() for t in key.get_texts()] == ["R_ult",
                                                      "Comp. Stress"]
    plt.close(fig)


def test_given_handles_and_nothing_to_say():
    from matplotlib.patches import Patch
    fig, ax = plt.subplots(figsize=(5, 4))
    key = key_under_axes(fig, ax, handles=[Patch(label="Compression (+)")])
    assert [t.get_text() for t in key.get_texts()] == ["Compression (+)"]
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot([0, 1], [0, 1])
    assert key_under_axes(fig, ax) is None
    assert fig.legends == []
    plt.close(fig)


def test_a_key_under_the_axes_leaves_the_curves_clear_of_it():
    """The GM-194 case: a key inside the axes had a rising curve through
    it at the printed width; under them nothing runs through it."""
    from geotech_common.layout_ink import draw_for_print, layout_check
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([1.0, 1.1, 1.2, 1.3], [4.3, 4.6, 4.8, 5.2], "o-",
            label="Converged trials")
    ax.axvline(1.35, ls="--", label="FOS = 1.350, failure basis: "
                                    "nonconvergence")
    key_under_axes(fig, ax)
    draw_for_print(fig, 5.0)
    found = layout_check(fig, "x", dpi=150)
    assert not found.get("text_on_a_line"), found
    assert not found.get("text_overlaps"), found
    plt.close(fig)


def test_halo_rings_the_lettering():
    from matplotlib import patheffects
    fig, ax = plt.subplots()
    text = ax.text(0.5, 0.5, "4.2")
    halo([text], colour="#f5e6c8")
    (effect,) = text.get_path_effects()
    assert isinstance(effect, patheffects.withStroke)
    plt.close(fig)
