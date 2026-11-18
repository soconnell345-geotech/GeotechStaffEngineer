"""A figure's interactive sidecar keeps its two-part suffix when its name is
already taken (BACKLOG TA-10).

``_unique_dest`` used to split ``plot.plotly.json`` at its LAST dot, so a
second copy became ``plot.plotly_1.json``: no longer a ``*.plotly.json``, it
classified as a plain file and no longer paired with its picture
``plot_1.png``, so the chat showed a picture card and a file card instead of
one chart. Made-up file names and contents only.
"""

import os

from aec_document_review import core


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def test_taken_sidecar_name_keeps_its_suffix(tmp_path):
    taken = tmp_path / "plot.plotly.json"
    _write(taken, "{}")
    first = core._unique_dest(str(taken))
    assert os.path.basename(first) == "plot_1.plotly.json"
    assert core.classify_artifact(first) == "plotly"
    _write(first, "{}")
    assert os.path.basename(core._unique_dest(str(taken))) == \
        "plot_2.plotly.json"


def test_suffix_is_matched_without_regard_to_case(tmp_path):
    taken = tmp_path / "Plot.PLOTLY.JSON"
    _write(taken, "{}")
    assert os.path.basename(core._unique_dest(str(taken))) == \
        "Plot_1.PLOTLY.JSON"


def test_other_names_unchanged(tmp_path):
    free = tmp_path / "free.plotly.json"
    assert core._unique_dest(str(free)) == str(free)        # free: as given
    for name, want in (("plot.png", "plot_1.png"),
                       ("table.json", "table_1.json"),
                       ("notes", "notes_1"),
                       ("plotly.json", "plotly_1.json")):
        _write(tmp_path / name, "x")
        assert os.path.basename(core._unique_dest(str(tmp_path / name))) \
            == want


def test_reimported_figure_stays_one_chart(tmp_path):
    """A figure re-made under names the conversation already holds: both
    files are copied under new names, and they still make ONE chart card."""
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, "plot.png"), "OLD PICTURE")
    _write(os.path.join(files_dir, "plot.plotly.json"), '{"old": 1}')
    work = str(tmp_path / "w")
    os.makedirs(work)
    before = core.snapshot_dir(work)
    _write(os.path.join(work, "plot.plotly.json"), '{"new": 1}')
    _write(os.path.join(work, "plot.png"), "NEW PICTURE")
    copied = core.import_external_artifacts(work, files_dir, before, [])
    names = sorted(os.path.basename(p) for p in copied)
    assert names == ["plot_1.plotly.json", "plot_1.png"]
    sidecar = next(p for p in copied if p.endswith(".plotly.json"))
    assert core.classify_artifact(sidecar) == "plotly"
    assert core.read_text(sidecar) == '{"new": 1}'
    picture = next(p for p in copied if p.endswith(".png"))
    os.utime(picture, (1_000_000, 1_000_000))   # the picture is the older
    cards = core.collect_turn_artifacts([], copied)
    assert cards == [sidecar]
    assert core.read_text(os.path.join(files_dir, "plot.plotly.json")) == \
        '{"old": 1}'                                        # never clobbered
