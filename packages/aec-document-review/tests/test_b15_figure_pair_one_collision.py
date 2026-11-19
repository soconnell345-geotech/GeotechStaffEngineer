"""A figure imported into a folder that holds a file with the name of only
ONE of its two files keeps its chart (the limit TA-10 left).

A figure is two files with one stem: its picture and its interactive file.
Named file by file, the one whose name was taken became ``plot_1.*`` while
the other stayed ``plot.*``: the two no longer paired, and the chat showed a
plain picture beside a separate chart. Made-up names and contents only.
"""

import os

import pytest

from aec_document_review import core

SIDECAR, PICTURE = "plot.plotly.json", "plot.png"


def _write(path, text):
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(text)


def _figure(folder, tag, names=(SIDECAR, PICTURE)):
    os.makedirs(folder, exist_ok=True)
    for name in names:
        _write(os.path.join(folder, name), f"{tag} {name}")
    return [os.path.join(folder, name) for name in names]


def _one_figure(copied, tag):
    """The copies are one figure: one stem, one chart card, its own bytes."""
    names = sorted(os.path.basename(p) for p in copied)
    sidecar = next(p for p in copied if p.endswith(".plotly.json"))
    picture = core.plotly_picture_twin(sidecar)
    assert picture in copied, names
    assert core.plotly_download_twin(sidecar) == picture
    assert core.read_text(sidecar).startswith(tag)
    assert core.read_text(picture).startswith(tag)
    os.utime(picture, (1_000_000, 1_000_000))       # the picture is the older
    assert core.collect_turn_artifacts([], sorted(copied)) == [sidecar], names
    return names


def _import(kind, tmp_path, files_dir, tag, names=(SIDECAR, PICTURE)):
    """Import a new made-up figure by one of the two import paths."""
    work = str(tmp_path / f"work_{tag}")
    os.makedirs(work)
    before = core.snapshot_dir(work)
    made = _figure(work, tag, names)
    if kind == "external":
        return core.import_external_artifacts(work, files_dir, before, [])
    return list(core.import_reported_outputs(
        made, files_dir, working_dir=work).values())


KINDS = ("external", "reported")


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("taken", (SIDECAR, PICTURE))
def test_one_name_taken_keeps_the_pair(tmp_path, kind, taken):
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, taken), "OLD")
    copied = _import(kind, tmp_path, files_dir, "NEW")
    assert _one_figure(copied, "NEW") == ["plot_1.plotly.json", "plot_1.png"]
    assert core.read_text(os.path.join(files_dir, taken)) == "OLD"
    assert sorted(os.listdir(files_dir)) == sorted(
        [taken, "plot_1.plotly.json", "plot_1.png"])


@pytest.mark.parametrize("kind", KINDS)
@pytest.mark.parametrize("taken", (SIDECAR, PICTURE))
def test_three_imports_in_a_row(tmp_path, kind, taken):
    """Each later import is one figure again, under the next free stem."""
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, taken), "OLD")
    for n, tag in enumerate(("A", "B", "C"), start=1):
        copied = _import(kind, tmp_path, files_dir, tag)
        assert _one_figure(copied, tag) == [
            f"plot_{n}.plotly.json", f"plot_{n}.png"]
    # the first figure is still whole after the later two
    first = os.path.join(files_dir, "plot_1.plotly.json")
    assert core.read_text(core.plotly_picture_twin(first)).startswith("A")


@pytest.mark.parametrize("kind", KINDS)
def test_stem_skips_a_number_only_one_file_holds(tmp_path, kind):
    """``plot_1.png`` alone is in the way: both copies take ``plot_2``."""
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, SIDECAR), "OLD")
    _write(os.path.join(files_dir, "plot_1.png"), "OLD")
    copied = _import(kind, tmp_path, files_dir, "NEW")
    assert _one_figure(copied, "NEW") == ["plot_2.plotly.json", "plot_2.png"]


@pytest.mark.parametrize("kind", KINDS)
def test_html_copy_moves_with_its_figure(tmp_path, kind):
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, "plot.html"), "OLD")
    copied = _import(kind, tmp_path, files_dir, "NEW",
                     (SIDECAR, PICTURE, "plot.html"))
    assert sorted(os.path.basename(p) for p in copied) == [
        "plot_1.html", "plot_1.plotly.json", "plot_1.png"]
    sidecar = next(p for p in copied if p.endswith(".plotly.json"))
    assert core.collect_turn_artifacts([], sorted(copied)) in (
        [sidecar], [sidecar[:-len(".plotly.json")] + ".png"])


def test_reported_same_picture_new_chart_stays_paired(tmp_path):
    """The reported-outputs path reuses a file of the same bytes: a figure
    whose picture is unchanged and whose chart is new must not be split
    between the reused picture and a renamed chart."""
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, PICTURE), f"NEW {PICTURE}")   # same bytes
    _write(os.path.join(files_dir, SIDECAR), "OLD")
    copied = _import("reported", tmp_path, files_dir, "NEW")
    assert _one_figure(copied, "NEW") == ["plot_1.plotly.json", "plot_1.png"]


def test_reported_same_figure_again_is_reused(tmp_path):
    """Unchanged from before: the very same figure reported twice is one
    pair of files, not two."""
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    work = str(tmp_path / "w")
    made = _figure(work, "SAME")
    for _ in range(2):
        copied = core.import_reported_outputs(made, files_dir,
                                              working_dir=work)
        assert sorted(os.path.basename(p) for p in copied.values()) == [
            SIDECAR, PICTURE]
    assert sorted(os.listdir(files_dir)) == [SIDECAR, PICTURE]


# ---- the neighbours: named file by file, as before -------------------------

@pytest.mark.parametrize("kind", KINDS)
def test_picture_without_interactive_file_is_named_alone(tmp_path, kind):
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, PICTURE), "OLD")
    copied = _import(kind, tmp_path, files_dir, "NEW", (PICTURE,))
    assert [os.path.basename(p) for p in copied] == ["plot_1.png"]
    assert core.collect_turn_artifacts([], copied) == copied


@pytest.mark.parametrize("kind", KINDS)
def test_file_that_only_shares_the_stem_is_left_alone(tmp_path, kind):
    """``plot.csv`` is no part of the figure: it keeps its free name while
    the figure moves, and it moves alone when only its own name is taken."""
    files_dir = core.conversation_files_dir("c", root=str(tmp_path))
    _write(os.path.join(files_dir, PICTURE), "OLD")
    copied = _import(kind, tmp_path, files_dir, "A",
                     (SIDECAR, PICTURE, "plot.csv"))
    assert sorted(os.path.basename(p) for p in copied) == [
        "plot.csv", "plot_1.plotly.json", "plot_1.png"]
    _write(os.path.join(files_dir, "plot_2.png"), "OLD")
    os.remove(os.path.join(files_dir, PICTURE))
    os.remove(os.path.join(files_dir, "plot_1.png"))
    os.remove(os.path.join(files_dir, "plot_1.plotly.json"))
    copied = _import(kind, tmp_path, files_dir, "B",
                     (SIDECAR, PICTURE, "plot.csv"))
    assert sorted(os.path.basename(p) for p in copied) == [
        "plot.plotly.json", "plot.png", "plot_1.csv"]
