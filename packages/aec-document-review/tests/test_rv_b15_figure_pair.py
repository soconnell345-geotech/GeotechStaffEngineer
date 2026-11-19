"""Review of B15: a lone file of the same name, from another folder of the
same import, must not take one of the two names of a whole figure. Made-up
names and bytes only."""
import os

import pytest

from aec_document_review import core


def _write(path, data):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as fh:
        fh.write(data)


def _read(path):
    with open(path, "rb") as fh:
        return fh.read()


def _import(mode, wd, fd, names):
    if mode == "external":
        return core.import_external_artifacts(wd, fd, set(), [])
    return list(core.import_reported_outputs(
        [os.path.join(wd, n) for n in names], fd, working_dir=wd).values())


@pytest.mark.parametrize("mode", ["external", "reported"])
@pytest.mark.parametrize("lone", ["plot.png", "plot.plotly.json"])
def test_lone_file_ahead_of_a_figure_does_not_split_it(tmp_path, mode, lone):
    fd = str(tmp_path / "conv" / "files")
    wd = str(tmp_path / "work")
    os.makedirs(fd)
    files = {os.path.join("a", lone): b"LONE",
             os.path.join("b", "plot.png"): b"PICTURE-B",
             os.path.join("b", "plot.plotly.json"): b'{"figure": "b"}'}
    for name, data in files.items():
        _write(os.path.join(wd, name), data)
    got = _import(mode, wd, fd, list(files))
    assert len(got) == len(set(got)) == 3
    # every source's bytes are in the folder, once
    assert sorted(_read(p) for p in got) == sorted(files.values())
    charts = [p for p in got if _read(p) == b'{"figure": "b"}']
    assert len(charts) == 1
    picture = core.plotly_picture_twin(charts[0])
    assert picture is not None and _read(picture) == b"PICTURE-B"
    assert _read(core.plotly_download_twin(charts[0])) == b"PICTURE-B"
    # the order of the result is the order of the sources, as before
    if mode == "external":
        assert [_read(p) for p in got] == [
            files[k] for k in sorted(files, key=lambda n: os.path.join(wd, n))]
    else:
        assert [_read(p) for p in got] == list(files.values())


@pytest.mark.parametrize("mode", ["external", "reported"])
def test_two_figures_of_one_stem_in_one_call_keep_every_earlier_file(
        tmp_path, mode):
    fd = str(tmp_path / "conv" / "files")
    wd = str(tmp_path / "work")
    _write(os.path.join(fd, "plot.png"), b"EARLIER")
    files = {"a/plot.png": b"PA", "a/plot.plotly.json": b"JA",
             "b/plot.png": b"PB", "b/plot.plotly.json": b"JB",
             "c/plot.png": b"LONE-C"}
    for name, data in files.items():
        _write(os.path.join(wd, name), data)
    got = _import(mode, wd, fd, list(files))
    assert _read(os.path.join(fd, "plot.png")) == b"EARLIER"
    assert len(set(got)) == 5 and len(os.listdir(fd)) == 6
    for chart, pic in ((b"JA", b"PA"), (b"JB", b"PB")):
        side = [p for p in got if _read(p) == chart][0]
        assert _read(core.plotly_picture_twin(side)) == pic
