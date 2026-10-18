"""The code kept beside each figure never becomes a card (BACKLOG GM-151).

plot_code and plot_data keep the code (or arguments) that drew each figure
in the working folder's hidden ``.figure_code/``, so a later helper can be
handed it. It is working data: no download card, not mirrored to
SharePoint. The draw here is a real plot_code child process.
"""

import os

import pytest

pytest.importorskip("matplotlib")

from funhouse_agent.plot_box import figures, run_plot_code  # noqa: E402
from webapp import core  # noqa: E402

DRAW = ("import matplotlib.pyplot as plt\nfig, ax = plt.subplots()\n"
        "ax.plot([0, 1], [0, 1])\nax.set_xlabel('x (m)')\n"
        "save_figure(fig, 'wall')\n")


def test_the_figure_gets_a_card_and_its_code_none(tmp_path):
    work = tmp_path / "files"
    work.mkdir()
    before = core.snapshot_dir(str(work))
    out = run_plot_code(DRAW, data={"x": [0, 1]}, root=str(work))
    assert out["status"] == "success", out
    code = work / figures.CODE_DIR / "wall.plot.py"
    data = work / figures.CODE_DIR / "wall.data.json"
    assert code.exists() and data.exists()
    new = core.new_artifacts(str(work), before, [])
    cards = core.collect_turn_artifacts([], new)
    names = {os.path.relpath(p, work).replace(os.sep, "/") for p in cards}
    assert "wall.png" in names
    assert not any(figures.CODE_DIR in n for n in names), names
    assert core.is_cache_path(str(code), str(work))
    assert core.mirror_skips_dir("files/" + figures.CODE_DIR)
