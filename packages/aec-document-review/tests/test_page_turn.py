"""The page builds and a turn runs through its real turn path (BACKLOG
INF-33, step 13's exit: the package installs and runs the review page with
nothing geotech present).

Each build is made the way ``app.py`` makes it (``core.build_agent`` with
the page's profile) on a fake model whose first answer calls the tools whose
code once reached into GeotechStaffEngineer (``plot_data``, ``calculate``,
``write_docx``) and the build's helper; the turn is streamed by
``core.stream_turn``, the page's own turn loop, to its ``turn_done``.
"""

import os

import pytest

from tests import page_builds


@pytest.mark.parametrize("name", sorted(page_builds.BUILDS))
def test_a_turn_runs_through_the_page(name, tmp_path):
    files_dir = tmp_path / "conv" / "files"
    files_dir.mkdir(parents=True)
    settings = {"GEOTECH_DEFAULT_OUTPUT_DIR": str(files_dir)}
    model = page_builds.fake_model(page_builds._turn_calls(name))
    with page_builds.page_environment(page_builds.BUILDS[name][1],
                                      **settings):
        agent, _, _ = page_builds.build(name, model=model,
                                        files_dir=str(files_dir),
                                        settings=settings)
        from aec_document_review import core
        events = list(core.stream_turn(
            agent, [{"role": "user", "content": "Plot, work out, write."}],
            f"turn-{name}", recursion_limit=max(100, int(getattr(
                agent, "geotech_min_recursion_limit", 0) or 0))))
    kinds = [e.get("kind") for e in events if isinstance(e, dict)]
    assert "turn_done" in kinds, kinds[-10:]
    assert "error" not in kinds, [e for e in events if e.get("kind") ==
                                  "error"]
    if name == "review_default":
        written = os.listdir(files_dir)
        assert "boundary_note.docx" in written, written
        assert any(n.endswith(".png") for n in written), written


def test_the_turn_writes_the_apps_name_in_its_word_file(tmp_path):
    """The Word file the turn wrote names this app, not the one it was
    split from (the core's app details, handed in by the package)."""
    from aec_agent_core import app_details, docx_renderer
    import aec_document_review  # noqa: F401 - hands in its details
    assert app_details.name() == "Document Review"
    out = docx_renderer.markdown_to_docx("# Note\n\nOne line.",
                                         str(tmp_path / "n.docx"))
    import docx
    assert docx.Document(out).core_properties.author == "Document Review"
