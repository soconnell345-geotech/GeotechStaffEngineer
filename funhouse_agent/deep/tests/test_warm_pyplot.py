"""``matplotlib.pyplot`` is imported before tools run side by side
(BACKLOG GM-172).

Live wave w13 (F18 [45]/[48]): two tools drawing in the same model turn both
made the process's first ``import matplotlib.pyplot``, and one failed with
"partially initialized module 'matplotlib.pyplot'". The agent build now makes
that import, with the Agg backend, before any tool runs.

The thread test runs in a fresh interpreter: in this one pyplot is long
imported.
"""

import os
import subprocess
import sys
import textwrap

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

_SCRIPT = textwrap.dedent("""
    import io, sys, threading, traceback
    sys.path.insert(0, {root!r})
    assert "matplotlib.pyplot" not in sys.modules
    from funhouse_agent.deep.warm_imports import warm_pyplot
    assert warm_pyplot() and warm_pyplot()
    import matplotlib
    assert matplotlib.get_backend().lower() == "agg", matplotlib.get_backend()

    start = threading.Barrier(8)
    errors = []

    def draw(i):
        try:
            start.wait()
            if i % 2:                    # as profile_figure and others do
                matplotlib.use("Agg", force=False)
            import matplotlib.pyplot as plt
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            fig.savefig(io.BytesIO(), format="png")
            plt.close(fig)
        except Exception:
            errors.append(traceback.format_exc())

    threads = [threading.Thread(target=draw, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    print("ERRORS", len(errors))
    print("\\n".join(errors))
""")


def test_tools_drawing_in_several_threads_after_the_warm_import():
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    run = subprocess.run([sys.executable, "-c", _SCRIPT.format(root=ROOT)],
                         capture_output=True, text=True, timeout=300,
                         env=env, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    assert "ERRORS 0" in run.stdout, run.stdout[-2000:]


#: The failure as the trace shows it: one tool's ``matplotlib.use("Agg")``
#: met pyplot half imported by another ("partially initialized module
#: 'matplotlib.pyplot' has no attribute 'switch_backend'"). A bare module in
#: ``sys.modules`` stands in for the half-done import, so the race is
#: reproduced every time rather than by chance. After the warm import a
#: tool's ``matplotlib.use("Agg")`` finds Agg set and never reaches into
#: pyplot at all.
_HALF_IMPORTED = textwrap.dedent("""
    import sys, types
    sys.path.insert(0, {root!r})
    import matplotlib
    if {warm}:
        from funhouse_agent.deep.warm_imports import warm_pyplot
        warm_pyplot()
    sys.modules["matplotlib.pyplot"] = types.ModuleType("matplotlib.pyplot")
    try:
        matplotlib.use("Agg", force=False)
        print("OK")
    except AttributeError as exc:
        print("RACE", exc)
""")


@pytest.mark.parametrize("warm, want", [(False, "RACE"), (True, "OK")])
def test_a_tool_meeting_a_half_imported_pyplot(warm, want):
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    run = subprocess.run(
        [sys.executable, "-c", _HALF_IMPORTED.format(root=ROOT, warm=warm)],
        capture_output=True, text=True, timeout=300, env=env, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().startswith(want), run.stdout


def test_a_backend_already_chosen_is_kept():
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {ROOT!r})
        import matplotlib
        matplotlib.use("svg")
        from funhouse_agent.deep.warm_imports import warm_pyplot
        assert warm_pyplot()
        print(matplotlib.get_backend())
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=300, env=env, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().lower() == "svg"


def test_pyplot_imported_with_no_backend_chosen_gets_agg():
    """A package imported before the agent is built may import pyplot
    without choosing a backend (Pynite does)."""
    env = {k: v for k, v in os.environ.items() if k != "MPLBACKEND"}
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {ROOT!r})
        import matplotlib, matplotlib.pyplot
        assert matplotlib.rcParams._get_backend_or_none() is None
        from funhouse_agent.deep.warm_imports import warm_pyplot
        assert warm_pyplot()
        print(matplotlib.get_backend())
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=300, env=env, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip().lower() == "agg"


def test_the_agent_build_warms_pyplot(monkeypatch):
    pytest.importorskip("deepagents")
    from langchain_core.language_models.fake_chat_models import (
        GenericFakeChatModel)
    from langchain_core.messages import AIMessage

    from funhouse_agent.deep import warm_imports
    from funhouse_agent.deep.agent import build_deep_agent
    calls = []
    monkeypatch.setattr(warm_imports, "warm_pyplot",
                        lambda: calls.append(1) or True)
    build_deep_agent(GenericFakeChatModel(
        messages=iter([AIMessage(content="ok")])))
    assert calls == [1]


def test_no_matplotlib_no_error():
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {ROOT!r})
        sys.modules["matplotlib"] = None        # import matplotlib fails
        from funhouse_agent.deep.warm_imports import warm_pyplot
        print(warm_pyplot(), warm_pyplot())
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True,
                         text=True, timeout=300, cwd=ROOT)
    assert run.returncode == 0, run.stderr[-2000:]
    assert run.stdout.strip() == "False False"
