"""Skip the deep/ (deepagents v2) suite when the optional [deep] stack isn't installed.

The deepagents / langchain dependencies live behind the ``[deep]`` extra. CI and clean
installs that don't install ``[deep]`` skip these tests (rather than erroring on import),
matching how the other optional-dependency agent tests behave.
"""
try:
    import deepagents  # noqa: F401
    import langchain_core  # noqa: F401
except ImportError:  # pragma: no cover - only when [deep] absent
    collect_ignore_glob = ["*.py"]
