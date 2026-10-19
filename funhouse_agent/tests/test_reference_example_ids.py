"""Every section number a reference tool's description gives as an example
is a section that tool returns (BACKLOG REF-01: ``retrieve_section`` said
'5.7.2', which is in no reference).

``retrieve_section`` is one tool per reference, so its examples must be
that reference's own sections; ``reference_db.reference_get`` takes the
reference as an argument, so each of its examples must be in at least one
reference.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

retrieval = pytest.importorskip("geotech_references._retrieval")

from funhouse_agent import dispatch  # noqa: E402
from funhouse_agent.adapters._reference_common import (  # noqa: E402
    example_section_ids,
)

#: A quoted section number in a description: '7.2.1.3.1', ``"4-2.1"``.
_QUOTED_ID = re.compile(r"""['"]([0-9P]+(?:[-.][0-9A-Za-z]+)*)['"]""")


def _section_params(agent: str):
    mod = dispatch._load_adapter(agent)
    for method, info in mod.METHOD_INFO.items():
        if info.get("alias_of"):
            continue
        for name, spec in (info.get("parameters") or {}).items():
            if name == "section_id" and isinstance(spec, dict):
                yield method, spec.get("description") or ""


def _examples(text: str) -> list[str]:
    return _QUOTED_ID.findall(text)


def _every_section_id() -> set[str]:
    root = Path(retrieval.__file__).parent
    ids: set[str] = set()
    for text_dir in root.glob("*/text"):
        for path in text_dir.glob("*.json"):
            data = json.loads(path.read_text(encoding="utf-8"))
            ids.update(str(e["section"].get("section_id"))
                       for e in retrieval.section_entries(data))
    return ids


def _text_agents() -> list[str]:
    return [name for name in sorted(dispatch.REFERENCE_MODULES)
            if "retrieve_section" in dispatch._load_adapter(name).METHOD_REGISTRY]


@pytest.mark.parametrize("agent", _text_agents())
def test_retrieve_section_examples_resolve(agent):
    found = []
    for method, text in _section_params(agent):
        examples = _examples(text)
        assert examples, f"{agent}.{method}: no example section number"
        for sid in examples:
            out = dispatch.call_agent(agent, method, {"section_id": sid})
            assert "error" not in out, f"{agent}.{method}({sid!r}): {out}"
            found.append(sid)
    assert found


def test_reference_get_examples_are_in_the_library():
    ids = _every_section_id()
    texts = [text for _, text in _section_params("reference_db")]
    assert texts
    examples = [sid for text in texts for sid in _examples(text)]
    assert examples
    missing = [sid for sid in examples if sid not in ids]
    assert not missing, f"reference_get examples in no reference: {missing}"


def test_examples_come_from_the_reference_itself():
    """GEC-12 shows a number at each level, down to the static methods."""
    assert example_section_ids("gec_12") == ["1.1", "4.2.3.1", "7.2.1.3.1"]
    assert "5.7.2" not in _every_section_id()
    assert example_section_ids("no_such_reference") == []
