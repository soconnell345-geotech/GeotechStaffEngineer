"""Which files did this turn write, and which did it only read?

Field feedback 2026-09-15 (Nairobi SOE re-run), N4/N6/N8. The agents wrote
the calc package, its figures and a plot to /tmp. The app offers downloads,
previews and the SharePoint mirror only for files in the conversation's
``files/`` folder, so the owner never received the PDF ("Where is the PDF?"),
the plot showed as a broken image, and nothing reached SharePoint. Meanwhile
the 23 MB submittal the agent downloaded from SharePoint to READ was shown as
if the agent had produced it.

:class:`OutputCollector` is a LangChain callback on the turn (it sees the
primary's and every sub-agent's tool results, as the activity log does) that
records:

* ``outputs`` -- paths a tool reports having written: ``"output_path"`` /
  ``"saved"`` / ``"plotly_json_path"`` in its JSON result, and the local side
  of a SharePoint upload;
* ``inputs``  -- the local copies of files downloaded from SharePoint.

After the turn ``core.import_reported_outputs`` copies outputs that landed
outside the conversation folder into ``files/``; inputs are kept off the
list of produced files. Streamlit-free and never raises.
"""

from __future__ import annotations

import json
import os
import re
from typing import List

try:
    from langchain_core.callbacks import BaseCallbackHandler
except Exception:  # pragma: no cover - langchain always present in the app
    BaseCallbackHandler = object  # type: ignore[misc,assignment]

# ``plotly_json_path`` is the interactive-chart sidecar a figure tool writes
# beside its image/HTML. It must be captured too: a sidecar left in /tmp is
# exactly the Nairobi failure mode — the chat would have nothing to render.
_JSON_KEY = re.compile(
    r'"(?:output_path|saved|plotly_json_path)"\s*:\s*"((?:[^"\\]|\\.)+)"')
_REPR_KEY = re.compile(
    r"'(?:output_path|saved|plotly_json_path)'\s*:\s*'([^']+)'")
_DOWNLOADED = re.compile(r"^Downloaded .+? -> (.+?) \([\d,]+ bytes\)", re.M)
_UPLOADED = re.compile(r"^Uploaded (.+?) -> ", re.M)


def _text_of(output) -> str:
    content = getattr(output, "content", output)
    if isinstance(content, list):
        content = " ".join(
            c.get("text", "") if isinstance(c, dict) else str(c) for c in content)
    return content if isinstance(content, str) else str(content)


def paths_in(text: str) -> tuple:
    """``(outputs, inputs)`` named in one tool result's text."""
    outputs: List[str] = []
    for raw in _JSON_KEY.findall(text):
        try:
            outputs.append(json.loads(f'"{raw}"'))
        except ValueError:
            outputs.append(raw)
    outputs += _REPR_KEY.findall(text)
    outputs += _UPLOADED.findall(text)
    inputs = _DOWNLOADED.findall(text)
    return outputs, inputs


class OutputCollector(BaseCallbackHandler):
    """Accumulates written (``outputs``) and fetched (``inputs``) paths."""

    def __init__(self):
        super().__init__()
        self.outputs: List[str] = []
        self.inputs: List[str] = []

    def on_tool_end(self, output, **kwargs):  # noqa: D401 - callback
        try:
            outs, ins = paths_in(_text_of(output))
            for p in outs:
                if p and p not in self.outputs:
                    self.outputs.append(p)
            for p in ins:
                ap = os.path.abspath(p)
                if ap not in self.inputs:
                    self.inputs.append(ap)
        except Exception:  # noqa: BLE001 - capture must never cost a turn
            pass


__all__ = ["OutputCollector", "paths_in"]
