"""The copy of the app split's step 13 (BACKLOG INF-33): Document Review's
files out of this repository into ``packages/aec-document-review``, each
under its new import name.

Run once, from the repository root, at the commit the copy was taken from::

    python module_work/inf33/copy_step13.py

It copies (never moves: GSE keeps its own review files until step 15) and
renames the imports mechanically; the hand edits that follow it (the
review-only profile, the one-page entry, the tasks without GSE's four, the
diagnostics without the analysis back ends) are commits of their own on the
same branch. Kept as a record of how the copy was made; running it again
over an edited package overwrites the hand edits.
"""

from __future__ import annotations

import os
import re
import shutil
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(
    __file__))))
DEST = os.path.join(REPO, "packages", "aec-document-review",
                    "aec_document_review")

#: Document Review's copy of the chat page (plan 2.9) and its host files,
#: leaving out webapp/geotech_page.py, the Databricks and Foundry host files
#: (databricks_launcher, foundry_entry, foundry_dataset_store,
#: reference_fetch) and the e-mail tool (OWN-69).
WEBAPP = [
    "app.py", "core.py", "turn_jobs.py", "turn_record.py",
    "output_capture.py", "ws_upload.py", "budget_panel.py", "feedback.py",
    "version_guard.py", "diagnostics.py", "profiles.py", "page_profile.py",
    "review_page.py", "pages_entry.py", "file_widgets.py", "page_errors.py",
    "conversation_export.py", "conversation_restore.py",
    "sharepoint_store.py", "sharepoint_tools.py", "sharepoint_log.py",
    "identity.py", "helper_models.py", "suite_page.py",
    # Tiny Apps host files
    "tinyapps_settings.py", "tinyapps_keys.py", "graph_sharepoint.py",
    "tinyapps_entry.py",
]
WEBAPP_DATA = ["ws_upload_component/index.html"]

#: Document Review's own files (plan 2.2): source -> name in the package.
REVIEW = {
    "funhouse_agent/review_prompt.py": "review_prompt.py",
    "funhouse_agent/review_build.py": "review_build.py",
    "funhouse_agent/deep/review_agent.py": "review_agent.py",
    "funhouse_agent/deep/minimal_agent.py": "minimal_agent.py",
    "funhouse_agent/review_eval/tasks.py": "review_eval/tasks.py",
}

#: Module renames, longest first.
RENAMES = [
    ("funhouse_agent.deep.review_agent", "aec_document_review.review_agent"),
    ("funhouse_agent.deep.minimal_agent",
     "aec_document_review.minimal_agent"),
    ("funhouse_agent.review_prompt", "aec_document_review.review_prompt"),
    ("funhouse_agent.review_build", "aec_document_review.review_build"),
    ("funhouse_agent.deep.notebook", "aec_document_review.stream_format"),
    ("funhouse_agent.review_eval.tasks",
     "aec_document_review.review_eval.tasks"),
    ("funhouse_agent/deep/review_agent.py",
     "aec_document_review/review_agent.py"),
    ("funhouse_agent/deep/minimal_agent.py",
     "aec_document_review/minimal_agent.py"),
    ("funhouse_agent/review_prompt.py", "aec_document_review/review_prompt.py"),
    ("funhouse_agent/review_build.py", "aec_document_review/review_build.py"),
]


def core_aliases() -> dict:
    """GSE's old names of the core's modules (funhouse_agent.CORE_ALIASES and
    webapp.CORE_ALIASES), read from the source so the two cannot drift."""
    out = {}
    for path in ("funhouse_agent/__init__.py", "webapp/__init__.py"):
        text = open(os.path.join(REPO, path), encoding="utf-8").read()
        for old, new in re.findall(r'"((?:funhouse_agent|webapp)[\w.]*)":'
                                   r'\s*"(aec_agent_core[\w.]*)"', text):
            out[old] = new
    return out


def rewrite(text: str, aliases: dict) -> str:
    for old, new in RENAMES:
        text = text.replace(old, new)
    # ``from funhouse_agent import review_flags`` and friends
    def _from_pkg(m):
        pkg, names = m.group(1), m.group(2)
        first = names.split(" as ")[0].split(",")[0].strip()
        if f"{pkg}.{first}" in aliases:
            return f"from {aliases[pkg + '.' + first].rsplit('.', 1)[0]} " \
                   f"import {names}"
        return m.group(0)
    text = re.sub(r"from (funhouse_agent(?:\.\w+)*) import ([\w ,]+?)"
                  r"(?=[ \t]*(?:#.*)?$)", _from_pkg, text, flags=re.M)
    # dotted old names, longest first
    for old in sorted(aliases, key=len, reverse=True):
        text = re.sub(re.escape(old) + r"(?![\w])", aliases[old], text)
    # ``from webapp import (a, core, engine_config)``: the core's modules
    # come from the core
    core_mods = {o.split(".", 1)[1]: n for o, n in aliases.items()
                 if o.startswith("webapp.")}

    def _from_webapp(m):
        indent, body = m.group(1), m.group(2)
        names = [n.strip() for n in body.strip("()").replace("\n", " ")
                 .split(",") if n.strip()]
        ours = [n for n in names if n.split(" as ")[0] not in core_mods]
        core = [n for n in names if n.split(" as ")[0] in core_mods]
        lines = []
        if ours:
            lines.append(f"{indent}from aec_document_review import "
                         + ", ".join(ours))
        if core:
            lines.append(f"{indent}from aec_agent_core import "
                         + ", ".join(core))
        return "\n".join(lines)
    text = re.sub(r"^([ \t]*)from webapp import (\([^)]*\)|[\w ,]+?)"
                  r"(?=[ \t]*(?:#.*)?$)", _from_webapp, text, flags=re.M)
    text = re.sub(r"\bwebapp\.(?=[a-z_])", "aec_document_review.", text)
    text = re.sub(r"\bimport webapp\b", "import aec_document_review", text)
    text = re.sub(r"\bwebapp/(?=[a-z_])", "aec_document_review/", text)
    return text


def main() -> int:
    aliases = core_aliases()
    os.makedirs(os.path.join(DEST, "review_eval"), exist_ok=True)
    pairs = [(f"webapp/{n}", n) for n in WEBAPP] + list(REVIEW.items())
    for src, dst in pairs:
        text = open(os.path.join(REPO, src), encoding="utf-8").read()
        out = os.path.join(DEST, dst)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        with open(out, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(rewrite(text, aliases))
        print(f"{src} -> aec_document_review/{dst}")
    for rel in WEBAPP_DATA:
        out = os.path.join(DEST, rel)
        os.makedirs(os.path.dirname(out), exist_ok=True)
        shutil.copyfile(os.path.join(REPO, "webapp", rel), out)
        print(f"webapp/{rel} -> aec_document_review/{rel}")
    # the stream formatters the page uses (deep/notebook.py, its first part)
    lines = open(os.path.join(REPO, "funhouse_agent/deep/notebook.py"),
                 encoding="utf-8").read().splitlines(keepends=True)
    start = next(i for i, l in enumerate(lines)
                 if l.startswith("from __future__"))
    end = next(i for i, l in enumerate(lines)
               if l.startswith("def _entry_to_plain"))
    head = ('"""The chat page\'s stream formatters (a copy of the first part '
            'of\nGeotechStaffEngineer\'s ``funhouse_agent/deep/notebook.py`` '
            'at the app\nsplit\'s step 13): a deep agent\'s streamed updates '
            'as the lines the\npage shows, the final answer of an update, and '
            'the token line."""\n\n')
    with open(os.path.join(DEST, "stream_format.py"), "w", encoding="utf-8",
              newline="\n") as fh:
        fh.write(head + "".join(lines[start:end]).rstrip() + "\n")
    print("funhouse_agent/deep/notebook.py (formatters) -> "
          "aec_document_review/stream_format.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
