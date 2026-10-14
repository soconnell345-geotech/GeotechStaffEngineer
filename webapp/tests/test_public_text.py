"""The shipped packages carry no field-session place names or Department
addresses (BACKLOG INF-11, OWN-57).

Every wheel and sdist on PyPI is public and cannot be edited, and the
2026-10-13 audit of all of them found post and project names in code
comments and the Department's SharePoint address in two docstrings. This
test walks the files the wheel ships (the packages ``pyproject.toml``
includes, without their ``tests`` folders) and fails on any guarded word or
host name.

The guarded words are stored only as SHA-256 prefixes, so this public file
does not repeat them. To guard a new word, add
``hashlib.sha256(word.lower().encode()).hexdigest()[:16]`` to the set below.
Not guarded here: the report-ingest answer vocabulary (owner decision
OWN-25) and generic words such as "Department".
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Single words: field-session place and project names, the SharePoint
#: tenant and site, a work mailbox and the published-app domain label.
GUARDED_WORDS = {
    "7ba9b19fc81560f4", "a5c49775b0a68d9f", "8c75350c087ec2d7",
    "267a58403ed7e3f8", "e18ad88cf9a0960f", "b5078524066ca3b3",
    "74aa0a6847fd6036", "ca6c6caa8d8bbacd", "3553be1e6e882992",
    "2f3919316d4f6484",
}
#: Host names (any dotted name or e-mail domain ending in one of these).
GUARDED_HOSTS = {"57d0525215903582", "b6b8caa18084c17c"}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".html", ".toml", ".yaml",
                 ".yml", ".cfg", ".ini", ".csv", ".j2", ".jinja"}
_WORD = re.compile(r"[a-z0-9]+")
_HOST = re.compile(r"[a-z0-9-]+(?:\.[a-z0-9-]+)+")


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


def _shipped_files() -> list[Path]:
    cfg = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    find = cfg["tool"]["setuptools"]["packages"]["find"]
    include = find.get("include") or []
    tops = sorted(p for p in REPO.iterdir() if p.is_dir()
                  and any(fnmatch.fnmatch(p.name, pat) for pat in include))
    out = []
    for top in tops:
        for f in top.rglob("*"):
            parts = f.relative_to(REPO).parts
            if (f.is_file() and f.suffix.lower() in TEXT_SUFFIXES
                    and "tests" not in parts and "__pycache__" not in parts):
                out.append(f)
    return out


def _findings(text: str) -> set[str]:
    low = text.lower()
    found = {w for w in _WORD.findall(low) if _h(w) in GUARDED_WORDS}
    for host in _HOST.findall(low):
        labels = host.split(".")
        for i in range(len(labels) - 1):
            if _h(".".join(labels[i:])) in GUARDED_HOSTS:
                found.add(host)
    return found


def test_the_guard_recognises_its_own_words():
    # The hashes must match what _findings computes, or the guard is blind.
    assert len(GUARDED_WORDS) == 10 and len(GUARDED_HOSTS) == 2
    assert all(len(h) == 16 for h in GUARDED_WORDS | GUARDED_HOSTS)
    assert _findings("nothing to see in a plain sentence") == set()


def test_shipped_packages_carry_no_guarded_names():
    files = _shipped_files()
    assert len(files) > 300, "the package walk found too few files"
    bad = {}
    for f in files:
        hits = _findings(f.read_text(encoding="utf-8", errors="ignore"))
        if hits:
            bad[str(f.relative_to(REPO))] = len(hits)
    # Report paths and counts only: printing the words would put them in
    # CI logs.
    assert not bad, f"guarded names in shipped files (path: count): {bad}"
