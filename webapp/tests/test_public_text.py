"""The shipped packages carry no field-session place names or Department
addresses (BACKLOG INF-11, OWN-57), and no tracked file in the public repo
names a Department address or a real deployment's host (INF-29).

Every wheel and sdist on PyPI is public and cannot be edited, and the
2026-10-13 audit of all of them found post and project names in code
comments and the Department's SharePoint address in two docstrings. The
first check walks the files the wheel ships (the packages ``pyproject.toml``
includes, without their ``tests`` folders) and fails on any guarded word or
host name.

The GitHub repos are public too, docs and notes included, and the INF-25
review found the Foundry enclave's host in ``docs/FOUNDRY.md`` and
``HANDOFF.md``. The second check walks every tracked file of this repo and
of the ``geotech-references`` submodule when it is checked out (CI checks
it out). It fails on the Department and deployment words, the guarded
hosts, and any Foundry enclave, Databricks workspace, SharePoint tenant or
Azure resource host that is not one of the placeholders listed below by
exact name. Place names are not guarded there: whether docs may name them
is OWN-26.

The guarded words are stored only as SHA-256 prefixes, so this public file
does not repeat them. To guard a new word, add
``hashlib.sha256(word.lower().encode()).hexdigest()[:16]`` to the right set
below.
Not guarded here: the report-ingest answer vocabulary (owner decision
OWN-25) and generic words such as "Department".
"""

from __future__ import annotations

import fnmatch
import hashlib
import re
import subprocess
import tomllib
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

#: Single words: field-session place and project names (shipped files only).
PLACE_WORDS = {
    "7ba9b19fc81560f4", "a5c49775b0a68d9f", "8c75350c087ec2d7",
    "267a58403ed7e3f8", "e18ad88cf9a0960f",
}
#: Single words: the SharePoint tenant and site, a work mailbox and the
#: published-app domain label (24b46df); the Foundry enclave's host label
#: and the Funhouse workspace number (INF-29). Guarded in every tracked file.
ADDRESS_WORDS = {
    "b5078524066ca3b3", "74aa0a6847fd6036", "ca6c6caa8d8bbacd",
    "3553be1e6e882992", "2f3919316d4f6484",
    "2b707a110b325f4c", "ebad111d7873d1f8",
}
GUARDED_WORDS = PLACE_WORDS | ADDRESS_WORDS
#: Host names (any dotted name or e-mail domain ending in one of these):
#: the Department's domain, its SharePoint host and the Foundry enclave.
GUARDED_HOSTS = {"57d0525215903582", "b6b8caa18084c17c", "3eadf5dd85bce17f"}

#: A host that names one real deployment: a Foundry enclave, a Databricks
#: workspace, a SharePoint tenant or an Azure resource. Docs write
#: ``<your-enclave>.palantirfoundry.com``, which this does not match.
DEPLOYMENT_HOST = re.compile(
    r"(?<![a-z0-9-])[a-z0-9-]+(?:\.[a-z0-9-]+)*\."
    r"(?:palantirfoundry\.(?:com|us)|palantirgov\.com"
    r"|azuredatabricks\.net|databricks\.azure\.us"
    r"|cloud\.databricks\.com|sharepoint\.(?:com|us)"
    r"|openai\.azure\.(?:com|us)|cognitiveservices\.azure\.(?:com|us)"
    r"|vault\.(?:azure|usgovcloudapi)\.net|azurewebsites\.(?:net|us))")
_DEPLOYMENT_KEYS = ("palantir", "databricks", "sharepoint", "azure",
                    "usgovcloudapi")
#: The placeholders tests and docs use, by exact host name.
PLACEHOLDER_HOSTS = {
    "stack.example.palantirfoundry.com", "yourstack.palantirfoundry.com",
    "adb-1234567890123456.7.databricks.azure.us",
    "adb-dp-1234567890123456.7.databricks.azure.us",
    "adb-dp-9.3.databricks.azure.us", "adb-99.11.azuredatabricks.net",
    "dbc-x.cloud.databricks.com", "dbc-deadbeef.cloud.databricks.com",
    "contoso.sharepoint.com", "host.sharepoint.com", "t.sharepoint.com",
    "x.sharepoint.com",
}

TEXT_SUFFIXES = {".py", ".md", ".txt", ".json", ".html", ".toml", ".yaml",
                 ".yml", ".cfg", ".ini", ".csv", ".j2", ".jinja"}
_WORD = re.compile(r"[a-z0-9]+")
#: A dotted name. The look-behind starts a match only at a label boundary,
#: which keeps long runs (base64 in the example pages) linear.
_HOST = re.compile(r"(?<![a-z0-9-])[a-z0-9-]+(?:\.[a-z0-9-]+)+")


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


def _tracked_files() -> list[Path]:
    """Every tracked file of the repo and of a checked-out submodule."""
    out = []
    for root in (REPO, REPO / "geotech-references"):
        if root != REPO and not (root / ".git").exists():
            continue
        names = subprocess.run(
            ["git", "-C", str(root), "ls-files", "-z"], capture_output=True,
            check=True).stdout.decode("utf-8").split("\0")
        out += [root / n for n in names if n and (root / n).is_file()]
    return out


def _findings(text: str, words: set[str] = GUARDED_WORDS) -> set[str]:
    low = text.lower()
    found = {w for w in set(_WORD.findall(low)) if _h(w) in words}
    for host in set(_HOST.findall(low)):
        labels = host.split(".")
        for i in range(len(labels) - 1):
            if _h(".".join(labels[i:])) in GUARDED_HOSTS:
                found.add(host)
    return found


def _deployment_hosts(text: str) -> set[str]:
    low = text.lower()
    if not any(k in low for k in _DEPLOYMENT_KEYS):
        return set()  # most files: skip the slower pattern
    return {h for h in DEPLOYMENT_HOST.findall(low)
            if h not in PLACEHOLDER_HOSTS}


def test_the_guard_recognises_its_own_words():
    # The hashes must match what _findings computes, or the guard is blind.
    assert len(GUARDED_WORDS) == 12 and len(GUARDED_HOSTS) == 3
    assert all(len(h) == 16 for h in GUARDED_WORDS | GUARDED_HOSTS)
    assert _findings("nothing to see in a plain sentence") == set()


def test_the_host_pattern_passes_placeholders_and_catches_the_rest():
    text = ("the stack host, e.g. `yourstack.palantirfoundry.com`; "
            "https://<your-enclave>.palantirgov.com/api; "
            "https://contoso.sharepoint.com/sites/x; "
            "suffixes such as .databricks.azure.us or .sharepoint.com/sites/")
    assert _deployment_hosts(text) == set()
    # Built from parts, so this file does not trip its own walk.
    real = [f"{label}.{suffix}" for label, suffix in (
        ("acme", "palantirfoundry.com"), ("adb-42.1", "databricks.azure.us"),
        ("acme-my", "sharepoint.com"), ("acme", "openai.azure.us"))]
    assert _deployment_hosts(" and ".join(f"https://{h}" for h in real)
                             ) == set(real)


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


def test_public_repo_carries_no_department_or_deployment_addresses():
    files = _tracked_files()
    assert len(files) > 1000, "the git walk found too few files"
    bad = {}
    for f in files:
        raw = f.read_bytes()
        if b"\0" in raw[:8192]:
            continue  # binary
        text = raw.decode("utf-8", errors="ignore")
        hits = _findings(text, ADDRESS_WORDS) | _deployment_hosts(text)
        if hits:
            bad[f.relative_to(REPO).as_posix()] = len(hits)
    # Paths and counts only, as above.
    assert not bad, (f"Department or deployment addresses in tracked files "
                     f"(path: count): {bad}")
