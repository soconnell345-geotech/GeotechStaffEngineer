"""The documents the review suite asks about, and where to find them.

A document is one of:

* ``{"file": "<name>.pdf"}`` — a real PDF, looked for (first hit wins) in the
  suite's ``docs_dir`` (and its sub-folders), ``$GEOTECH_REVIEW_EVAL_DOCS``,
  ``$GEOTECH_REFERENCES_DOCS``, the source checkout's own folders, and last
  through the app's reference-PDF fetcher (the SharePoint
  ``primary_references/`` folder the app already reads UFCs from);
* ``{"concat": [ids...], "name": ...}`` — several of the above bound into one
  PDF, the way a drawing set arrives;
* ``{"fixture": "<name>"}`` — a synthetic PDF built by ``planlens.testing``
  (shipped with planlens, so it needs no upload).

On the cluster, copy the public files once into a folder (or a SharePoint
folder synced to one) and pass it as ``docs_dir``;
:func:`collect_public_docs` gathers them from a source checkout.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

DOCS_ENV = "GEOTECH_REVIEW_EVAL_DOCS"

_MECK = ["10.17a.pdf", "10.25a.pdf", "10.31A.pdf", "11.01.pdf", "2000a.pdf",
         "2000b.pdf", "21.01.pdf", "3000.pdf", "3001.pdf", "5003.pdf"]

DOCUMENTS: Dict[str, Dict[str, Any]] = {
    "meck_10.17a": {"file": "10.17a.pdf"},
    "meck_10.25a": {"file": "10.25a.pdf"},
    "meck_10.31a": {"file": "10.31A.pdf"},
    "meck_11.01": {"file": "11.01.pdf"},
    "meck_20.00a": {"file": "2000a.pdf"},
    "meck_20.00b": {"file": "2000b.pdf"},
    "meck_21.01": {"file": "21.01.pdf"},
    "meck_30.00": {"file": "3000.pdf"},
    "meck_30.01": {"file": "3001.pdf"},
    "meck_50.03": {"file": "5003.pdf"},
    "meck_set": {"concat": ["meck_10.17a", "meck_10.25a", "meck_10.31a",
                            "meck_11.01", "meck_20.00a", "meck_20.00b",
                            "meck_21.01", "meck_30.00", "meck_30.01",
                            "meck_50.03"],
                 "name": "mecklenburg_standard_details.pdf"},
    "ufc_3_220_04fa": {"file": "ufc_3_220_04fa_2004.pdf"},
    "ufc_3_220_07": {"file": "ufc_3_220_07.pdf"},
    "ufc_3_301_01": {"file": "UFC_3-301-01_2023_c4.pdf"},
    "ufc_3_260_02": {"file": "ufc_3_260_02_2001.pdf"},
    "calc_bearing": {"file": "sample_calc_package.pdf"},
    "calc_retaining_wall": {"file": "retaining_walls.pdf"},
    "fixture_review_document": {"fixture": "review_document",
                                "name": "review_set.pdf"},
    "fixture_submittal": {"fixture": "submittal", "name": "submittal.pdf"},
    # Three 11x17 sheets whose 0.06 in lettering is DRAWN (no text layer):
    # penetration-style tags with leaders, look-alikes, a legend on each.
    "fixture_tags": {"fixture": "tags", "name": "tag_set.pdf"},
    # The same sheets, 24 of them at an even density, with ONE FPG callout on
    # pages 4, 12 and 20 only (FBG, its look-alike, is on every sheet).
    "fixture_tags_long": {"fixture": "tags_long", "name": "long_tag_set.pdf"},
    # A 30-page geotechnical report (review_eval/report_fixture.py): new
    # boring logs as vector pages, older ones as scans, a laboratory appendix
    # of a summary table and 13 sheets - the shape of the 2026-10-06 field
    # session's report, every name and number invented.
    "fixture_report": {"fixture": "report",
                       "name": "harbour_road_geotechnical_report.pdf"},
    # Two pages with known scales (planlens.testing.visual_scale_fixtures,
    # W3): a scanned boring log with no text layer, and a site plan
    # re-plotted at half size whose scale note is wrong and whose scale bar
    # is right. The upload names say nothing of either.
    "fixture_scale_log": {"fixture": "scale_log", "name": "boring_log.pdf"},
    "fixture_scale_plan": {"fixture": "scale_plan", "name": "site_plan.pdf"},
}

#: The long tag set's rare tag and the 0-based pages that carry it.
LONG_SET_PAGES = 24
LONG_SET_RARE = {"FPG": [3, 11, 19]}

#: The scanned boring log of ``scale-log-depth`` (planlens ``LogVariant``
#: arguments). Parameters of its OWN, not one of the variants planlens'
#: measuring harness was gated on: an image-only page (no text layer) stored
#: ``/Rotate 270`` and skewed 0.4 deg as the 2026-10-06 session's scans are,
#: its depth labels printed on their baselines 1.5 pt ABOVE the depth each
#: marks - so a label's centre sits 4.7 pt (about 0.10 m) above its depth
#: and reading the centre as the depth is biased by that much.
SCALE_LOG = {"name": "suite_log_depth", "label_anchor": "baseline",
             "rotate270": True, "skew_deg": 0.4,
             "contacts": (1.64, 3.62, 4.87, 6.93), "seed": 211}
#: The contact the task asks about (index into ``contacts``): the top of the
#: third layer, "Brown, medium dense, silty sandy GRAVEL."
SCALE_LOG_CONTACT = 1

#: The site plan of ``scale-plan-distance`` (planlens ``PlanVariant``
#: arguments): drawn at half size on the same sheet, as a re-plot onto
#: smaller paper is. Its note still says 1" = 20' and is wrong; its graphic
#: bar shrank with the drawing and is right (1" = 40').
SCALE_PLAN = {"name": "replot_half", "replot": 0.5}
#: The two borings the task asks about.
SCALE_PLAN_PAIR = ("B-1", "B-4")


def scale_fixture(which: str):
    """The planlens ``ScaleFixture`` behind ``fixture_scale_log``
    (``which="log"``) or ``fixture_scale_plan`` (``"plan"``): the page and
    everything true about it. Raises ImportError on a planlens without the
    visual-scale fixtures."""
    try:
        from planlens.testing import visual_scale_fixtures as V
    except ImportError as exc:          # a planlens before the visual scales
        raise ImportError(f"needs a planlens with the visual-scale fixtures "
                          f"({exc})")
    if which == "log":
        return V.build_log(V.LogVariant(**SCALE_LOG))
    if which == "plan":
        return V.build_plan(V.PlanVariant(**SCALE_PLAN))
    raise KeyError(f"no scale fixture {which!r}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _dev_dirs() -> List[Path]:
    root = _repo_root()
    return [root / "module_work" / "drawing_ground_truth" / "mecklenburg",
            root / "geotech-references" / "docs", root / "sample_pdfs", root]


def _search_dirs(docs_dir: Any = None) -> List[Path]:
    out: List[Path] = []
    for d in (docs_dir, os.environ.get(DOCS_ENV),
              os.environ.get("GEOTECH_REFERENCES_DOCS")):
        if d:
            out.append(Path(os.fspath(d)))
    return out + _dev_dirs()


def find_file(name: str, docs_dir: Any = None) -> Optional[str]:
    """The path of a public document called ``name``, or ``None``."""
    for base in _search_dirs(docs_dir):
        if not base.is_dir():
            continue
        direct = base / name
        if direct.is_file():
            return str(direct)
        # One or two folders down, so a docs folder may keep its own layout.
        for sub in list(base.glob(f"*/{name}")) + list(base.glob(f"*/*/{name}")):
            if sub.is_file():
                return str(sub)
    try:
        # The app registers its SharePoint primary_references/ fetcher when
        # it builds an agent; a suite run looks documents up BEFORE that.
        from webapp.core import _register_reference_fetcher
        _register_reference_fetcher()
    except Exception:
        pass
    try:
        from funhouse_agent import reference_docs
        fetched = reference_docs.fetch(name)
    except Exception:
        fetched = None
    return fetched if fetched and os.path.isfile(fetched) else None


def _fixture(name: str) -> bytes:
    if name == "review_document":
        from planlens.testing.document_fixtures import (
            build_synthetic_review_document)
        return build_synthetic_review_document().pdf
    if name == "submittal":
        from planlens.testing.submittal_fixtures import (
            build_synthetic_submittal)
        return build_synthetic_submittal().pdf
    if name == "tags":
        from planlens.testing.tag_fixtures import build_synthetic_tag_set
        return build_synthetic_tag_set().pdf
    if name == "tags_long":
        from planlens.testing.tag_fixtures import build_synthetic_tag_set
        try:
            return build_synthetic_tag_set(
                n_pages=LONG_SET_PAGES, gce_growth=0,
                extra_callouts=LONG_SET_RARE).pdf
        except TypeError as exc:        # planlens older than 0.11
            raise ImportError(f"needs planlens 0.11 ({exc})")
    if name == "report":
        from funhouse_agent.review_eval.report_fixture import (
            build_synthetic_extraction_report)
        return build_synthetic_extraction_report().pdf
    if name == "scale_log":
        return scale_fixture("log").pdf
    if name == "scale_plan":
        return scale_fixture("plan").pdf
    raise KeyError(f"unknown fixture {name!r}")


def _concat(members: Sequence[bytes]) -> bytes:
    import fitz
    out = fitz.open()
    try:
        for data in members:
            src = fitz.open(stream=data, filetype="pdf")
            try:
                out.insert_pdf(src)
            finally:
                src.close()
        return out.tobytes(garbage=3, deflate=True)
    finally:
        out.close()


class MissingDocument(FileNotFoundError):
    """A task's document is not available here."""


def resolve(doc_id: str, docs_dir: Any = None) -> Tuple[str, bytes]:
    """``(upload name, PDF bytes)`` for ``doc_id``; raises
    :class:`MissingDocument` when it cannot be found."""
    spec = DOCUMENTS.get(doc_id)
    if spec is None:
        # A private task may name a file directly.
        spec = {"file": doc_id}
    if "file" in spec:
        path = find_file(spec["file"], docs_dir)
        if not path:
            raise MissingDocument(
                f"{spec['file']} not found (docs_dir={docs_dir!r}); copy the "
                f"public documents there with collect_public_docs()")
        with open(path, "rb") as fh:
            return spec.get("name") or os.path.basename(path), fh.read()
    if "concat" in spec:
        parts = [resolve(m, docs_dir)[1] for m in spec["concat"]]
        return spec.get("name") or f"{doc_id}.pdf", _concat(parts)
    if "fixture" in spec:
        try:
            data = _fixture(spec["fixture"])
        except ImportError as exc:
            raise MissingDocument(f"fixture {spec['fixture']}: {exc}")
        return spec.get("name") or f"{doc_id}.pdf", data
    raise MissingDocument(f"document {doc_id!r} has no source")


def public_files() -> List[str]:
    """Every file name the shipped tasks need (no fixtures, no sets)."""
    return sorted({s["file"] for s in DOCUMENTS.values() if "file" in s})


def collect_public_docs(dest: Any, docs_dir: Any = None) -> Dict[str, Any]:
    """Copy every public document the suite uses into ``dest`` (for upload).

    Returns ``{"copied": [...], "missing": [...]}``. Run it once on a machine
    with the source checkout, then upload ``dest`` to the cluster.
    """
    dest = Path(os.fspath(dest))
    dest.mkdir(parents=True, exist_ok=True)
    copied, missing = [], []
    for name in public_files():
        path = find_file(name, docs_dir)
        if not path:
            missing.append(name)
            continue
        target = dest / name
        if not target.exists() or target.stat().st_size != os.path.getsize(path):
            shutil.copy2(path, target)
        copied.append(name)
    return {"copied": copied, "missing": missing, "dest": str(dest)}


__all__ = ["DOCUMENTS", "DOCS_ENV", "MissingDocument", "resolve", "find_file",
           "public_files", "collect_public_docs", "scale_fixture",
           "SCALE_LOG", "SCALE_LOG_CONTACT", "SCALE_PLAN", "SCALE_PLAN_PAIR"]
