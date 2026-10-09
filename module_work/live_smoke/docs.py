"""PUBLIC documents only: resolve a flow's document references, and refuse
anything that is not public.

A reference is one of:

* a Document Review suite id (``meck_10.17a``, ``meck_set``,
  ``ufc_3_220_04fa``, ``calc_bearing``, ``fixture_report``, ...), resolved by
  ``funhouse_agent.review_eval.documents.resolve`` -- the Mecklenburg County
  standard details, UFC manuals, the app's own sample calc packages and
  planlens' synthetic fixtures;
* ``{"file": "<name>"}`` -- a PDF in ``geotech-references/docs`` (FHWA / UFC
  only, see :data:`PUBLIC_REFERENCE_PREFIXES`);
* ``{"sample": "<name>"}`` -- a file in ``funhouse_agent/eval_samples``;
* ``{"render": {"doc": <ref>, "page": 0, "clip": [x0, y0, x1, y1],
  "zoom": 2}}`` -- a PNG cut from one of the above (a "screenshot").

Every file read from disk passes :func:`assert_public`, which refuses the
private folders (field feedback, any ``raw/``, Tiny Apps reference,
Funhouse reference) outright and accepts only the known public roots.
"""

from __future__ import annotations

import os
from typing import Any, Tuple

from live_smoke.watch import REPO

#: Folders whose contents may be sent to the API.
PUBLIC_ROOTS = (
    os.path.join(REPO, "module_work", "drawing_ground_truth", "mecklenburg"),
    os.path.join(REPO, "geotech-references", "docs"),
    os.path.join(REPO, "sample_pdfs"),
    os.path.join(REPO, "funhouse_agent", "eval_samples"),
)
#: Single public files at the repo root (the app's own sample output).
PUBLIC_FILES = (os.path.join(REPO, "sample_calc_package.pdf"),)

#: Of geotech-references/docs, only US federal publications (FHWA, UFC, and
#: the USACE/USDA/GSA ones) are public domain. AASHTO and the Eurocodes are
#: not, and are refused.
PUBLIC_REFERENCE_PREFIXES = ("fhwa", "gec ", "ufc", "em ", "em_",
                             "usda", "gsa_", "california trenching",
                             "fema")

#: Never, whatever the path looks like.
FORBIDDEN_PARTS = ("field_feedback", os.sep + "raw" + os.sep,
                   os.path.join("tinyapps", "reference"),
                   "funhouse_for_reference", "report_ingest_harness",
                   "drawing_ground_truth" + os.sep + "private")


class NotPublic(PermissionError):
    """A document outside the public roots (never sent to the API)."""


def _norm(p: str) -> str:
    return os.path.normcase(os.path.abspath(p))


def assert_public(path: str) -> str:
    """``path`` if it is a public document, else raise :class:`NotPublic`."""
    ap = _norm(path)
    low = ap.lower()
    for bad in FORBIDDEN_PARTS:
        if os.path.normcase(bad).lower() in low:
            raise NotPublic(f"refused (private folder): {path}")
    if ap in {_norm(f) for f in PUBLIC_FILES}:
        return path
    for root in PUBLIC_ROOTS:
        r = _norm(root)
        if ap.startswith(r + os.sep):
            if r == _norm(PUBLIC_ROOTS[1]):
                name = os.path.basename(ap).lower()
                if not name.startswith(PUBLIC_REFERENCE_PREFIXES):
                    raise NotPublic(f"refused (not a public-domain "
                                    f"reference): {os.path.basename(path)}")
            return path
    raise NotPublic(f"refused (outside the public roots): {path}")


def _render(spec: dict) -> Tuple[str, bytes]:
    import fitz
    name, data = resolve(spec["doc"])
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc[int(spec.get("page", 0))]
        r = page.rect
        x0, y0, x1, y1 = spec.get("clip") or (0.0, 0.0, 1.0, 1.0)
        clip = fitz.Rect(r.x0 + x0 * r.width, r.y0 + y0 * r.height,
                         r.x0 + x1 * r.width, r.y0 + y1 * r.height)
        z = float(spec.get("zoom", 2.0))
        pix = page.get_pixmap(matrix=fitz.Matrix(z, z), clip=clip)
        return spec.get("name") or "image.png", pix.tobytes("png")
    finally:
        doc.close()


def resolve(ref: Any) -> Tuple[str, bytes]:
    """``(upload name, bytes)`` for a flow's document reference."""
    if isinstance(ref, dict) and "render" in ref:
        return _render(ref["render"])
    if isinstance(ref, dict) and "file" in ref:
        path = os.path.join(PUBLIC_ROOTS[1], ref["file"])
        assert_public(path)
        with open(path, "rb") as fh:
            return ref.get("name") or os.path.basename(path), fh.read()
    if isinstance(ref, dict) and "sample" in ref:
        path = os.path.join(PUBLIC_ROOTS[3], ref["sample"])
        assert_public(path)
        with open(path, "rb") as fh:
            return ref.get("name") or os.path.basename(path), fh.read()
    if isinstance(ref, str):
        from funhouse_agent.review_eval import documents as D
        spec = D.DOCUMENTS.get(ref)
        if spec is None:
            raise KeyError(f"unknown document id {ref!r}")
        _check_spec(ref, spec, D)
        return D.resolve(ref)
    raise TypeError(f"not a document reference: {ref!r}")


def _check_spec(ref: str, spec: dict, D) -> None:
    """Every on-disk file behind a suite id must be public."""
    if "file" in spec:
        path = D.find_file(spec["file"])
        if not path:
            raise FileNotFoundError(f"{ref}: {spec['file']} not found")
        assert_public(path)
    for member in spec.get("concat") or []:
        _check_spec(member, D.DOCUMENTS[member], D)


def input_roots() -> list:
    """Where the app may legitimately READ from outside a conversation."""
    return list(PUBLIC_ROOTS) + list(PUBLIC_FILES)


__all__ = ["resolve", "assert_public", "NotPublic", "PUBLIC_ROOTS",
           "input_roots"]
