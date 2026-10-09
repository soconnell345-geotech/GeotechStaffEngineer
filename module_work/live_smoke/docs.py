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
  "zoom": 2}}`` -- a PNG cut from one of the above (a "screenshot");
* ``{"doc": <ref>, "name": "C-101.pdf"}`` -- any of the above under another
  upload name;
* ``{"make": {"kind": ..., ...}}`` -- a file generated here from the above or
  from invented text (:func:`_make`): ``set`` (sheets bound into one PDF,
  each stamped with a project sheet number), ``spec`` (an invented
  specification section, PDF or DOCX, written to agree with some Mecklenburg
  details and contradict others), ``submittal_log`` (an XLSX),
  ``dxf_plan`` (a small DXF site plan), ``photo`` (a JPEG "phone photo" of a
  page), ``image_pdf`` (pages re-made as images: a scan), ``rotated`` (a
  sideways sheet), ``blank`` (empty pages) and ``planlens_report`` (planlens'
  22-page synthetic report).

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


# ---------------------------------------------------------------------------
# Generated documents ({"make": {...}})
# ---------------------------------------------------------------------------

#: The invented project every generated document belongs to.
PROJECT = "RIVERSIDE DRIVE STREETSCAPE AND DRAINAGE IMPROVEMENTS"
PROJECT_NO = "24-117"

#: An INVENTED specification section written against the Mecklenburg
#: standard details in ``meck_set`` (sheet C-n = the n-th detail of the set,
#: :data:`SET_ORDER`). Some paragraphs agree with the drawings; these do NOT
#: (the drawing's own value in brackets): 2.1 A concrete 3,000 psi [10.25A
#: note 1: 3600 psi]; 2.2 A underdrain 4-inch SDR 35, perforations 6 in. on
#: centre [21.01 note 5: min. 6-inch Schedule 40, 3 in. on centre]; 2.3 A
#: filter media 18 in. [21.01: 2'-0" to 4'-0"]; 3.2 B breakover 10 % [10.25A
#: note 4: 8 % or less]; 3.3 A ramp running slope 1:10 [10.31A: 8.33 % max];
#: 3.6 A sediment trap 1,800 cf/acre [30.01: 3600 cf/acre]; 3.7 A monuments
#: 24 in. deep [50.03: 30 in.]. 3.3 A's 2.0 % cross slope is stricter than
#: 10.31A's 2.1 %.
SPEC_SECTION = "32 16 00"
SPEC_TITLE = ("SECTION 32 16 00 - SITE CONCRETE, STREET DETAILS AND SITE "
              "DRAINAGE")
SPEC_LINES = [
    ("h", "PART 1 - GENERAL"),
    ("p", "1.1 SUMMARY"),
    ("i", "A. This Section covers concrete curb and gutter, sidewalks, "
          "driveway aprons, curb ramps, street pavement, bioretention cells, "
          "temporary sediment traps and survey control monuments shown on "
          "the Drawings (sheets C-1 through C-10)."),
    ("i", "B. Standard details are the Mecklenburg County Land Development "
          "Standards (MCLDS) as referenced on the Drawings."),
    ("p", "1.2 SUBMITTALS"),
    ("i", "A. Concrete mix design for each class of concrete, with 28-day "
          "compressive strength test data."),
    ("i", "B. Product data for underdrain pipe, filter fabric, detectable "
          "warning surfaces and washed stone."),
    ("i", "C. Gradation and soil analysis of the bioretention filter media."),
    ("p", "1.3 QUALITY ASSURANCE"),
    ("i", "A. Where this Section and the Drawings disagree, the more "
          "stringent requirement governs unless the Engineer directs "
          "otherwise in writing."),
    ("h", "PART 2 - PRODUCTS"),
    ("p", "2.1 CONCRETE"),
    ("i", "A. Concrete for curb and gutter, sidewalks and driveway aprons: "
          "Class B, minimum 28-day compressive strength 3,000 psi, air "
          "entrained 5 to 7 percent."),
    ("i", "B. Expansion joint filler: 1/2-inch preformed, non-extruding."),
    ("p", "2.2 UNDERDRAIN AND STONE"),
    ("i", "A. Bioretention underdrain: 4-inch perforated PVC, SDR 35, with "
          "3/8-inch perforations spaced 6 inches on center in four rows."),
    ("i", "B. Washed stone for the underdrain envelope and the sediment trap "
          "outlet: NCDOT No. 57."),
    ("p", "2.3 BIORETENTION FILTER MEDIA"),
    ("i", "A. Filter media depth: 18 inches minimum, placed in 9-inch lifts "
          "and not compacted."),
    ("p", "2.4 DETECTABLE WARNING SURFACES"),
    ("i", "A. Detectable warning mats per MCLDS 10.35B."),
    ("h", "PART 3 - EXECUTION"),
    ("p", "3.1 CURB AND GUTTER"),
    ("i", "A. Standard curb and gutter: 2'-6\" wide per MCLDS 10.17A; 2'-0\" "
          "valley gutter where shown."),
    ("i", "B. Remove existing curb to the nearest joint, or saw cut "
          "perpendicular to the edge of pavement."),
    ("p", "3.2 SIDEWALKS AND DRIVEWAYS"),
    ("i", "A. Sidewalks: 4 inches thick, a minimum of 4 feet from back of "
          "curb, cross slope 1/4 inch per foot."),
    ("i", "B. Driveway aprons: 6 inches thick. The algebraic difference in "
          "grade at the driveway breakover shall not exceed 10 percent."),
    ("p", "3.3 CURB RAMPS"),
    ("i", "A. Ramp running slope shall not exceed 1:10 (10 percent). Landing "
          "and cross slope shall not exceed 2.0 percent."),
    ("i", "B. Provide a flush transition from the ramp to the gutter."),
    ("p", "3.4 STREET PAVEMENT"),
    ("i", "A. Local residential street: 8-inch compacted aggregate base "
          "course, 1-1/2 inch S9.5B intermediate course and 1-1/2 inch S9.5B "
          "surface course."),
    ("p", "3.5 BIORETENTION CELLS"),
    ("i", "A. Maximum ponding depth 12 inches. Underdrain cleanouts shall "
          "extend at least 6 inches above the maximum ponding elevation."),
    ("i", "B. Provide a 20-foot access easement to a public right-of-way."),
    ("p", "3.6 TEMPORARY SEDIMENT TRAPS"),
    ("i", "A. Provide a minimum storage volume of 1,800 cubic feet per acre "
          "of disturbed drainage area."),
    ("i", "B. Outlet of NCDOT No. 57 washed stone, 12 inches minimum; "
          "emergency bypass 6 inches below the settled top of the dam."),
    ("p", "3.7 SURVEY CONTROL MONUMENTS"),
    ("i", "A. Concrete control monuments with brass plate, minimum 24 inches "
          "deep."),
    ("h", "END OF SECTION"),
]

#: The details of ``meck_set`` in order: sheet C-n is the n-th.
SET_ORDER = ["meck_10.17a", "meck_10.25a", "meck_10.31a", "meck_11.01",
             "meck_20.00a", "meck_20.00b", "meck_21.01", "meck_30.00",
             "meck_30.01", "meck_50.03"]

#: An invented submittal log (rows) for the same project, with the same
#: disagreements as the spec plus a duplicate number and a missing status.
SUBMITTAL_LOG = [
    ("32 16 00-001", "Concrete mix design, Class B (3,000 psi, 6% air)", 0,
     "2026-09-02", "Approved", "J. Patel", ""),
    ("32 16 00-002", "Expansion joint filler, 1/2 in. - product data", 0,
     "2026-09-02", "Approved", "J. Patel", ""),
    ("32 16 00-003", "Detectable warning mats - product data", 0,
     "2026-09-09", "Revise and Resubmit", "J. Patel", "Colour not shown"),
    ("32 16 00-004", "Curb ramp layout shop drawing (10% max running slope)",
     0, "2026-09-22", "", "", ""),
    ("32 16 00-005", "Bioretention underdrain - 4 in. perforated PVC SDR 35",
     0, "2026-09-11", "Approved as Noted", "R. Okafor", "Verify perf spacing"),
    ("32 16 00-006", "Bioretention filter media gradation + soil analysis",
     1, "2026-09-18", "Under Review", "R. Okafor", ""),
    ("32 16 00-007", "Sediment trap sizing (1,800 cf/ac) and No. 57 stone",
     0, "2026-09-05", "Approved", "R. Okafor", ""),
    ("32 16 00-001", "Concrete mix design, Class B - resubmittal", 1,
     "2026-09-25", "Under Review", "J. Patel", "Now 3,600 psi"),
    ("32 16 00-008", "Survey control monument detail, 24 in. brass cap", 0,
     "2026-09-26", "", "", ""),
]


def _spec_pdf() -> bytes:
    import fitz
    doc = fitz.open()
    W, H, L, R, TOP, BOT = 612.0, 792.0, 72.0, 540.0, 96.0, 720.0
    page, y = None, 0.0

    def new_page():
        p = doc.new_page(width=W, height=H)
        p.insert_text((L, 48), f"{PROJECT}", fontsize=7.5, fontname="helv")
        p.insert_text((L, 58), f"Project No. {PROJECT_NO}", fontsize=7.5,
                      fontname="helv")
        p.insert_text((W / 2 - 30, 760),
                      f"{SPEC_SECTION} - {doc.page_count}", fontsize=8,
                      fontname="helv")
        return p, TOP

    page, y = new_page()
    page.insert_text((L, 80), SPEC_TITLE, fontsize=10.5, fontname="hebo")
    for style, text in SPEC_LINES:
        font = "hebo" if style in ("h", "p") else "helv"
        size = 9.5
        x0 = L + (24 if style == "i" else 0)
        words, line, lines = text.split(), "", []
        for w in words:
            trial = (line + " " + w).strip()
            if fitz.get_text_length(trial, fontname=font,
                                    fontsize=size) > R - x0:
                lines.append(line)
                line = w
            else:
                line = trial
        lines.append(line)
        y += 6 if style in ("h", "p") else 2
        for ln in lines:
            if y > BOT:
                page, y = new_page()
            page.insert_text((x0, y), ln, fontsize=size, fontname=font)
            y += size * 1.35
    return doc.tobytes(garbage=3, deflate=True)


def _spec_docx() -> bytes:
    import io
    import docx
    d = docx.Document()
    sec = d.sections[0]
    sec.header.paragraphs[0].text = f"{PROJECT} - Project No. {PROJECT_NO}"
    sec.footer.paragraphs[0].text = f"Section {SPEC_SECTION}"
    d.add_heading(SPEC_TITLE, level=1)
    for style, text in SPEC_LINES:
        if style == "h":
            d.add_heading(text, level=2)
        elif style == "p":
            d.add_paragraph().add_run(text).bold = True
        else:
            d.add_paragraph(text).paragraph_format.left_indent = \
                docx.shared.Inches(0.4)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()


def _submittal_xlsx() -> bytes:
    import io
    import openpyxl
    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "Submittal Log"
    ws.append([f"{PROJECT} - SUBMITTAL LOG"])
    ws.append([f"Project No. {PROJECT_NO}", "", "", "Updated 2026-09-28"])
    ws.append([])
    ws.append(["Submittal No.", "Description", "Rev", "Received", "Status",
               "Reviewer", "Remarks"])
    for row in SUBMITTAL_LOG:
        ws.append(list(row))
    for col, width in zip("ABCDEFG", (15, 52, 5, 12, 20, 11, 22)):
        ws.column_dimensions[col].width = width
    buf = io.BytesIO()
    wb.save(buf)
    return buf.getvalue()


def _dxf_plan() -> bytes:
    """A small boring-location plan in feet (invented)."""
    import io
    import ezdxf
    doc = ezdxf.new("R2010")
    doc.header["$INSUNITS"] = 2                       # feet
    msp = doc.modelspace()
    for name in ("PROPERTY", "BUILDING", "BORINGS", "TEXT", "TITLE"):
        doc.layers.add(name)
    msp.add_lwpolyline([(0, 0), (300, 0), (300, 200), (0, 200)], close=True,
                       dxfattribs={"layer": "PROPERTY"})
    msp.add_lwpolyline([(80, 60), (200, 60), (200, 140), (80, 140)],
                       close=True, dxfattribs={"layer": "BUILDING"})
    msp.add_text("PROPOSED BUILDING  FF EL 714.50", height=4,
                 dxfattribs={"layer": "TEXT", "insert": (95, 98)})
    borings = [("B-1", 70, 50, 712.4), ("B-2", 210, 50, 711.8),
               ("B-3", 210, 150, 713.1), ("B-4", 70, 150, 713.6),
               ("B-5", 140, 100, 712.9)]
    for tag, x, y, el in borings:
        msp.add_circle((x, y), 2.5, dxfattribs={"layer": "BORINGS"})
        msp.add_text(tag, height=4,
                     dxfattribs={"layer": "BORINGS", "insert": (x + 4, y + 2)})
        msp.add_text(f"GS EL {el}", height=2.5,
                     dxfattribs={"layer": "BORINGS",
                                 "insert": (x + 4, y - 3)})
    msp.add_line((280, 170), (280, 190), dxfattribs={"layer": "TEXT"})
    msp.add_text("N", height=6, dxfattribs={"layer": "TEXT",
                                            "insert": (277, 192)})
    for i, t in enumerate((PROJECT, "BORING LOCATION PLAN",
                           "SCALE: 1\" = 40'", "SHEET C-2",
                           "BORING LOCATIONS APPROXIMATE - BY OTHERS")):
        msp.add_text(t, height=4 if i < 2 else 3,
                     dxfattribs={"layer": "TITLE",
                                 "insert": (0, -12 - 7 * i)})
    buf = io.StringIO()
    doc.write(buf)
    return buf.getvalue().encode("utf-8")


def _set_pdf(spec: dict) -> Tuple[str, bytes]:
    """Sheets bound into one PDF, each stamped bottom-right with the
    project and ``SHEET <prefix><n> OF <N>`` (in the text layer, in the
    displayed frame, so a rotated sheet's stamp reads upright)."""
    import fitz
    refs = spec.get("docs") or SET_ORDER
    prefix = spec.get("sheet_prefix", "C-")
    out = fitz.open()
    try:
        for r in refs:
            src = fitz.open(stream=resolve(r)[1], filetype="pdf")
            try:
                out.insert_pdf(src)
            finally:
                src.close()
        n = out.page_count
        for i, page in enumerate(out, 1):
            r = page.rect                                  # displayed
            box = fitz.Rect(r.width - 312, r.height - 44, r.width - 12,
                            r.height - 12)
            unrot = box * page.derotation_matrix
            unrot.normalize()
            page.draw_rect(unrot, color=(0, 0, 0), fill=(1, 1, 1),
                           width=0.6)
            left = page.insert_textbox(
                unrot + (4, 3, -4, -3),
                f"{spec.get('project', PROJECT)}\n"
                f"SHEET {prefix}{i} OF {n}", fontsize=7, fontname="helv",
                rotate=page.rotation)
            if left < 0:
                raise ValueError(f"sheet stamp does not fit on page {i}")
        return (spec.get("name") or "civil_set.pdf",
                out.tobytes(garbage=3, deflate=True))
    finally:
        out.close()


def _photo(spec: dict) -> Tuple[str, bytes]:
    """A JPEG "phone photo" of a page: tilted, a little soft, compressed."""
    import io
    import fitz
    from PIL import Image, ImageFilter
    _n, data = resolve(spec["doc"])
    doc = fitz.open(stream=data, filetype="pdf")
    try:
        page = doc[int(spec.get("page", 0))]
        pix = page.get_pixmap(dpi=int(spec.get("dpi", 150)))
        img = Image.frombytes("RGB", (pix.width, pix.height), pix.samples)
    finally:
        doc.close()
    img = img.rotate(float(spec.get("tilt", 4.0)), expand=True,
                     resample=Image.BICUBIC, fillcolor=(156, 142, 120))
    img = img.filter(ImageFilter.GaussianBlur(float(spec.get("blur", 0.8))))
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=int(spec.get("quality", 70)))
    return spec.get("name") or "IMG_4471.jpg", buf.getvalue()


def _image_pdf(spec: dict) -> Tuple[str, bytes]:
    """The document's pages re-made as greyscale JPEG images (a scan: no
    text layer), optionally a fraction of a degree askew."""
    import io
    import fitz
    from PIL import Image
    name, data = resolve(spec["doc"])
    src = fitz.open(stream=data, filetype="pdf")
    out = fitz.open()
    try:
        pages = spec.get("pages") or list(range(src.page_count))
        for pno in pages:
            page = src[int(pno)]
            pix = page.get_pixmap(dpi=int(spec.get("dpi", 110)),
                                  colorspace=fitz.csGRAY)
            img = Image.frombytes("L", (pix.width, pix.height), pix.samples)
            skew = float(spec.get("skew", 0.0))
            if skew:
                img = img.rotate(skew, resample=Image.BICUBIC, fillcolor=255)
            buf = io.BytesIO()
            img.save(buf, "JPEG", quality=int(spec.get("quality", 55)))
            new = out.new_page(width=page.rect.width,
                               height=page.rect.height)
            new.insert_image(new.rect, stream=buf.getvalue())
        return (spec.get("name") or f"scan_{name}",
                out.tobytes(garbage=3, deflate=True))
    finally:
        src.close()
        out.close()


def _rotated(spec: dict) -> Tuple[str, bytes]:
    """A sideways sheet. ``mode="content"`` (default): each page's content
    drawn turned ``rotate`` degrees on a page of swapped size, as a sheet
    plotted sideways arrives (use an unrotated source); ``"flag"``: the
    page's ``/Rotate`` changed by ``rotate``."""
    import fitz
    name, data = resolve(spec["doc"])
    rot = int(spec.get("rotate", 90))
    src = fitz.open(stream=data, filetype="pdf")
    try:
        if spec.get("mode", "content") == "flag":
            for page in src:
                page.set_rotation((page.rotation + rot) % 360)
            return spec.get("name") or name, src.tobytes(garbage=3,
                                                          deflate=True)
        out = fitz.open()
        try:
            for pno, page in enumerate(src):
                w, h = page.rect.width, page.rect.height
                new = out.new_page(width=h, height=w) if rot % 180 else \
                    out.new_page(width=w, height=h)
                new.show_pdf_page(new.rect, src, pno, rotate=rot)
            return spec.get("name") or name, out.tobytes(garbage=3,
                                                         deflate=True)
        finally:
            out.close()
    finally:
        src.close()


def _blank(spec: dict) -> Tuple[str, bytes]:
    import fitz
    doc = fitz.open()
    for _ in range(int(spec.get("pages", 1))):
        doc.new_page(width=612, height=792)
    return spec.get("name") or "scan_0001.pdf", doc.tobytes()


def _make(spec: dict) -> Tuple[str, bytes]:
    """``(name, bytes)`` for a generated document (see the module doc)."""
    kind = spec.get("kind")
    if kind == "set":
        return _set_pdf(spec)
    if kind == "spec":
        if spec.get("format", "pdf") == "docx":
            return (spec.get("name") or f"Section {SPEC_SECTION}.docx",
                    _spec_docx())
        return spec.get("name") or f"Section {SPEC_SECTION}.pdf", _spec_pdf()
    if kind == "submittal_log":
        return spec.get("name") or "Submittal Log.xlsx", _submittal_xlsx()
    if kind == "dxf_plan":
        return spec.get("name") or "C-2 Boring Plan.dxf", _dxf_plan()
    if kind == "photo":
        return _photo(spec)
    if kind == "image_pdf":
        return _image_pdf(spec)
    if kind == "rotated":
        return _rotated(spec)
    if kind == "blank":
        return _blank(spec)
    if kind == "planlens_report":
        from planlens.testing.report_fixtures import build_synthetic_report
        return (spec.get("name") or "geotechnical_report_with_appendices.pdf",
                build_synthetic_report().pdf)
    raise KeyError(f"unknown generated document kind {kind!r}")


def resolve(ref: Any) -> Tuple[str, bytes]:
    """``(upload name, bytes)`` for a flow's document reference."""
    if isinstance(ref, dict) and "make" in ref:
        return _make(ref["make"])
    if isinstance(ref, dict) and "doc" in ref and "name" in ref:
        return ref["name"], resolve(ref["doc"])[1]
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
           "input_roots", "SPEC_LINES", "SET_ORDER", "SUBMITTAL_LOG"]
