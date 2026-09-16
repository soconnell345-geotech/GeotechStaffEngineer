"""Probe: what measurement calibration do real PDFs actually store?

Step 1 of the planlens "read the stored scale" feature. Before designing a
parser we look at what Bluebeam / Acrobat really write into these files:

- page level: the page dict's /VP array of Viewport dicts, each with /BBox,
  optional /Name and a /Measure dict (/Subtype /RL rectilinear, or /GEO);
- annotation level: measurement markups (/IT ...Dimension) carrying their own
  /Measure and a /Contents that states the measured value.

Run:
    <python> probe_measure.py            # both corpora
    <python> probe_measure.py <pdf> ...  # specific files

Nothing here is imported by planlens; it exists to produce the numbers the
design is built on. Output goes to stdout only (no files written).
"""

from __future__ import annotations

import glob
import re
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import fitz

SUBMITTAL = (r"C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\module_work"
             r"\field_feedback\2026-09-09_nairobi-soe_v5.11.2\raw\files"
             r"\315000-001-00 Excavation Support Dwg & Calcs.pdf")
MECKLENBURG = (r"C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\module_work"
               r"\drawing_ground_truth\mecklenburg\*.pdf")


# ---------------------------------------------------------------------------
# A tolerant parser for the subset of PDF object syntax we need.
# ---------------------------------------------------------------------------

class Ref:
    __slots__ = ("num",)

    def __init__(self, num: int) -> None:
        self.num = num

    def __repr__(self) -> str:
        return f"Ref({self.num})"


_TOKEN = re.compile(rb"""
    (?P<dict_open><<) | (?P<dict_close>>>) |
    (?P<arr_open>\[)  | (?P<arr_close>\]) |
    (?P<name>/[^\s/\[\]<>(){}%]*) |
    (?P<num>[+-]?(?:\d+\.\d*|\.\d+|\d+)) |
    (?P<hex><[0-9A-Fa-f\s]*>) |
    (?P<kw>true|false|null|R|obj|endobj|stream) |
    (?P<junk>\S)
""", re.VERBOSE)


def _lex(data: bytes) -> List[Tuple[str, bytes]]:
    out: List[Tuple[str, bytes]] = []
    i, n = 0, len(data)
    while i < n:
        c = data[i:i + 1]
        if c.isspace():
            i += 1
            continue
        if c == b"%":                       # comment to end of line
            j = data.find(b"\n", i)
            i = n if j < 0 else j + 1
            continue
        if c == b"(":                       # literal string, balanced parens
            depth, j = 1, i + 1
            buf = bytearray()
            while j < n and depth:
                ch = data[j:j + 1]
                if ch == b"\\":
                    buf += data[j:j + 2]
                    j += 2
                    continue
                if ch == b"(":
                    depth += 1
                elif ch == b")":
                    depth -= 1
                    if not depth:
                        break
                buf += ch
                j += 1
            out.append(("str", bytes(buf)))
            i = j + 1
            continue
        m = _TOKEN.match(data, i)
        if not m:
            i += 1
            continue
        kind = m.lastgroup
        out.append((kind, m.group(0)))
        i = m.end()
    return out


def _parse(tokens: List[Tuple[str, bytes]], i: int = 0) -> Tuple[Any, int]:
    kind, tok = tokens[i]
    if kind == "dict_open":
        d: Dict[str, Any] = {}
        i += 1
        while i < len(tokens) and tokens[i][0] != "dict_close":
            if tokens[i][0] != "name":
                i += 1
                continue
            key = tokens[i][1][1:].decode("latin-1")
            val, i = _parse(tokens, i + 1)
            d[key] = val
        return d, i + 1
    if kind == "arr_open":
        a: List[Any] = []
        i += 1
        while i < len(tokens) and tokens[i][0] != "arr_close":
            val, i = _parse(tokens, i)
            a.append(val)
        return a, i + 1
    if kind == "name":
        return "/" + tok[1:].decode("latin-1"), i + 1
    if kind == "num":
        # "12 0 R" is an indirect reference, not three tokens.
        if (i + 2 < len(tokens) and tokens[i + 1][0] == "num"
                and tokens[i + 2] == ("kw", b"R")):
            return Ref(int(float(tok))), i + 3
        txt = tok.decode("latin-1")
        return (int(txt) if re.fullmatch(r"[+-]?\d+", txt)
                else float(txt)), i + 1
    if kind == "str":
        return tok.decode("latin-1"), i + 1
    if kind == "hex":
        raw = re.sub(rb"\s", b"", tok[1:-1])
        try:
            return bytes.fromhex(raw.decode("ascii")).decode(
                "utf-16-be" if raw[:4].lower() == b"feff" else "latin-1"), i + 1
        except Exception:
            return tok.decode("latin-1"), i + 1
    if kind == "kw":
        return {b"true": True, b"false": False}.get(tok, None), i + 1
    return None, i + 1


def parse_object(raw: str) -> Any:
    data = raw.encode("latin-1", "replace")
    tokens = _lex(data)
    # skip a leading "N G obj"
    for j in range(len(tokens)):
        if tokens[j] == ("kw", b"obj"):
            tokens = tokens[j + 1:]
            break
    if not tokens:
        return None
    try:
        val, _ = _parse(tokens, 0)
    except Exception as exc:               # pragma: no cover - probe only
        return {"__parse_error__": str(exc)}
    return val


def resolve(doc, val, depth: int = 0):
    """Follow indirect references (one object deep at a time, bounded)."""
    while isinstance(val, Ref) and depth < 8:
        val = parse_object(doc.xref_object(val.num, compressed=False))
        depth += 1
    return val


def raw_of(doc, val) -> str:
    if isinstance(val, Ref):
        return doc.xref_object(val.num, compressed=False)
    return str(val)


# ---------------------------------------------------------------------------
# Probing
# ---------------------------------------------------------------------------

def page_vp_raw(doc, page) -> Optional[str]:
    kind, val = doc.xref_get_key(page.xref, "VP")
    if kind in (None, "null") or not val:
        return None
    if kind == "xref":
        num = int(val.split()[0])
        return doc.xref_object(num, compressed=False)
    return val


def probe(path: str, max_dump: int = 1) -> Dict[str, Any]:
    doc = fitz.open(path)
    t0 = time.perf_counter()
    vp_pages: List[int] = []
    empty_vp_pages: List[int] = []
    bbox_frame: List[str] = []
    ratios: Dict[str, int] = {}
    measure_keys: Dict[str, int] = {}
    subtypes: Dict[str, int] = {}
    vp_keys: Dict[str, int] = {}
    n_viewports = 0
    dumped_vp: List[str] = []

    for i in range(doc.page_count):
        page = doc[i]
        raw = page_vp_raw(doc, page)
        if not raw:
            continue
        arr = parse_object(raw)
        if not isinstance(arr, list):
            arr = [arr]
        if not arr:
            # Bluebeam writes /VP [] on pages where no scale was stored.
            empty_vp_pages.append(i)
            continue
        vp_pages.append(i)
        for entry in arr:
            entry_raw = raw_of(doc, entry)
            vp = resolve(doc, entry)
            if not isinstance(vp, dict):
                continue
            n_viewports += 1
            for k in vp:
                vp_keys[k] = vp_keys.get(k, 0) + 1
            bbox_frame.append(_which_frame(page, vp.get("BBox")))
            meas = resolve(doc, vp.get("Measure"))
            if isinstance(meas, dict):
                for k in meas:
                    measure_keys[k] = measure_keys.get(k, 0) + 1
                st = str(meas.get("Subtype"))
                subtypes[st] = subtypes.get(st, 0) + 1
                r = meas.get("R")
                if isinstance(r, str):
                    ratios[r] = ratios.get(r, 0) + 1
            if len(dumped_vp) < max_dump:
                dumped_vp.append(_dump_vp(doc, entry_raw, vp))
    vp_seconds = time.perf_counter() - t0

    # -- annotations ------------------------------------------------------
    t1 = time.perf_counter()
    ann_pages: List[int] = []
    it_values: Dict[str, int] = {}
    ann_measure_keys: Dict[str, int] = {}
    ann_subtypes: Dict[str, int] = {}
    ann_ratios: Dict[str, int] = {}
    contents: List[str] = []
    n_meas_annots = 0
    n_annots = 0
    annot_types: Dict[str, int] = {}
    dumped_ann: List[str] = []

    for i in range(doc.page_count):
        page = doc[i]
        hit = False
        for annot in page.annots() or []:
            n_annots += 1
            xref = annot.xref
            obj = parse_object(doc.xref_object(xref, compressed=False))
            if not isinstance(obj, dict):
                continue
            it = obj.get("IT")
            has_measure = "Measure" in obj
            if not has_measure and not (isinstance(it, str)
                                        and it.endswith("Dimension")):
                continue
            n_meas_annots += 1
            hit = True
            annot_types[str(obj.get("Subtype"))] = (
                annot_types.get(str(obj.get("Subtype")), 0) + 1)
            if isinstance(it, str):
                it_values[it] = it_values.get(it, 0) + 1
            meas = resolve(doc, obj.get("Measure"))
            if isinstance(meas, dict):
                for k in meas:
                    ann_measure_keys[k] = ann_measure_keys.get(k, 0) + 1
                st = str(meas.get("Subtype"))
                ann_subtypes[st] = ann_subtypes.get(st, 0) + 1
                r = meas.get("R")
                if isinstance(r, str):
                    ann_ratios[r] = ann_ratios.get(r, 0) + 1
            c = obj.get("Contents")
            if isinstance(c, str) and c.strip() and len(contents) < 25:
                contents.append(f"p{i}: {c.strip()[:80]!r}")
            if len(dumped_ann) < max_dump:
                dumped_ann.append(
                    doc.xref_object(xref, compressed=False)[:4000])
        if hit:
            ann_pages.append(i)
    ann_seconds = time.perf_counter() - t1
    n_pages = doc.page_count
    doc.close()
    return {
        "path": path, "n_pages": n_pages,
        "vp_pages": vp_pages, "empty_vp_pages": empty_vp_pages,
        "bbox_frame": bbox_frame, "n_viewports": n_viewports,
        "vp_keys": vp_keys, "ratios": ratios, "measure_keys": measure_keys,
        "subtypes": subtypes, "vp_seconds": vp_seconds,
        "n_annots": n_annots, "n_measure_annots": n_meas_annots,
        "annot_pages": ann_pages, "annot_types": annot_types,
        "it_values": it_values, "ann_measure_keys": ann_measure_keys,
        "ann_subtypes": ann_subtypes, "ann_ratios": ann_ratios,
        "contents": contents, "ann_seconds": ann_seconds,
        "dumped_vp": dumped_vp, "dumped_ann": dumped_ann,
    }


def _which_frame(page, bbox) -> str:
    """Does the /VP /BBox fit the UNROTATED mediabox or the displayed rect?

    The spec says unrotated user space. On a rotated page the two differ, so a
    real rotated page settles it. Reported as "unrotated", "displayed", "both"
    (an unrotated page, where they coincide) or "neither".
    """
    if not (isinstance(bbox, list) and len(bbox) == 4):
        return "no-bbox"
    try:
        xs = [float(bbox[0]), float(bbox[2])]
        ys = [float(bbox[1]), float(bbox[3])]
    except Exception:
        return "unparsable"
    tol = 1.0
    mb, dr = page.mediabox, page.rect
    fits_u = (max(xs) <= mb.width + tol and max(ys) <= mb.height + tol)
    fits_d = (max(xs) <= dr.width + tol and max(ys) <= dr.height + tol)
    if fits_u and fits_d:
        return "both(unrotated page)" if page.rotation % 180 == 0 else "both"
    if fits_u:
        return "unrotated"
    if fits_d:
        return "displayed"
    return "neither"


def _dump_vp(doc, entry_raw: str, vp: dict) -> str:
    """The viewport entry plus every nested object it references."""
    parts = [f"--- /VP entry (raw) ---\n{entry_raw[:3000]}"]
    meas_val = vp.get("Measure")
    if isinstance(meas_val, Ref):
        parts.append("--- /Measure (raw, indirect) ---\n"
                     + doc.xref_object(meas_val.num, compressed=False)[:3000])
    meas = resolve(doc, meas_val)
    if isinstance(meas, dict):
        for key in ("X", "Y", "D", "A", "T"):
            v = meas.get(key)
            if isinstance(v, Ref):
                parts.append(f"--- /Measure/{key} (raw, indirect) ---\n"
                             + doc.xref_object(v.num, compressed=False)[:2000])
            elif isinstance(v, list):
                for n, item in enumerate(v):
                    if isinstance(item, Ref):
                        parts.append(
                            f"--- /Measure/{key}[{n}] (raw, indirect) ---\n"
                            + doc.xref_object(item.num, compressed=False)[:1500])
    return "\n".join(parts)


def report(res: Dict[str, Any], name: str, full_dump: bool) -> None:
    print(f"\n=== {name} ===")
    print(f"pages: {res['n_pages']}")
    print(f"pages with a POPULATED /VP: {len(res['vp_pages'])} "
          f"{res['vp_pages'][:40]}{' ...' if len(res['vp_pages']) > 40 else ''}"
          f"  (viewports: {res['n_viewports']})")
    print(f"pages with an EMPTY /VP []: {len(res['empty_vp_pages'])} "
          f"{res['empty_vp_pages'][:40]}")
    if res["vp_keys"]:
        print(f"  viewport keys: {sorted(res['vp_keys'])}")
        print(f"  /Measure keys: {sorted(res['measure_keys'])}")
        print(f"  /Measure /Subtype: {res['subtypes']}")
        print(f"  distinct /R: {res['ratios']}")
        print(f"  /BBox fits which frame: {res['bbox_frame']}")
    print(f"  scan cost: {res['vp_seconds']:.3f} s for {res['n_pages']} pages")
    print(f"annotations: {res['n_annots']} total, "
          f"{res['n_measure_annots']} with /Measure or a *Dimension /IT "
          f"on pages {res['annot_pages'][:40]}")
    if res["n_measure_annots"]:
        print(f"  annot /Subtype: {res['annot_types']}")
        print(f"  /IT values: {res['it_values']}")
        print(f"  /Measure keys: {sorted(res['ann_measure_keys'])}")
        print(f"  /Measure /Subtype: {res['ann_subtypes']}")
        print(f"  distinct /R: {res['ann_ratios']}")
        print("  /Contents samples:")
        for c in res["contents"]:
            print(f"    {c}")
    print(f"  annot scan cost: {res['ann_seconds']:.3f} s")
    if full_dump:
        for d in res["dumped_vp"]:
            print("\n" + d)
        for d in res["dumped_ann"]:
            print("\n--- measurement annotation (raw) ---\n" + d)


def main(argv: List[str]) -> int:
    if argv:
        paths = argv
    else:
        paths = [SUBMITTAL] + sorted(glob.glob(MECKLENBURG))
    for p in paths:
        try:
            res = probe(p)
        except Exception as exc:
            print(f"\n=== {p} ===\n  FAILED: {exc!r}")
            continue
        name = p.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        report(res, name, full_dump=bool(res["dumped_vp"] or res["dumped_ann"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
