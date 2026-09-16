"""MEASURE what PDF vector paths carry before the IR is asked to keep it.

The PDF ingest leg drops two things ``page.get_drawings()`` hands it: the
optional-content-group (OCG) name of each path — a PDF's version of a CAD
layer — and whether the path is FILLED. Before designing how the IR should
carry them, this script measures how much of either is actually there, on
real sheets:

* how many OCGs the document declares, their names, and which are OFF by
  default (content a viewer hides);
* the share of paths that carry a layer at all, and how many distinct layer
  names a page uses;
* the fill/stroke split (``type`` "f" fill-only, "s" stroke-only, "fs" both);
* how many filled CLOSED paths look like a boring dot (circle-like ring) or
  an arrowhead (small few-vertex polygon) — the shapes the construct finders
  care about;
* the wall-clock cost of ``get_drawings()``, so the document layer's page map
  can be told what a layer tally would cost it.

Usage:
    <venv>/python module_work/drawing_ground_truth/probe_layers_fill.py
    <venv>/python module_work/drawing_ground_truth/probe_layers_fill.py \
        --pdf "<some other file>" --pages 6-12

With no ``--pdf`` it runs the ten public Mecklenburg County corpus sheets.
Results for any OTHER document are PRINTED ONLY — nothing here writes a file,
so a private submittal's sheet names and layer names never land in a repo.
"""
import argparse
import glob
import math
import os
import time

HERE = os.path.dirname(os.path.abspath(__file__))
MECK = os.path.join(HERE, "mecklenburg")

#: A filled ring counts as "circle-like" when a centroid circle fits it this
#: well (rms radial deviation / radius) — the same 0.08 the bubble-callout
#: finder uses, so the count means what that finder would see.
CIRCLE_RMS_FRAC = 0.08
#: "Small" polygon = bbox diagonal at most this fraction of the page diagonal.
#: Arrowheads and boring dots sit far below it; a filled title-block panel or
#: a hatched region sits far above.
SMALL_DIAG_FRAC = 0.01


def _path_points(items):
    """Every point a drawing path's items visit (bezier ENDPOINTS only).

    The probe only needs shape statistics, so curves are not subdivided the
    way the ingest leg subdivides them; a 4-bezier circle still arrives as a
    4-point ring whose points lie on the circle, which the circle fit reads
    correctly.
    """
    pts = []
    for it in items:
        k = it[0]
        if k == "l":
            pts.append((it[1].x, it[1].y))
            pts.append((it[2].x, it[2].y))
        elif k == "c":
            pts.append((it[1].x, it[1].y))
            pts.append((it[4].x, it[4].y))
        elif k == "re":
            r = it[1]
            pts.extend([(r.x0, r.y0), (r.x1, r.y0), (r.x1, r.y1), (r.x0, r.y1)])
        elif k == "qu":
            q = it[1]
            pts.extend([(p.x, p.y) for p in (q.ul, q.ur, q.lr, q.ll)])
    out = []
    for p in pts:
        if not out or abs(p[0] - out[-1][0]) > 1e-6 or abs(p[1] - out[-1][1]) > 1e-6:
            out.append(p)
    return out


def _circle_like(pts):
    """True when the points sit on a common circle about their centroid."""
    if len(pts) < 4:
        return False
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    rs = [math.hypot(p[0] - cx, p[1] - cy) for p in pts]
    r = sum(rs) / len(rs)
    if r <= 0:
        return False
    rms = math.sqrt(sum((v - r) ** 2 for v in rs) / len(rs))
    return rms / r <= CIRCLE_RMS_FRAC


def probe_page(doc, pno, ocg_names):
    """One page's layer/fill statistics (see the module docstring)."""
    page = doc[pno]
    t0 = time.perf_counter()
    paths = page.get_drawings()
    dt = time.perf_counter() - t0

    diag = math.hypot(page.rect.width, page.rect.height)
    small = diag * SMALL_DIAG_FRAC
    kinds = {"f": 0, "s": 0, "fs": 0}
    n_layered = 0
    layers = {}
    n_filled = n_ring = n_flag = n_circle = n_small_poly = 0
    for d in paths:
        kinds[d.get("type", "s")] = kinds.get(d.get("type", "s"), 0) + 1
        # PyMuPDF reports the EMPTY STRING, not None, for a path that belongs
        # to no optional-content group — so "carries a layer" has to test for
        # a non-empty name or every path on every sheet counts as layered.
        lay = d.get("layer") or None
        if lay is not None:
            n_layered += 1
            layers[lay] = layers.get(lay, 0) + 1
        if d.get("type") not in ("f", "fs"):
            continue
        pts = _path_points(d.get("items", ()))
        if len(pts) < 3:
            continue
        # A FILLED path is closed by PDF semantics whatever ``closePath``
        # says: the fill operator closes every open subpath. Both weaker
        # signals are counted so the gap between them is visible.
        n_filled += 1
        if bool(d.get("closePath")):
            n_flag += 1
        if (abs(pts[0][0] - pts[-1][0]) <= 1e-6
                and abs(pts[0][1] - pts[-1][1]) <= 1e-6):
            n_ring += 1
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        d_diag = math.hypot(max(xs) - min(xs), max(ys) - min(ys))
        if _circle_like(pts):
            n_circle += 1
        elif 3 <= len(pts) <= 8 and d_diag <= small:
            n_small_poly += 1
    return {
        "page": pno,
        "n_paths": len(paths),
        "seconds": dt,
        "kinds": kinds,
        "n_layered": n_layered,
        "layers": layers,
        "n_filled_closed": n_filled,
        "n_closepath_flag": n_flag,
        "n_repeats_first_point": n_ring,
        "n_circle": n_circle,
        "n_small_poly": n_small_poly,
        "keys": sorted(paths[0].keys()) if paths else [],
        "ocgs_on_page": sorted(set(layers) & set(ocg_names)),
    }


def probe_document(path, pages=None):
    """Probe one document; returns ``(doc_facts, [page stats])``."""
    import fitz
    doc = fitz.open(path)
    ocgs = doc.get_ocgs() or {}
    names = {v.get("name"): v for v in ocgs.values()}
    facts = {
        "n_pages": len(doc),
        "n_ocgs": len(ocgs),
        "ocgs": [(v.get("name"), bool(v.get("on")), v.get("intent"))
                 for v in ocgs.values()],
        "n_layer_configs": len(doc.get_layers() or []),
    }
    wanted = pages if pages is not None else range(len(doc))
    rows = [probe_page(doc, p, names) for p in wanted]
    doc.close()
    return facts, rows


def _fmt_row(name, r):
    k = r["kinds"]
    share = (100.0 * r["n_layered"] / r["n_paths"]) if r["n_paths"] else 0.0
    return ("  %-10s %7d %6.1f%% %6d | %6d %6d %6d | %6d %6d %6d %6d | %6.2f"
            % (name, r["n_paths"], share, len(r["layers"]),
               k.get("f", 0), k.get("s", 0), k.get("fs", 0),
               r["n_filled_closed"], r["n_circle"], r["n_small_poly"],
               r["n_repeats_first_point"], r["seconds"]))


HEAD = ("  %-10s %7s %6s %6s | %6s %6s %6s | %6s %6s %6s %6s | %6s"
        % ("sheet", "paths", "w/layer", "layers", "f", "s", "fs",
           "filled", "circle", "smpoly", "ring", "sec"))


def report(title, path, pages=None, show_layer_names=True):
    facts, rows = probe_document(path, pages)
    print(title)
    print("  %d pages; %d OCGs; %d layer configs"
          % (facts["n_pages"], facts["n_ocgs"], facts["n_layer_configs"]))
    if facts["ocgs"]:
        for nm, on, intent in facts["ocgs"]:
            print("    OCG %-30s default=%s intent=%s"
                  % (str(nm)[:30], "ON " if on else "OFF", intent))
    print(HEAD)
    for r in rows:
        print(_fmt_row("page %d" % r["page"], r))
    if rows:
        print("  get_drawings(): %.2f s on the biggest page (%d paths)"
              % (max(rows, key=lambda r: r["n_paths"])["seconds"],
                 max(r["n_paths"] for r in rows)))
        print("  path dict keys: %s" % ", ".join(rows[0]["keys"]))
    if show_layer_names:
        seen = sorted({lay for r in rows for lay in r["layers"]})
        print("  distinct layer names seen: %s"
              % (", ".join(seen) if seen else "(none — no path carries one)"))
    print()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", help="probe this document instead of the corpus")
    ap.add_argument("--pages", help="0-based page range, e.g. 6-12")
    ap.add_argument("--no-layer-names", action="store_true",
                    help="suppress layer names (a confidential document's "
                         "layer names can name the project)")
    args = ap.parse_args()

    if args.pdf:
        pages = None
        if args.pages:
            a, _, b = args.pages.partition("-")
            pages = range(int(a), int(b or a) + 1)
        report("DOCUMENT: %s" % os.path.basename(args.pdf), args.pdf, pages,
               show_layer_names=not args.no_layer_names)
        return

    print("MECKLENBURG COUNTY CORPUS (public) — layers and fill per sheet")
    print()
    print(HEAD)
    totals = {"paths": 0, "layered": 0, "f": 0, "s": 0, "fs": 0,
              "fillcl": 0, "circle": 0, "smpoly": 0, "ring": 0, "flag": 0}
    ocg_lines, all_layers, slowest = [], set(), (0.0, "", 0)
    per_layer = []
    for pdf in sorted(glob.glob(os.path.join(MECK, "*.pdf"))):
        name = os.path.basename(pdf)[:-4]
        facts, rows = probe_document(pdf)
        for r in rows:
            print(_fmt_row(name if len(rows) == 1 else "%s p%d" % (name, r["page"]),
                           r))
            totals["paths"] += r["n_paths"]
            totals["layered"] += r["n_layered"]
            for k in ("f", "s", "fs"):
                totals[k] += r["kinds"].get(k, 0)
            totals["fillcl"] += r["n_filled_closed"]
            totals["circle"] += r["n_circle"]
            totals["smpoly"] += r["n_small_poly"]
            totals["ring"] += r["n_repeats_first_point"]
            totals["flag"] += r["n_closepath_flag"]
            all_layers |= set(r["layers"])
            if len(r["layers"]) > 1:
                per_layer.append((name, sorted(r["layers"].items(),
                                               key=lambda kv: -kv[1])))
            if r["seconds"] > slowest[0]:
                slowest = (r["seconds"], name, r["n_paths"])
        ocg_lines.append("  %-10s %d pages, %d OCGs%s"
                         % (name, facts["n_pages"], facts["n_ocgs"],
                            "" if not facts["ocgs"] else ": " + ", ".join(
                                "%s (default %s)" % (nm, "on" if on else "OFF")
                                for nm, on, _ in facts["ocgs"])))
    share = 100.0 * totals["layered"] / totals["paths"] if totals["paths"] else 0.0
    print("  %-10s %7d %6.1f%% %6d | %6d %6d %6d | %6d %6d %6d %6d"
          % ("TOTAL", totals["paths"], share, len(all_layers),
             totals["f"], totals["s"], totals["fs"],
             totals["fillcl"], totals["circle"], totals["smpoly"],
             totals["ring"]))
    print("  filled paths whose closePath flag is True: %d of %d"
          % (totals["flag"], totals["fillcl"]))
    print()
    print("OPTIONAL CONTENT GROUPS per sheet")
    for line in ocg_lines:
        print(line)
    if per_layer:
        print()
        print("PATHS PER LAYER (sheets with more than one)")
        for name, items in per_layer:
            print("  %-10s %s" % (name, ", ".join("%s=%d" % kv for kv in items)))
    print()
    print("  slowest get_drawings(): %.2f s on %s (%d paths)"
          % (slowest[0], slowest[1], slowest[2]))
    print("  distinct layer names across the corpus: %s"
          % (", ".join(sorted(all_layers)) if all_layers
             else "(none — no path carries one)"))


if __name__ == "__main__":
    main()
