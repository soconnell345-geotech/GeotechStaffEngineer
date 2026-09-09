"""Re-measure the corpus figures that the planlens docs PUBLISH.

Companion to score_compositions.py (recall) and ocr_coverage_check.py
(OCR). This one exists because published numbers drifted twice: two
successive remediation rounds printed blunt-terminator and layer figures
into README.md / ir/DESIGN.md / DRAWING_INTELLIGENCE_DESIGN.md that no
run reproduced. On this project a documented number nobody verified is a
defect, so the numbers now have a command.

Usage:  <venv>/python module_work/drawing_ground_truth/doc_claims_check.py

Anything printed here that disagrees with the three documents is a
DEFECT IN THE DOCUMENTS — re-measure, then edit the prose; never the
other way round.
"""
import glob
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MECK = os.path.join(HERE, "mecklenburg")

#: Entity types from_dxf turns into IR entities (mirrors ingest._handle).
SUPPORTED = {"LINE", "LWPOLYLINE", "POLYLINE", "ARC", "CIRCLE", "ELLIPSE",
             "SPLINE", "TEXT", "MTEXT", "HATCH", "LEADER", "MULTILEADER",
             "DIMENSION"}


def blunt_terminator_counts():
    """Proposals carrying `blunt_terminators` evidence, per threshold.

    The branch is validated only by synthetic fixtures, so what matters
    is whether it reaches the CALL threshold on real drafting.
    """
    from planlens.ir import from_pdf_vector, queries as q
    rows, totals = [], [0, 0, 0]
    for pdf in sorted(glob.glob(os.path.join(MECK, "*.pdf"))):
        ir = from_pdf_vector(pdf)
        row = [sum(1 for p in q.find_dimensions(ir, min_confidence=c)
                   if p["evidence"].get("blunt_terminators"))
               for c in (0.0, 0.3, 0.5)]
        for i, v in enumerate(row):
            totals[i] += v
        if any(row):
            rows.append((os.path.basename(pdf)[:-4], row))
    return rows, totals


def layer_inheritance():
    """Entities re-homed by layer-"0" inheritance, and the layer counts.

    Two DIFFERENT quantities the docs must not conflate: the distinct
    layers CARRYING GEOMETRY (which the inheritance moves) and the
    `n_layers` METADATA field (every layer name seen during ingest,
    INSERTs and unsupported types included), which does not follow it.
    """
    import ezdxf
    out = []
    for f in sorted(glob.glob(os.path.join(MECK, "dxf", "*.dxf"))):
        doc = ezdxf.readfile(f)
        raw, inherited, state = set(), set(), {"rehomed": 0}

        def walk(ent, placed, depth=0):
            et = ent.dxftype()
            lay = getattr(ent.dxf, "layer", None)
            if et == "INSERT":
                if depth >= 8:
                    return
                here = placed if placed is not None else lay
                try:
                    kids = list(ent.virtual_entities())
                except Exception:
                    return
                for k in kids:
                    walk(k, here, depth + 1)
                return
            if et not in SUPPORTED:
                return
            raw.add(lay)
            eff = placed if (placed is not None and lay == "0") else lay
            inherited.add(eff)
            if eff != lay:
                state["rehomed"] += 1

        for e in doc.modelspace():
            walk(e, None)
        out.append((os.path.basename(f)[:-4], state["rehomed"],
                    len(raw), len(inherited)))
    return out


def main():
    rows, totals = blunt_terminator_counts()
    print("BLUNT-TERMINATOR proposals (evidence 'blunt_terminators')")
    print("  %-8s %6s %6s %6s" % ("sheet", "@0.0", "@0.3", "@0.5"))
    for name, row in rows:
        print("  %-8s %6d %6d %6d" % (name, row[0], row[1], row[2]))
    print("  %-8s %6d %6d %6d   <- ALL TEN SHEETS"
          % ("total", totals[0], totals[1], totals[2]))
    print("  (sheets not listed contribute none at any threshold)")

    print()
    print('LAYER-"0" INHERITANCE (needs the gitignored dxf/ conversions)')
    lay = layer_inheritance()
    if not lay:
        print("  dxf/ absent - run the ODA conversion first (see README).")
        return
    print("  %-8s %8s | %s" % ("sheet", "re-homed", "geometry layers raw->inh"))
    for name, rehomed, nraw, ninh in lay:
        print("  %-8s %8d | %d -> %d" % (name, rehomed, nraw, ninh))


if __name__ == "__main__":
    main()
