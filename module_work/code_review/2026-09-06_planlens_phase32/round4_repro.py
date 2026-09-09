"""Self-contained repro fixtures for the planlens round-4 adversarial review.

Usage (from anywhere):
    <venv>/python round4_repro.py work     # the uncommitted round-4 working tree
    <venv>/python round4_repro.py tip      # committed tip 1f6551c
    <venv>/python round4_repro.py v010     # published PyPI 0.1.0

The `tip` and `v010` trees are produced READ-ONLY from the planlens repo with:
    git archive 1f6551c | tar -x -C <dir>
    git archive v0.1.0  | tar -x -C <dir>
Point TIP_DIR / V010_DIR below at those directories.

venv used for every measurement in the report:
    C:/Users/socon/OneDrive/dev/GeotechStaffEngineer/.venv/Scripts/python.exe
"""
import math
import sys

WORK_DIR = r"C:\Users\socon\OneDrive\dev\planlens"
TIP_DIR = r"C:\Users\socon\.claude\jobs\69be0e95\tmp\base"
V010_DIR = r"C:\Users\socon\.claude\jobs\69be0e95\tmp\v010"

which = sys.argv[1] if len(sys.argv) > 1 else "work"
sys.path.insert(0, {"work": WORK_DIR, "tip": TIP_DIR, "v010": V010_DIR}[which])

import planlens  # noqa: E402
print("[%s] planlens: %s" % (which, planlens.__file__))
from planlens.ir import queries as q  # noqa: E402
from planlens.ir.results import (  # noqa: E402
    Dimension, DrawingIR, Line, Polyline, TextItem)

MAS = 9.31  # the validation sheets' arrowhead scale, points


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def ir_of(entities, source="pdf_vector"):
    ir = DrawingIR(units="pt", coordinate_space="page", origin="bottom_left",
                   source=source, width=612.0, height=792.0)
    for i, e in enumerate(entities):
        e.id = "e%d" % i
        ir.add(e)
    return ir


def chevron(apex, d, leg=7.3, base=2.4):
    """[barb, apex, barb] open chain - the native-leader arrowhead flavor."""
    dx, dy = d
    px, py = -dy, dx
    h = math.sqrt(leg * leg - (base / 2) ** 2)
    bx, by = apex[0] - dx * h, apex[1] - dy * h
    return Polyline(vertices=[(bx + px * base / 2, by + py * base / 2), apex,
                              (bx - px * base / 2, by - py * base / 2)],
                    closed=False)


def triangle(apex, d, leg=6.0, ratio=0.33, rot=0):
    """Closed filled triangle. ratio = base/leg. rot rotates the VERTEX ORDER
    (a plotter is free to start the outline at any corner)."""
    dx, dy = d
    px, py = -dy, dx
    base = ratio * leg
    h = math.sqrt(max(leg * leg - (base / 2) ** 2, 1e-9))
    bx, by = apex[0] - dx * h, apex[1] - dy * h
    v = [apex, (bx + px * base / 2, by + py * base / 2),
         (bx - px * base / 2, by - py * base / 2)]
    return Polyline(vertices=v[rot:] + v[:rot], closed=True)


def diamond(apex, d, leg=6.0, w=2.4):
    """Rhombus/diamond terminator (a centrally symmetric quad)."""
    dx, dy = d
    px, py = -dy, dx
    return Polyline(vertices=[
        apex,
        (apex[0] - dx * leg / 2 + px * w / 2,
         apex[1] - dy * leg / 2 + py * w / 2),
        (apex[0] - dx * leg, apex[1] - dy * leg),
        (apex[0] - dx * leg / 2 - px * w / 2,
         apex[1] - dy * leg / 2 - py * w / 2),
    ], closed=True)


def trapezoid(apex, d, leg=6.0, tip_w=0.8, base=3.0):
    """Truncated-triangle ('flat tip') terminator."""
    dx, dy = d
    px, py = -dy, dx
    bx, by = apex[0] - dx * leg, apex[1] - dy * leg
    return Polyline(vertices=[
        (apex[0] + px * tip_w / 2, apex[1] + py * tip_w / 2),
        (apex[0] - px * tip_w / 2, apex[1] - py * tip_w / 2),
        (bx - px * base / 2, by - py * base / 2),
        (bx + px * base / 2, by + py * base / 2)], closed=True)


def dart(apex, d, leg=7.3, base=3.0, notch=0.45):
    """Concave swallowtail / barbed arrowhead."""
    dx, dy = d
    px, py = -dy, dx
    bx, by = apex[0] - dx * leg, apex[1] - dy * leg
    nx, ny = apex[0] - dx * leg * notch, apex[1] - dy * leg * notch
    return Polyline(vertices=[apex, (bx + px * base / 2, by + py * base / 2),
                              (nx, ny),
                              (bx - px * base / 2, by - py * base / 2)],
                    closed=True)


def fill_cluster(tip, d, n=7, span=3.0):
    """Micro-dot stipple arrowhead straddling `tip` (the Mecklenburg anatomy)."""
    out = []
    for i in range(n):
        t = (i / (n - 1.0)) * span - 0.5 * span
        px, py = -d[1], d[0]
        for k in (-0.6, 0.0, 0.6):
            x = tip[0] + d[0] * t + px * k
            y = tip[1] + d[1] * t + py * k
            out.append(Line(start=(x, y), end=(x + 0.06, y + 0.06)))
    return out


def core():
    """A 100 pt dimension: shaft, two witness lines, value text. Terminators
    are added by each scenario. TRUTH: end_a=[0,0] end_b=[100,0] length 100."""
    return [Line(start=(0.0, 0.0), end=(100.0, 0.0)),
            Line(start=(0.0, -5.0), end=(0.0, 25.0)),
            Line(start=(100.0, -5.0), end=(100.0, 25.0)),
            TextItem(content="100'", position=(50.0, 6.0), height=6.0)]


def report(tag, entities, min_confidence=0.0):
    props = q.find_dimensions(ir_of(entities), max_arrowhead_size=MAS,
                              min_confidence=min_confidence)
    if not props:
        print("  %-46s NO PROPOSAL" % tag)
        return
    p = props[0]
    flags = ",".join(k for k in p["evidence"]
                     if k.startswith("arrow_") or "blunt" in k)
    print("  %-46s b=%-18s len=%-9s conf=%-6s [%s]"
          % (tag, p["end_b_xy"], p["length"], p["confidence"], flags))


# ---------------------------------------------------------------------------
# FINDING 1 - box-like terminators: correct apex discarded, construct capped
# ---------------------------------------------------------------------------
print("\nFINDING 1  box-like terminators (truth b=[100,0] len 100)")
for name, mk in (("slender triangle (control)", lambda a, d: triangle(a, d)),
                 ("diamond terminator", diamond),
                 ("trapezoid (flat tip)", trapezoid),
                 ("dart (concave)  [= FINDING 6]", dart)):
    report(name, core() + [mk((0.0, 0.0), (-1.0, 0.0)),
                           mk((100.0, 0.0), (1.0, 0.0))])

# ---------------------------------------------------------------------------
# FINDING 2 - a fill cluster loses its end to a FARTHER directional arrowhead
# ---------------------------------------------------------------------------
print("\nFINDING 2  fill-cluster arrowhead vs foreign directional chevron")
A = triangle((0.0, 0.0), (-1.0, 0.0))
report("cluster alone (centroid 0.04 pt from the tip)",
       core() + [A] + fill_cluster((100.0, 0.0), (1.0, 0.0)))
report("cluster + foreign chevron apex=(108,0.5)",
       core() + [A] + fill_cluster((100.0, 0.0), (1.0, 0.0))
       + [chevron((108.0, 0.5), (1.0, 0.0))])
ir = ir_of(core() + [A] + fill_cluster((100.0, 0.0), (1.0, 0.0))
           + [chevron((108.0, 0.5), (1.0, 0.0))])
for p in q.find_dimensions(ir, max_arrowhead_size=MAS, min_confidence=0.5):
    print("    arrowhead_ids=%s   <- named for exclude_dimensions arbitration"
          % p["arrowhead_ids"])

# ---------------------------------------------------------------------------
# FINDING 3 - ON-SPINE axial outlier: F2's other half, uncapped
# ---------------------------------------------------------------------------
print("\nFINDING 3  on-spine axial outlier (F2's axial half)")
B = chevron((100.0, 0.0), (1.0, 0.0))
report("clean", core() + [chevron((0.0, 0.0), (-1.0, 0.0)), B])
for ax, ay in ((108.0, 0.0), (108.0, 1.0), (108.0, 5.0)):
    report("junk chevron apex=(%g,%g)" % (ax, ay),
           core() + [chevron((0.0, 0.0), (-1.0, 0.0)), B,
                     chevron((ax, ay), (1.0, 0.0))])
print("  y=5 is the ORIGINAL F2 report and IS fixed; y<=1.2 (inside the arrow's")
print("  own half-width) reproduces F2 verbatim at full confidence.")

# ---------------------------------------------------------------------------
# FINDING 4 - nearer OFF-SPINE junk caps a true dimension out of the band
# ---------------------------------------------------------------------------
print("\nFINDING 4  nearer off-spine junk shadows the true arrow")
for oy in (2.0, 3.0, 4.0, 5.0):
    report("junk chevron apex=(103,%g)" % oy,
           core() + [chevron((0.0, 0.0), (-1.0, 0.0)), B,
                     chevron((103.0, oy), (1.0, 0.0))])
print("  junk centroid 4.39 pt from the tip vs the true arrow's 4.80 pt;")
print("  both tier 0, so distance alone decides and the true arrow loses.")

# ---------------------------------------------------------------------------
# FINDING 5 - apex vote order-dependent at 60 deg, inverted above it
# ---------------------------------------------------------------------------
print("\nFINDING 5  equilateral / wide triangle apex vote")
for rot in (0, 1, 2):
    report("base/leg=1.00 (60 deg tip) vertex-rot=%d" % rot,
           core() + [triangle((0.0, 0.0), (-1.0, 0.0), ratio=1.0, rot=rot),
                     triangle((100.0, 0.0), (1.0, 0.0), ratio=1.0, rot=rot)])
for ratio in (0.95, 1.05, 1.20):
    tip_deg = 2 * math.degrees(math.asin(min(ratio / 2.0, 1.0)))
    report("base/leg=%.2f (%.0f deg tip) any order" % (ratio, tip_deg),
           core() + [triangle((0.0, 0.0), (-1.0, 0.0), ratio=ratio),
                     triangle((100.0, 0.0), (1.0, 0.0), ratio=ratio)])

# ---------------------------------------------------------------------------
# ROUND-4 ITEM 4 (verified SOUND) - native-dimension suppression
# ---------------------------------------------------------------------------
print("\nROUND-4 ITEM 4  native suppression = CAP on the same SPAN (sound)")
if not hasattr(q, "_native_span_at"):
    print("  (_native_span_at absent on this tree - round-4 code only)")
else:
    def native_scene(defpoints):
        e = core() + [chevron((0.0, 0.0), (-1.0, 0.0)),
                      chevron((100.0, 0.0), (1.0, 0.0))]
        if defpoints is not None:
            e.append(Dimension(defpoints=defpoints, measurement=100.0,
                               text="100'"))
        return e

    cases = (("same span", [(0.0, 0.0), (100.0, 0.0)], 0.0),
             ("same span REVERSED", [(100.0, 0.0), (0.0, 0.0)], 0.0),
             ("same span, min_conf=0.5", [(0.0, 0.0), (100.0, 0.0)], 0.5),
             ("ONE shared end (0,0)-(60,0)", [(0.0, 0.0), (60.0, 0.0)], 0.0),
             ("enclosing (-9,0)-(109,0)", [(-9.0, 0.0), (109.0, 0.0)], 0.0),
             ("enclosing (-10,0)-(110,0)", [(-10.0, 0.0), (110.0, 0.0)], 0.0))
    for tag, dps, mc in cases:
        props = q.find_dimensions(ir_of(native_scene(dps), source="dxf"),
                                  max_arrowhead_size=MAS, min_confidence=mc)
        bits = []
        for p in props:
            sup = p["evidence"].get("superseded_by_native")
            bits.append("%s=%s%s" % (p["evidence"].get("path"),
                                     p["confidence"],
                                     "/sup:" + sup if sup else ""))
        print("  %-46s %s" % (tag, " | ".join(bits)))
