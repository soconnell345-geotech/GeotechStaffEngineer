"""Verifier's adversarial fixtures for the round-5 repair. Usage: attack.py <tree>  (work | v_r4 | v_tip)"""
import math, os, sys
TREE = sys.argv[1]; TMP = r"C:\Users\socon\.claude\jobs\69be0e95\tmp"
if TREE != "work": sys.path.insert(0, os.path.join(TMP, TREE))
import planlens
from planlens.ir import queries as q
from planlens.ir.results import DrawingIR, Line, Polyline, TextItem
print("[%s] %s" % (TREE, planlens.__file__))
MAS = 9.31
H = math.sqrt(7.3 ** 2 - 1.2 ** 2)   # 7.2007 apex-to-base of the standard arrow

def ir_of(ents):
    ir = DrawingIR(units="pt", coordinate_space="page", origin="bottom_left", source="pdf_vector", width=612.0, height=792.0)
    for i, e in enumerate(ents): e.id = "e%d" % i; ir.add(e)
    return ir
def chevron(apex, d, leg=7.3, base=2.4):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    return Polyline(vertices=[(bx+px*base/2, by+py*base/2), apex, (bx-px*base/2, by-py*base/2)], closed=False)
def base_leg(apex, d, leg=7.3, base=2.4):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    return Polyline(vertices=[(bx+px*base/2, by+py*base/2), (bx-px*base/2, by-py*base/2), apex], closed=False)
def tri_closed(apex, d, leg=7.3, base=2.4, rot=0):
    dx, dy = d; px, py = -dy, dx; h = math.sqrt(leg*leg - (base/2)**2); bx, by = apex[0]-dx*h, apex[1]-dy*h
    v = [apex, (bx+px*base/2, by+py*base/2), (bx-px*base/2, by-py*base/2)]
    return Polyline(vertices=v[rot:]+v[:rot], closed=True)
def rect(cx, cy, w, h, ang=0.0, open_ring=False):
    """w along `ang`, h across. open_ring: drop one SHORT edge (a PDF 're' ingests as an open 4-corner chain)."""
    c, s = math.cos(ang), math.sin(ang)
    loc = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    v = [(cx + x*c - y*s, cy + x*s + y*c) for x, y in loc]
    if open_ring:
        v = [v[1], v[0], v[3], v[2]]
        return Polyline(vertices=v, closed=False)
    return Polyline(vertices=v, closed=True)
def diamond(apex, d, leg=6.0, w=2.4):
    dx, dy = d; px, py = -dy, dx
    return Polyline(vertices=[apex, (apex[0]-dx*leg/2+px*w/2, apex[1]-dy*leg/2+py*w/2), (apex[0]-dx*leg, apex[1]-dy*leg), (apex[0]-dx*leg/2-px*w/2, apex[1]-dy*leg/2-py*w/2)], closed=True)
def fill_cluster(tip, d, n=7, span=3.0):
    out = []; px, py = -d[1], d[0]
    for i in range(n):
        t = (i/(n-1.0))*span - 0.5*span
        for k in (-0.6, 0.0, 0.6):
            x = tip[0]+d[0]*t+px*k; y = tip[1]+d[1]*t+py*k
            out.append(Line(start=(x, y), end=(x+0.06, y+0.06)))
    return out
def core(text=True):
    e = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0))]
    if text: e.append(TextItem(content="100'", position=(50.0, 6.0), height=6.0))
    return e
def dims(ents, mc=0.0, shaft="e0"):
    ps = q.find_dimensions(ir_of(ents), max_arrowhead_size=MAS, min_confidence=mc)
    return [p for p in ps if p["shaft_id"] == shaft or p["evidence"]["path"] == "split_shaft"]
def leaders(ents, mc=0.5):
    return q.find_leaders(ir_of(ents), max_arrowhead_size=MAS, min_confidence=mc, exclude_dimensions=True)
def show(tag, ents, mc=0.0, shaft="e0"):
    ps = dims(ents, mc, shaft)
    if not ps: print("  %-58s NO PROPOSAL" % tag); return None
    p = ps[0]; ev = p["evidence"]
    flags = ",".join(sorted(k for k in ev if k.startswith("arrow_") or "blunt" in k or "oriented" in k or k == "below_extent_floor"))
    print("  %-58s b=%-16s len=%-8s conf=%-6s ids=%s [%s]" % (tag, p["end_b_xy"], p["length"], p["confidence"], p["arrowhead_ids"], flags))
    return p
class _NA:
    state = "n/a"; score = float("nan"); apex = (float("nan"), float("nan"))
def attach(shape, sdir=(1.0, 0.0), tip=(100.0, 0.0)):
    if not hasattr(q, "_arrow_attach"):
        return _NA()
    return q._arrow_attach([tuple(v) for v in shape.vertices], "triangle", sdir, tip, MAS)

A = chevron((0.0, 0.0), (-1.0, 0.0))
B = chevron((100.0, 0.0), (1.0, 0.0))

print("\nP1  ORIENTED/BLUNT gate: closed rectangle terminators (w along line x h across) at both ends, drawn INSIDE the line (far edge at the defpoint); truth b=[100,0]")
for w, h in ((3.0, 3.0), (3.9, 3.0), (4.5, 3.0), (4.77, 3.0), (4.8, 3.0), (4.83, 3.0), (6.0, 3.0), (6.0, 1.0), (6.0, 0.75)):
    ents = core() + [rect(w/2, 0.0, w, h), rect(100.0 - w/2, 0.0, w, h)]
    st = attach(rect(100.0 - w/2, 0.0, w, h)).state
    show("rect %.2fx%.2f (elong %.2f) state=%s" % (w, h, w/h, st), ents)
print("  -- same 6x3 rect rotated about its centre (fold cone: cos30 = %.4f)" % math.cos(math.radians(30)))
for deg in (25.0, 29.0, 29.9, 30.0, 30.1, 31.0, 35.0, 45.0):
    r = rect(97.0, 0.0, 6.0, 3.0, math.radians(deg))
    print("     %5.1f deg -> state=%s score=%.4f" % (deg, attach(r).state, attach(r).score))
print("  -- diamond width sweep (leg 6): elong = leg/w")
for w in (2.4, 3.0, 3.6, 3.75, 3.8, 4.0, 6.0):
    d = diamond((100.0, 0.0), (1.0, 0.0), leg=6.0, w=w)
    a = attach(d); print("     w=%.2f elong=%.2f -> state=%s apex=%s" % (w, 6.0/w, a.state, tuple(round(x, 3) for x in a.apex)))

print("\nP2  SEAT DOMINANCE: cluster (tier 1) genuinely terminates the line, foreign chevron (tier 0) beyond the end on-axis; cluster seat s1 = splash-centre offset from the end; chevron seat s0 = base-centre gap. truth b~[100,0]; foreign apex = 100+s0+H")
for s1 in (0.04, 0.1, 0.3, 0.5, 1.0, 2.0):
    for s0 in (0.94, 2.7, 3.0, 3.3, 5.0, 9.0, 11.0):
        ents = core() + [A] + fill_cluster((100.0 + s1, 0.0), (1.0, 0.0)) + [chevron((100.0 + s0 + H, 0.0), (1.0, 0.0))]
        p = dims(ents, 0.0)[0]
        who = "CLUSTER" if any(i.startswith("cluster") for i in p["arrowhead_ids"]) else "chevron"
        print("  s1=%.2f s0=%5.2f ratio=%5.1f -> %-7s b=%-16s conf=%-6s ids=%s" % (s1, s0, s0/max(s1, 1e-9), who, p["end_b_xy"], p["confidence"], p["arrowhead_ids"]))
print("  -- finding-2 anatomy exactly (foreign apex (108,0.5), seat 0.94) with the cluster shifted by s1 along the line, PLUS the leader that owns the chevron:")
for s1 in (0.0, 0.05, 0.1, 0.5, 1.0, 2.0):
    ents = core() + [A] + fill_cluster((100.0 + s1, 0.0), (1.0, 0.0)) + [chevron((108.0, 0.5), (1.0, 0.0)), Line(start=(108.0 - H, 0.5), end=(60.0, -10.0)), TextItem(content="CB #4", position=(54.0, -12.0), height=3.0)]
    p = dims(ents, 0.0)[0]; L = leaders(ents, 0.5)
    print("  shift=%.2f -> b=%-16s conf=%-6s ids=%-28s leader 'CB #4' survives exclude_dimensions: %s" % (s1, p["end_b_xy"], p["confidence"], p["arrowhead_ids"], any(l.get("text") == "CB #4" for l in L)))

print("\nP3  SEAT for a pointed terminator: vertex order / flavour invariance (apex at end, and base at end)")
for name, mk in (("chevron [b,apex,b]", chevron), ("base-leg [b,b,apex]", base_leg), ("closed rot0", lambda a, d: tri_closed(a, d, rot=0)), ("closed rot1", lambda a, d: tri_closed(a, d, rot=1)), ("closed rot2", lambda a, d: tri_closed(a, d, rot=2))):
    for tag, apex in (("apex@end", (100.0, 0.0)), ("base@end", (100.0 + H, 0.0))):
        v = [tuple(x) for x in mk(apex, (1.0, 0.0)).vertices]
        if hasattr(q, "_seat_distance"):
            s = q._seat_distance(v, "triangle", (1.0, 0.0), (100.0, 0.0), q._arrow_geometry(v))
            print("  %-22s %-9s seat=%.6f" % (name, tag, s))
        else:
            print("  %-22s %-9s (no _seat_distance on this tree)" % (name, tag))

print("\nP4  TIE at one end: two tier-0 chevrons both base-seated at the end (seat 0, 0), one on-axis, one 20 deg up; entity ORDER swapped")
for order in ("straight-first", "skew-first"):
    a20 = math.radians(20.0)
    c1 = chevron((100.0 + H, 0.0), (1.0, 0.0)); c2 = chevron((100.0 + H*math.cos(a20), H*math.sin(a20)), (math.cos(a20), math.sin(a20)))
    ents = core() + [A] + ([c1, c2] if order == "straight-first" else [c2, c1])
    show("order=%s" % order, ents)

print("\nP5  NO-TEXT sheet (the corpus's own class): true chevron at one end, an ORIENTED oblong glyph fragment at the other, witnesses both ends")
for tag, frag in (("closed 2.0x0.8 oblong on the end", rect(100.6, 0.0, 2.0, 0.8)), ("open near-ring 2.0x0.8 (PDF re)", rect(100.6, 0.0, 2.0, 0.8, open_ring=True)), ("corpus-like 1.98x0.96 open ring", rect(100.5, 0.0, 1.98, 0.96, open_ring=True)), ("same fragment ACROSS the line (90 deg)", rect(100.4, 0.0, 2.0, 0.8, math.radians(90))), ("square 1.0x1.0 tile", rect(100.5, 0.0, 1.0, 1.0))):
    ents = core(text=False) + [A, frag]
    p = show(tag, ents)
    if p is not None:
        print("      -> CALLED at 0.5: %s" % (p["confidence"] >= 0.5))
print("  -- same on a TEXT-bearing sheet")
show("text sheet: closed 2.0x0.8 oblong on the end", core() + [A, rect(100.6, 0.0, 2.0, 0.8)])
print("  -- split leg on a no-text sheet: real half (arrow outside, base at tip) + a half whose 'arrow' is an oblong fragment beyond its tip")
ents = [Line(start=(0.0, -5.0), end=(0.0, 25.0)), Line(start=(100.0, -5.0), end=(100.0, 25.0)),
        Line(start=(H, 0.0), end=(40.0, 0.0)), base_leg((0.0, 0.0), (-1.0, 0.0)),
        Line(start=(60.0, 0.0), end=(98.0, 0.0)), rect(99.0, 0.0, 2.0, 0.8)]
ps = [p for p in q.find_dimensions(ir_of(ents), max_arrowhead_size=MAS, min_confidence=0.0) if p["evidence"]["path"] == "split_shaft"]
for p in ps: print("  split: a=%s b=%s conf=%s ids=%s kinds=%s flags=%s" % (p["end_a_xy"], p["end_b_xy"], p["confidence"], p["arrowhead_ids"], p["evidence"]["arrowhead_kinds"], sorted(k for k in p["evidence"] if k.startswith("arrow_") or "blunt" in k or "oriented" in k)))
if not ps: print("  split: NO PROPOSAL")

print("\nP6  GRAPHIC SCALE BAR (text sheet): baseline 0-100 with filled end blocks INSIDE the line, ticks every 25, labels")
ticks = [Line(start=(x, -4.0), end=(x, 4.0)) for x in (0.0, 25.0, 50.0, 75.0, 100.0)]
labels = [TextItem(content="0", position=(0.0, 7.0), height=3.0), TextItem(content="50", position=(50.0, 7.0), height=3.0), TextItem(content="100", position=(100.0, 7.0), height=3.0)]
ents = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), rect(10.0, 0.0, 20.0, 3.0), rect(90.0, 0.0, 20.0, 3.0)] + ticks + labels
show("scale bar (20x3 blocks, 6.7:1)", ents)
ents2 = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), rect(3.0, 0.0, 6.0, 3.0), rect(97.0, 0.0, 6.0, 3.0)] + ticks + labels
show("scale bar (6x3 blocks, 2:1)", ents2)
ents3 = [Line(start=(0.0, 0.0), end=(100.0, 0.0)), rect(1.5, 0.0, 3.0, 3.0), rect(98.5, 0.0, 3.0, 3.0)] + ticks + labels
show("scale bar (3x3 blocks, 1:1)", ents3)

print("\nP7  EXACT threshold: tier-1 seat*10 == tier-0 seat (float)")
if hasattr(q, "_award_end"):
    for s1, s0 in ((0.1, 1.0), (0.3, 3.0), (0.07, 0.7), (1e-9, 1e-8)):
        r = q._award_end({0: {"tier": 0, "seat": s0}, 1: {"tier": 1, "seat": s1}})
        print("  s1=%g s0=%g -> tier %d wins  (s1*10=%r)" % (s1, s0, r["tier"], s1 * 10))
else: print("  (no _award_end on this tree)")

print("\nP8  Tipless attach/seat with a duplicate closing vertex and a reversed traversal")
d0 = diamond((100.0, 0.0), (1.0, 0.0)); v = [tuple(x) for x in d0.vertices]
for tag, vv in (("plain", v), ("dup-closing", v + [v[0]]), ("reversed", v[::-1])):
    if not hasattr(q, "_arrow_attach"):
        print("  (no _arrow_attach on this tree)"); break
    a = q._arrow_attach(vv, "triangle", (1.0, 0.0), (100.0, 0.0), MAS)
    extra = ""
    if hasattr(q, "_seat_distance"):
        extra = " seat=%.4f" % q._seat_distance(vv, "triangle", (1.0, 0.0), (100.0, 0.0), q._arrow_geometry(vv))
    print("  %-12s state=%s score=%.4f apex=%s%s" % (tag, a.state, a.score, tuple(round(x, 3) for x in a.apex), extra))
