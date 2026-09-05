"""Verifier item 4: reimplement the OCR coverage validation vs native DXF
TEXT truth. usage: ocr_coverage.py <sheet>"""
import json, math, os, re, statistics, sys

from planlens.ir import from_pdf_vector
from planlens.ir.align import fit_plot_transform
from planlens.ocr import ocr_text_items

MECK = r"C:\Users\socon\OneDrive\dev\GeotechStaffEngineer\module_work\drawing_ground_truth\mecklenburg"
sheet = sys.argv[1]

truth = json.load(open(os.path.join(MECK, sheet + ".truth.json")))
m = truth["spaces"]["model"]
pdf = os.path.join(MECK, sheet + ".pdf")
ir = from_pdf_vector(pdf)

chains = [L["vertices"] for L in m.get("leaders", []) if L.get("vertices")]
chains += [ln for ML in m.get("multileaders", [])
           for ln in ML.get("leader_lines", []) if ln]
anchors = [v for c in chains for v in c]
anchors += [d["defpoint"] for d in m.get("dimensions", []) if d.get("defpoint")]
fit = fit_plot_transform(anchors, ir, anchor_chains=chains)
print(f"{sheet}: fit r{fit['rotation_deg']} s={fit['scale']:.2f} "
      f"rms={fit['rms']} matched={fit['n_matched']}/{len(anchors)}")


def norm(s):
    s = re.sub(r"%%[UuDdOo]", "", s)
    s = re.sub(r"%%[Cc]", "DIA", s)
    s = s.replace("\\P", " ").replace("\\p", " ")
    s = re.sub(r"\\[A-Za-z][^;]*;", "", s)  # other MTEXT codes
    s = re.sub(r"[{}]", "", s)
    return re.sub(r"\s+", " ", s).strip().upper()


items = ocr_text_items(filepath=pdf, rotate="auto")
print(f"ocr items: {len(items)}")
oc = [(norm(i.content), i.position) for i in items if norm(i.content)]

exact = partial = missed = 0
errs = []
miss_list = []
for t in m.get("text", []):
    ts = norm(t.get("text", ""))
    if not ts:
        continue
    tp = fit["apply"](tuple(t["insert"]))
    best = None  # (kind_rank, dist, pos)
    for s, pos in oc:
        d = math.hypot(pos[0] - tp[0], pos[1] - tp[1])
        if s == ts:
            kind = 0
        elif len(s) >= 3 and (s in ts or ts in s):
            kind = 1
        else:
            continue
        if d > 200:  # a "match" across the sheet is not a match
            continue
        if best is None or (kind, d) < (best[0], best[1]):
            best = (kind, d, pos)
    if best is None:
        missed += 1
        miss_list.append(ts[:50])
    else:
        if best[0] == 0:
            exact += 1
        else:
            partial += 1
        errs.append(best[1])

n = exact + partial + missed
print(f"truth texts: {n}  exact: {exact}  partial: {partial}  "
      f"missed: {missed}  coverage(exact+partial): {100*(exact+partial)/n:.0f}%"
      f"  exact-only: {100*exact/n:.0f}%")
if errs:
    print(f"median coord err: {statistics.median(errs):.1f} pt  "
          f"p90: {sorted(errs)[int(0.9*len(errs))]:.1f} pt")
print("missed:", miss_list)
