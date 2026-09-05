"""B3 set-of-marks A/B harness: does numbering IR endpoints on a snip help?

The design memo (DRAWING_INTELLIGENCE_DESIGN.md B3) conditions default-on
set-of-marks rendering on an A/B: SoM gains are proven on natural images,
plausible-not-proven on engineering drawings. This harness builds that
comparison on REAL ground truth (native leader tips from the Mecklenburg
DXFs, located on the plotted PDF via the verified plot-transform fit):

- Arm A (marked): render_region snip with numbered marks at the true tip
  plus nearby decoy endpoints; the model answers a MULTIPLE-CHOICE
  question ("which mark sits at the arrow tip?").
- Arm B (plain): the same snip, no marks; the model answers with pixel
  coordinates, scored by distance to the true tip.

OFFLINE by default: generates the paired PNGs + manifest.json (questions,
truth, crop geometry) into som_ab_build/ so the comparison is inspectable
and re-runnable. The LIVE arm (actually querying a vision model and
printing the A-vs-B score table) is opt-in, house pattern:

    set RUN_LIVE_TESTS=1  (+ ANTHROPIC_API_KEY in the environment)
    python module_work/drawing_ground_truth/som_ab_check.py 21.01

Costs API money on the live arm — owner-gated, never run it in CI.

RESULTS LEDGER (fill in after each live run; do not tune to it):
- 2026-09-05: harness built, offline fixtures verified on 21.01
  (13 questions; decoys drawn from real endpoint clutter). Live A/B not
  yet run (owner-gated).
"""

from __future__ import annotations

import base64
import json
import math
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, ROOT)

MECK = os.path.join(HERE, "mecklenburg")
OUT = os.path.join(HERE, "som_ab_build")
DPI = 300
CROP_HALF = 80.0     # pt half-window around each tip
PAD_FRAC = 0.10
N_DECOYS = 4
PIXEL_TOL_PT = 12.0  # arm-B correctness radius, page points
MODEL = "claude-opus-5"

PROMPT_A = (
    "This is a crop of a construction drawing. Small numbered circles "
    "(marks) have been drawn on it. Exactly one mark sits at the TIP of "
    "an arrowhead (the point a leader or dimension arrow points AT). "
    "Reply with that mark's number only."
)
PROMPT_B = (
    "This is a crop of a construction drawing containing at least one "
    "arrowhead (a leader or dimension arrow). Reply with the pixel "
    "coordinates of the arrowhead TIP in this image (origin at the "
    "image's top-left), formatted exactly as x,y with no other text."
)


def _truth_tips_page(sheet):
    from planlens.ir import from_pdf_vector
    from planlens.ir.align import fit_plot_transform

    truth = json.load(open(os.path.join(MECK, sheet + ".truth.json")))
    m = truth["spaces"]["model"]
    ir = from_pdf_vector(os.path.join(MECK, sheet + ".pdf"))
    chains = [L["vertices"] for L in m.get("leaders", []) if L.get("vertices")]
    chains += [ln for ML in m.get("multileaders", [])
               for ln in ML.get("leader_lines", []) if ln]
    anchors = [v for c in chains for v in c]
    anchors += [d["defpoint"] for d in m.get("dimensions", [])
                if d.get("defpoint")]
    fit = fit_plot_transform(anchors, ir, anchor_chains=chains)
    if fit is None:
        raise SystemExit(f"{sheet}: no plot-transform fit — cannot score")
    tips = [fit["apply"](tuple(L["vertices"][0]))
            for L in m.get("leaders", []) if L.get("vertices")]
    for ML in m.get("multileaders", []):
        for ln in ML.get("leader_lines", []):
            if ln:
                tips.append(fit["apply"](tuple(ln[0])))
                break
    return ir, tips


def _decoys(ir, tip_ir, rng):
    """Nearby REAL endpoint clutter (long entities only) as decoy marks."""
    from planlens.ir import queries as q
    hits = q.entities_ending_near(ir, tip_ir, CROP_HALF * 0.9,
                                  entity_types=["line", "polyline"])
    pts = []
    for h in hits:
        if h.get("length", 0.0) < 15.0:
            continue
        p = tuple(h["end_point"])
        if math.hypot(p[0] - tip_ir[0], p[1] - tip_ir[1]) < 10.0:
            continue  # too close to the truth to be a fair decoy
        if all(math.hypot(p[0] - o[0], p[1] - o[1]) > 8.0 for o in pts):
            pts.append(p)
    rng.shuffle(pts)
    return pts[:N_DECOYS]


def build_fixtures(sheet):
    from planlens.ir.render import clip_rect_for_bbox, render_region

    ir, tips = _truth_tips_page(sheet)
    page_h = ir.height
    pdf = os.path.join(MECK, sheet + ".pdf")
    outdir = os.path.join(OUT, sheet)
    os.makedirs(outdir, exist_ok=True)
    rng = random.Random(20260905)

    manifest = {"sheet": sheet, "dpi": DPI, "pixel_tol_pt": PIXEL_TOL_PT,
                "prompt_marked": PROMPT_A, "prompt_plain": PROMPT_B,
                "questions": []}
    for i, tip_ir in enumerate(tips):
        # ingest frame (bottom-left) -> PyMuPDF page frame (top-left)
        tip_pdf = (tip_ir[0], page_h - tip_ir[1])
        bbox = (tip_pdf[0] - CROP_HALF, tip_pdf[1] - CROP_HALF,
                tip_pdf[0] + CROP_HALF, tip_pdf[1] + CROP_HALF)
        clip = clip_rect_for_bbox(bbox, (0.0, 0.0, ir.width, page_h),
                                  pad_frac=PAD_FRAC)

        cands = [tip_pdf] + [(p[0], page_h - p[1])
                             for p in _decoys(ir, tip_ir, rng)]
        order = list(range(len(cands)))
        rng.shuffle(order)
        marks = [(cands[j][0], cands[j][1], str(k + 1))
                 for k, j in enumerate(order)]
        true_label = str(order.index(0) + 1)

        png_a = render_region(filepath=pdf, bbox=bbox, dpi=DPI,
                              pad_frac=PAD_FRAC, marks=marks)
        png_b = render_region(filepath=pdf, bbox=bbox, dpi=DPI,
                              pad_frac=PAD_FRAC)
        fa = os.path.join(outdir, f"q{i:02d}_marked.png")
        fb = os.path.join(outdir, f"q{i:02d}_plain.png")
        open(fa, "wb").write(png_a)
        open(fb, "wb").write(png_b)

        zoom = DPI / 72.0
        tip_px = ((tip_pdf[0] - clip[0]) * zoom, (tip_pdf[1] - clip[1]) * zoom)
        manifest["questions"].append({
            "id": i, "marked_png": os.path.basename(fa),
            "plain_png": os.path.basename(fb),
            "n_marks": len(marks), "true_mark": true_label,
            "tip_page_xy": [round(v, 2) for v in tip_pdf],
            "tip_px_xy": [round(v, 1) for v in tip_px],
            "clip": [round(v, 2) for v in clip],
        })

    mpath = os.path.join(outdir, "manifest.json")
    json.dump(manifest, open(mpath, "w"), indent=1)
    print(f"{sheet}: {len(manifest['questions'])} question pairs -> {outdir}")
    return outdir, manifest


def _ask_vision(client, png_path, prompt):
    data = base64.standard_b64encode(open(png_path, "rb").read()).decode()
    resp = client.messages.create(
        model=MODEL, max_tokens=256,
        messages=[{"role": "user", "content": [
            {"type": "image",
             "source": {"type": "base64", "media_type": "image/png",
                        "data": data}},
            {"type": "text", "text": prompt},
        ]}])
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def _ask_vision_chat_model(chat_model, png_path, prompt):
    """Ask through any LangChain chat model (the Funhouse/Prompter path).

    Accepts the SAME model object the deployed app runs on
    (``PrompterChatModel`` from funhouse_agent.deep.databricks_bridge) —
    which makes a cluster-side A/B measure the PRODUCTION vision model,
    the comparison that actually gates default-on marks for users.
    OpenAI-style image_url content block; works for any multimodal
    BaseChatModel.
    """
    from langchain_core.messages import HumanMessage
    data = base64.standard_b64encode(open(png_path, "rb").read()).decode()
    msg = HumanMessage(content=[
        {"type": "image_url",
         "image_url": {"url": f"data:image/png;base64,{data}"}},
        {"type": "text", "text": prompt},
    ])
    resp = chat_model.invoke([msg])
    return (resp.content if isinstance(resp.content, str)
            else "".join(str(b) for b in resp.content)).strip()


def run_live(outdir, manifest, chat_model=None):
    """Run the A/B. ``chat_model=None`` -> Anthropic SDK (needs
    ANTHROPIC_API_KEY). Pass a LangChain chat model to run against any
    engine instead — FUNHOUSE NOTEBOOK RECIPE (production-model A/B,
    metered through the existing Prompter account, no new keys)::

        from funhouse.services.prompter.prompter_api import PrompterAPI
        from funhouse_agent.deep.databricks_bridge import PrompterChatModel
        from module_work.drawing_ground_truth import som_ab_check as ab

        cm = PrompterChatModel(prompter=PrompterAPI(chat_model="funhouse-gpt-high"),
                               model="funhouse-gpt-high")
        outdir, manifest = ab.build_fixtures("21.01")
        ab.run_live(outdir, manifest, chat_model=cm)

    (Run from a checkout with planlens installed; ~26 vision calls.)
    """
    if chat_model is not None:
        client = None
        ask = lambda path, prompt: _ask_vision_chat_model(  # noqa: E731
            chat_model, path, prompt)
    else:
        import anthropic
        client = anthropic.Anthropic()
        ask = lambda path, prompt: _ask_vision(client, path, prompt)  # noqa: E731
    zoom = DPI / 72.0
    a_right = b_right = n = 0
    for qd in manifest["questions"]:
        n += 1
        ans_a = ask(os.path.join(outdir, qd["marked_png"]),
                    manifest["prompt_marked"])
        m = re.search(r"\d+", ans_a)
        ok_a = bool(m) and m.group(0) == qd["true_mark"]
        a_right += ok_a

        ans_b = ask(os.path.join(outdir, qd["plain_png"]),
                    manifest["prompt_plain"])
        m = re.search(r"(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", ans_b)
        ok_b = False
        if m:
            px = (float(m.group(1)), float(m.group(2)))
            d_pt = math.hypot(px[0] - qd["tip_px_xy"][0],
                              px[1] - qd["tip_px_xy"][1]) / zoom
            ok_b = d_pt <= PIXEL_TOL_PT
        b_right += ok_b
        print(f" q{qd['id']:02d}: A={'OK ' if ok_a else 'no '}({ans_a!r:>8}) "
              f"B={'OK' if ok_b else 'no'}({ans_b!r})")
    print(f"\nARM A (set-of-marks, multiple choice): {a_right}/{n}")
    print(f"ARM B (plain snip, coordinates):        {b_right}/{n}")
    print("(record in the RESULTS LEDGER; the design memo gates default-on "
          "marks on this comparison)")


def main():
    sheet = sys.argv[1] if len(sys.argv) > 1 else "21.01"
    outdir, manifest = build_fixtures(sheet)
    if os.environ.get("RUN_LIVE_TESTS") == "1":
        run_live(outdir, manifest)
    else:
        print("offline only (set RUN_LIVE_TESTS=1 + ANTHROPIC_API_KEY for "
              "the live A/B; costs API)")


if __name__ == "__main__":
    main()
