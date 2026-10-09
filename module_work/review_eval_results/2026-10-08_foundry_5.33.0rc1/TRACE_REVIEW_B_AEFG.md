# Foundry brief 5 (5.33.0rc1 + planlens 0.13.0rc1): parts B, A, E, F, G and the engine, read in full

The AI FDE ran brief 5 on Palantir Foundry on 2026-10-08 against the test
wheels **geotech-staff-engineer 5.33.0rc1** (app `39716a8`; master `fcc5c38`
adds docs only) and **planlens 0.13.0rc1** (`c5bdb8d`, planlens main). Both
models went through the package's own engine
(`webapp.palantir_sdk_engine.PalantirSdkChatModel`, Responses route), every
`GEOTECH_VISION_*` unset except in part F. Code is cited at those two
commits; reference data at `geotech-references` `d8ff52e`.

**This file covers** part B (the two markup tasks, × 3 per model), part A
(label crops), part E (GEC-12 Figure 7-15, second reading), part F (patch
alignment), part G (report-ingest rescore and visual scales; private data,
quoted as report IDs, counts and rates only) and the engine's reasoning
request. Parts C and D are reviewed in the other files beside this one.

**Sources.** Raw hand-back (git-ignored):
`module_work/field_feedback/2026-10-08_foundry_brief5_5.33.0rc1/raw/`.
Brief 4 for comparison: `.../2026-10-08_foundry_brief4_5.32.1rc3/raw/` and
`module_work/review_eval_results/2026-10-07_foundry_5.32.1rc3/TRACE_REVIEW.md`
(cited as **brief 4**).

**Method.**
* I read every event of the 12 part B runs: 1,068 records, every vision side
  call's text, all 19 `annotate_document` calls with their check verdicts,
  every answer.
* Every ring in the six produced circle PDFs was placed against the
  fixture's truth (`build_synthetic_tag_set()`, page 0). Every comment's
  arrow tip was traced back through the zoom it came from.
* The `found` lists of all 12 page looks were replayed offline through the
  app's own box parser.
* Figure 7-15 was measured from its embedded image with no model, and
  planlens `find_scales` was run on that page.
* Part F's per-box rows were refitted. Part G's rows and both RESULTS.md files
  were read alongside the report-ingest code.

`Lnn` is a line of that run's `activity.jsonl`; `t=` is seconds into the turn.

---

## In short

1. **The markup arrow is not a coordinate bug.** In both failing GPT-5.4
   reps the tip is the centre of the section label "RAMP SLOPE UP TO 7.5 %
   (8.3 % MAX.)", to within 1 pt: (514.9, 444.4) and (516.0, 445.0) against
   the label's printed box 490–543 × 429–460. Every conversion in the chain
   is exact.
   * The repeat is systematic because the same three things happened in each:
     - GPT-5.4's whole-page answer named the section label (3 of 3 reps);
     - `found` threw away the tile that boxed note 4 (6 of 6 markup runs, both
       models);
     - the markup check confirmed the label, because the agent's `target` was
       generic ("ramp slope note"), or, in r2, was renamed to the check's own
       reading after an "unsure".
   * Sol asked its page look for "the note stating an 8.33 % maximum" and
     went straight to note 4 (3 of 3).
2. **`found` drops long text lines (P).** `vision_view.REGION_GRID = 300`
   treats any box wider *or* taller than 300 on the 0–999 grid as a region.
   A notes line seen in a tile is about 400 wide, so its box is discarded.
   Half of all `found` rows (61 of 121) also carry a useless label ("?",
   "Tag", "image_box").
3. **The new markup check judges against whatever the agent calls the
   target (P).** Rings, against truth:
   * 56 of 69 good rings confirmed;
   * **13 good rings rejected** (Sol r2: exact `find_like` rings, "beside the
     mark", three rounds running);
   * **12 of 12 blankets confirmed** (brief 4: 9 of 9 rejected);
   * the one QCE ring rejected rightly.

   In each wrong verdict the agent's `target` said "tag **with leader**", the
   look boxed tag and leader as "the thing", and the check measured both
   enclosure and size against that box. The same wording gave right verdicts
   when GPT-5.4's look boxed the lettering alone (r1, r2), so the verdict
   hinges on how a look reads an ambiguous target. Comments: 4 of 4 on note 4
   confirmed (brief 4: 0 of 10).
4. **Circles: GPT-5.4 6/7, 6/7, 5/7; Sol 7/7, 2/7, 7/7.**
   * GPT-5.4's two passing reps both zoomed T6 twice (tile overlap) and never
     zoomed T1. The duplicate flag (fix I) fired and the agent dropped the
     copy, but did not look for the tag it had meant.
   * GPT-5.4 r3 asked its zooms for "tag + leader" boxes. It drew blanket
     rings, which the check confirmed, and it missed T5.
   * Sol r2 is the check failure in item 3.
   * Fixes C, D, E and G held: no `dpi` on any of 68 zooms (13–17 px/pt on
     the tags), 0 refused calls, T5 ringed in 5 of 6 runs, 0 bracketed zoom
     readings.
5. **Part A:** both models read 32 of 32 labels, one call per sheet.
6. **Part E: the Figure 7-15 table is wrong, and both models and a pixel
   read agree.**
   * Read from the figure's own pixels (±1 tsf), the curve gives 7.1 / 16.0 /
     35.9 / 75.2 / 208.5 / 296.0 tsf at 30 / 32 / 34 / 36 / 40 / 42°.
   * The table says 10 / 20 / 40 / 75 / 200 / 280. That is 41 % / 25 % /
     11 % high at 30–34° and 3–6 % low at 38–43°.
   * The printed axis starts at 30° and the curve ends at 43.77° (369 tsf).
     The table's nodes at 26, 28, 44 and 45° are not on the chart.
   * `code_reading` never measured: `no_scale` on 9 of 9 READ lines. planlens
     treats the page as a non-raster page (images cover 43.6 % < 50 %).
7. **Part F: no signal on the brief's statistic, a clear one on the
   hypothesis's own.**
   * The script's y slope (an OLS fit with an intercept): OFF 1.014–1.017,
     ON 1.010–1.024.
   * Fitted as a pure stretch through the origin (what "whole 32-px patches"
     predicts): OFF 1.010–1.015 (brief 4: 1.008–1.016), ON 0.994–1.000. The
     two clean ON reps give 0.9999.
   * n = 3 per arm, and one tag sits alone in the top third. Not decided; the
     section says what more would decide it.
8. **Part G1 settles brief 4's open items, at no model cost.**
   * Blind log index: floor 0/16, model 0/16, after 0/16. The floor never
     held those values, so nothing was overruled; 13 of the 16 are on one
     log (R13_p45).
   * The merge never lost: after ≥ max(floor, model) on all 15 logs.
   * Gradation: 3 of the 4 collapsed sheets are link failures, read 72–100 %
     against after 3–9 %. R15_p82 is a reading failure (read 21 %).
9. **Part G2: the visual-scales voters changed nothing.**
   * No label-reading call was charged in either stage. The totals equal the
     per-item reader calls: 29 and 36.
   * The grid's own blind `layer_top` is 22/36 OFF and ON.
   * The voters act only on raster pages, and label reads only where a scan
     has no text. On this corpus Azure DI gives the scans text.
   * Logs moved 466 → 468 of 521 on one log's index values (model variance).
10. **Engine: confirmed, no reasoning was asked for.**
    * `_reasoning_request` takes the first SDK name that exists. That is
      `Reasoning` (an output class) and `ReasoningSummary` (a content class
      with no `AUTO`), so it returns `None`.
    * The offline test passed because its fake SDK uses those same wrong
      names.
    * 0 of 399 `model_end` records in part B carry reasoning. Its 69
      tool-calling steps carry no text either.

---

## 1. Part B — the two markup tasks

### 1.1 Results against brief 4

| model | task | brief 4 | brief 5 (r1, r2, r3) | input / output tokens, seconds per run |
|---|---|---|---|---|
| GPT-5.4 | circle | 3/7 ✗, 7/7, 6/7 (2/3) | **6/7, 6/7, 5/7 ✗** (2/3) | 244K/5.4K 104 s; 195K/6.1K 168 s; 148K/3.7K 113 s |
| GPT-5.4 | markup | 3/3 (2/3 under the new check) | **✗, ✗, ✓** (1/3) | 99K/1.6K 56 s; 121K/2.3K 66 s; 97K/1.2K 27 s |
| Sol | circle | 7/7, 7/7, 6/7 (3/3) | **7/7, 2/7 ✗, 7/7** (2/3) | 229K/70K 416 s; 365K/35K 266 s; 305K/26K 231 s |
| Sol | markup | 3/3 | **✓, ✓, ✓** | 105K/2.9K 55 s; 90K/2.2K 45 s; 91K/3.6K 56 s |

Totals: GPT-5.4 904K input, Sol 1.18M input; the brief estimated 2.4M.
GPT-5.4's 45-a-minute bucket waited 61 times (428 s), which shows up as
30–80 s check calls in GPT-5.4 r3 (L73–L80). That cost time, not
correctness.

### 1.2 The markup arrow, traced

**The truth.**
* Note 4's second line, "RAMP SLOPE CANNOT EXCEED 8.33 % MAX.", sits at
  x 60.1–225.5, y 224.5–230.3 (the suite's target, `tasks.MECK_1031A_RAMP_NOTE`).
* The section label "RAMP SLOPE UP TO 7.5 % (8.3 % MAX.)" is a text-layer
  object at 490–543 × 429–460 (`p0.c33` in `read_document`).
* The sheet carries four statements of a maximum ramp slope.

**GPT-5.4 baseline, step by step.**
1. **t=30, page look** (2×2 tiles), prompt "Find the ramp slope note … provide
   the image_box".
   * The whole-page answer names the section label (`px=[1238,1118,1448,1233]`
     → page 491–545 × 433–458).
   * Tile r1c1 names note 4: "RAMP SLOPE CANNOT EXCEED 8.33 % MAX. … the second
     line of Note 4", box `[139, 658, 539, 712]` on the tile's grid → page
     59.5–230.8 × 217.7–235.6, which holds the target line.
   * Tile r1c2 names the SLOPE "B" row; tile r2c2 names the section label.
2. **`found` (fix E):** two rows. The section label ("Note text", page +
   r2c2), and the r2c1 "TYP. MATCH RAMP SLOPE" labelled **"image_box"**.
   **Note 4 and the SLOPE "B" row are missing.** Their tile boxes are 400 and
   326 wide on the 0–999 grid, and `answer_boxes` discards any box wider or
   taller than `REGION_GRID = 300` as "a region, not a thing"
   (`vision_view.py:775`, `:804`; used by `_merge_found`,
   `vision_tools.py:946-953`).
3. **t=48, zoom on the r2c2 tile box.** The view is 448.3–587.3 ×
   400.0–490.6 pt (139 × 91 pt, 2,046 px, 14.7 px/pt). The answer reads
   "RAMP SLOPE UP TO 7.5 % (8.3 % MAX.)", `px=[638, 469, 1322, 839]`, which
   converts to page 491.7–537.9 × 431.8–456.9.
4. **t=52, `annotate_document`:** a callout on that view + `image_box`, with
   `target: "ramp slope note"`.
   * planlens anchors a box at its centre (`markup_writer.py:897-900`) and
     puts the callout's tip at the anchor point (`:1057`).
   * Tip (514.9, 444.4) is the centre of step 3's box to 0.1 pt, and the
     centre of the printed label.
5. **The check:** `{"same_thing": true, "comment_fits": true, "inside": "RAMP
   SLOPE UP TO 7.5% (8.3% MAX.)", "sure": true}` → confirmed. The answer says
   "the placement check confirmed the markup is on the intended item".

**GPT-5.4 r2** is the same through step 4.
* The page answer again names the section label.
* The zoom (view 448.7–623.3 × 399.6–490.2) carried an `aim_note`: its
  nearest box was 14 pt from the aim, the leader. The agent did not act on it.
* The check came back **unsure** (`"inside": "UP TO 7.5%", "sure": false`,
  L42). The agent wrote the **same mark** again with `target: "UP TO 7.5%"`,
  i.e. the check's own reading, and the check confirmed it (L48).

**GPT-5.4 r3** got the same page answer.
* Its next step zoomed tile r1c1's box: view 17.2–266.7 × 184.3–274.6, 8.2
  px/pt. That box was in the tile's text but not in `found`.
* Target "the note 'RAMP SLOPE CANNOT EXCEED 8.33% MAX.'". Tip (143.1, 228.9),
  0.0 pt from the line.
* The model's text gives no reason for the different choice (no reasoning was
  recorded, §6).

**Sol (3 of 3)** asked its page look for "the ramp slope note **stating an
8.33 % maximum**".
* The whole-page answer named note 4 directly, `[75, 365, 280, 378]`.
* It zoomed tile r1c1's box and anchored there.
* Tips: (142.8, 227.4) on the line; (133.4, 223.0) and (133.3, 222.7) 1.5–1.8 pt
  above it. Those last two zoom boxes held both lines of note 4 and the
  centre fell between them; the 4 pt slack passes them.

**So, what made the repeat systematic.**
* **(M)** GPT-5.4's page look, given "the ramp slope note" without the
  figure, picks the label that literally reads "RAMP SLOPE" (3 of 3). The
  agent follows the page answer over the tiles (2 of 3).
* **(P)** The tile that found note 4 is invisible in `found` in 6 of 6
  runs. The replay drops exactly these boxes:

  | run | r1c1 box (note 4) → page | r1c2 box (SLOPE "B") → page |
  |---|---|---|
  | GPT-5.4 ×3 | 59.5–230.8 × 217.7–235.6 (and two near-identical) | 621–790 × 280–326 |
  | Sol ×3 | 59.5–230.8 × 224.3–230.9 (0–1 pt from the line) | 622–757 × 290–298 |

* **(P)** The check confirmed a comment on the wrong note twice. Once the
  `target` was a generic description; once it was the check's own `seen` text
  after an "unsure". `target` is free text that never moves a mark
  (planlens: "text, not drawn"). It only changes what the check compares
  against, so naming the thing you happened to hit always passes.

### 1.3 The circles, ring by ring

Truth (sheet 1): seven GCE callouts T1–T7 (T1 on a heavy grid line, T5 turned
90°), with look-alikes GCG × 6, QCE, GPE × 3, FBG × 3, a bare GCE and the
legend row. The suite counts a hit when the tag's centre is inside the ring
± 4 pt and the ring is at most 60 × the tag's area
(`review_eval/checks.py`, `check_markups_on_targets`). Below, "×" is the
ring's area over the tag's (≈ 44 pt²).

| run | final rings (centre distance to tag, area ×) | missing | why |
|---|---|---|---|
| GPT-5.4 r1 | T2 0.7, T3 0.9, T4 1.0, T5 0.8, T6 0.7, T7 0.8 pt; all 6× | **T1** | T6 zoomed twice, T1 never zoomed |
| GPT-5.4 r2 | T2 0.9, T3 0.4, T4 0.7, T5 2.2, T6 0.8, T7 0.1 pt; 6–7× | **T1** | the same |
| GPT-5.4 r3 | T1 12.0 (26×), T2 21.9 (58×), T3 23.9 (**77×**), T4 15.9 (36×), T6 18.7 (59×), T7 17.8 (41×) | **T3** (blanket), **T5** | tag + leader rings; T5 not pursued |
| Sol r1 | all seven 0.0–0.5 pt, 5.5–6× | — | — |
| Sol r2 | T1 11.8 (55×), T5 9.8 (38×); T2–T4, T6, T7 69–119× | **5** (blankets) | check rejected exact rings, agent widened them |
| Sol r3 | T1–T4, T6, T7 0.1 pt (6×); T5 10.2 (21×) | — | — |

Every tight ring is within 0.1–2.2 pt of its tag. No ring came from a
whole-page or tile box: all came from zooms, or from `find_like`'s page boxes
in Sol r2's first round.

**GPT-5.4 r1 and r2 (L41–L113 and L41–L103): the duplicate that hid a
miss.**
* The page answer listed T1 in both reps. In r2 it said "I found 7 visible
  GCE callouts".
* The agent zoomed seven boxes taken from the tiles:
  - four from r2c1 (T2, T3, T5, T6);
  - one from r2c2 (T4);
  - two from r3c1 (T6 again, from the tile overlap, and T7).
* Tile r1c2's T1 box (`[241, 482, 264, 501]`, also on the page answer's
  list) was never zoomed.
* All seven zooms were legible (16.7 px/pt) and read "GCE" without brackets.
* The first `annotate_document` wrote 7 rings. planlens flagged index 5 as
  index 4 again (`overlap 0.99` / `0.95`; fix I, `duplicates_note`,
  `planlens/tools/toolkit.py:928`).
* The agent rewrote with 6 rings and answered "6 qualifying callouts"; the
  check confirmed all 6.
* The duplicate was the clue that one of the seven it meant was unmarked.
  The note says "count each thing once", which the agent did, and nothing
  points it at the gap.
* `found` listed T1 (seen in page and r1c2) as "Tag", the same label as five
  other rows, so it could not serve as a checklist.

**GPT-5.4 r3 (L33–L72): tag-plus-leader rings, confirmed.**
* Every zoom prompt asked for "a tight image_box around the **whole
  tag+leader** suitable for marking"; the six answers boxed tag and leader.
* The annotate call named `target: "GCE penetration tag with leader"`. Each
  look boxed tag + leader as the thing (`thing_px` 450 × 240 px of the crop,
  against about 180 × 75 for a tag alone), so the size rule saw 1–2 ×, not
  26–77 ×. All six were confirmed (L60–L80).
* T5 was not among the six:
  - the page answer listed six (no T5);
  - tile r2c1 listed it ("Mid-right area, vertical text: text GCE",
    `px=[1491, 835, 1518, 899]`), and `found_note` named it;
  - but its label came out as **"?"**, because the box sat on the line after
    its description (`_label_text`, `vision_view.py:778-786`, takes only the
    words before the box on the same line).

**Sol r1 (L41–L133).**
* Eight `render_region` calls straight on page boxes of 80 × 70 pt (windows
  of about 104 × 94 pt, 16.7 px/pt), each answered "GCE — px=[…]". One
  answered "QCE" and was dropped.
* `find_like` used T1's box as the example. Fix H left out 24 ink pixels of
  grid line: 128 candidates (brief 4: 400), 7 contact sheets, **231 s**.
* It reported 14 "instances": the 7, the bare GCE, and 6 sub-tag fragments of
  3–5 pt read as "GCE".
* Sol zoomed five of the extras and rejected them all (GPE × 2, GCG × 3).
  Seven rings, seven confirmed.

**Sol r2 (L42–L179): the check talks a correct run into a wrong one.**
1. `find_like` with T3 as example: 40 s, 2 sheets, exactly the 7 callouts
   plus the bare GCE (`seen_as: other`).
2. Annotate #1 (L62): seven rings on `find_like`'s **exact** boxes, 0.0–0.3
   pt from every tag, with `target: "GCE callout tag with leader"`.
   * The check: **5 misplaced, "GCE — beside the mark, not inside it"**.
   * Each look boxed tag *and leader* as the thing (`thing_px` about 700–850
     px wide, against about 180 px for a tag). That box's centre lies outside
     a ring drawn round the tag, so `_encloses` (`markup_check.py:413-415`)
     said "beside".
3. Annotate #2 and #3: the agent re-zoomed (13–17 px/pt) and asked "is GCE
   inside or beside a small circle?". It got "plain text, not inside a
   circle", rewrote tight rings and got **4 misplaced** each time. All 13
   rejections were wrong.
4. Annotate #4 (L164): it widened every ring to tag + leader with `target:
   "complete GCE callout tag and its leader"`. 7 of 7 confirmed; the suite
   saw 5 blankets (69–119 ×) and 2 hits.
* The tool's note says "Widening a mark until it takes the thing in does not
  place it". The check itself then rewarded exactly that, because the
  widened target made the widened ring the right size.

**Sol r3 (L42–L153).**
* `find_like`'s example was a tile's whole-callout box (52 × 22 pt, height
  21.6 pt; 227 px of "linework" left out). It ran 60 s for 12 wrong-size
  hits, and Sol ignored it.
* Fifteen 13–17 px/pt zooms. Annotate #1 had **no `target`** (label "GCE",
  comment "GCE").
* The check looked for "GCE": seven confirmed, and the eighth ring, on the
  **QCE** (Sol's zoom had read it "GCE"), was **rejected rightly**
  ("seen: QCE").
* Rewrite: 7 of 7.

### 1.4 The markup check against truth

| marks checked | confirmed | rejected rightly | rejected wrongly |
|---|---|---|---|
| rings that enclose their tag at ≤ 30 × its area | **56 of 69** | — | **13** (all Sol r2, "beside the mark") |
| rings over 30 × (tag + leader blankets) | **12 of 12** (wrong) | **0** | — |
| ring on a look-alike (QCE) | 0 | 1 of 1 | — |
| comments on note 4's line | **4 of 4** | — | 0 |
| comments on the section label | 2 (wrong) + 1 unsure | — | — |

Against brief 4:
* **Fixed:**
  - the "- GCE" leader-end rejections are gone;
  - comments on the right line are now confirmed (0 of 10 → 4 of 4).
* **New:**
  - a blanket is confirmed whenever the target describes the blanket;
  - a correct ring is rejected whenever the target includes something the
    ring does not.
* **The common cause:** `_target` puts the agent's free-text `target` ahead
  of the `label` (`markup_check.py:238`, `for key in ("target", "label",
  "quote")`), and `_verdict` measures enclosure (`:413`) and size
  (`_too_wide`, `:159-177`) against the box the look gives for *that*
  description.
* **The pattern:**
  - all 25 wrong ring verdicts (13 rejections and 12 confirmations) came from
    the two runs whose target described tag + leader *and* whose look then
    boxed tag + leader: Sol r2 (the look is Sol) and GPT-5.4 r3 (the rings
    were already tag + leader);
  - GPT-5.4 r1 and r2 used much the same wording ("… callout tag with
    leader", "… tag with leadered callout"), but GPT-5.4's look boxed the
    lettering alone there (`thing_px` ≈ 180 × 75 px), so every verdict was
    right;
  - Sol r1 ("GCE callout tag") and Sol r3 (no target, label "GCE") produced
    no wrong verdict.

  So the check's answer depends on how a look reads an ambiguous,
  agent-written description.

### 1.5 The brief's look-fors

* **Refused `annotate_document` calls:** 0 of 19 (brief 4: 8 of 14 runs lost
  a round trip). No `color`, `comment` kind or `anchor` object was tried, so
  fix D's notes never fired.
* **Duplicate rings:** 2 flagged (GPT-5.4 r1 and r2, T6), both removed on the
  next call. See §1.3 for what they hid.
* **Zoom resolution:**
  - 68 zooms, none with an agent `dpi` (the schema no longer offers it);
  - circle zooms 13.1–16.7 px/pt (median 16.7);
  - markup zooms on 140–250 pt windows 8.0–14.7 px/pt;
  - brief 4: 45 of 106 at a median 4.2.
* **T5 once the tiles-only note names it:** ringed in 5 of 6 runs.
  - GPT-5.4 r1 found it only through the tiles, named "Tag" in `found_note`.
  - GPT-5.4 r3 missed it: named, but as "?".
* **Bracketed readings:** none in any zoom answer (fix G). The two bracketed
  reads were `find_like`'s own verifier (Sol r1), where brackets are wanted.
* **`find_like`:** 3 calls, all Sol:
  - T1 example: 231 s, 7 sheets;
  - T3 example: 40 s, 2 sheets, exact;
  - a whole-callout example: 60 s, useless.

  None flooded (fix H's limit is 200; the most was 128). Brief 4: 46, 180 and
  289 s.
* **Aim notes (fix F):** fired 5 times (GPT-5.4 r2 ×5, including the markup
  zoom). No agent changed course on one.
* **The record:**
  - 1,068 lines; 399 model starts and 399 ends; 123 tool starts and 123 ends;
  - **0 `reasoning`** on any `model_end`, and no text on any of the 69
    tool-calling steps;
  - `run.json` now carries `versions`, `commits` and `vision_profile` (fix L,
    record part).

---

## 2. Part A — label crops

Both models: **32 of 32 right, 0 unread, 0 wrong**, one call per sheet. The
sheets:
* the log ruler, 10 labels;
* the grading plot, 15;
* the log-log scan, 7.

GPT-5.4 used 8 calls (5 are the probe) for 13.3K in / 0.26K out in 16.6 s;
Sol used 8 calls for 11.6K / 0.46K in 22.0 s. This meets the brief's "good".
The labels `measure` depends on are read reliably on both models.

---

## 3. Part E — GEC-12 Figure 7-15, second reading

### 3.1 The figure, measured without a model

* **Where it is.** Page index 283 of `GEC 12 vol 1.pdf` carries Figure 7-14
  and Figure 7-15 as two embedded RGB images. Figure 7-15 is image xref 820,
  765 × 587 px, shown at 113.25–480.43 × 419.88–701.26 pt.
* **The axes.** Linear: φ 30–45° with a gridline every 1°; qL 0–400 tsf with
  a gridline every 25 tsf.
* **The fit.** I fitted x to the 16 vertical lines (frame included) and y to
  the 17 horizontal lines (max residual 0.44 and 0.35 px). One pixel is
  0.025° and 0.85 tsf.
* **The curve.** I read it as the centre of its black stroke (about 3 px)
  in the column at each angle. Where the "Very Loose" arrow shares a column
  I took the upper run, which is the curve; the arrow is a separate run near
  qL ≈ 10.
* **Precision.** About ±1 tsf, plus ±0.5 tsf from x on the steep part.
* **A check by eye.** Zooms at 32° and 42° confirm it: at 32° the curve
  crosses two-thirds of the way from 0 up to the 25-tsf line (≈ 16); at 42°
  it crosses just under the 300-tsf line.

| φ (°) | figure, code (tsf) | table `gec_12/figures.py` | table vs figure | GPT-5.4 | Sol | earlier session's code read (VISUAL_SCALES_DESIGN E8) |
|---|---|---|---|---|---|---|
| 28 | not on the chart (axis starts at 30) | 5 | — | **110** | "outside the axis, cannot be read" | — |
| 30 | 7.1 | 10 | **+41 %** | — | — | — |
| 32 | 16.0 | 20 | **+25 %** | 16 | 17 | ≈ 16 (−19 %) |
| 34 | 35.9 | 40 | +11 % | — | — | −10 % |
| 36 | 75.2 | 75 | 0 % | 75 | 75 | same |
| 38 | 133.6 | 130 | −3 % | — | — | +3 to +6 % |
| 40 | 208.5 | 200 | −4 % | 205 | 210 | |
| 42 | 296.0 | 280 | −5 % | 300 | 280 | ≈ 296 |
| 43 | 339.2 | 320 | −6 % | — | — | |
| 43.77 (curve ends) | 368.8 | 351 | −5 % | — | — | — |
| 44, 45 | no curve | 360, 400 | — | — | — | — |

### 3.2 Verdict

**The digitisation is wrong.** All three readers agree with each other and
not with the table at 32° (16, 17 and 16.0 against 20). The table is high at
the loose end (+41 % at 30°, +25 % at 32°, +11 % at 34°), right at 36°, and
3–6 % low from 38° to the curve's end. Its nodes at 26, 28, 44 and 45° are
not on the chart:
* the printed axis begins at 30°;
* the curve ends at 43.77° and 369 tsf.

The same table is copied in kPa into `axial_pile/nordlund.py:312-318`. The
docstring there says "the chart spans 26-45 deg" (`:308`); it spans 30–45 and
the curve 30–43.8.

**Exact correction** (1° nodes; linear interpolation between them is within
±2 % of the curve except +4 % at 33.5°, checked against half-degree reads):

```
phi (deg)  30    31    32    33    34    35    36    37     38     39     40     41     42     43     43.75
qL (tsf)   7.1   10.1  16.0  24.5  35.9  53.7  75.2  102.3  133.6  168.3  208.5  251.6  296.0  339.2  368
qL (kPa)   680   967   1532  2346  3438  5142  7201  9796   12794  16116  19966  24093  28345  32482  35240
```

**Where to change it:**
* `geotech_references/gec_12/figures.py:194-195` (and the range in
  `:198-220`);
* `axial_pile/nordlund.py:312-318` (×95.76; docstring `:299-309`).

**Effect on the Nordlund toe limit:** −29 % at 30°, −20 % at 32°, −10 % at
34°, 0 at 36°, +3 to +6 % at 38–43°.

**Owner's call:** what to do below 30° and above 43.75°, where the chart gives
nothing.
* Today the reference raises outside 26–45; `axial_pile` clamps to 192 kPa
  below 26 and to 38,304 kPa above 45.
* The honest options are to refuse outside 30–43.75°, or to clamp to the end
  values (7.1 / 368 tsf) with a warning that says the chart was left.

### 3.3 The models' readings and `code_reading`

* **Values.** Where a value exists, both models are within 6 % of the code
  read (GPT-5.4 at 32 / 36 / 40 / 42: 0, 0, −2, +1 %; Sol: +6, 0, +1,
  −5 %).
* **φ = 28°, off the printed axis.**
  - Sol refused, rightly (no READ line, so `code_reading` "no_read_lines").
  - GPT-5.4 answered **110 tsf** and boxed a point at φ 36.5°, qL ≈ 135, while
    saying in the same answer that the axis runs from 30 (at φ = 40 and 42).
    **(M)**
* **READ boxes are poor in y.** GPT-5.4's are 42–85 pt off the curve point
  (x 2–15 pt). Sol's are exact at 32° (0.3 pt) but 99–108 pt low at 36, 40
  and 42° (x within 0.6 pt). Both models answered on a 1,582 × 2,048 portrait
  view of the whole page, with the chart in its lower half. **(M)**
* **`code_reading` returned `no_scale` on all 9 READ lines**, so nothing was
  measured or flagged. Three causes, each reproducible offline on this public
  page:
  1. **(P)** planlens `find_scales(doc, 283)` finds no scale at all, in
     0.1 s.
     - `page_facts` reads pixels only when `is_raster_page(page)`
       (`scalefinder.py:449-456`).
     - `is_raster_page` needs image coverage ≥ 0.5
       (`planlens/document/raster.py:282-291`).
     - A textbook page with two figures covers 0.436, so the pixels are
       never looked at.
     - The design measured that 71 % of catalogued reference figures are
       embedded images (VISUAL_SCALES_DESIGN E9). Many share their page with
       text or a second figure, so this gate blinds `code_reading` on most
       reference charts.
     - `log_grid` uses the same gate (`loggrid.py:2679-2681`).
  2. **(P)** With the gate forced open, `find_scales` finds Figure 7-15's
     frame (170.6–463.1 × 432.6–659.2 pt) and its **x** scale (16 gridlines,
     4 labels to read). It finds **no y scale**: 17 gridlines, labelled only
     on every 4th line (0, 100, … 400).
  3. **(P)** Even with both scales, `code_reading` measures within
     `pad = max(3, 0.008 × 792) = 6.3 pt` of the READ box
     (`chart_reading.py:62`, `:240`, `:270`). Seven of the eight boxes at
     angles on the axis are 42–108 pt off the curve, so it would report
     "no_curve".
     - For a single-curve chart the input value alone fixes where to look
       (the curve's crossing of x = φ inside the frame).
     - The box is needed only to choose between several curves.

Tokens: GPT-5.4 5 calls, 19.5K / 0.75K, 14.1 s; Sol 5 calls, 21.9K / 4.4K,
60.4 s. Every answer except Sol's 28° carried a `READ … px=[…]` line.

---

## 4. Part F — patch alignment (GPT-5.4)

`GEOTECH_VISION_PATCH_ALIGN=1` did what it says: the page went as 2,048 ×
1,344 (view 1,224 × 803.4 pt) instead of 2,048 × 1,325. Costs:
* OFF: 23 calls, 65.2K in, 28 s;
* ON: 18 calls + 16 retries, 53.8K in, 53.7 s.

| repeat | sent px | median / max error (pt) | script's y scale (OLS + intercept) | through the origin | OLS without T1 |
|---|---|---|---|---|---|
| OFF 1 | 2048 × 1325 | 5.3 / 7.3 | 1.016 | 1.0108 | 1.009 |
| OFF 2 | 2048 × 1325 | 5.5 / 8.8 | 1.017 | 1.0148 | 1.014 |
| OFF 3 (tag + leader boxes) | 2048 × 1325 | 17.2 / 23.3 | 1.014 | 1.0097 | 1.031 |
| ON 1 (tag + leader boxes) | 2048 × 1344 | 20.1 / 24.9 | 1.012 | 0.9940 | 1.020 |
| ON 2 | 2048 × 1344 | **1.5 / 5.2** | 1.010 | **0.9999** | 1.004 |
| ON 3 (5 of 7 tags) | 2048 × 1344 | 3.5 / 5.0 | 1.024 | **0.9999** | 1.024 |
| brief 4 OFF 1–3 | 2048 × 1325 | 19.2, 5.1, 3.5 | 1.015, 1.017, 1.014 | 1.0076, 1.0160, 1.0122 | — |

**Verdict.**

* **On the brief's own statistic there is no signal.** The hypothesis was
  "within about 0.003 of 1.000". ON gave 1.010, 1.012 and 1.024; OFF gave
  1.014–1.017. The spread within the ON arm (0.014) is larger than the
  difference of the means (0.0005).
* **That statistic does not test the hypothesis (S).**
  `location_remeasure_foundry.py:103-109` fits a slope *with an intercept*.
  "The model works in a frame rounded up to whole 32-px patches" predicts a
  pure stretch about the image's top edge, `reported = k × true`:
  - k = 1,344/1,325 = 1.0143 OFF;
  - k = 1.000 ON.
* **Fitted that way there is a clear signal:**
  - OFF 1.0097–1.0148 this round, 1.0076–1.0160 in brief 4 (6 repeats, mean
    1.012);
  - ON 0.9940–0.9999 (mean 0.998);
  - the difference (0.014) is about four times the per-repeat standard
    deviation (≈ 0.003 in each arm).
* **Errors in the clean repeats** (median / max): OFF 5.3–5.5 / 7.3–8.8 pt,
  ON 1.5–3.5 / 5.0–5.2 pt.
* **What the through-origin fit leaves out.** In two of three ON repeats T1,
  the only tag in the top third of the sheet, came back 4–5 pt high. That is
  what drags the intercept fit up, and it may be a second, small effect.
* **Not decided, for three reasons:**
  - 3 repeats per arm;
  - 7 targets with one carrying most of the leverage;
  - one repeat in each arm boxed tag + leader.

**What would decide it** (GPT-5.4 only, no agent; about 60–80 calls,
≈ 250K input):
1. 10 repeats per arm, OFF and ON interleaved in one process.
2. A target set spread evenly over the full sheet height: 20 or more tags in a
   grid, built with `planlens.testing.tag_fixtures` as the tag set is.
3. A third arm at another aspect ratio, so that k OFF differs (e.g. an image
   2,048 × 1,100 → 1,120, k = 1.018). If the OFF stretch tracks the padding
   ratio across both shapes, the patch explanation holds.
4. Reported per repeat: the through-origin k, residuals by vertical band
   (top, middle, bottom thirds), and a flag on answers whose boxes are tag +
   leader (box wider than 2 × the tag).

Keep the switch OFF until then; the owner's rule is "on only when measured".

---

## 5. Part G — report ingest (private: IDs, counts, rates)

### 5.1 G1: brief 4's run files rescored, no model call

| reader | set | after | model alone | floor alone | values read (unlinked) |
|---|---|---|---|---|---|
| logs | open 6 | 301/307 (98.0 %) | 292/307 (95.1 %) | 248/307 (80.8 %) | — |
| logs | blind 9 | 166/214 (77.6 %) | 151/214 (70.6 %) | 146/214 (68.2 %) | — |
| logs, index | open | 32/32 | 32/32 | 18/32 | — |
| logs, index | blind | **0/16** | **0/16** | **0/16** | — |
| lab | open 16 | 188/367 (51.2 %) | identical, every sheet | not kept by rc3 | 261/301 (86.7 %) |
| lab | blind 15 | 359/458 (78.4 %) | identical, every sheet | not kept by rc3 | 278/366 (76.0 %) |

These match FOUNDRY RUN 2 (MEASUREMENTS.md:4257-4284) exactly where both
report a figure: logs 98 % / 78 %, lab after 51 % / 78 %, gradation 111/303
= 37 %.

**Brief 4 item 1: logs index 0 % blind. Settled: the floor never held the
values.**
* Floor alone 0/16, model alone 0/16, merged 0/16. There was no floor value
  for `model_wins` to replace, so the per-object evidence rule
  (`floor.py:117-145`) did no harm here. Per-value evidence is **not
  needed** for this.
* 13 of the 16 values are on one log (R13_p45); the others are R21_p96 (2)
  and R34_p49 (1).
* The before-score's 7/16 credits numbers in the right column inside the
  depth window. Neither the floor (`seed_from_grid` binds index cells only
  inside the sample's row group) nor the model attaches them to a sample on
  that form.
* On the open set the model filled every index value the floor lacked
  (floor 18/32 → 32/32: R07, R15, R28).
* **The merge never lost:** after ≥ max(floor, model) on all 15 logs, e.g.
  R30_p65 floor 19, model 16, after 20; R34_p49 floor 29, model 23, after
  29.

**Brief 4 item 2: gradation links. Three of four are links; one is a
reading failure.**

| sheet | set | after | values read | verdict |
|---|---|---|---|---|
| R28_p176 | open | 1/37 | 35/35 (100 %) | link |
| R28_p177 | open | 1/37 | 35/35 (100 %) | link |
| R17_p114 | open | 5/56 | 33/46 (72 %) | link (plus some reading) |
| R15_p82 | blind | 1/30 | 6/28 (21 %) | **reading** |

* **The same link signature** (read ≥ 70 %, after ≤ 20 %) appears on three
  more open sheets: R17_p136 compaction 7/8 read → 1/10, R36_p56
  consolidation 7/8 → 1/10, R28_p172 chemical 3/3 → 1/5. On these six open
  sheets, **120 values read became 10** after linking.
* **The reading failures are all blind:**
  - R15_p82 gradation 21 %, R25_p56 gradation 28 %;
  - R25_p142 direct shear 30 %;
  - R27_p85 triaxial 17 %, R35_p124 triaxial 13 %.
* **Linking is unstable run to run.** In G2 (5.33.0rc1, same sheets),
  R17_p136 linked 9/10 in one arm and 1/10 in the other, and R36_p54 14/15
  and 1/15. A single run cannot measure a link fix.
* **Not settled from G1:** whether the hole matched after folding and the
  depth difference per specimen. G1's rows had the `misses` lists removed
  for privacy, rightly. Those two facts can be handed back as counts per
  specimen (hole matched yes/no; |Δdepth| in bands 0–0.15 / 0.15–1 / > 1 m /
  feet-vs-metres).

### 5.2 G2: visual scales off and on

| | OFF | ON |
|---|---|---|
| logs, all 15 (after) | 466/521 (89 %) | 468/521 (90 %) |
| logs, blind (after) | 166/214 | 168/214 |
| blind `layer_top`: before / floor / model / after | 22 / 22 / 22 / 22 of 36 | 22 / 22 / 22 / 22 of 36 |
| blind index, model and after | 0/16 | 2/16 (R13_p45) |
| log calls, input, time | 28, 298.5K, 616 s | 29, 304.6K, 695 s |
| lab, all 31 (after / read) | 520/825 / 552/667 | 516/825 / 552/667 |
| lab calls, input, time | 35, 403.1K, 693 s | 36, 415.9K, 720 s |

**Did the voters fire? On the evidence, no, not in any way that reached a
score.**
* **No label-reading call.**
  - `log_grid_for` charges label reads to the log's cost, so they show in the
    stage total but not in the per-log `calls` column
    (`visual_scales.py:280-310`, `log_scoring.py:721-748`).
  - In both arms the total equals the per-item sum: logs 28 = 28 and 29 = 29;
    lab 35 = 35 and 36 = 36.
  - The one extra call in each ON stage is a reader's own (R36_p38 1 → 2 at
    +11K input; R35_p29 1 → 2 at +14K), well above what a label sheet costs
    (Part A: about 2K per sheet).
* **No pixel-found layer top reached the grid.** `before` is scored on the
  very grid the setting shapes (OFF folds pixel layers away; ON keeps them).
  It is 56/70 overall and 22/36 blind in both arms.
* **The +2 on logs** is R13_p45's index values in the model column (60 → 62).
  Index has nothing to do with visual scales; this is run-to-run variance.
* **The lab differences** are the link flips above (±8 to ±13 values on
  single sheets). `plot_check` only adds QA entries; the record is
  unchanged.
* **`plot_vs_table` and `layer_votes` counts:** RESULTS.md does not print QA
  entries or the run file's `visual_scales` block, so they cannot be
  reported (P, a reporting gap).

**Why `layer_top` could not move.**
* **The voters only act on scans.**
  - Pixel layer tops exist only where planlens' raster leg ran, i.e. where
    `is_raster_page` is true (`loggrid.py:2679-2681`, same 50 % gate as
    §3.3).
  - Label reads happen only where the grid `needs_values`: a scan whose depth
    labels have no text.
* **On this corpus the scans have text.**
  - The operator passed `di_dir` (glue `datasets/ri_runs.py:124`).
  - The corpus attaches Azure DI text to every page that needs OCR
    (`report_ingest/corpus.py:540-567`, `di="auto"`).
  - So a scanned log's ruler labels are text and no label read can arise.
* **On text-layer (vector) logs, ON and OFF are the same by construction.**
* **The blind misses (14 of 36) are not the voters' business.** Floor, model
  and after all score 22/36, so the model never moves a layer top either. The
  misses sit where the visual-scales lever does not reach, or where its
  stratum lines matched existing tops.
* **R25_p19** (0/14 in every column, both arms; 8 unresolved) is the one log
  that looks like a scan nothing reads.
* **To locate the misses at no model cost**, the FDE can hand back, from the
  ON run files, per log as counts:
  - `visual_scales.pixel_layers`, `label_calls` and `layer_votes`
    agree/disagree;
  - the `layer_top` found/total;
  - whether each page was raster (`is_raster_page`) and had DI text.

---

## 6. Engine — reasoning summaries

**Verified.** The SDK names on Foundry (`setup/setup_checks.csv`,
`sdk_reasoning_types`) are:

```
Reasoning, ReasoningConfig, ReasoningContent, ReasoningContentVisitor, ReasoningEffort,
ReasoningSummary, ReasoningSummaryTextDeltaChunk, ReasoningSummaryVisitor,
ReasoningTextDeltaChunk, SummaryConfig
```

**Why nothing is requested.**
* `_reasoning_request` (`webapp/palantir_sdk_engine.py:477-493`) takes the
  first existing name from `_REASONING_TYPES` (`:463-464`) and from
  `_SUMMARY_TYPES` (`:465-466`).
  - Those are `Reasoning`, which the FDE reports is not the request type,
    and `ReasoningSummary`, a content class with no `AUTO`.
  - `getattr(ReasoningSummary, "AUTO", None)` is `None`, so the function
    returns `None`.
  - `_generate_responses` then sends no `reasoning` (`:760-769`) and records
    `reasoning_summary: "not requested"`, as both first calls show.
* The request field is `ReasoningConfig(effort, summary)`. The level enum is
  `SummaryConfig` (AUTO / CONCISE / DETAILED / UNKNOWN).
* **Why the offline test passed.**
  `webapp/tests/test_palantir_reasoning.py:38-44` fakes exactly the guessed
  names: `Reasoning` as the request class and `ReasoningSummary` with AUTO.

**The exact change** (not made). In `palantir_sdk_engine.py`, put the real
names first:

```python
_REASONING_TYPES = ("ReasoningConfig", "Reasoning", "ResponsesReasoning",
                    "OpenAiResponsesReasoning", "ReasoningParams")
_SUMMARY_TYPES = ("SummaryConfig", "ReasoningSummary", "ReasoningSummaryType",
                  "ReasoningSummaryMode", "Summary")
```

and make `_reasoning_request` probe instead of trusting the first name. Take
the first summary type that actually has the level, then the first request
class that accepts it:

```python
    want = level.upper()
    value = next((getattr(getattr(r, n), want) for n in _SUMMARY_TYPES
                  if hasattr(getattr(r, n, None), want)), None)
    if value is None:
        return None
    for n in _REASONING_TYPES:
        cls = getattr(r, n, None)
        if cls is None:
            continue
        for kwargs in ({"summary": value}, {"effort": None, "summary": value}):
            try:
                return cls(**kwargs)
            except TypeError:
                continue
    return None
```

`ReasoningConfig` must stay first. `Reasoning` might well accept
`summary=` and build, and then fail on the service. That would trip the
refusal path, which stops asking for the rest of the run.

**The read side is unverified too.** `_reasoning_text` reads
`item.reasoning.summary[*].text`. The SDK's output union member and its
content class (`ReasoningSummary` / `ReasoningContent`) are not known
offline. When a summary was requested and none was parsed, the engine should
log the output items' field names once, so the first Foundry call says
which.

**An offline test that pins it.**
* Install a fake `…_v3_responses` module carrying all ten real names in their
  real roles:
  - `Reasoning` and `ReasoningSummary` as plain content classes with no
    `AUTO`;
  - `ReasoningConfig(effort=None, summary=None)` as the request class;
  - `SummaryConfig` with AUTO / CONCISE / DETAILED / UNKNOWN;
  - `ReasoningEffort` with the usual levels;
  - the visitor and chunk classes as stubs.
* Assert that the captured `OpenAiResponsesRequest.reasoning` is a
  `ReasoningConfig` whose `summary is SummaryConfig.AUTO`, and that
  `generation_info["reasoning_summary"] == "requested"`.
* Keep the existing test as the old-name case.
* Add one where `SummaryConfig` exists but no request class takes it, so
  nothing is sent.
* The live check is one Foundry call: `reasoning_summary` "requested" and a
  non-empty `model_end.reasoning`.

---

## 7. Fixes

None is fitted to the tag sheet, 10.31A or Figure 7-15. Each changes a
general capability, and the tool says what it does in its own description or
result.

**Classes:** P = plumbing or code (reproducible offline or with any model);
M = model behaviour only Foundry can measure; S = scorer or task.

**Claude API check:** Haiku 5.5 / Sonnet 5.5, public documents only. It
qualifies only for P items whose path a fake model cannot exercise.

| id | failure | class | code site | proposed general fix | how to measure | Claude API check? |
|---|---|---|---|---|---|---|
| B1 | `found` drops long text lines (note 4 lost in 6/6 markup runs; the SLOPE "B" row too) | P | `vision_view.py:775`, `:804` (`REGION_GRID` on either side); `vision_tools.py:946-953` | Call a box a region only when it is large in **both** directions (or by area), so a long thin line of text is a thing | Replay the 12 saved page looks offline: the 6 note-4 boxes come back and the tag rows are unchanged; then Part B markup × 3 on GPT-5.4 | n: the replay of saved answers is exact |
| B2 | Half the `found` rows (61/121) labelled "?", "Tag", "image_box"; T5 named as "?" (GPT-5.4 r3) | P | `vision_view.py:778-786` (`_label_text`), `:789-812` | When the words before a box are empty or a field word (box, px, tag box, image_box, location), take the label from the item's heading line above | Offline replay: generic labels 61 → near 0 | n: offline |
| B3 | Markup check measures against the agent's free-text `target`: 13 exact rings rejected, 12 blankets confirmed, a wrong-note comment confirmed after `target` was renamed to the check's own reading | P | `markup_check.py:234-243` (`_target`: target before label), `:383-434` (`_verdict`, `_encloses` `:413`, `_too_wide` `:159-177`) | (a) For enclosing marks, judge **enclosure and size against the label's thing** (the printed name the ring is for), and ask the look for the box of the printed name or symbol itself, never its leader; use `target` for identity only. (b) A re-write of the same mark whose only change is a target equal to the previous verdict's `seen` keeps that verdict, and the result says why. | Offline scripted-look tests: a look that boxes tag + leader when told "with leader". Live: Part B circle × 3 per model, wrong ring verdicts 25 → 0, blankets confirmed 12 → 0 | **y**: Sonnet on the public tag fixture (about 30 looks, under $1) shows what a real look boxes for "tag with leader" against label "GCE"; a fake cannot. It is not a GPT measurement. |
| B4 | A duplicate ring hid a missed tag (T1, GPT-5.4 r1 and r2): the agent removed the copy and stopped | M (P note) | planlens `tools/toolkit.py:928` (`duplicates_note`) | The note says that a duplicate means one intended thing may be unmarked, and to compare the rings with the things found | Part B circle on GPT-5.4: T1 ringed after a duplicate flag | n: model behaviour |
| B5 | Arrow on the section label (GPT-5.4 r1, r2): the page answer followed, the tile ignored | M | — (B1, B3 remove the plumbing half) | none beyond B1 and B3 | Part B markup × 3 on GPT-5.4 after B1 and B3 | n |
| B6 | `find_like` given a whole-callout example (21.6 pt tall, 227 px of "linework" left out): 60 s, 12 wrong-size hits | M (P guard) | `funhouse_agent/find_like.py` (example check) | When most of the example's ink is left out as linework, or the box is much taller than the page's lettering, say "box the mark alone" before searching | Offline replay of Sol r3's example | n: offline |
| E1 | `code_reading` blind on figures that share a page (`no_scale` 9/9) | P | planlens `document/raster.py:282-291`; `scalefinder.py:449-456`; `loggrid.py:2679-2681` | Decide raster **per region**: where the query or the frame lies inside an embedded image, read that image's pixels whatever the page's coverage | Offline: `find_scales(GEC-12, 283)` finds Figure 7-15's frame and scales; then sample 20 catalogued figure pages | n: offline, public PDF |
| E2 | Figure 7-15's y scale not found even with pixels (labels on every 4th gridline) | P | planlens `scalefinder.py` (axis fit from labels and gridlines) | Fit the label values to gridlines by spacing when labels mark only every n-th line | Offline on page 283: y fitted, `measure` at 32° / 42° gives 16.0 / 296 ± 1 tsf | n |
| E3 | `code_reading` searches ±6.3 pt round a READ box; boxes were 42–108 pt off in 7 of the 8 on-axis reads | P (box placement M) | `chart_reading.py:62`, `:240`, `:254`, `:270` | When the input value fixes the line (x = φ), search the whole frame along it; use the box only to choose among several crossings, and report the box's distance from the crossing | Offline with the 9 saved Part E answers once E1 and E2 land: 9/9 measured | n |
| E4 | GPT-5.4 read 110 tsf at φ 28°, off the printed axis | M (P guard) | `chart_reading.py` | Mark a READ input outside the fitted axis range `out_of_range`, with no value | Offline with the saved answer | n |
| E5 | Figure 7-15 table wrong: +41 / +25 / +11 % at 30 / 32 / 34°, −3 to −6 % at 38–43°; invented nodes at 26, 28, 44, 45 | P (data) | `geotech_references/gec_12/figures.py:194-195`; `axial_pile/nordlund.py:299-318` | Replace with §3.2's 1° table, 30–43.75°; the owner decides refuse or clamp outside | Unit test pins 32° = 16.0 and 42° = 296.0 tsf; the Nordlund regression shows the toe-limit change | n |
| F1 | The patch-alignment statistic does not test the hypothesis | S | `foundry_handoff/location_remeasure_foundry.py:103-109` | Report the through-origin k and residuals by band beside the OLS slope; add a spread-out target set and a second aspect ratio | §4's design, GPT-5.4 only, about 70 calls | n: the effect is GPT-specific |
| G1 | Lab links lost on read values (6 open sheets: 120 read → 10 kept) and unstable run to run | M (P lever) | `report_ingest/lab_scoring.py` (diagnostics); `lab_floor.py` (a link floor) | First hand back per-specimen hole-match and depth-gap counts. Then consider a deterministic link floor from the sheet's own printed hole and depth, which the model may correct only with evidence. | Two repeats per sheet, so a single run's flip is not read as a fix | n: private data |
| G2 | Blind index values on R13_p45's form are held by neither floor nor model | P | `report_ingest/log_floor.py` (`seed_from_grid` row-group binding) | Bind an index cell to the nearest sample within the depth tolerance, not only inside its row group | Offline `floor_alone` on the local truth (no model): blind index 0/16 → ? | n: private, no model needed |
| G3 | RESULTS.md cannot show what visual scales did | P | `report_ingest/cluster_scoring.py` (log and lab tables) | Print per stage: label calls, pixel layers, `layer_votes` agree/disagree, `plot_vs_table` count, and the raster/DI page counts | Rescore saved run files | n |
| L1 | No reasoning summary ever requested | P | `webapp/palantir_sdk_engine.py:463-466`, `:477-493`; the test fakes `test_palantir_reasoning.py:38-44` | §6: real names first, probe for the level and a class that takes it; log the output field names once when nothing parses | Offline fake SDK with the 10 real names; one Foundry call | n: only Foundry's SDK can show it |

**Before the next Foundry round,** in order:
1. **Offline:** B1, B2, B3, E1–E5, G3 and L1. Each can be tested on saved
   answers, public PDFs or a fake SDK.
2. **One Sonnet check, about $1:** B3's look behaviour on the public tag
   fixture.
3. **Next round, on Foundry:** Part B × 3 both models; Part E both models;
   §4's patch design; one L1 call.
