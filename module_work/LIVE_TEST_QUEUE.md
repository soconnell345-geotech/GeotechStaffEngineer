# Live test queue — small checks to run in Funhouse later

Owner, 2026-10-04: "For small tests moving forward, just start keeping a list
of tests and we'll do it in funhouse in the future." Foundry is for the big
suite runs; a change that only needs a few live runs to confirm goes on this
list instead of a new Foundry round. Run them on the release that carries the
change (the Funhouse cluster installs from PyPI), tick them off here with the
date and result.

Each entry: what changed, the exact runs, what passing looks like, and what
to do if it fails.

## Open

1. **Markup check asks "drawn closely round it?"** (app 4ea4bd0, in 5.32).
   - Runs: `score_review_suite(prompter=fh_prompter, model_name="funhouse-gpt-high",
     ids=["produce-circle-tags", "produce-markup"], arms=("baseline",
     ("baseline_r2", {}), ("baseline_r3", {})), out_dir=...)` — 6 runs, two tasks.
   - Pass: produce-circle-tags rings on target in all three (the
     `markups_on_targets` check), produce-markup passes, and no run needs more
     than ~3 `annotate_document` calls.
   - Fail modes: good rings reported "drawn far wider" (too strict: the agent
     redraws or drops correct marks) → loosen `_CLOSE_Q` in
     `funhouse_agent/markup_check.py` or treat `close` as advisory; oversized
     rings still confirmed (too lenient) → no worse than 5.31.
2. **OpenCV test-load on a normal host** (planlens 999f82c + app ad2f67d).
   - Run: one Document Review question that uses `find_like` on a stroke-lettered
     sheet (e.g. the IZD-style "where are the GCE penetrations?").
   - Pass: `find_like` is offered and runs; the first call in the process is a
     second or two slower (the child-process test-load), later ones are not.
   - Fail: `find_like` missing from the tool list on Funhouse → read
     `planlens.opencv.available()` there and report its reason.
3. **Vision call cap of 8** (app be742b0). No separate run: watch the minutes
   column of item 1 against the same tasks on 5.31 (cap changes timing only).
4. **Narrative "mention" conventions read from the answer keys** (app
   8e428ce; still DRAFT for the owner).
   - Run: `score_on_cluster(stages=("narrative",), truth_dir=..., ...)` on the
     eight keyed reports (~0.4M input tokens on GPT-5.4).
   - Pass: siteResponseMention right on most of 8 (was 0/8 on 2026-10-02),
     soilCorrosion better than 2/6, geophysicalTestingMention still 8/8,
     overall recall/precision not below 75 % / 77 %.
   - Fail: the owner's reading of the fields differs — rule in
     `report_ingest/narrative_glossary.py`, one place.
5. **Lab reader link** — PENDING the cause (the Foundry run files asked for
   2026-10-04): once fixed, `stages=("lab",)` on the 31 keyed sheets; pass =
   open-set link back near the blind set's 100 % and gradation sheets
   R17_p114 / R28_p176 / R28_p177 back above 80 %.

## Done

(none yet)
