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
4. **Report-ingest stage d on 5.32: narrative conventions, lab link, log
   floor** — one run answers three questions (~1.1M input tokens on GPT-5.4;
   the Foundry stage d of 2026-10-02 is the comparison).
   - Run: `score_on_cluster(stages=("logs", "lab", "narrative"),
     truth_dir=..., ...)` on the keyed items, into a NEW out_dir (5.32 run
     files keep each item's records and misses; old ones do not).
   - Narrative (conventions re-read from the answer keys: 8e428ce, 571bcfa,
     d7d6652 — still DRAFT for the owner). Pass: siteResponseMention right on
     most of 8 (was 0/8), soilCorrosion better than 2/6, propertyType no
     longer empty on outside projects, geophysicalTestingMention still 8/8,
     overall recall/precision not below 75 % / 77 %. Fail → the owner's
     reading differs; the rule is in `report_ingest/narrative_glossary.py`.
   - Lab link (open set 55 % vs blind 100 % on Foundry; the merge was NOT the
     cause; the scorer's exact-string hole match was fixed in 5e70823). Pass:
     open-set link near the blind set's, gradation R17_p114 / R28_p176 /
     R28_p177 above 80 %. Fail → the run files now show the boring and depth
     the model wrote (`after.record`, the `link` misses): read them before
     changing anything.
   - Logs: recovery printed as a length now scores (46be026). Pass: blind
     recovery no longer 0/15-ish; each run file's `floor_alone` (via
     `report_ingest.log_scoring.rescore_saved`) shows whether the floor held
     what the grid's cells did (R30 N values, R13 water contents).

## Done

(none yet)
