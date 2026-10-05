# Foundry brief 3 — 5.32.0rc2/rc3 on GPT-5.6 Sol, full resolution (2026-10-02 to 10-04)

Run by the AI FDE in Palantir Foundry: Responses route, `original` sent as
AUTO (full size, ≥ 4096 px), numpy 1.26.4 (wheels declare ≥ 2.0), a forked
child process per task, ≤ 8 model calls in flight (glue, then also the
package). The original 35 tasks ran on rc2 and were RESCORED on rc3 (no model
calls); the two new tasks ran on rc3 with `find_like` hidden (OpenCV aborts
the process on this host: FIPS self-test, exit -6). Not comparable with the
5.31 runs (chat route, ~768 px).

## Scores after every check fault is fixed

`sweep_r3 / set-long-rare-tag` failed in `RESULTS_repeats.md` on a bulleted
page list the check could not read (fixed after the run, 7e605ba); rescored
locally it passes. No other result changes.

| arm | 35 original tasks | 2 new tasks × 3 runs | input tokens (originals / new) | minutes (originals / new) |
|---|---|---|---|---|
| baseline | 35/35 | 5/6 | 4.53M / 2.38M | 45.5 / 30.5 |
| sweep | 35/35 | 6/6 | 10.81M / 2.50M | 124.6 / 11.1 |
| minimal | 33/35 | 4/6 | 18.45M / 9.46M | 67.6 / 34.2 |
| baseline_high | 15/15 (drawn lettering) | 1/2 (one run) | 0.88M / 0.70M | 18.6 / 20.8 |

## What the misses are

- **baseline_r2 / produce-circle-tags (0/7 on target).** Every ring does take
  its tag in, but the rings are 70-125 pt across round 10 pt tags (the good
  runs drew ~20 pt rings). The markup check reported the marks misplaced six
  times; the agent widened them each time until the check confirmed all
  seven, then handed the file over. Fixed in the check (4ea4bd0): boxes and
  circles are also asked whether they are drawn closely round the thing.
  1,220 s, 7 `annotate_document` calls.
- **set-long-rare-tag, the real misses:** minimal named pages 2, 3, 8 extra
  (precision 0.50) and, in r2, missed page 12; baseline_high named 9 and 22
  extra. Page 19 appears in most answers (baseline, sweep, baseline_high) and
  is wrong: it carries only FBG callouts — one stroke from FPG — and is within
  the check's 0.75 precision.
- **minimal on the originals:** set-3600-psi found 1 of 3 sheets (no text
  search); fixture-markups could not name "Contractor B" (a reply's author is
  in the annotation data, not drawn on the page).

## Reading

- Baseline as shipped holds: 35/35 on the originals and 5/6 on the new tasks;
  its one new-task miss is the markup-widening case, now fixed in the check.
- Sweep ties baseline on the originals at 2.4× tokens and 2.7× time, and its
  edge on the new tasks (6/6 vs 5/6) is the markup case, not coverage. No
  case for making it the default.
- The harness's text and markup tools carry real answers on a strong model:
  minimal loses two originals and two new-task runs at about 4× baseline's
  tokens.
- Resolution: no score change on 15 drawn-lettering tasks (full size +15 %
  tokens, ~15 % faster), but on the 24-sheet rare-tag task HIGH (~768 px)
  added two false pages where full size had none. Keep full size.
