# REF-53 check: section titles weighted double in BM25

**Decision: rejected and reverted.** One recall-eval case leaves the top 5
in `auto` and `rerank`. The gold set and the subsection cases lose none.
The search order on the branch is plain `bm25(sections_fts)` again, the
same as main (commit 4f0710c makes the change, f6e6a4f reverts it).

## The change measured

`reference_search` ordered by `bm25(sections_fts, 2.0, 1.0, 1.0, 1.0, 1.0)`:
title 2, summary, body, key_points and applicability 1 each. Figure search
was not touched (its index is a separate one), so the two figure cases in
the recall eval cannot move.

Method: one process, the same database, the weight switched between 1
(before) and 2 (after). Each case ran in the four modes the repo runs
offline (`off`, `fill`, `auto`, `rerank`; `EXPANSION_PIN` 1, `FUSION_K` 3).
Gold-set ranks come from `limit=10`, as `eval_retrieval_gold.py` counts
them. For the subsection and recall-eval cases, the top-5 test uses
`limit=5`, as their own scripts do, and ranks 6-10 come from `limit=10`.
Date: 2026-11-05, on main at 32bbd77.

## MRR@10 before -> after

| mode | gold set (54) | subsection cases (15) | recall eval, text (7 trusted) |
|---|---|---|---|
| off | .477 -> .475 | .719 -> .761 | .014 -> .000 |
| fill | .524 -> .526 | .719 -> .761 | .014 -> .000 |
| auto | .569 -> .569 | .752 -> .794 | .128 -> .092 |
| rerank | .514 -> .521 | .719 -> .761 | .205 -> .157 |

Recall, before -> after:

- **Gold set, R@3 / R@5 / R@10:**
  - off 30/32/33 -> 30/33/34
  - fill 34/37/39 -> 34/38/42
  - auto 37/41/44 -> 37/42/44
  - rerank 32/35/40 -> 33/36/42
- **Subsection cases, R@5:** 14/15 -> 14/15 in off, fill and rerank, and 15/15 -> 15/15 in auto. R@3 rises by one in every mode.
- **Recall eval, R@5 (text):** 0/7 -> 0/7 in off and fill; 3/7 -> 2/7 in auto and rerank. The script's overall figure under `auto`, figures included, goes from 4/9 to 3/9.

The gain that the REF-48 review measured (on an earlier library and
earlier fusion) has mostly gone. On the gold set `auto` is flat and `off`
is slightly lower.

## Every case whose rank moved

### Gold set (rank before -> after; `-` = not in the top 10)

| case | off | fill | auto | rerank |
|---|---|---|---|---|
| pvd-spacing | | | 4 -> 5 | |
| pvd-smear | | - -> 10 | 3 -> 2 | - -> 10 |
| dsm-geometry | | - -> 9 | | - -> 9 |
| mse-pullout | 5 -> 4 | 5 -> 4 | | |
| dmt-def | - -> 8 | - -> 8 | 6 -> 5 | |
| cpt-def | 8 -> 5 | 8 -> 5 | 3 -> 2 | |
| su-spt | 1 -> 2 | 1 -> 2 | 1 -> 2 | |
| design-cbr | 3 -> 2 | 3 -> 2 | 3 -> 2 | 3 -> 2 |
| stone-col-capacity | | | | 6 -> 3 |
| cfa-piles | | | | 2 -> 3 |

No gold case leaves the top 5. `pvd-spacing` in `auto` drops to rank 5,
the last place that still counts.

### Subsection cases (same moves in all four modes)

| reference | target | query | rank |
|---|---|---|---|
| gec_12 | 7.2.1.3.1 | Nordlund method | 5 -> 3 |
| gec_12 | 7.2.3.2.2 | group uplift cohesionless | 2 -> 1 |

### Recall eval (`scripts/eval_retrieval_recall.py`, text cases)

The eval's truth is the `off` top-1 of each canonical query, kept only if
it passes the gate word. Two truths move:

| concept | canonical query | truth before | truth after |
|---|---|---|---|
| undrained su | undrained shear strength | dm7_1 3-2.4.2 "Undrained or Total Stress Strength Tests" | dm7_1 8-5.2.2 "Correlations with Undrained Shear Strength" |
| permeability | permeability coefficient | eurocode_7_2 5.11.1 "Objective" | ufc_pavement 20-3.3.3 "Coefficient of Permeability" |

The "undrained su" synonym query ("cohesion cu of clay") finds neither
truth in any mode, before or after, so its score does not change.

**The case that left the top 5: permeability.** The synonym query
"hydraulic conductivity" returns the same list before and after under
`auto` and `rerank`, with eurocode_7_2 5.11.1 at rank 4 in `auto` and 3 in
`rerank`. The truth has moved to a pavement section, which that query does
not return in its top 10. So the case goes from hit to miss in `auto`
(4 -> -) and `rerank` (3 -> -). In `off` and `fill` it goes from 10 to
outside the top 10.

If the truth is held at its "before" value, nothing in the recall eval
moves except that `off`/`fill` rank 10 -> -, and its recall@5 is unchanged
in every mode. The miss comes from the eval's truth moving, not from the
synonym query ranking worse. The rule as set counts it anyway, and that is
why the change was reverted.

**Precision proxy** (literal top-1 of the canonical queries disturbed): the
same two canonical queries change their top-1 in `off`, `fill` and `auto`,
and `rerank` changes none. Both new top-1s are sections titled with the
query's own words, where before they were a section titled "Objective" and
one about test types.

## Also measured (not part of the acceptance rule)

- **Title ranking guard** (`eval_subsection_retrieval.py --titles`, every
  top-level section searched for by its own title, `auto`): of 7,466
  sections, 3,757 found themselves first before. 33 of those no longer do;
  609 more find themselves first only after; the first hit changes for 964.
  The net is +576 in favour of the weight.
- **Tests with the weight on:** the five retrieval test files
  (`test_query_expansion`, `test_subsection_retrieval`, `test_result_size`,
  `test_retrieval_db`, `test_fts_safe_query`) passed 283/283. No test pins
  the recall eval's truth: no test imports `eval_retrieval_recall.py` or
  names its targets. So no test needed updating, and none was changed.

## For the lead

The change failed only on a self-labelled truth that it moved. On the
gold set and the subsection cases it loses nothing, and the title guard
nets +576. It can be accepted again in two ways:

- fix the recall eval's truth (pin each case's target section, as the gold
  set does), or
- accept the permeability truth moving to the section titled "Coefficient
  of Permeability".

Either way, re-applying is the single ORDER BY change in commit 4f0710c.
Without one of those, the rule as written keeps it out.

## Full suite

`python -m pytest -q -p no:cacheprovider tests/`, run serially on the
reverted branch, exited with code 0: 6,434 passed, 37 skipped. All 37
skips are tests that read the PDFs with PyMuPDF (`fitz`), which is not
installed here. None of them is a retrieval test.
