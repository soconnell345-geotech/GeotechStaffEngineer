# PL-11 check: `Label (unit): value` in `find_quantities`

Branch `claude/pl-11-label-unit-value`, with the review fixes (REVIEW_PL-11
items 1-8). For the lead to fold into the app's notes at merge; delete it
from planlens then.

## What changed

- `planlens/document/quantities.py`
  - New pattern `_RE_LABEL_UNIT`: `(unit)` holding exactly one unit from
    `_UNIT_TABLE`, then `:` or `=`, then an optionally signed number (the
    same `_NUM` grammar as before) or a range (the same range tail as
    `_RE_VALUE`, so "Sample Depth (ft): 5.0-6.5" is 5 to 6.5 ft and a
    backwards range, "12 - 3", is arithmetic and refused). The number may not
    be followed by `:digit` (a time), `/digit` (a date), `+digit` (a station)
    or a digit after `.` or `,`; a `.` or `,` that ends the value is
    punctuation ("Phi (deg) = 30, c (psf) = 200" reads both).
  - A fraction is read only for `in` and `ft` ("Bar Dia. (in): 3/4" is
    0.75 in); "Ratio (%): 45/55" is refused.
  - `_value_ends_its_field`: the value is refused unless what follows it in
    its field is nothing (line break, column gap, `,` `;` `.`), a hedge, the
    next field's label ("Tc (kip):", "M (kN-m):", "FS:"), a datum
    ("NAVD88"), or prose resuming ("in", "at", "OK", ...). An operator
    (`x × * · ^ / + - ( =`), a second `=`, a comparison, a series
    ("0, 1.5, 3.0") or any other word (a symbol: "0.5 Ka gamma H") refuses
    it. This is what keeps "Vertical stress (psf) = 120 x 10 = 1,200" from
    being read as 120 psf.
  - A reading another pattern made of the same number is taken over or
    blocks: feet-inches become the value with the label ("Wall Height (ft):
    12'-6\"" is 12.5 ft); a different unit refuses the label form and the
    other reading stands.
  - A unit after the value counts only on the same line, without a column
    gap, and when it is not the next field's label (a token followed by its
    own `(...)` and `:`/`=`, or by `:`/`=`). Same unit: one mention ("Cover
    (in): 3.0 in"). Different unit: the label form yields ("Depth (m): 12.5
    ft" stays 12.5 ft, no label). This also fixes the pre-existing item 8:
    "P (kN): 437.0 M (kN-m): 120.0" was 437 m and is now P: 437 kN.
  - `_label_before` reads the label back from the text before the
    parenthesis. It stops at a line break, tab, two-space column gap,
    `: ; = |`, a word that starts with a digit (the previous field's value,
    also "3.0ksf"), a word ending a sentence, or six words. Leading joiners
    ("and", "the", ...) are dropped. No label means no mention.
  - A label holding "el"/"elev"/"elevation" gives kind `elevation`, by the
    word `_RE_ELEVATION` already uses (now the shared `_ELEVATION_WORD`).
  - `scan_text(text, line_breaks=())`: offsets of the characters that join
    printed lines. Only the label form uses them. `Document.quantities`
    passes the joins of each block (`_join_group` joins lines with one
    space), so on a page the label stays on its own printed line: "PILE
    HEAD CONDITIONS / Lateral Load (kip): 25.0" gives the label "Lateral
    Load" and the second line's box only. Every other form still reads
    across the joins.
  - `_canonical` finds "sq. ft." (its own spelling ends in a full stop).
    Before, "450 sq. ft." raised KeyError on 0.14.0, and the label form
    would have added a new way to reach it ("Area (sq. ft.): 1,250").
  - Priority 0 in the overlap resolution. When a label would run back into
    a value another pattern already took, the label is trimmed to the words
    after that value rather than dropped.
  - `QuantityMention` gains `label: Optional[str] = None`. `to_dict()` emits
    `"label"` only when set. `__str__` prefixes `label: ` when set.
- `planlens/document/document.py`: `quantities()` passes the line joins.
- `planlens/document/DESIGN.md`: one paragraph after "What is not read" on
  the form, the formula trap and the corpus-B re-run still owed.
- `CHANGELOG.md`: Unreleased entry. No version change.
- `planlens/tools/` untouched. `find_quantities` renders each mention with
  `render_quantity` (`planlens/tools/formatting.py`), which has no label
  slot. The label reaches the model through the quoted raw text, which for
  this form now begins at the label: `p0 437 kip [force] "Axial Load (kip):
  437.0"`. A separate label slot would repeat those words, so none was
  added.

## What a caller sees change

- Mentions 0.14.0 already read in this form keep their value, range, unit
  and hedge, and gain the label; their `text` widens to start at the label,
  and the box and `line_ids` widen with it ("Bearing (psf): 2,500 psf" was
  the mention "2,500 psf"). If the label names an elevation, the kind
  becomes `elevation`.
- The next field's label spelled like a unit (item 8) no longer gives a
  false length or force.
- "sq. ft." reads instead of raising.
- Nothing else. Checked with a side-by-side run of 0.14.0's `scan_text`
  and this one (`quantities.py` is the same on main 9cfe472 and 0.14.0):

  | text | strings | 0.14.0 mentions | identical |
  |---|---|---|---|
  | 0.14.0's own `test_quantities.py` strings | 113 | 56 | 56 (all 113 strings serialize byte-identical) |
  | reference library manual text (`geotech_references/*/text/*.json`) | 73,151 | 25,092 | 25,092 |
  | Markdown in planlens and the app's `docs/`, `module_work/` | 43,096 | 283,418 | 283,418 |

  Neither corpus gained a labelled mention. Two Markdown strings that hold
  "sq. ft." raise KeyError on 0.14.0 and read on the branch.

## The formula trap, measured

All 44 matches of `(unit) [:=] number` in the reference library's text are
formulas ("q_p (tsf) = 0.6 N60", "Mr (psi) = 1000 + 555*R", "Egrout (MPa) =
4732 x (f'c (MPa))^(1/2)"). The first version of this branch read 37 of them
as values (0.6 tsf, 1000 psi, 4732 MPa, ...). This version reads none.

## Tests (`planlens/document/tests/test_quantities.py`, all values made up)

- `Label (unit): value` rows: force, length, pressure, unit weight, angle,
  area; `=` instead of `:`; a negative value; thousands separators; an
  exponent; a space before the colon; the unit repeated after the value; a
  fraction in inches; three elevation labels; "sq. ft.".
- In context: a closing full stop, prose after the value, a check mark,
  hedges, a datum.
- Ranges: five rows, plus main's range reading kept with a label added.
- Overlaps: feet-inches taken over (two spellings) and refused under another
  unit; the previous field's attached-unit value kept ("qa=3.0ksf",
  "Unit Wt=120pcf", "Wall 3ft").
- Fields: two on one line, column gaps, comma-separated, joined with "and",
  the label stopping at the previous value and at a sentence end.
- The next field's label spelled like a unit: four rows (joined, column
  gap, line break, "Mn").
- Line breaks: `scan_text(..., line_breaks=...)` directly, and through
  `Document.quantities` on a made-up PDF block.
- Near misses that yield nothing: the 8 from the first version plus the
  hand-calculation lines (`x`, `×`, `(...)^2`, `*`, symbols, a backwards
  range, a second `=`), a comparison, two ratios and a series.
- Every existing row of the phrase table and the NOTHING list is unchanged
  and passes.

Run from the worktree root with `planlens.__file__` in the worktree and
`-p no:cacheprovider`, each judged by its exit code:
`planlens/document/tests/test_quantities.py` 128 passed (exit 0);
`planlens/tools` 122 passed (exit 0); `pytest planlens -q` 1851 passed in
8 min 5 s (exit 0).

## Published numbers

- `../GeotechStaffEngineer/module_work/drawing_ground_truth/doc_claims_check.py`
  measures drawing-geometry figures only and does not call `scan_text`, so
  nothing it prints can move with this change.
- **Needs a re-run, not regenerated by any script:** the bake-off table in
  `planlens/document/DESIGN.md`, row "B — 40 real | regex | 16 | 15 | 0 |
  1.00 | 0.52". The 15 misses are the form this reads, but the 40 real
  sentences are not in either repository (and the submittal is not on this
  laptop), so the number stays as measured. The re-run should count new
  false positives on the same pages as well as recall. Corpus A has no
  `Label (unit): value` sentence and is not expected to move. Neither row
  is guarded by `planlens/tests/test_readme_claims.py`.
