# Backlog tidy, 2026-10-20: proposal for the lead

**Status: applied on `wip/backlog-tidy` the same day, at the lead's
request.** The triage, every hand edit below, OWN-69 and its card, and the
sweep (`## Swept 2026-10-20 (app master 116ddb1)`) are in the branch. The
merged cards were folded: INF-08 into INF-01, OWN-13 into GM-140. REF-14,
REF-19 and OWN-24 were deleted, because OWN-40, OWN-21 and VIS-20 say the
same. The rest of this file is the record of what was done and why.

## 1. The Done archive

**Path:** `BACKLOG_DONE.md` at the repo root. CI's `paths-ignore` (`**.md`)
skips it, so the archive costs no CI runs.

**Measured on the live BACKLOG.md of 2026-10-20 (master 116ddb1), in the
worktree, then restored:**

| Order | BACKLOG.md | BACKLOG_DONE.md | test pair |
|---|---|---|---|
| sweep only | 287,799 -> 129,788 bytes (238 entries, 227 distinct lead IDs moved) | 158,535 bytes | 16 passed, exit 0 |
| sweep, then the triage | 129,788 -> 132,072 bytes | 158,535 bytes | exit 0 once HANDOFF:219 is edited (EV-12, below) |
| triage, then sweep (recommended) | 287,799 -> 290,083 -> 127,789 bytes (-56 %) | 162,826 bytes | 16 passed, exit 0 |

`sweep_done.py --next GM` gives GM-222 before and after the sweep, and
`--next OWN` gives OWN-69. `build.py` builds the desk from the swept
backlog: 322 items, 79 owner items, 79 cards.

**What changed on the branch:**
- `module_work/backlog_dashboard/sweep_done.py` (new) moves every Done entry,
  text unchanged and in order, to the top of `BACKLOG_DONE.md` under
  `## Swept <date> (<note>)`. Earlier sweeps stay below it. The Done heading
  stays in BACKLOG.md with a one-line pointer. It keeps the file's line
  endings, refuses to write if any entry line would be lost, and has
  `--dry-run`. `--next PREFIX` reads both files.
- `module_work/backlog_dashboard/apply_triage.py` (new) applies a triage
  file:
  - MERGE and CLOSE-COVERED rows always;
  - DROP? rows only when named in `--drop`.
  Each removed item gets a Done entry starting `- **ID**`, so whatever cites
  it still resolves and the number is never reused. A merge text goes on the
  end of the target as "Also (was ID, merged DATE): ...". Rows already
  applied are skipped. It refuses, changing nothing, when an item or a target
  is not open, or a target is itself leaving. It prints the HANDOFF
  §0a-current lines that cite a removed ID.
- `webapp/tests/test_backlog.py`:
  - closed IDs come from the Done list plus `BACKLOG_DONE.md`, so archived
    IDs still count as defined and are never reused (the message names
    `--next`);
  - new `test_archive_holds_only_swept_entries`: the archive has a
    `## Swept` heading and no tiered open item;
  - new `test_sweep_keeps_every_closed_id_and_open_item`: a sweep of copies
    of today's files keeps every closed ID and open item, and the same next
    free GM ID;
  - new `test_apply_triage_moves_merged_and_closed_items_to_done`: on a
    synthetic backlog.
  - HANDOFF §0a-current still accepts open IDs only, as before.
- `build.py` reads the archive's IDs, entry count and sweep dates. The "cards
  for closed IDs" note counts the archive as closed; a card for an ID found
  in neither file gets its own note. The page gets `DATA.archive`, and the
  Done tab (`template.html`) names the file, the count and the last sweep
  date.

### 1a. Replacement for "How every session uses this file", items 2 to 5 and the test list

Item 1 is unchanged. Replace from "2. **While working.**" down to the end of
the test list ("- CLAUDE.md and HANDOFF still point here.") with:

```markdown
2. **While working.** Anything new goes HERE the moment it is found, under its area with the next free ID: a bug, a gap, a follow-up, an unverified claim, an owner question, a fix list from a review or a field session. The next free ID is the highest number in use for that prefix in this file or in `BACKLOG_DONE.md`, plus one (`.venv/Scripts/python module_work/backlog_dashboard/sweep_done.py --next GM` prints it); never reuse one.
   - One line per item: what it is and why it matters, then `*Src:*` the document holding the detail.
   - A detailed write-up elsewhere is fine. A to-do list elsewhere is not.
3. **Finishing an item.** Delete its line and add it to "Done since the last sweep" with the commit. If a §0 item changed, update HANDOFF §0a-current's slice too.
4. **Releasing.** Every item the release closed moves to Done. Raise or lower tiers for the next train. Then sweep: `.venv/Scripts/python module_work/backlog_dashboard/sweep_done.py --note "app master <sha>"` moves the whole Done list, text unchanged, to the top of `BACKLOG_DONE.md` under a dated heading. Sweep on demand too, whenever the Done list grows long.
5. **A full sweep** (on the owner's ask, or every few weeks): re-read the plan documents against this file, as on 2026-10-12. Triage the Later and Parked items into a file (keep, merge, close as covered with the commit, or ask the owner about a drop) and apply it with `module_work/backlog_dashboard/apply_triage.py`, as on 2026-10-20. Then run `sweep_done.py` and record the date below.

After any change to this file, rebuild and republish the Backlog Desk (`module_work/backlog_dashboard/build.py`, same URL). Every `[owner]` item needs a plain-English card in `module_work/backlog_dashboard/context/`; the guard test checks this.

`webapp/tests/test_backlog.py` (part of the gate) checks:
- the IDs are unique, and each has a tier;
- every ID cited here exists, open or closed (in Done or in `BACKLOG_DONE.md`); every ID cited in HANDOFF §0a-current is open;
- no closed ID is reused, and `BACKLOG_DONE.md` holds only swept entries;
- CLAUDE.md and HANDOFF still point here, and every `[owner]` item has a desk card.
```

Also update the key line to read
`**Last full sweep:** 2026-10-20 (app master <sha>; Done swept to BACKLOG_DONE.md; Later/Parked triaged, module_work/backlog_dashboard/triage_2026-10-20.json)`.

### 1b. CLAUDE.md lines

- **Trigger row "bumps the version / cuts a release".** Replace its last sentence with:

  > Then sweep `BACKLOG.md`: every item the release closed moves to "Done
  > since the last sweep" with its commit, and
  > `module_work/backlog_dashboard/sweep_done.py` moves that list to
  > `BACKLOG_DONE.md` (the archive; its IDs are never reused).

- **Trigger row "finds a bug, a gap...".** Change "with the next free ID" to:

  > with the next free ID (`sweep_done.py --next PREFIX`, which reads
  > `BACKLOG.md` and `BACKLOG_DONE.md`)

- **State block, line ~171 (`BACKLOG.md` "Done since the last sweep" has
  every item with its commit).** Change it to:

  > `BACKLOG_DONE.md` (swept 2026-10-20) has every item with its commit

  Lines ~295 (`see BACKLOG "Done"`) and ~2259 can read
  `BACKLOG.md "Done" and BACKLOG_DONE.md`.
- **HANDOFF's day bullets** say "every item in BACKLOG 'Done' with its
  commit" seven times. They stay true if read as "Done, or the archive after
  a sweep". Rewording is optional.

### 1c. Notes checklist (memory `feedback_notes_checklist.md`, step 1)

> 1. BACKLOG: close the item with `close_item2.py`, and put the commit ID on its Done entry (close before committing, then add `Commit \`id\`.` in the next commit, or batch). At a release, or when the Done list grows long, run `module_work/backlog_dashboard/sweep_done.py`. The next free ID is `sweep_done.py --next PREFIX`, never a count of BACKLOG.md alone.

Memory index line for `feedback_one_backlog_list.md`: add "closed items
archived in BACKLOG_DONE.md (sweep_done.py, 2026-10-20)".

### 1d. `close_item2.py` (scratchpad)

The sweep rewrites the Done intro, so the anchor at line 11 must change.
Accept both, so the script works on either side of the sweep:

```python
ANCHORS = (
    "Older docs still list these as open. Items closed before the last sweep "
    "are in `BACKLOG_DONE.md`, newest sweep first.\n\n",
    "Older docs still list these as open. Delete this section at the next sweep.\n\n",
)
anchor = next(a for a in ANCHORS if s.count(a) == 1)
```

The sweep keeps the blank line after the intro, so the anchor is the same
whether the list is empty or full.

## 2. Triage of the Later and Parked items

`module_work/backlog_dashboard/triage_2026-10-20.json` holds one row per
item: `{id, tier, owner, action, target, reason, evidence}`, plus
`merge_text` on a MERGE. All 237 items are covered (227 Later, 10 Parked).

| Action | Count |
|---|---|
| KEEP | 207 |
| MERGE | 15 |
| CLOSE-COVERED | 6 |
| DROP? (owner) | 9 |

Rules followed: an item about a wrong number, an unsafe default, a missing
printed method or a measured agent failure is KEEP or MERGE. Every
CLOSE-COVERED names a commit or Done entry checked in code or in git log.

### MERGE (15)

| Item | Into | Why |
|---|---|---|
| PL-21 | VIS-11 | VIS-11 already says it merges PL-21 (OCR as a text source). |
| REF-13 | REF-29 | Same NHI-06-088/089 indexing; REF-13 adds "no figure catalog either". |
| REF-14 | OWN-40 | Duplicate (no text appended). |
| REF-19 | OWN-21 | OWN-21 already lists "structural future rounds". |
| OWN-24 | VIS-20 | The owner's standing answer for VIS-20's knobs. |
| INF-04 | INF-03 | The openai<3 comment (pyproject.toml:46-49, still wrong) belongs to INF-03's caps. |
| GM-102 | GM-219 | Same Story-PDF header-shading defect; html_to_pdf renders through fitz.Story. |
| GM-22 | GM-199 | GM-191 pinned anchors to GEC-4 Examples 1-2 (b1613af); the rest is GM-199's worked-example check. |
| GM-106 | GM-216 | The same question: which plane the wall pressure is drawn on. |
| VIS-15 | VIS-24 | Location rows keep only the page_bbox computed in code (cf43c90). What is left (page coordinates in the prose) is VIS-24. |
| VIS-13 | DR-28 | DR-28 already makes VIS-13 its first step; inline_images.py:71 still asks for 0-999 boxes. |
| OWN-13 | GM-140 | A slope method question; all slope work waits for GM-140. |
| INF-08 | INF-01 | Both leftovers wait on OWN-27. Every one still exists (checked 2026-10-20). |
| REF-20 | TA-03 | gpt-5.1 is no longer the rollout model. A chart read-off on the rollout model is a TA-03 check. |
| DOC-19 | DOC-10 | The manual's HTML and PDF come from one build (`docs/user_manual/`). |

### CLOSE-COVERED (6)

| Item | Covered by |
|---|---|
| TA-12 | TA-20, 8cd38f4: AgentBusyRetry on every engine, "tested on ... the Funhouse bridge". |
| EV-28 | 8b9abba (the three keys and the scratch-write detector), then 006ed10 (`"within": "judgment_calls"` on 8 keys in flows.json). |
| EV-22 | EV-30, 6e452d3: the check for answer numbers that no tool computed. The leftover is EV-32. |
| GM-33 | c9752a0 (GM-141): seismic_geotech/calc_steps.py ~281-298 prints PGA when given and notes when Fpga is read against Ss. |
| EV-12 | A standing rule, already a CLAUDE.md read-first trigger (4fd1987). **HANDOFF.md:219 cites EV-12; delete " (BACKLOG EV-12)" in the same commit, or the HANDOFF test fails.** |
| INF-09 | DOC-18, 1f278d4: the Harness Map was rebuilt 2026-10-19 (`dev/harness_viz/out/data.js`). Its stale summaries are DOC-20. |

### Most notable KEEP notes

- **Trim the text of GM-132:** the first clause ("tag the other modules")
  was done by GM-165 (0a44e7b) and GM-184 (8ab231a). The `[True]` vs `"set"`
  part and the undeclared names remain.
- **Trim the text of GM-46:** G13 was done (3195ede). G3(d), G12 (a unique
  guess is still not resolved by `describe_method`) and PAV-6/7 remain.
- **Trim the text of DOC-11:** "currently disconnected" and "no adapter yet"
  are gone. "machine precision" (lateral_pile/DESIGN.md:63) and the rest
  remain.
- **GM-89 and GM-126** are slope items without the "**Deferred to GM-140**"
  tag their siblings carry. Add it.
- **Confirmed open, wrong-result items:**
  - GM-11 (fem2d side surcharge only sizes the ramp);
  - GM-32 (a descending backfill is taken as level, silently);
  - RI-01 (citations print 0-based pages);
  - GM-161 (weak-rock defaults uncited, `choices` empty);
  - GM-04 (downdrag β = 0.3 uncited, downdrag/analysis.py:1108);
  - PL-11 (`scan_text("Pc (kip): 437.0")` still returns `[]`).

### After applying: things the script reports

- 19 open lines still cite a merged or closed ID. They resolve through the
  Done entries. Optionally repoint two:
  - the switches table's OWN-24 -> VIS-20;
  - INF-01's own "OWN-27" text, which now covers INF-08.
- Desk cards for merged owner IDs: INF-08, OWN-13, OWN-24, REF-14 and REF-19
  have cards in `context/*.json`. `build.py` reports them as "cards for
  closed IDs". Fold each card's text into its target's card, or delete it.
- **BACKLOG.md line 493** (live file) is an orphan fragment of INF-17's old
  line, split by a literal `\n` in 3cd8fc1. It starts
  `-" for "A-1", one table type lost); ...`. Delete it. INF-17 is in Done.

## 3. DROP? owner card (lead adds the OWN ID: `sweep_done.py --next OWN` -> OWN-69)

The BACKLOG line:

```markdown
- **OWN-69** [Next] [owner] **May nine old backlog items be dropped?** OWN-23, OWN-37, OWN-42, OWN-20, TA-09, TA-15, VIS-18, REF-17, GM-44; reasons in `module_work/backlog_dashboard/triage_2026-10-20.json`. On the answer: `apply_triage.py ... --drop <IDs>`. *Src:* `module_work/backlog_dashboard/tidy_proposal_2026-10-20.md` §3
```

The card, for `context/own_decisions.json`:

```json
"OWN-69": {
 "kind": "decision",
 "ask_who": null,
 "question": "May these nine old backlog items be dropped?",
 "why": "The backlog holds 237 'later' and 'parked' items. A review of every one found nine that no longer earn a place: later work overtook them, or they are ideas nothing has needed. Dropping them makes the list shorter to read. Each moves to the Done list with its reason, so nothing is lost and its number is never reused.",
 "background": "Only items that are your call are listed here, and none of them fixes a wrong or unsafe result. The same review merged 15 items into others that cover the same work and closed 6 that were already done (the commit named for each). Those needed no decision.",
 "parts": [
  {
   "label": "Overall",
   "options": [
    {"label": "Drop all nine", "effect": "All nine move to the Done list as 'dropped by the owner', each with its reason."},
    {"label": "Drop some", "effect": "Tick the ones to drop below; the rest stay open."},
    {"label": "Keep all", "effect": "Nothing changes; all nine stay open."}
   ]
  },
  {
   "label": "If some: which to drop",
   "multi": true,
   "options": [
    {"label": "OWN-23: a separate multi-user screen", "effect": "Each user already gets their own conversations on Tiny Apps and on Funhouse; the old idea was a different web framework."},
    {"label": "OWN-37: follow-ups from the security scan", "effect": "The app now installs beside the newer pandas, the quarantined package is no longer used, and the matplotlib flag is a labelling quirk of the scanner."},
    {"label": "OWN-42: the Funhouse admin note about refused keys", "effect": "The app now retries a refused key once and says so plainly; Funhouse is only the backup host now."},
    {"label": "OWN-20: switches that copy worked examples' own conventions", "effect": "The module's defaults are correct as they stand (the validation notes say so). Bearing depth factors already have such a switch, and pile alpha has its own question (OWN-63)."},
    {"label": "TA-09: email from the app on Tiny Apps", "effect": "Email works only through Funhouse's own toolkit, and nobody has asked for it on Tiny Apps."},
    {"label": "TA-15: old live checks from versions 5.11 to 5.18", "effect": "Those features have run in every live test wave and in daily use since."},
    {"label": "VIS-18: a public CAD drawing benchmark", "effect": "Measuring against our own review tasks matters more, and that work stays (VIS-17)."},
    {"label": "REF-17: a backup for chart reading and a smarter search, only if needed", "effect": "Nothing has needed it since June; the rule stays written in the project notes."},
    {"label": "GM-44: standard multi-step 'playbooks'", "effect": "An idea with no failure behind it; the worked-examples library already shows the agent multi-step work."}
   ]
  }
 ],
 "recommendation": "Drop all nine.",
 "if_no_answer": "All nine stay open; nothing changes.",
 "message_to_send": null,
 "steps": [],
 "unblocks": [],
 "sources": [
  "module_work/backlog_dashboard/triage_2026-10-20.json",
  "module_work/backlog_dashboard/tidy_proposal_2026-10-20.md"
 ]
}
```

OWN-20, OWN-23, OWN-37 and OWN-42 have their own cards now. If they are
dropped, delete those cards too.

## 4. Commands (main checkout, one commit)

```bash
cd C:/Users/socon/OneDrive/dev/GeotechStaffEngineer
git merge --no-ff wip/backlog-tidy
PY=.venv/Scripts/python
$PY module_work/backlog_dashboard/apply_triage.py module_work/backlog_dashboard/triage_2026-10-20.json --dry-run
$PY module_work/backlog_dashboard/apply_triage.py module_work/backlog_dashboard/triage_2026-10-20.json
#   then by hand:
#   - HANDOFF.md:219: delete " (BACKLOG EV-12)"
#   - BACKLOG.md: delete the orphan line after INF-21
#   - §1a text; the GM-132 / GM-46 / DOC-11 trims; "Deferred to GM-140" on GM-89 and GM-126
#   - the OWN-69 line and its card (§3); fold or delete the five merged cards
#   - CLAUDE.md lines (§1b)
$PY module_work/backlog_dashboard/sweep_done.py --note "app master $(git rev-parse --short HEAD)"
$PY -m pytest webapp/tests/test_backlog.py webapp/tests/test_docs_currency.py -q; echo "exit=$?"
$PY module_work/backlog_dashboard/build.py      # then republish the desk, same URL
git add BACKLOG.md BACKLOG_DONE.md HANDOFF.md CLAUDE.md module_work/backlog_dashboard/context/own_decisions.json module_work/backlog_dashboard/context/other_owner.json module_work/backlog_dashboard/context/own_actions.json
git commit -m "Backlog tidy: triage of Later/Parked applied; Done swept to BACKLOG_DONE.md"
```

When the owner answers OWN-69:

```bash
$PY module_work/backlog_dashboard/apply_triage.py \
    module_work/backlog_dashboard/triage_2026-10-20.json --drop OWN-23,OWN-37,...
```

This applies only the named DROP? rows. Rows already applied (their ID is
closed in Done or in `BACKLOG_DONE.md`) are skipped, so the same file serves
both runs.
