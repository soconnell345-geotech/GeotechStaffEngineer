# Review of brief 14: three planlens parts on the Document Review page

## 1. Verdict

**MERGE AFTER FIXES**, reviewed head `868c6a8` (code `7a50cf8`, base `69d6c2c`). No must-fix.
Two should-fix findings, both repaired on this branch (`claude/rv-b14-wiring`: `efa5568`,
`6984f48`); with them the branch can merge. planlens for the review:
`origin/claude/b13-integration` at `a99192c` ("the integration branch" below), and the released
0.16.0 wheel.

## 2. Findings

| # | Severity | Where | What is wrong | Input that shows it | Fix |
|---|---|---|---|---|---|
| 1 | should-fix, **fixed `efa5568`** | `document_tools.dxf_plots_available` | It read the toolkit's signature, `planlens.dxf.plot` and ezdxf. planlens' plot now needs Pillow and refuses every plot without it (`PlotUnavailable`). On such a host with `GEOTECH_REVIEW_DXF_PLOTS=1` the page said "A DXF is ALSO opened as a plot", `open_document` carried `plot`, and every look then failed. | Pillow made unimportable, `make_vision_tools(dxf_plots=True)`, `analyze_pdf_page(attachment_key="site.dxf")`: "could not plot ... PlotUnavailable", under a description promising the plot. | The detection also needs `PIL`; without it the page is the build it was. |
| 2 | should-fix, **fixed `6984f48`** | `vision_kit._dxf_plot_for_look` | A plot was kept by the file's digest, a failure was not: each look at a DXF past a bound of the plot read and drew it again (up to planlens' 30 s, plus the read) before refusing it. | `MAX_ENTITIES` patched to 1, three looks at one DXF: `dxf_to_pdf` ran three times. | The reason is kept beside the plots, as text; a later look is refused at once in the same words. |
| 3 | note (planlens' words, left) | planlens `review/conversation.py` 1401 | `open_document`'s `plot` line on a DXF still reads "render_page, render_region and the other page tools take this file too". The page has no `render_page`, and `find_like` on a DXF still raises `FileDataError` out of the tool (as on the base), `read_pdf_text` answers "Could not open ... as a PDF". The core's own description names the right tools. | `open_document(source="site.dxf")` with the switch on. | A planlens decision: let the host supply the tool names, or name only what the host layer serves. |
| 4 | note (planlens, left) | planlens page map | The plot's one page is `kind: "blank"`, `words: 0`, though it holds the drawing. | `document_page_map(handle="site.dxf")`, switch on. | planlens: a kind for a plotted page. |
| 5 | note (design, left) | `_dxf_plot_for_look`, planlens `ReviewToolkit.open` | One DXF is plotted twice: by planlens for the page tools (under its toolkit's guard) and by the core for the two looks. The core's copies are held per build with no count or size limit and leave with the build. Two looks at one new DXF in the same step both plot (no harm, double work). | A page-map call then a look at one DXF: two plots. | Design: planlens hands the host the plot it made, or the core bounds its table. |
| 6 | note | CHECK.md | Written against the preview `99af95f`; see section 6. | | The lead updates it or files this report beside it. |

Nothing in the wiring fails to match the planlens changes named in the prompt, other than
finding 1: `dxf_plots` as a callable (the host still asks with `True`; `_new_toolkit` wraps
it); the bounds and `PlotUnavailable` (both come back as `{"error", "hint"}`); `sheets` on a
Word, Excel or DXF file (planlens' refusal comes through whole); `from` in a `sheet_pages` row
and the new `drawing_geometry` keys (nothing in the core, the page or the app reads or trims
them: grep of `packages/`, `webapp/`, `funhouse_agent/`).

## 3. Paths checked

1. **Unset, byte for byte: holds.** A dump of all 24 page tools (description and schema), every
   helper's tool list, the system prompt's hash and 18 fixed calls on planlens' made-up
   five-sheet set and a made-up DXF, base worktree against the branch: identical on 0.16.0 and
   on the integration branch (deepagents 0.7.24); the only difference is a temp-folder name in
   a contact sheet's path. On deepagents 0.6.8 the builder's unset hash test and Document
   Review's fingerprints pass on both planlens. Both fingerprint files are unchanged and both
   tests pass on both planlens.
2. **`_SHEET_ADDRESS`: holds.** It is set and reset inside the tool's own call and read down
   the same stack, so it never outlives a call. Eight threads, 15 rounds each, three different
   `sheets` beside plain `pages` calls on one kit: 0 wrong `sheet_pages`, none leaked into a
   plain call. A call that raises before the dispatcher (`start_line="abc"`): the value is
   `None` afterwards and the next call carries no `sheet_pages`. `sheets=""` hands nothing on.
   Each wrapped tool dispatches once, to itself. A helper holds the same tool objects; two
   testers are two contexts.
3. **The DXF plot: holds after findings 1 and 2.** Per build (a closure of
   `make_vision_tools`), so no other build can reach it; the attachments dict is copied, not
   written to; a look by file name and by the `doc_...` handle planlens returns both see the
   plot; page 3 of a plot is refused in words; a broken DXF, a DXF past a bound and a missing
   Pillow all return an error with a hint, no raise. The core's look holds no lock while it
   plots. Unset: `analyze_pdf_page` and `render_region` on a DXF raise `FileDataError("Failed
   to open stream")` out of the tool function on base and branch alike.
4. **`_DXF_PLOT_REWRITE`: holds.** The swapped sentence is the core's own note
   (`_DOCUMENT_TOOL_NOTES["open_document"]`), not planlens' words, so a planlens release cannot
   unseat it. With the DXF switch and each of the other 20 switches in turn, the page's
   description carries the new words and no "do not apply to it". The builder's test pins both
   that the old sentence is there unset and that the swap happened.
5. **Detections: hold after finding 1.** On 0.16.0 all three say no and every switch set gives
   the build it was (`sheets` refused by name, no `drawing_geometry`, a DXF has no `plot`).
   None reads a version.
6. **`strict_tool`: holds.** `bogus=1` is refused by name with switches on and off; `sheets` is
   refused by name on `open_document`, `sheet_index` and `analyze_pdf_page` with every switch
   on, and on every tool when unset or on 0.16.0.
7. **Switch lists: hold.** Both new names are in `LATER_ENVS` only; `switches` clears them,
   `describe` names them, and the build record takes `describe()` (`deep/assembly.py` 560).
8. **Characters: supported.** My count (the JSON of each tool's call schema, description
   included): unset 43,870 on both; sheet tools +5,788 on 0.16.0 and +9,344 on the integration
   branch; drawing +1,732; DXF +433; all three and the plan tools +10,347 and +16,989; `sheets`
   adds 409 to a tool. The builder's figures are 1 to 2 percent lower throughout (another way
   of counting); 409 and 433 match exactly.
9. **Merge with `origin/claude/b13-integration` (`1826bdf`): three conflicts, one file.**
   `review_flags.py`: both sides add switch names after `LEAN_PROMPT_ENV`, entries to
   `LATER_ENVS`, and functions after `plan_tools()`. Keeping both sides (one comma added in
   `LATER_ENVS`) resolves it; in a scratch worktree the b14 tests and Document Review's
   fingerprints then pass on both planlens. Nothing was pushed.
10. **The geotech page and `minimal`: hold.** Only `review_build.py` passes the three arguments.

## 4. Tests

On the fixed head `6984f48`, each exit 0:

| Venv | planlens | Command | Result |
|---|---|---|---|
| deepagents 0.7.24 | 0.16.0 | `packages/aec-document-review`: `pytest -q tests` | 1179 passed, 18 skipped |
| deepagents 0.7.24 | integration | the same | 1194 passed, 3 skipped |
| deepagents 0.6.8, langchain 1.3.4 | 0.16.0 | the two b14 files, `test_review_fingerprints.py`, `test_boundary.py` | 18 passed, 15 skipped |
| deepagents 0.6.8 | integration (`PYTHONPATH`) | the same | 32 passed, 1 skipped |
| core with the app (venv b) | 0.16.0 | `packages/aec-agent-core/tests`, `webapp/tests/test_page_fingerprints.py`, `test_wheels.py`, `test_tinyapps_wiring.py`, `funhouse_agent/deep/tests/test_document_gate.py` | 716 passed, 5 skipped |
| core with the app | integration (`PYTHONPATH`) | the same | 716 passed, 5 skipped |

On the reviewed head `868c6a8`: the whole Document Review folder 1179 passed, 16 skipped
(0.16.0) and 1192 passed, 3 skipped (integration), as CHECK.md says for the preview. Guards
(`test_public_text.py`, `test_review_freeze.py`): 13 passed before each commit.

On the base (`69d6c2c`), the builder's test file and mine do not import (`document_tools` has
no `sheet_address_args`): collection errors, exit 2, so no test passes there, and none is
shown failing one by one. My two tests each fail on `868c6a8` and pass after their fix
(`tests/test_b14_review_fixes_offline.py`). The builder's tests check behaviour (results of
calls, schemas against planlens' own), not only text.

Fingerprints: not moved; neither file is touched by the branch or by the fixes.

## 5. Scope

Seven files on the reviewed branch: three core files, `review_build.py`, the package README,
one new test file, CHECK.md. No existing test edited, no frozen file, no notes file. The call
to make the two looks take the plot goes past "pass the option on" and is argued in CHECK.md
("Calls made"); without it the switch would point the model at a tool that raises. The fixes
add one test file and touch `document_tools.py` and `vision_kit.py`.

## 6. CHECK.md: untrue or unsupported

- "The planlens with the three parts is `claude/b14-pl-preview` at 99af95f": superseded;
  everything here ran on `a99192c`.
- `dxf_plots_available`: now also needs Pillow (finding 1).
- "kept per build by the file's digest": now the plot or the reason there is none (finding 2).
- "Gaps found in planlens": gaps 1, 2 and 3 are all still there on `a99192c` (findings 3, 4;
  no public list of host options).
- The character table: supported within 1 to 2 percent (section 3, item 8). The result sizes
  (`open_document` 1,584 to 4,758; +360 for a DXF) were not recounted.
- The runs in `/tmp/dr68` with the preview: repeated here for the b14 files, the fingerprints
  and the boundary test only, not the whole folder.

## 7. Could not check

- A model's use of any of this: no arm was run. Suite arms as CHECK.md names them.
- A real drawing's plot (legible lettering, a file near the bounds): the lead uploads a DXF on
  a host with `GEOTECH_REVIEW_DXF_PLOTS=1` and asks what it shows.
- Whether the testers' host has Pillow: `pip show pillow` in the wrapper's environment; the
  thirdparty requirements bring it here (12.3.0).
- The full eight jobs: push as `ci/<name>` after the merge with the integration branch.
