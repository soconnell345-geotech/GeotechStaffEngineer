# traces: what the recorded runs show about the references helper — findings register

Filed by the lead from reviewer `traces`'s report messages (2026-11-10).
Source: every unique `activity.jsonl` under `module_work/live_smoke/runs/` and
the field sessions' private raw folders (1,675 files, 1,169 unique records); 41
unique consults read in full (39 `references`, 2 `reviewer`), 702 helper tool
calls; all 524 recorded library lookups run again on today's code. Supporting
files: `C:/dev/tmp/refs_review_2026-11-10/traces/` (consults.jsonl,
consults.md, rerun.jsonl, stats.json, notes.md; field-session consult text
withheld). Consult numbers (#n) are the reviewer's. Counts: **A 4, B 6, C 2.**
No client, project, place or person detail is recorded here.

## Counts

- Wrong or unsupported content reached the main agent in 16 consults: the library returned it in 11 (#6, 9, 12, 13, 14, 15, 17, 27, 28, 31, 38); the helper asserted it with no lookup behind it in 5 (#1, 4, 10, 21, 40).
- A failed, empty or never-made lookup came back as "not found" in 12 (#1, 3, 4, 5, 10, 16, 20, 21, 31, 38, 39, 40).
- The main agent then added or upgraded a claim in 5 flows (#1, 3, 6, 15, 18/19).
- Unit problems that changed a number: 0. Limits dropped by the helper: 0. Helper caveats dropped by the main agent: 4 (#1, 3, 6, 15).
- Clean or an honest "not held": 19. Never returned: 1 (#41).
- The 39 references consults ended: 28 on the forced last call (19 at budget 8, 3 at 10, 6 at 6), 7 after the low-budget warning, 4 freely.
- Chart reads: 32 calls, 10 failed, 22 succeeded; the code voter returned no value in 22 of 22 (20 `no_scale`, 2 `no_read_lines`). 7 of the 9 chart name errors resolve today.
- Results cut at the size cap: 11 (6 at 8,000 characters, 5 at 16,000).
- Empty `reference_search` results: 14, of which 11 were the `reference="GEC-10"` scope (REF-41, fixed).

Waves: w1-geotech-haiku and w1-flows-haiku (2026-10-08; Haiku 5.5, budget 8;
20 consults), w13b/w14 judgment (Sonnet 5.5, budget 10; 5), w15-model-split
(2026-10-20; Opus 5.5 or Sonnet 5.5, budget 6; 13), three field sessions (GPT
through the Prompter; 3).

## DM7 lookups in the records

No recorded run called any DM 7.1 lookup, nor any function `dm71` or `dm72`
named. 27 dm7 calls in all (26 in six consults, 1 by a calc helper), all UFC
3-220-20 chapters 4 and 5: bearing capacity factors (match Table 5-2 to its
printed digits), Meyerhof and Hansen shape and depth factors (not checked
against Tables 5-6, 5-8), Rankine Ka. 9 of the 27 were errors (3 method names
that do not exist, 1 no arguments, 4 wrong parameter names, 1 chart call on
"dm7"). No wrong dm7 value reached an answer in the records.

## A

- **traces-01** (live; = dm72-02) `dm7_2/chapter5.py:2747-2817` `figure_5_18a_Ncm_strip`: (0.3, 0.75) -> 3.4, print 2.67; (0.2, 1.0) -> 3.8, print 2.53 (+50 %, unsafe). The figure's box prints Nc,m,s = 1.5 H/B + 5.14 su2/su1, not more than 5.14 (`ufc_3_220_20_2025.pdf` PDF p. 348, printed 310). In consult #4 the vision read of the figure gave the printed formula's values; the helper never called the function.
- **traces-02** (live; = core-07) `_retrieval.py:610-693`: a reference module's own `search_sections` returns an empty list with no hint where the library holds the answer. 19 recorded in 12 consults (#4, 6, 7, 11, 12, 13, 14, 15, 17, 18, 19, 40); 13 still empty today; `reference_search` scoped to the same reference returns sections for all 13, the wanted section in the first three hits in 12 (gec9 A.3/A.2; micropile 5.9.2; gec5 7.4.8, 7.4.6; gec11 4.4.6.c, 4.4.9; gec10 11.5; gec6 5.3.3). Helpers then said "not found" or left the item unconfirmed; #40 wrote from memory.
- **traces-03** (live) A lookup that failed or was never made reaches the main agent as "not found" (12 consults). Examples: #1 no reference tool called (ls and grep on the empty scratch space), told the user nothing was available though GEC-5 5.6 held it; #3 "Table 5-2" search died on the hyphen, a method name that does not exist, a call with no argument; the main agent then published a "Terzaghi" table holding the Meyerhof and Brinch Hansen values (Nc 30.14, Nq 18.40 at 30 degrees; `terzaghi_Nq(30)` is 22.46); #16 two searches failed on "(" and "/", the listing was cut at 8,000 characters; #18/#19 the main agent filled a gap "from general practice" with the Reese, Cox and Koop (1975) criterion, which Appendix A.2 says is for stiff clay with free water; #31, #38, #39 ten searches scoped "GEC-10" returned empty (REF-41). Gone today: scratch tools hidden from the helper (`packages/aec-agent-core/aec_agent_core/deep/limits.py:86`); queries with hyphen, apostrophe, slash, bracket work; a wrong method name gets an alias or redirect; long listings shortened with a note. Still there: traces-02; the budget (10 calls, 6 for a judgment consult: 6 of 13 w15 consults were forced at 6); the last-call instruction (`limits.py:218-225`) says only "If something could not be confirmed, say so in one line rather than guessing"; nothing in the consult's return tells the main agent how many lookups failed. Fix proposed: code adds one line to every consult's return (lookups made, errored, empty, cut, ended on budget or not); tell the helper "not found" is only for a lookup that ran and returned nothing, and that it names it; tell the main agent to say "the lookup failed" and not to fill the gap from memory.
- **traces-04** (recorded; fixed in today's library) The library returned content the print does not say, and it reached the user in 10 of 11 consults. Ten came from chapter text that was a model-written summary under section numbers the manual does not have ("5.2-5.3", "7.1-7.4", "7.2.1.3.4-7.2.1.3.7"); one from a digitized chart. The helper relayed the tool result faithfully each time.
  1. #12 GEC-8 safety factor "2.0-2.5 without load tests" (print PDF p. 148, 176: 2.0 requires load testing; unsafe). Today the function defaults to 2.5.
  2. #17 GEC-5 "Nkt about 12-17", "S about 0.25-0.30, m 0.8-0.9" (print PDF p. 294: Nkt often 10 to 20; p. 297: S slightly less than 0.2 to less than 0.3, m 0.6 to 1.0). The library's text overrode the user's correct S = 0.22; strength about 25 % higher; unsafe.
  3. #14, #15 micropile "FS typically 2.0 to 2.5" (print PDF p. 114: 2.5 for creeping, high plasticity, weak rock or limited experience), and a table title that is not the print's.
  4. #27, #28 GEC-12 "beta = 0.2-0.3 for NC clay" (print Table 7-9, PDF p. 302: clay beta 0.15-0.35).
  5. #38 GEC-10 "side resistance factors of 0.45 without a load test and 0.65 with one" (print Table 8-4, PDF p. 222: rock 0.50 compression, 0.40 uplift; text p. 10-21 says 0.55).
  6. #13 GEC-4 anchor creep test without the 60-minute extension (print PDF p. 171).
  7. #6 GEC-10 alpha text under the 2018 title that was the 2010 edition's; the main agent wrote "verified ... High confidence" though the helper said it could not confirm the edition. Not print-checked by the reviewer.
  8. #31 GEC-10 "beta = 1.5 - 0.245 z^0.5" from the old section 10.3.5 (today 10.3.5.1 carries Eq. 10-9 to 10-12); the one consult where the wrong content was not repeated to the user.
  9. #9 UFC 3-250-01 Figure E-1: the digitized function returned 6.9 in at CBR 8, 500,000 passes; print by pixels 8.4 in (18 % low). Today 8.3 in (GM-298, GM-303).
  Today all nine return different content, matching the print where checked (rows 1 to 6 and 9). Why it still matters: traces-05.

## B

- **traces-05** (suspected, live) 20 references still serve text of the older form (no printed-page marker in any section body): dm7_1 (457 sections), dm7_2 (438 of 446), gec_11 (182 of 183), ufc_pavement (317), eurocode_7_1 (208), ufc_concrete_practice (424), ufc_flexible_practice (294), ufc_collapse (132), gsa_collapse (135), ufc_stabilization (101), ufc_structural (71), california_trenching (51), aashto_1993 (44), em_2104 (39), em_2107 (35), wood_handbook (37), fema_p2082 (28), fhwa_pavements (23), gec_14 (7), gec_10 chapters 2, 6, 8 (46 of 395). Every section has one in gec_4, 5, 6, 7, 8, 9, 12, 13, micropile, eurocode_7_2, fhwa_soils_foundations. A missing marker does not prove the text wrong; two spot-checks held (UFC 3-220-10 section 7-8 and Table 7-2, PDF p. 393; GEC-11 eccentricity and settlement limits, Vol 1 PDF p. 165, 173, 323). Order by use in the consults: dm7_2 chapters 4-6, dm7_1 chapters 4, 5, 7, gec_11 chapter 4.
- **traces-06** (live; = core-05) `_retrieval_db.py:594-606`: `reference_search` answers a query nothing matches with unrelated sections and the note "more match"; the description still says "Plain words are AND-matched". Related: a section's table titles are stored but not searched (`_retrieval_db.py:328`).
- **traces-07** (live) `funhouse_agent/adapters/_reference_common.py:109-116, :150-152`: a lookup's number comes back with no unit and no name; `describe_method` stops reading the docstring at "Returns". `gec12.figure_7_15_limiting_toe_resistance(phi=34)` -> {"result": 35.9} (tsf, per `gec_12/figures.py:248-251`); `dm7.meyerhof_shape_factors` described as "tuple" with the order only in the docstring. Of 1,339 reference methods described today: 423 float, 13 tuple, 26 list, 425 state no return. Recorded: #5 helper got the bare 18.40 and said "the printed Nq equation was not retrieved"; #22, #25 the helper assigned the three shape factors by assumption (right); #35, #36 neither arm used the Figure 7-15 function, both read the chart by eye.
- **traces-08** (live) A chart read is one model's reading. `funhouse_agent/reference_vision.py:264-285`: only a page located by estimate is asked to confirm the figure; the prompt does not say to report only what is printed. 32 calls: 10 failed (9 on a name that was not a catalog id, 7 of which resolve today; 1 asked for a table); 22 succeeded, all describing the captioned figure. **Code voter: no value in 22 of 22.** 20 `no_scale` (61 readings, each `{"code": null, "status": "no_scale", "note": "no fitted scale answers to 'x' on this page"}`), 2 `no_read_lines` (schematics). Known as VIS-10. The voter also parsed "x = s_u2/s_u1 = 0.30" as x = 2.0 (the digit in the symbol's name). Accuracy: within 10 % for GEC-12 Fig 7-15 at 34-36 degrees, NHI-06-089 Fig 9-14, UFC 3-220-20 Fig 4-13, Fig 5-5 Nc and Nq; outside for Fig 7-15 at 32 degrees (-13 %) and 30 degrees (5 tsf against 7.1, -30 %, reached the user in #35); Figure E-1 at CBR 3 high on every curve, up to +32 %; NHI-06-089 Fig 10-10 one read +47, +32, +21 % (caught by the helper on a second read); Fig 5-5 Terzaghi N-gamma +14 %. Stated tolerances ("about 10 %") are the vision model's own. Values not on the page: #3 "tabulated Terzaghi values" from the vision model's memory (N-gamma 5.0; `terzaghi_Ngamma(20)` is 4.41); #36 unit conversions done by the vision model. Fix proposed: every read confirms the figure and its axis units first; report only what is printed and mark anything else; tolerance from the chart's grid; where the library holds a digitized function for the figure, return its value beside the reading.
- **traces-09** (known; backstop is a prompt only) Field session 2026-09-15 (5.15.0), reviewer helper: all four lookups empty or in error, then a full review citing a wrong document about ten times. Today `funhouse_agent/reviewer.py:69-76` forbids citing what a tool did not return; `reviewer.py:91-92` still says "Limit yourself to 3-4 reference lookups maximum" (budget is 10, TA-25); no code compares a review's citations with the tool results. Second reviewer consult (#41) never returned (TA-25 runaway).
- **traces-10** (wave 1 only) The main agent (Haiku 5.5) dropped the helper's caveat in #1, 3, 6, 15. In w15 (Opus or Sonnet main agent) every traced caveat was carried to the user.

## C

- **traces-11** the chart tool gives `pdf_page_index` without saying it counts from 0 (one arm told the user "283 or 284").
- **traces-12** a document's own number ("UFC 3-220-20") is not accepted as a `reference_search` scope; loud (an error with every id as suggestion). The same class of name cost the chart tool 7 calls in the records ("UFC 3-250-01"; resolves today).

## Chart reads one by one (part 7; full lines in the reviewer's message and `traces/notes.md`)

22 reads returned a reading; code voter no value in any (20 no_scale, 2 no_read_lines).
Read against today's digitized function or the print:
- UFC 3-220-20 Fig 5-5 (Terzaghi factors): within 2 % except N-gamma at 20 degrees +14 %; also carried values from the vision model's memory.
- UFC 3-220-20 Fig 5-18: read matches the printed formulas within 1 % (the function is the wrong one: dm72-02).
- UFC 3-250-01 Fig E-26, E-28, E-2: within 0.5 in of the function. Fig E-1 at CBR 3, eight curves: +7, +12, +3, +8, +16, +32, +18, +12 % against the print by pixels.
- UFC 3-220-20 Fig 4-27, 4-53, 4-54: schematics or printed expressions, answered in words.
- NHI-06-089 Fig 9-18 (Nt at 38 degrees): 97, 101; no digitized value.
- Passive charts (UFC 3-220-20 Fig 4-13; NHI-06-089 Fig 10-10; Caltrans Fig 4-20), level ground: -6 % to +1 % in six reads; one read of Fig 10-10 +43 % (discarded by the helper after a second read).
- NHI-06-089 Fig 9-14 and GEC-12 Fig 7-15 (limiting toe resistance): 34 to 36 degrees within 8 %; 32 degrees -13 %; 30 degrees -30 %.
10 failed: 9 on a reference name that was not a catalog id ("UFC 3-250-01" x6, "gec10", "dm7", the manual's full title; 7 resolve today, "dm7" and a full title still error, loudly); 1 asked the figure tool for "Table 5-3" (REF-34).
Figure E-1 measurement: `ufc_3_250_01_2016.pdf` PDF p. 212 (printed 192) at 400 dpi; 0 and 30 in gridlines at pixel rows 3408 and 1034; the figure's own worked arrow (CBR 5, curve 6) reads 10.5 in, measured 10.6.

## The 41 consults (part 8; one line each in the reviewer's message and `traces/consults.md`)

Verdict codes: W library returned wrong content, relayed; H helper asserted without a lookup; N false "not found"; M main agent added or upgraded a claim; G honest "not held"; ok clean.
- Haiku 5.5 helper, 20 consults (wave 1): W 7, N 5, H 3; ok or G 7.
- Sonnet 5.5 helper, 12: W 1 (#28), N 1 (#39); ok or G 10.
- Opus 5.5 helper, 6: W 3 (#27, #31, #38), N 2 (#31, #38); ok 3.
- GPT, 3: N and H in two; one never returned.
- Every W on Opus and Sonnet was the library's text relayed word for word; their N cases were the silent empty scope (REF-41, fixed).
- By ending: of 28 forced consults 17 carry W, N or H; of 7 warned, 1; of 4 free, 2.

## Tool-level problems (part 9)

702 helper tool calls in the 41 consults: call_agent 524 (reference_search 192, reference_get 134, figure_search 22, the rest module calls), list_methods 47, describe_method 37, read_reference_figure 31, list_agents 25, grep 14, ls 8, glob 6, calculate 5, record_feedback 4, one invented tool name. Problems: 66 errors, 35 empty results, 11 cut results, 1 repeated call, 28 scratch-space calls. None of the 66 errors returned a value; each was visible to the helper.
- Search query syntax, 14 (hyphen, apostrophe, bracket, slash): all fixed today.
- **A lookup called with no arguments, 15**: the helper called a table to see what it holds (`table_5_3_alpha_bond()`, `table_a1_k_stiff_clay()`, ...). Still an error today, now naming the parameters. In #18, #19 the helper then said it "could not read the values".
- Unknown method, 12 (e.g. `dm7.bearing_capacity_factors`, `dm7.search_sections`, `dm7.rankine_active_coefficient`, `ufc_pavement.table_7_2_minimum_thickness`, a category name used as a method). Today most get an alias, a redirect or the right method's description.
- Chart tool "Figure not found", 9 (part 7).
- Malformed calls, 6 (module name `reference_get` with no method; two "placeholder" names on the last turns of a 50-call consult).
- Wrong parameter name, 5 (`phi`, `friction_angle` for `phi_deg`; today "Did you mean" for `phi` only).
- A key not in the table, 3 (raised with the valid keys, as it should).
- `reference_query` "SQL error: no such column: id", 1 (same today). A guessed section id, 1.
- Cut at the size cap, 11: `list_methods("dm7")` cut at 8,000 before the bearing capacity and earth pressure functions (#3, #5, #16); `list_methods("ufc_pavement")`; two `describe_method` replies carrying the whole list; five text results at 16,000. Today listings are shortened with a note naming the categories left out (the dm7 listing is 7,742 characters and leaves out four categories, among them "DM7.2 Ch5 - Shallow Foundations", so those functions take a second call to list) and text comes back as fitted hits (REF-26).
- Scratch-space calls (ls, grep, glob on the empty scratch space), 28 in 17 consults; hidden from the helper today.
- 109 of 702 calls went on finding out what exists (list_agents 25, list_methods 47, describe_method 37); in w15 the helpers made 0 list_agents calls.

## Empty results and the re-run (part 10)

35 empty results: 19 from a module's `search_sections`, 14 from `reference_search`, 2 from `list_methods` (an invented category; today a note and the module's methods).
- Of the 33 empty searches: in 26 the content was in the library at the time under other words or another scope name; in 5 it was not there then and is today; 2 unknown. On today's code 19 return hits as written and 14 return nothing; for all 14 the content is reachable another way.
- New silent empties on today's code from the same recorded calls: `gec5.search_sections("Nkt cone factor undrained shear strength")` found the section in wave 1 and is empty today (the rebuilt text no longer packs every keyword into one section, so the all-words rule now misses; `reference_search` with the full query puts 7.4.8 first); `micropile.search(...)` and `gec9.text_search(...)` were "Unknown method" errors then and are accepted as the module search today, returning an empty list (a loud error became a silent empty). So the recorded queries give **16 silent empties from module search today**.
- The quoted exact title "Factors of Safety for New Earth and Rockfill Dams" returns nothing though dm7_1 section 7-8 lists that table (table titles are not indexed).

Re-run of all 524 recorded library lookups: 369 changed (nearly all the new result shape and the rebuilt text), 63 the same, 32 an error then and now, 19 empty then and hits now, 17 an error then and fine now, 14 empty then and now, 7 fine then and an error now (five are old summary section ids that no longer exist, each now a KeyError naming the nearest printed section), 2 an error then and empty now, 1 hits then and empty now.

Value lookups only (tables, equations, digitized charts; 82 calls): 45 the same value, 27 errors then and now, 10 changed, eight of them in shape only. Two changed their number: `ufc_pavement.figure_e1_flexible_thickness` (6.9 to 8.3 in at CBR 8, 500,000 passes; 0.3 to 1.1 in at CBR 30). **Apart from Figure E-1, no value a consult received from a lookup function has changed since it was recorded; the wrong content in the records came from the chapter text, not from the functions.**

Listing calls by wave (list_agents, list_methods, describe_method): wave 1 83 of 334 calls; w13b and w14 23 of 179; w15 none of 179.

## Documents and topics asked for and not found (part 11)

Not held today: USACE EM 1110-2-1902 Slope Stability (1 consult; UFC 3-220-10 Table 7-2 quotes four of its values; the app's slope work cites it, OWN-102); liquefaction triggering from SPT (Youd et al. 2001) and a pseudo-static slope factor of safety (2 consults; GEC-3 would cover both, REF-35); ASCE 7 site class and seismic parameters (1; PDF in docs/, untracked, not public); AASHTO LRFD articles by number (1 directly; not public); downdrag methods by name (Briaud and Tucker 1997, CGPR #56, Endo, Poulos, PILENEG; 2 consults; NCHRP Report 393 is public); a source for the 25 mm tolerable settlement (2); finite element modelling guidance (boundary distances, dilation angle; 1); GEC-10 2010 edition and the 1997 micropile manual (superseded); "NHI-05-054" (number unconfirmed); the printed DM 7.2 tables as tables (3; REF-34).
Held today, not found at the time: Ito and Matsui (micropile Eq. 6-12); GEC-9 Appendix A p-y criteria; NHI-06-089 and a cut-slope factor of safety; GEC-12 7.2.10 downdrag; GEC-10 2018 beta method and rock socket text.
No reference states it: a rule for rounding pile length; the buoyant unit weight step in the beta method; K0 behind sloping ground in the US manuals (EN 1997-1 and Caltrans give one and differ in sign).
By frequency the two additions the records argue for are EM 1110-2-1902 and GEC-3.

## Patterns

1. Summary text standing in for the manual: 10 of the 11 wrong-content consults; fixed where rebuilt (7 references); 20 references still in the older form (traces-05).
2. "Not found" that means "the lookup failed": 12 consults. Causes then, by calls: scratch-space tools 28, module search returning nothing 19, query syntax 14, a scope name not resolved 11, a listing cut at the cap 6, and the budget. Not fixed: the module search (traces-02); no count of failed lookups in the consult's return (traces-03).
3. The budget ends most consults: 28 of 39 forced; 17 of those 28 carry a wrong or unsupported statement or a false "not found".
4. Bare numbers (traces-07). 5. Chart reads are one voter (traces-08).
6. The better models relay faithfully: in w15 the wrong statements were the library's, word for word.
7. Value lookups were stable (apart from Figure E-1).

## Right, though it looked wrong

UFC 3-220-20 Table 5-2 functions match the printed digits (misprints are REF-21); the vision read of Figure 5-18 was right and the function wrong; UFC 3-220-10 section 7-8 and GEC-11 chapters 4 and 7 statements checked match the print; GEC-10 rock side 0.50 follows OWN-81; unit handling right in every conversion traced; the 23 calls with no arguments, a wrong parameter name or an unknown key all raised and none returned a default; no chart read landed on a wrong page (22 of 22); old summary section ids now fail naming the nearest printed section.

Not reached: "clean" means every number was in a tool result and returns the same today, not that each was print-checked; chart reads compared with today's digitized functions (only Figure E-1 measured off the print); run outputs (PDF, Word) not opened; dm7 shape and depth factors not checked against Tables 5-6 and 5-8; consult #6's alpha relation not checked against the 2018 print.
