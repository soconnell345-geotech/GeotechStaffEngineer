# GM-295: the soil correlations against the printed references

Checked `geotech_common/soil_properties.py`, `geotech_common/alpha_method.py`,
the places `geotech_common/soil_profile.py` uses them, and the subsurface
tools `spt_correction` and `stress_dilatancy`, against the manuals in
`geotech-references/docs/`, as printed:

- FHWA NHI-06-088, Soils and Foundations Vol. I (Dec 2006), §3.7 and §4.1.1;
- FHWA NHI-06-089, Vol. II, §10.2;
- FHWA GEC-12 Vol. I (FHWA-NHI-16-009), §5.1.1 and §5.2.3;
- FHWA GEC-5 (FHWA-NHI-16-072, Nov 2016), §5.6.2 and §7.6.8;
- FHWA GEC-6, §5.3.4.1;
- UFC 3-220-10 (16 Jan 2026), §8-2.1.2 and §8-3.4;
- UFC 3-220-20 (16 Jan 2025), §6-5.4.2;
- EN 1997-2:2007, Annex F.

"Fixed" items changed the code and are pinned in
`geotech_common/tests/test_gm295_correlations.py` (and, for the tool,
`funhouse_agent/tests/test_subsurface_adapter.py::TestSptCorrectionChoices`).
No held manual has a worked example that uses these correlations, so each
pin is the printed equation or table at made-up inputs. "Noted" items were
not changed.

Not held, so not checked: Peck, Hanson & Thornburn (1974), Terzaghi & Peck
(1967) and Kulhawy & Mayne (1990) themselves (each is checked through the
held manual that reprints it, named below); Youd et al. (2001) Table 2;
EN ISO 22476-3; Bolton (1986); Hara et al.'s paper (checked through UFC
3-220-10 Fig 8-32).

## soil_properties

### C1. SPT to phi', Peck et al. (1974), the default (`spt_to_phi`). Fixed
- UFC 3-220-10 Fig 8-4, p. 399: the Peck et al. (1974) curve against N1,60,
  0 to 60. Eq 8-5 (Wolff 1989): phi' = 27.1 + 0.3 N1,60 − 0.00054 N1,60².
  UFC: "this correlation is conservative" (Ameratunga et al. 2016).
- Module: a hand-drawn piecewise line on N60 (26 deg at 0 to 42 deg at 50,
  then +0.1 deg a blow to a 50 deg cap). Its own comment said the slopes were
  "author-chosen" and not checked. It ran up to 2.4 deg above Eq 8-5 (at 30).
- Now Eq 8-5 on N1,60. Past the chart's N1,60 = 60 the angle is held at
  60's (43.2 deg) with a warning.
- GEC-12 Table 5-5 (p. 123, AASHTO 2014) gives ranges by (N1)60; Eq 8-5
  falls inside them at 4, 10, 30 and 50 (pinned).

### C2. SPT to phi', "meyerhof". Fixed
- GEC-5 Table 7-5, p. 7-54 (after Meyerhof 1956; also GEC-12 Table 5-6,
  p. 123, and UFC 3-220-10 Table 8-3, p. 396): phi' < 30, 30-35, 35-40,
  40-45 and > 45 deg for N < 4, 4-10, 10-30, 30-50 and > 50.
- Module: 25 + 0.3 N60, capped at 50 deg. No held manual prints that form.
- Now the band edges (30, 35, 40, 45 deg at N = 4, 10, 30, 50), linear
  between them (that reading is the function's). Below 4 and above 50 the
  table gives only a bound, so those are refused. The notes (UFC's under
  Table 8-3: 5 deg less for clayey sand, 5 more for gravelly sand, which
  GEC-12 p. 123 credits to GEC-5's 2002 edition; GEC-5 2016's: only the N'
  correction in saturated fine or silty sand) are in the docstring, not
  applied. Moves +3.8 to +7.0 deg over N 4 to 50 (table below).

### C3. SPT to phi', the other printed correlations. Added as options
- `hatanaka_uchida`: GEC-5 Eq 7.23, p. 7-55, sqrt(15.4 (N1)60) + 20 "for
  an average state-of-practice hammer with 60 percent efficiency" (also
  UFC Eq 8-6).
- `kulhawy_mayne`: GEC-5 Eq 7.22, p. 7-54, tan phi' = [N60 / (12.2 + 20.3
  σ'v/pa)]^0.34, pa = 100 kPa (GEC-5 p. 6-19); also GEC-12 Eq. 5-14 and UFC
  Eq 8-4. Needs σ'v.
- `kulhawy_chen`: GEC-12 Eq. 5-16, p. 124, 27.5 + 9.2 log (N1)60.
- `shioi_fukui_roads` and `shioi_fukui_buildings`: UFC 3-220-10 Eq 8-2 and
  8-3, p. 397, sqrt(15 N60) + 15 and 0.3 N60 + 27.

### C4. SPT to su, Terzaghi & Peck, the default (`spt_to_cu`). Matches
- NHI-06-088 Table 4-2, p. 4-5 (after Peck et al. 1974): qu 25, 50, 100,
  200 and 400 kPa at N60 = 2, 4, 8, 15 and 30. With su = qu/2 that is 6.25
  kPa a blow at 2, 4 and 8, and 6.67 at 15 and 30.
- Module: su = 6.25 N60 kPa. Matches at 2, 4 and 8; 6 % under the table at
  15 and 30. Basis restated; no value change.

### C5. SPT to su, Hara. Fixed
- UFC 3-220-10 Fig 8-32, p. 429: "su = 0.297 N^0.72 (Hara et al. 1974)",
  drawn through about 800 kPa at N = 100, so in kgf/cm².
- Module: 29 N^0.72 kPa, cited as Hara et al. (1971). Now 0.297 kgf/cm² =
  29.13 kPa (+0.4 %), cited as UFC prints it.

### C6. SPT to su, Stroud & Butler. Added as an option
- UFC 3-220-10 Eq 8-26, p. 428 (Carter & Bentley 2016, a fit to Stroud &
  Butler 1975, overconsolidated clays): su/N = 8910/PI³ + 4.36 kPa. Fig 8-31
  draws it for PI 15 to 62; outside that range a warning says so.

### C7. SPT to relative density (`spt_to_relative_density`). Fixed (a bug)
- EN 1997-2 Table F.1, p. 125 (normally consolidated natural sands): ID 0-15,
  15-35, 35-65, 65-85 and 85-100 % for (N1)60 0-3, 3-8, 8-25, 25-42 and
  42-58. Table F.2 (Skempton 1986, fine sands): (N1)60/ID² = 35 in lab
  tests, 40 in recent fills, 55 in natural deposits.
- Module: Dr = sqrt(N60 / (0.46 σ'v/pa)) × 100, called a simplified
  Kulhawy & Mayne (1990). The docstring's own figure was 46, not 0.46. With
  0.46 it returned 100 % for any N60 above about 0.5 at 100 kPa. No module
  or tool called it.
- Now `spt_to_relative_density(N1_60, method="en1997", age="natural")`.
  The default reads Table F.1's band edges, linear between them. Option
  "skempton" uses Table F.2. The `sigma_v` argument is gone, because both
  tables take (N1)60. NHI-06-088 Table 4-1, UFC Table 8-4 and GEC-5 Table 7-5
  print class tables close to Table F.1.

### C8. Ka, Kp, K0 (`phi_to_Ka`, `phi_to_Kp`, `phi_to_K0`). Matches; one option added
- NHI-06-089 Eq 10-4 and 10-5, p. 10-12: Rankine Ka and Kp. Eq 10-2,
  p. 10-10: Jaky's K0 = 1 − sin φ'. All three match.
- Eq 10-3, p. 10-11: K0 = (1 − sin φ') OCR^Ω, with Ω = sin φ'. Added as
  `phi_to_K0(phi, OCR=1)`.

### C9. Energy correction (`spt_energy_correction`). CE matches
- NHI-06-088 Eq 3-2, p. 3-55, and GEC-12 Eq. 5-1, p. 108: N60 = (ER/60) N.
  Matches.
- NHI p. 3-55 gives the typical ER: about 35-85 for cathead systems, about
  60 on average, and 80-100 for automatic trip hammers. Without a
  measurement AASHTO takes 60 for rope and cathead and 80 for automatic
  hammers. The module's "donut 45-60, safety 55-72, automatic 80-100" was
  not printed; the tool and docstring now use the NHI text.
- CB, CR and CS come from Youd et al. (2001) Table 2. Not held, so not
  checked. All three are 1 at their defaults.

### C10. Overburden correction (`spt_overburden_correction`). Form matches; Peck added; cap noted
- Liao & Whitman, C_N = (pa/σ'v)^0.5: GEC-5 Eq 5.10, p. 5-31; GEC-12 Eq. 5-4,
  p. 108 (n = 0.5 in sands); pa = 100 kPa (GEC-5 p. 6-19). The form matches.
- The 1.7 cap is Youd et al. (2001). Not held, so not checked. GEC-6 Figure
  5-18, p. 90, draws the same correction with pa = 95.76 kPa (1 tsf), to
  "2.0 max." Kept at 1.7; see owner question Q1.
- GEC-12 p. 108 calls Peck et al. (1974), Eq. 5-3, "the recommended
  correction in AASHTO specifications". NHI-06-088 Eq 3-3, p. 3-56, prints
  the same correction: C_N = 0.77 log10(20/p_o), p_o in tsf, C_N < 2.0. The
  module did not offer it. It is now `method="peck"`. C_N reaches zero at
  20 tsf (1915 kPa), so that stress and above are refused.
- `iso`, sqrt(98/σ'v) capped at 2.0, comes from EN ISO 22476-3. Not held,
  so not checked. EN 1997-2 §4.6.3(4) only points to it.

### C11. Bolton stress-dilatancy (`stress_dilatancy_bolton`). Not checked
- No held manual prints Bolton (1986). The docstring and the tool's brief
  now say so. The triaxial dilation angle, taken equal to the friction
  excess, is the function's own step.

## alpha_method

### A1. UFC 3-220-20 §6-5.4.2 and GEC-12 Figure 7-17. Matches, no change
- UFC pp. 415-417 match the module text: Table 6-16's su bands (800, 1500,
  3000, 6000 psf; su/Pa 0.38, 0.71, 1.42, 2.83, so Pa ≈ 2116 psf, the
  module's 101.325 kPa); α limited to 0-1; Eq 6-9, 0.28 (su/Pa)^−0.44;
  the soft-clay fs taken where it is higher; α = 1 in the sand/gravel
  transition zone; "judgment" for firm clay.
- GEC-12 p. 248 Figure 7-17 matches the module text. The curve nodes were
  measured off the figures in GM-121 and GM-229 (2026-10-18 and 20) and
  were not measured again here.

## soil_profile

### P1. phi from N in `fill_missing_from_correlations`. Fixed
- Module: `spt_to_phi(N60)`. The Peck curve is drawn on N1,60 (C1).
- Now N60 is first corrected to N1,60 at the layer's mid-depth: C_N by Liao
  & Whitman (the `spt_overburden_correction` default) at σ'v there. phi is
  estimated after every unit weight has been estimated. If σ'v cannot be
  worked out (a layer down to it has no unit weight), phi is left
  unestimated and the log says why. The cu estimate is unchanged (C4).
- `validate()`: the phi check uses N1,60 the same way, and skips a layer
  without σ'v. The cu check calls `spt_to_cu`, which is the same 6.25 N.

## Tool text (subsurface)

- `spt_correction`:
  - The brief now names Eq 3-2 and the three C_N choices.
  - `energy_ratio_pct` now gives the NHI p. 3-55 text (C9).
  - The rod, borehole and sampler descriptions now name Youd et al. (2001)
    Table 2.
  - `overburden_method` gains `peck`, and each choice is described with
    its cap.
  - The result gains `choices`: the energy ratio, and the overburden
    correction with its alternatives. The basis is GEC-12 p. 108, which
    names Peck as AASHTO's correction.
- `stress_dilatancy`: the brief says Bolton (1986) is not in the library.
- No calc steps use these functions.

## What moved

Made-up cases, before (master 1199f93) and after; script
`cases.py`, same inputs both sides.

| Case | Before | After |
|---|---|---|
| `spt_to_phi` default, N = 0 | 26.0 | 27.1 |
| N = 4 | 28.0 | 28.3 |
| N = 10 | 31.0 | 30.0 |
| N = 20 | 34.5 | 32.9 |
| N = 30 | 38.0 | 35.6 |
| N = 40 | 40.0 | 38.2 |
| N = 50 | 42.0 | 40.8 |
| N = 60 | 43.0 | 43.2 |
| N = 80 | 45.0 | 43.2 (held, warning) |
| `spt_to_phi` "meyerhof", N = 4 / 10 / 20 / 30 / 50 | 26.2 / 28.0 / 31.0 / 34.0 / 40.0 | 30.0 / 35.0 / 37.5 / 40.0 / 45.0 |
| "meyerhof", N = 0, 2, 60, 80 | 25.0, 25.6, 43.0, 49.0 | refused |
| `spt_to_cu` "hara", N60 = 10 | 152.19 kPa | 152.85 kPa (+0.4 % at every N) |
| `spt_to_relative_density`, N = 3 / 10 / 25 / 40 | 100 / 100 / 100 / 100 % | 15 / 38.5 / 65 / 82.6 % |
| `fill_missing_from_correlations`, loose sand 0-4 m, N60 8, σ'v 36 kPa | phi 30.0 | 31.0 (N1,60 13.3) |
| medium sand 4-10 m, N60 20, σ'v 86 kPa | phi 34.5 | 33.3 (N1,60 21.6) |
| dense sand 14-20 m, N60 40, σ'v 191 kPa | phi 40.0 | 35.3 (N1,60 29.0) |
| `spt_correction` tool at its defaults | N60, CN, (N1)60 | unchanged; adds `choices` |

The default `spt_to_phi` takes the same number in, but it now means N1,60,
not N60.

Unchanged:
- Ka, Kp, and K0 at OCR 1;
- the energy correction;
- Liao & Whitman and ISO C_N;
- `spt_n1_60` at its defaults;
- Bolton;
- the `spt_to_cu` default;
- the profile's cu, γ, γsat and Es estimates;
- `alpha_method`.

Through soil_profile, no module's results move in a live run.
`fill_missing_from_correlations` and `validate` are reached only from
`SoilProfileBuilder.build(fill_correlations=True)`, `from_table` and the
tests. No adapter or analysis module calls them. A library user who fills
a profile and then calls a `to_*_input` adapter would pass the new phi on.

Recorded calls, replayed: every `soil_properties` call made by the tests
that touch these functions was recorded before and after. The tests were
the soil_properties, soil_profile, tier-3 crosscheck, subsurface, phase 3-4
adapter, defaults_used, unused-input and top-level-param tests: 202 before,
204 after. Results:
- 60 calls give the same result.
- 3 moved, all direct `spt_to_phi` calls in
  `foundry_test_harness/test_tier3_crosscheck.py`:
  - default at 20: 34.5 to 32.9;
  - "peck" at 25: 36.3 to 34.3;
  - "meyerhof" at 25: 32.5 to 38.8.

  Each still falls inside the test's range.
- 11 calls now go in with other inputs:
  - soil_profile's fill and validate now pass N1,60, not N60;
  - defaults_used now tries `peck` as the overburden alternative, where it
    tried `iso` before.

## Noted, not fixed

- GEC-12 Eq. 5-15, p. 124, prints Hatanaka & Uchida as sqrt(20 (N1)60) + 20
  and does not name the hammer. GEC-5 Eq 7.23 gives 15.4 for a 60 % hammer.
  UFC Eq 8-6 prints 15.4, but says it was found "using an SPT hammer with an
  efficiency of 78%". The option follows GEC-5: (N1)60 at 60 %, factor 15.4.
- GEC-12 Eq. 5-19, p. 128, prints Stroud's su = f1 N60 pa/100, with f1 =
  4.5 at PI 50 and 5.5 at PI 15. UFC Eq 8-26 gives 4.4 and 7.0 there. The
  option follows Eq 8-26, the only form continuous in PI.
- NHI-06-089 App. A.6, p. A.6-9, takes N1,60 = 2 N60 at p_o = 500 psf,
  citing Eq 3-3. Eq 3-3 gives C_N = 1.47 there. A print slip, not pinned.
- soil_profile's other estimates are outside this item:
  - γ by N or cu;
  - Es = 600 (N60 + 6) and 250 or 400 cu;
  - Cc = 0.009 (LL − 10);
  - Cr = Cc/6.

  Held prints exist for them: GEC-5 Table 6-9 (Cc against LL), GEC-5 Table
  8-11 (Es against (N1)60) and GEC-12 Table 5-9 (γ). No live run uses them,
  so no backlog item is proposed.

## Owner questions

Q1. The overburden correction the `spt_correction` tool uses by default.
Which should it be?
- (a) Liao & Whitman capped at 1.7. This is today's default; the 1.7 is from
  Youd et al. (2001), which is not held.
- (b) Liao & Whitman capped at 2.0, as GEC-6 Fig 5-18 draws it.
- (c) Peck et al. (1974), capped at 2.0. GEC-12 (p. 108) and NHI-06-088
  (Eq 3-3) give it as AASHTO's correction.

How C_N compares:

| σ'v | Liao & Whitman | Peck |
|---|---|---|
| 50 kPa | 1.41 | 1.22 |
| 100 kPa | 1.00 | 0.99 |
| 200 kPa | 0.71 | 0.76 |

The cap matters only below 35 kPa (a) or 25 kPa (b). soil_profile follows
the same default.
