# slope_stability VALIDATION (branch le-modern)

Ours-vs-published validation of the modernized LE module (rigorous GLE /
Fredlund-Krahn engine, Janbu+f0, searches, probabilistic). All "ours"
values regenerated on this branch, 2026-06-11.

Runnable: `pytest slope_stability/tests/test_validation.py -v`
(the fem2d SRM cross-check is `-m slow`, ~6 min; everything else is fast).
The same benchmarks are also asserted piecewise in test_gle.py,
test_janbu.py, test_search_modern.py and test_probabilistic.py.

---

## B1 — Fredlund & Krahn (1977) homogeneous slope
[Slide2 Verification #21. Surface (0,60)-(60,60)-(140,20)-(180,20) ft;
c'=600 psf, phi'=20 deg, gamma=120 pcf; specified circle xc=120, yc=90,
R=80. 50 slices.]

### Case 1 — dry

| Method                    | F&K (1977) | Slide2 | Ours  | vs F&K |
|---------------------------|-----------:|-------:|------:|-------:|
| Ordinary (Fellenius)      | 1.928      | 1.931  | 1.926 | -0.1%  |
| Bishop simplified         | 2.080      | 2.079  | 2.075 | -0.2%  |
| Janbu simplified (w/ f0)  | 2.041      | —      | 2.017 | -1.2%  |
| Janbu simplified (no f0)  | —          | —      | 1.877 | (f0=1.075) |
| Spencer                   | 2.073      | 2.075  | 2.074 | +0.0%  |
| Morgenstern-Price (half-sine) | 2.076  | 2.075  | 2.076 | +0.0%  |

Ours M-P lambda = 0.331; Spencer theta = 14.7 deg. Note: F&K's published
"Janbu simplified" 2.041 includes the f0 correction (standard practice).

### Case 2 — ru = 0.25

| Method                    | F&K (1977) | Ours  | vs F&K |
|---------------------------|-----------:|------:|-------:|
| Ordinary (Fellenius)      | 1.607      | 1.606 | -0.1%  |
| Bishop simplified         | 1.766      | 1.758 | -0.5%  |
| Spencer                   | 1.761      | 1.763 | +0.1%  |
| Morgenstern-Price (half-sine) | 1.764  | 1.764 | +0.0%  |

## B2 — F&K (1977) weak-layer composite surface
[Slide2 Verification #22. Same surface; weak layer el. 15-16 ft (c'=0,
phi'=10); composite surface = same circle clipped at z=15, an 82-point
polyline; 60 slices asked for, 85 cut (a boundary at each corner, GM-131),
moment axis at (120, 90). Published sources disagree in the 2nd decimal
(weak-layer position sensitivity); gate 3% on F&K.]

| Case      | Method            | F&K   | Slide2 | Ours  | vs F&K |
|-----------|-------------------|------:|-------:|------:|-------:|
| dry       | Ordinary          | 1.288 | 1.300  | 1.313 | +1.9%  |
| dry       | Spencer           | 1.373 | 1.382  | 1.386 | +0.9%  |
| dry       | Morgenstern-Price | 1.370 | 1.372  | 1.381 | +0.8%  |
| ru = 0.25 | Spencer           | 1.118 | 1.124  | 1.138 | +1.8%  |
| ru = 0.25 | Morgenstern-Price | 1.118 | —      | 1.132 | +1.3%  |

Before GM-131 (60 even slices across the polyline's corners): 1.283 / 1.354 /
1.349 / 1.109 / 1.102. On 960 slices or more the surface reads Spencer 1.389
(dry) and 1.140 (ru 0.25) with either slicing, so the old values were the
slicing's, not the method's.

## B3 — ACADS 1(a) critical-circle search
[Giam & Donald 1989; Slide2 Verification #1. Surface (20,25)-(30,25)-
(50,35)-(70,35) m; c'=3 kPa, phi'=19.6 deg, gamma=20 kN/m3, dry.
Published answer 1.00. Ours: entry-exit search, 12x12 windows
(20-32 / 46-68), ~6000 trial circles, 30 slices.]

| Method            | Published | Slide2 | Ours (search min) |
|-------------------|----------:|-------:|------------------:|
| Bishop            | 1.00      | 0.987  | 0.986             |
| Spencer           | 1.00      | 0.986  | 0.985             |
| GLE (half-sine)   | 1.00      | 0.986  | 0.985             |
| Janbu corrected   | 1.00      | 0.990  | 0.989             |

## B4 — Duncan (2000) reliability anchor
[Duncan, "Factors of safety and reliability in geotechnical engineering",
JGGE 126(4). Lognormal reliability index for F_MLV = 1.50, COV_F = 0.17.]

| Quantity      | Duncan (2000) | Ours   |
|---------------|--------------:|-------:|
| beta_LN       | 2.32          | 2.318  |
| pf            | ~1%           | 1.02%  |

Cross-checks (test_probabilistic.py): undrained closed form COV_F ==
COV_cu exactly; Monte Carlo pf matches FOSM-lognormal pf within sampling
error (n = 4000, seeded).

## B5 — Duncan, Wright & Brandon textbook suite
test_duncan_verification.py: 19 tests, all passing on this branch
(includes the documented behavior change from P7: external pool water now
buttresses the slope — the Duncan Ex. 6 assertion was inverted to the
physical direction).

## B6 — LE vs fem2d Strength Reduction Method (Griffiths & Lane 1999 style)
[ACADS 1(a) geometry, fem2d.analyze_slope_srm at this branch state
(read-only import), 40x12 mesh, SRF bisection tol 0.01.]

| Engine                      | FOS   |
|-----------------------------|------:|
| LE search (Bishop/Spencer/GLE) | 0.985-0.986 |
| LE search (Janbu corrected) | 0.989 |
| fem2d SRM                   | 1.053 |
| Published (Giam & Donald)   | 1.00  |

LE-vs-FEM spread ~6-7%: consistent with the usual SRM-vs-LE comparisons
(Griffiths & Lane 1999 report agreement within a few percent; coarse
meshes bias SRM high). The cross-check test is `-m slow` (~6 min).

## B7 — Rapid drawdown (Corps 2-stage / Duncan-Wright-Wong 3-stage) + search
[`rapid_drawdown_fos` (single specified surface) and `search_rapid_drawdown`
(critical-surface search under drawdown strengths, v5.4 E1). Low-permeability
layers carry the R-envelope (`R_c`/`R_phi`); the FOS is a GLE/Spencer solve at the
drawn-down pool with each undrained slice's base overridden to `c = tau_ff, phi =
0`. Feet inputs converted to SI. Full theory in DESIGN.md.]

### Single specified surface (validation_examples V-037 / V-038)

| Problem | Method | Ours | Published | vs pub |
|---------|--------|-----:|----------:|-------:|
| #95 EM 1110-2-1902 dam (seepage stage-1) | Corps 2-stage | 1.34  | 1.347 | -0.5% |
| #95 (flat full-pool stage-1, conservative bound) | Corps 2-stage | 1.21 | — | — |
| #96 same dam (seepage stage-1), default stage-3 | DWW 3-stage | 1.273 | 1.443 | -12% |
| #96 same dam (seepage stage-1), `stage3_effective_normal='gle'` | DWW 3-stage | 1.370 | 1.443 | -5% |

**#96 3-stage refinement (v5.4 E2).** The default (Fellenius stage-3 effective
normal) is preserved byte-for-byte (flat 1.235 / seepage 1.273). The residual to
the published 1.443 was traced NOT to the Kc interpolation (which alone yields
~1.45) but to the stage-3 drained substitution: the Fellenius `W*cos(a)/l - u`
normal under-predicts N' and fires the substitution on 17/50 slices. The optional
`stage3_effective_normal='gle'` uses the rigorous GLE normal (consistent with
stage 1; 9 genuine substitutions) → 1.306 flat / 1.370 seepage, closing most of
the gap. Gated behind the new parameter; the lead decides whether to flip the
default. ~5% residual left, within the representative-flow-net + LE-N'-at-FOS
sensitivity band (cf. the +0.6% Corps residual on #95).

### Critical-surface search (validation_examples V-041, v5.4 E1)

`search_rapid_drawdown` reuses the ordinary search machinery (grid / entry-exit /
random / DE) with the drawdown FOS wired in per surface via a new `fos_fn` hook;
degenerate/non-converged trial surfaces are rejected (a stage-1+stage-3
convergence gate + a min-slice / stage-1-FOS floor kill the shallow near-flat
slivers a circular search would otherwise pick with a spurious low FOS).

| Problem | Method | Ours (search min) | Published (search min) | vs pub |
|---------|--------|------------------:|-----------------------:|-------:|
| #98 Walter Bouldin (approx geom) | Corps 2-stage | 0.837 | 0.931 | -10% |
| #98 (approx geom) | DWW 3-stage (default) | 0.938 | 1.039 | -10% |
| #98 (approx geom) | DWW 3-stage (`gle` stage-3) | 0.959 | 1.039 | -8% |

(Since GM-87, 2026-10-15, the drawdown search also refines its lowest grid centre;
#98 moved by up to 0.3 % further down, e.g. Corps 0.8370 -> 0.8341, B13. The pinned
values above hold.)

Wrapper MECHANICS are exact on the #95/#96 EXACT section (the circles' least 0.90
<= the published specified-circle single-surface FOS; the stage detail recomputed
on the winning surface reproduces the search FOS; since GM-131 a shallow planar
slide on the upper face governs at 0.783, against the infinite slope's 0.772, B15). #98's ~10% low minima are a geometry
effect — the cross-section is only RECOVERED (flat-stacked layers, real pinch-outs
simplified, riprap veneer omitted; INVENTORY #98), and two independent search
types (grid + entry-exit) both land ~0.85 Corps, with the correct ordering
(DWW > Corps). NOT tuned; the recovered geometry is left as-is.

## B8 — Pore-pressure GRID / TIN input (v5.4 E3, validation_examples V-029)
[Slide2 Verification #10 = ACADS 5: homogeneous 1:2 excavation, c'=11 kPa /
phi'=28 deg / gamma=20 kN/m3, pore pressure from a flow-net grid + ponded water.
Published Bishop 1.498 / referee 1.53.]

`SlopeGeometry.pore_pressure_points` (scattered (x,z,u) triples) interpolated at
each slice base (`build_pore_pressure_interpolator`: Delaunay-linear + nearest
fallback, suction clamped). Overrides the piezometric-line / ru base pressure;
ponded water still from `gwt_points`; wired through the search.

| Check | Result |
|-------|--------|
| Grid encoding a hydrostatic field vs `gwt_points` piezometric line | identical base u + FOS (Bishop/Spencer/GLE) to ~1e-15 |
| TIN linear interpolation of a linear field u=3+0.5x+2z | exact (err ~2e-15) |
| Outside convex hull / < 3 points / suction | finite nearest value, u clamped >= 0 |
| ACADS-5-style 1:2 cut + flow-net grid (reconstructed) | dry 1.89 -> grid 1.25 (Bishop) |

VERDICT: **CAPABILITY BUILT — flips V-029 from N/A-scope.** Previously the module
had only a piezometric surface + per-layer ru; the pore-pressure grid is the
missing capability. Validated by construction (hydrostatic-grid == piezo-line
equivalence, TIN exactness, search wiring). The published 1.498 is NOT pinned: the
manual's Figure 10.2 flow-net grid + excavation extent are a figure that did not
survive text extraction, so an exact referee match would require inventing them
(forbidden). The ACADS-5-style run is a capability DEMONSTRATION, not a match.
The funhouse slope adapter now takes `pore_pressure_points` too.

## B9 — Tension crack: exit-side + mass-truncation (v5.4 E4, validation_examples V-026 E4)
[Slide2 Verification #2 = ACADS 1(b): ACADS #1 slope, c'=32/phi'=10/gamma=20,
water-filled tension crack, Rankine depth zc=3.81 m. Published (WITH the water
crack): Bishop 1.596 / Spencer 1.592 / GLE 1.592; referee 1.65.]

`tension_crack_side` ('entry'|'exit') + `tension_crack_model` ('strength'|
'truncation'), both default-preserving.

| Case | Bishop | Spencer | GLE | vs Slide2 |
|------|-------:|--------:|----:|-----------|
| entry, strength (V-026 default) | 1.497 | 1.498 | 1.494 | -6% (conservative) |
| exit, strength (un-mirrored geom) | 1.497 | 1.498 | 1.494 | = entry (mirror-symmetric, ~1e-16) |
| entry, **truncation** | **1.597** | **1.593** | **1.594** | **<0.1%** |

VERDICT: **PASS — mass-truncation matches Slide2; exit-side is mirror-symmetric.**
The V-026 CONVENTION residual (the module was ~6% conservative because it kept the
cracked wedge as zero-strength DRIVING soil) is resolved: with the truncation
model — the cracked wedge REMOVED from the sliding mass, as Slide2 does — the
published water-crack FOS is reproduced to <0.1%. The exit-side option removes the
old need to mirror the slope (V-026 analysed the mirror image). Default (entry,
strength) byte-identical to V-026. NOT tuned; the match falls out of the mass model.
Since GM-134 (B16) the crack's edge is a slice boundary: strength 1.485 / 1.486 /
1.482 (-7%), truncation 1.603 / 1.599 / 1.600 (+0.4%), on 60 slices; the old
slicing reads the same on 3,840 (1.603 / 1.603 / 1.600 truncated), so the
<0.1% match above was in part a 60-slice value.

## B10 — Stabilizing-pile passive support convention (v5.4 E6, validation_examples V-040 E6)
[Slide2 Verification #54 (Yamagami 2000): homogeneous slope, single micro-pile row
(1 m spacing, 10.7 kN/pile). Published: no-pile 1.102, with-pile 1.193.]

`StabilizingPile.support_convention` = 'active' (default) | 'passive'.

| Convention | with-pile Bishop | vs published 1.193 |
|-----------|-----------------:|-------------------:|
| active (default, V-040) | 1.223 | +2.5% |
| **passive** (Method B)  | **1.203** | **+0.8%** |

VERDICT: **PASS — the passive option narrows the residual.** The default active
convention (pile force reduces the driving moment) over-credits the pile vs a passive
force. The opt-in passive convention (force added to the resisting side, Slide2 Method
B) drops the with-pile Bishop FOS to 1.203, +0.8% over the published 1.193 (vs active's
+2.5%). Only the circular moment methods honour it; the GLE applies the pile force as an
external equilibrium force. Default (active) byte-identical to V-040. NOT tuned.

## B11 — Multiple surcharge zones (v5.4 E8, validation_examples V-028 E8)
[Slide2 Verification #9 = ACADS 4: weak-seam noncircular slope + piezometric surface
+ TWO external loads (Table 9.2): a bench load ~20 kPa (x 23-43) and a crest load
~20->40 kPa (x 70-80). Published no-optimization Spencer 0.760 / GLE 0.720; referee
0.78.]

`SlopeGeometry.surcharges` = list of (pressure, x_start, x_end) zones, summed by
`surcharge_at` on top of the single surcharge. slices + search route through
`surcharge_at`, so no change there.

| V-028 pinned seam surface | Spencer | GLE |
|---------------------------|--------:|----:|
| no surcharge (V-028)      | 0.808   | 0.802 |
| + 2 zones (bench + crest) | 0.806   | 0.799 |

(Since GM-131, B15; before, on 50 even slices across the surface's corners:
0.792 / 0.786 and 0.788 / 0.782.)

VERDICT: **PASS — capability built, small seam-governed movement.** The two ACADS-4
zones lower the FOS by ~0.3% (Spencer 0.808 -> 0.806): the crest-side zone (x 70-80,
~11 slices on the sliding mass) adds driving weight; the bench zone (x 23-43) barely
overlaps the mass (~1 slice). The move is small because the weak seam (c'=0, phi'=10)
governs the FOS, as expected. Default (no `surcharges`) byte-identical to V-028. The
crest 20->40 kPa ramp is represented as a uniform 30 kPa (its mean). NOT tuned.

## B12 — Search: toe circles and the infinite slope (GM-13, GM-14, 2026-10-14)
[`search_critical_surface` now scores circles through each slope toe and, on any
face of c' = 0 soil, shallow planar surfaces parallel to the face, and reports
the lowest (DESIGN.md "Supplementary families"). Tests:
`tests/test_search_supplements.py`.]

| Case | Before (search alone) | Now | True / published | Governs |
|------|----------------------:|----:|-----------------:|---------|
| V-054 isotropic: H 5 m, 45 deg, su 40, Bishop, 40 slices | 2.784 | 2.286 | 2.287 brute force (FLAC 2.18, Taylor ~2.2) | toe circle (2.346 since GM-134, B16: 2.287 was a slicing error at the arc's vertical end) |
| Slide2 #86 without supports: c'=0, phi' 37, 1.25H:1V, Bishop search | 1.566 | 0.942 | tan 37 / tan 38.66 = 0.942 | infinite slope |
| ACADS 1(a) default circular grid, Bishop | 0.990 | 0.985 | 1.00 (Slide2 0.987) | toe circle |
| V-032 (Slide2 #61) entry/exit 10x10, Bishop | 1.320 | 1.315 | Slide2 1.366 / Baker 1.35 (see below) | toe circle |
| V-029 grid wiring check (6x6 grid, given windows) | 1.209 | 1.130 | (wiring test, 0.5-2.5) | toe circle |

The planar surfaces are real LE surfaces scored by Janbu's simplified method (force
equilibrium, as the infinite slope is). Against the closed form
F = tan phi' (cos b - kh sin b - ru/cos b)/(sin b + kh cos b):

| Face | Load | Planar (30 slices) | Closed form | Diff |
|------|------|-------------------:|------------:|-----:|
| #86, 1.25H:1V, phi' 37 | dry | 0.9430 | 0.9419 | +0.11 % |
| #86 | ru 0.2 | 0.6338 | 0.6330 | +0.13 % |
| #86, both face directions | kh 0.1 | 0.7712 | 0.7703 | +0.12 % |
| D&W Ex 3, 2H:1V, phi' 40 | dry | 1.6804 | 1.6782 | +0.13 % |
| D&W Ex 3, both directions | kh 0.15 | 1.1959 | 1.1941 | +0.15 % |
| D&W Ex 4 sand, 3H:1V, phi' 38 | dry | 2.3471 | 2.3439 | +0.14 % |

(The excess is Janbu's f0 and the end ramps, each a slice of its own since
GM-131, about 0.75 / (20 n) of the slide's weight at each end. Before GM-131
the surfaces were a tenth of a slice width deep with every slice base parallel
to the face: 0.9443 / 0.6346 / 0.7722 / 1.6816 / 1.1965 / 2.3474, +0.15 to
+0.25 %.)

### Reinforced searches (GM-53, GM-59, GM-60, 2026-10-14)
Searches now score each trial surface WITH its reinforcement (before, every
search returned the unreinforced critical). Since GM-59 each support acts at
the base of the slice it crosses in every equation the method satisfies
(Wright & Duncan 1991: Bishop counts its moment and its vertical component),
active (Method A, default) or passive (Method B) in every method; since
GM-60 Spencer / GLE solve a fixed-lambda state with support forces by the
direct march. See DESIGN.md "Support forces in each method".

| Case | Method | Search now | Published | Note |
|------|--------|-----------:|----------:|------|
| Slide2 #86, five 800 lb/ft tiebacks, windows entry 0-4 / exit 30-75 ft | Bishop | 1.612 | 1.629 (D&W 1.61) | -1.0 %; crosses all five |
| same | Spencer | 1.606 | 1.620 | -0.9 % |
| same | GLE | 1.606 | 1.622 | -1.0 % |
| Slide2 #85, one 9,000 lb/ft support, active, default ranges | GLE | 1.584 (GM-60 and the searches since; 1.581 before GM-60, on a state not in equilibrium) | 1.575 (D&W 1.51) | +0.6 % |
| same | Spencer | 1.542 (1.605 before GM-60); no solution on the same circle at 60 slices, which the search says (GM-127) | 1.884 | within the published spread 1.51-1.884 |
| same | Bishop | 1.304 + warnings (1.372 since GM-134, B16) | 1.531 | reverse-curvature toe circle (centre 6 ft below the crest, 5.9 ft dry crack, as Slide2's default closes it); the support passes 4.1 ft below the centre, 29 % counted (GM-59). Its last slice has m_alpha = cos(a) 0.18 on phi = 0, which Slide2 refuses with its 'Check malpha < 0.2' setting on (error -112); phi = 0 slices are not checked by default, as m_alpha does not change the FOS there (GM-127). 1.325 at 60 slices, which the search says |
| same, `reverse_curvature='discard'` | Bishop | 1.432 (1.503 before GM-122, 1.467 before GM-127) | 1.531 (D&W 1.51) | a circle passing above the toe centred at the crest's level, m_alpha 0.195 on its last (phi = 0) slice; 1.484 at 240 slices (-1.7 % on D&W) |
| same, discard, `m_alpha_rule='every_slice'` | Bishop | 1.467 | 1.531 (D&W 1.51) | Slide2's check on every slice: a circle passing above the toe, 1.508 at 240 slices |
| same, on Slide2's printed GLE circle | Bishop = Spencer = GLE | 1.564 | GLE 1.575 | Spencer / GLE fell back to the unreinforced 0.913 before GM-60 |
| Slide2 #85, passive (Method B) | Bishop, discard / default | 1.274 / 1.208 (discard 1.305 before GM-122, 1.296 before GM-127) | 1.324 (D&W 1.32) | discard 1.307 at 240 slices (-1.0 %); with `m_alpha_rule='every_slice'` 1.296, 1.318 at 240 |
| same | Spencer / GLE | 1.346 / 1.379 | 1.872 / 1.378 | GLE +0.1 % |
| ACADS 1(a) + 200 kN/m geosynthetic at el 29 | Bishop | 1.123 | (no reference) | unreinforced 0.985; the critical now avoids the geosynthetic |

V-043 (Slide2 #39 sand fill): the published reinforcement force is the force that
raises the UNREINFORCED critical surface to FS 1.35 (single surface; pinned in
test_published_v043_slope_reinforced.py). A reinforced search cannot reach 1.35
with any basal force: surfaces in the sand fill above the reinforcement govern
(entry/exit search 1.248 at 150 kN/m; infinite slope 1.259).

V-032: Slide2's printed critical circle for #61 (-1.665, 9.968, R 10.106) gives
Bishop 1.324 / Spencer 1.322 here and 1.3244 in an independent hand Bishop, 3.2 %
below its printed 1.366, so the search's 1.315 toe circle is a genuine lower minimum
for the stated inputs; the old <0.1 % PASS is on a different circle.

## B13 — Search: local refinement of the centre grid (GM-87, 2026-10-15)
[`search_critical_surface` and `search_rapid_drawdown` now refine the lowest centres
of the grid by a compass search on the centre (`refine=True`, the default;
`refine=False` gives the grid alone). DESIGN.md "Local refinement of the centre
grid". Tests: `tests/test_duncan_verification.py`, `tests/test_search_robustness.py`,
`tests/test_search_supplements.py`.]

| Case | Grid alone | Refined (default) | Reference |
|------|-----------:|------------------:|----------:|
| Duncan Ex 1 (phi = 0), D = 2, 4 H / 5 H level ground each side, Bishop | 0.739 / 0.755 | 0.703 / 0.703 | Taylor 0.706 |
| same, D = 3, 4 H / 5 H / 6 H | 0.690 / 0.696 / 0.703 | 0.676 | Taylor 0.679 |
| Duncan Ex 1 on its own short profile (1 H each side) | 0.727 | 0.726 + the profile-end warning (GM-63) | |
| Duncan Ex 4, sand over clay | 1.319 | 1.249 + the profile-end warning | |
| ACADS 1(a), default Bishop, grid family | 0.990 | 0.988 (the toe circle 0.985 still governs) | 1.00 |
| V-054 window-limited circle (toe circles off) | 2.784 | 2.770 (the toe circle 2.286 governs; 2.346 since GM-134, B16) | 2.287 |
| #98 rapid drawdown, Corps / DWW / LK | 0.8370 / 0.9352 / 0.9632 | 0.8341 / 0.9349 / 0.9619 | 0.931 / 1.039 / 1.075 |
| Pilarcitos true section, Corps | 0.7258 | 0.7236 | 0.823 |

Every number that moved went down; no published-benchmark test needed a change
(the V-041 / V-048 pins hold at their ±0.03). Spencer / M-P / GLE walk with Bishop
and re-score with the method itself; a reinforced model walks with the method itself
(Slide2 #85 anchored, Spencer: grid 1.581, refined 1.574).

## B14 — Search: ranges set from the slope (GM-57, 2026-10-16)
[Ranges the caller does not give are set from the slope's toe, crest, height H and
depth to the base d, not from the model's width (DESIGN.md "Automatic search
ranges"; `search_ranges.py`). Tests: `tests/test_search_ranges.py`,
`tests/test_slope_stability.py::TestPublishedBenchmarks`.]

The same slope drawn narrow and wide (Bishop, default search):

| Model | Before (windows from the model width) | Now | Reference |
|-------|-------------------------------------:|----:|----------:|
| c-phi 2:1 (c' 10, phi' 25), circle search alone, 2 H each side | 1.645 | 1.645 | toe circle 1.645 |
| same, 10 H each side | no admissible circle | 1.655 | |
| same, 0.5 H behind the crest / 6 H in front of the toe, and the reverse | 3.160 / 3.198 | 1.647 / 1.651 | |
| soft foundation (Taylor D = 3), 4 H or 12 H each side | 0.676 / 0.676 (+ a warning at 12 H) | 0.676 / 0.676 | Taylor 0.679 |
| same, 4 H one side and 12 H the other (both ways) | 0.782 (a toe circle) | 0.676 | Taylor 0.679 |

What moved in the suites (every published-benchmark test still passes):

| Case | Before | Now | Note |
|------|-------:|----:|------|
| Slide2 #85 one support, `reverse_curvature='discard'`, Bishop, active / passive | 1.503 / 1.305 (toe circle) | 1.485 / 1.302 | a circle tangent to the base entering just above the toe, out at x = 39.2, beyond the old exit window's start (40.2); at 240 slices it gives 1.511 / 1.318 against D&W 1.51 / 1.32, where the old toe circle gives 1.557 / 1.338 |
| ACADS 1(a) + 200 kN/m geosynthetic at el 29 | 1.115 | 1.105 | the circle above the geosynthetic, entering 0.41 of the way up the face |
| V-054 (steep phi = 0 from the toe), circle search alone, grid / refined | 2.784 / 2.770 | 2.360 / 2.344 | the toe circle 2.286 still governs (its centre is below the crest); since GM-134 (B16) refined 2.348 against the toe circle 2.346 |
| Duncan Ex 1 on its own short profile | 0.726 | 0.720 + the profile-end warning | Taylor 0.706 needs a longer profile |
| Slide2 #86 without supports, default search | 0.944 (planar) | 0.942 (a circle 0.4 ft long) | the closed form 0.942; the circle search now reaches the infinite slope itself and the result says that is what it is (the planar surfaces, Janbu, 0.944) |
| same, circle search alone | 1.550 | 0.942 | |
| Duncan Ex 2 / Ex 4 sand (c' = 0) | 1.6816 / 2.3474 (planar) | 1.6782 / 2.3442 (shallow circles) | closed forms 1.678 / 2.344 |
| calc package, c' = 0 sand slope, Spencer | 0.936 (planar) | 0.933 (a shallow circle) | said to be the infinite-slope mechanism; closed form 0.933 |
| V-041 #95/#96 drawdown wrapper | 0.902 | 0.895 | pin 0.902 +- 0.03 |
| c' = 0 shell under drawdown, families off, 4 x 4 | 1.487 | 0.944 | closed form 0.942 |
| WE-SLOPE-5 (Pilarcitos 2.5:1, given centre grid), Corps / LK / DWW | 0.787 / 1.353 / 1.349 (recorded) | 0.787 / 1.344 / 1.233 | LK: a 4 m circle on the face, said to be the infinite-slope mechanism; DWW: the drawdown planar slide governs (since GM-54) |
| V-042 entry/exit 10 x 10 | 1.325 | 1.332 | UP: the 10 x 10 points over wider windows (20 x 20: 1.326); published 1.334 / 1.340 |
| V-043 random noncircular, 2000 trials, seeds 1-5 | 1.172 / 1.208 / 1.199 / 1.217 / 1.247 | 1.194 / 1.270 / 1.183 / 1.195 / 1.149 | seed-dependent; the test now takes the median (1.194, published 1.188 / 1.192) and checks no draw goes below differential evolution (seed 1: 0.717 before, a spurious solver state, now refused (below); now 1.092) |
| 10 m 1:1 cut, 4 m weak soil over rock, default search | 5.25 | 4.82 | stays high: the critical circle (0.885, 4.6 m long in the weak layer) lies above the rock's outcrop, which the default windows do not reach; directed (entry 25.5-28, exit 29-40) 0.895 on the default grid, 0.889 on 20 x 20, 0.888 by entry / exit; `governing_location` names the layers the governing surface crosses (rock, weak) |
| Pilarcitos true section, drawdown, automatic windows: Corps / LK / DWW | 0.726 / 1.279 / 1.266 | 0.718 / 1.075 / 1.075 | LK / DWW: the rapid-drawdown infinite slope, a sliver of a circle about an inch deep on the lower face at (gamma'/gamma) tan(phi')/tan(beta) = 1.075, reported as that mechanism (the planar family's own drawdown solve fails on that face); not the observed deep slide, so the other V-052 tests give the deep slide's windows (the old automatic ones) and are unchanged |

### Rapid drawdown: the Fellenius normal under the remaining pool
The fully automatic DWW search of the true Pilarcitos section reported 0.052 on a
toe circle wholly under the remaining pool: the stage-3 Fellenius normal counted
the pond's pressure on a slice base in full and its weight only at cos^2(alpha),
clamped sigma' to zero on all 20 slices, and the drained substitution left no
strength (2.048 with the rigorous 'gle' normal). With the pond taken out of both
terms (a pond changes no effective stress; FHWA NHI-06-088 p. 6-13) that circle
gives 1.57 and the automatic search 1.074, on a small circle above the pool,
as Lowe-Karafiath. No published drawdown benchmark moved (V-037, V-038, V-041 #98,
V-048, V-052: same values); the default stays 'fellenius'.

### Rigorous noncircular solutions must hold
Searches now keep a new best noncircular surface only if its Spencer / GLE
solution holds about a second moment axis and with one slice more or twice as
many (DESIGN.md "Rigorous noncircular solutions must hold"). Three surfaces the
searches had reported as critical were solver artefacts: 13 m deep in sandstone
under a weak clay (0.90 at HEAD, 1.06 mid-GM-57; no Spencer solution by
`analyze_slope`, Fellenius 21.7), V-043's DE seed 1 (0.717; 1.55-1.90 at other
slice counts) and a surface diving at 75 deg into rock (0.731; no solution at 29,
31 or 60 slices). The lowest refused surface is kept in
`SearchResult.refused_lowest` and, when it is below the reported FOS or nothing
held, a warning says so with its FOS, where it lies and why. Tests:
`tests/test_search_robustness.py`. All three came from even slices across a
corner; sliced at its corners (GM-131, B15) none has such a state (two have no
solution; V-043's solves at 1.715 at any count), and the check stays as the
guard.

### Taylor's phi = 0 benchmark (`test_taylor_undrained_nc`)
The test compared a 1:1 (45 deg) slope, H = 8 m, cu = 50 kPa, gamma 18, with
1.92 = cu x 5.52 / (gamma H), and a fine grid read 2.039 (now 2.055 on the test's
own centre box, toe circle). Two things, neither a search gap:

- **The chart value.** 1/5.52 = 0.181 is Taylor's limit for an unlimited depth
  (FHWA NHI-06-088 Fig. 6-15, after Taylor 1948: the curves for beta < 53 deg
  rise to 0.181). This model's clay ends at a firm base 10 m below the toe,
  D = 18/8 = 2.25, where the 45 deg curve reads about 0.1785: F = 0.347 / 0.1785
  = 1.945, the circle tangent to the base and out about 2 H beyond the toe
  (n = 2). An independent optimisation of phi = 0 circles tangent to the base
  (200 slices) gives N = 0.1783, F = 1.947, out 1.96 H beyond the toe and in
  2.45 H behind the crest.
- **The model.** Its profile ended 0.6 H behind the crest and 1.5 H beyond the
  toe, so no circle it holds reaches Taylor's (the best: a toe circle closed by
  a crack, 1.970 by default; 2.04-2.06 on the old centre box). The profile-end
  check gives a circle past the end at 1.928.

The test now draws 4 H of level ground each side: the default search gives 1.935
at 30 slices (0.6 % below the converged 1.947, the slicing's own bias, on the safe
side, as Duncan Ex 1's 0.703 against 0.706), tangent to the base, out 1.8 H
beyond the toe, no warning. The old short profile is kept as a second test, which
must read high and warn. (phi = 0: Bishop, Spencer and the ordinary method give
the same F on a circle, so the method is not a factor.)

## B15 — Polyline slices cut at their corners (GM-131, 2026-10-17)
[Duncan, Wright & Brandon (2014) Ch. 6; Slide2 help: "Slice at piezo-surface
intersection", and its non-vertical slice method's "slice boundaries will be
located at each vertex of a failure surface". `slices.slice_boundaries`;
DESIGN.md "Where a noncircular surface is sliced". Tests:
`tests/test_slice_boundaries.py`.]

A noncircular surface now has a slice boundary at each of its corners, each
break in the ground and each crossing of a layer boundary or the water table;
the stretches between share out about `n_slices` slices in proportion to their
width, and each slice takes its base angle and length from its chord. Circles
were cut evenly until GM-134 (B16). Cut evenly, a slice across a corner took one
segment's angle for its whole weight, and a fixed polyline could err either
way with the slice count; a search's minimum picked out the low errors.

| Surface | Before | After |
|---|---:|---:|
| Taylor D = 2 profile, a random search's seed-3 polyline, Spencer, 29 / 30 / 31 / 45 slices | 0.807 / 0.650 / 1.182 / 0.862 (29-45: 84 % spread) | 0.970 / 0.971 / 0.971 / 0.973 (0.29 %) |
| Taylor D = 2, `noncircular_de` seed 1, Spencer (Taylor's circle 0.706) | 0.610 (-13.6 %) | 0.680 (-3.7 %; its surface 0.680 on 30 to 240 slices) |
| Taylor D = 2, `noncircular` seeds 1 / 2 / 3 | 0.668 / 0.683 / 0.650 | 0.720 / 0.761 / 0.760 |

Values with a published or recorded answer that moved (slope "not slow"
suite, the 21 validation files that use slope_stability, `funhouse_agent -k
slope`; every other value stayed within 1e-4):

| Case | Before | After | Published | Why |
|---|---:|---:|---:|---|
| V-028 Slide2 #9 / ACADS 4 seam surface, Spencer / GLE / Janbu | 0.792 / 0.786 / 0.804 | 0.808 / 0.802 / 0.819 | referee 0.78; Slide2 no-optimization Spencer 0.760 | old: 50 even slices (0.830 on 51; 0.806-0.815 on 100 to 1,600); now 0.808 on 30 to 1,600 |
| V-028 E8, both loads, Spencer / GLE (B11) | 0.788 / 0.782 | 0.806 / 0.799 | | the same |
| V-028 weak-layer search, 30 / 40 slices | 0.792 / 0.820 | 0.808 / 0.808 | referee 0.78 | the same surface at both counts now |
| V-028 `noncircular_de` seeds 1 / 2, 6 points, 30 slices, the seam windows (not pinned) | 0.715 / 0.738 | 0.715 / 0.714 | Slide2 optimized Spencer 0.707 | |
| V-030 Duncan (2000) Lash surface, Spencer / GLE / Janbu | 1.190 / 1.192 / 1.182 | 1.218 / 1.220 / 1.207 | Slide2 1.157 / 1.160 / 1.168; Duncan 1.17 | old slicing 1.226 at 240 slices, 1.215 at 960, 1.218 at 3,840; the +5.3 % on Slide2 is not slicing |
| B2 F&K weak layer, ordinary / Spencer / M-P / Spencer ru / M-P ru | 1.283 / 1.354 / 1.349 / 1.109 / 1.102 | 1.313 / 1.386 / 1.381 / 1.138 / 1.132 | F&K 1.288 / 1.373 / 1.370 / 1.118 / 1.118 | an 82-point polyline: 85 slices; 1.389 on 960 slices either way |
| V-043 sand embankment, `noncircular_de` seeds 1-3 | 1.09-1.10 | 1.234 | Slide2 1.188, Tandjiria 1.192 | old DE surfaces dived into the soft clay and read 1.24-1.32 with one slice more; a KNOWN GAP now (GM-117): 1.234 is above our converged critical circle (1.213) and a 41-point polyline inscribed in it (1.2145) |
| V-041 #95/#96 dam, Corps 2-stage search | 0.895 (circle) | 0.783 (planar slide) | infinite slope 0.772; specified circle 1.207 | the drawdown planar family solves stage 1 now (none did) |
| WE-SLOPE-5 Pilarcitos, DWW search | 1.233 (planar) | 1.344 (4 m circle, as LK) | ~1.05 for the observed deep slide; infinite slope 1.343 | the old planar slide read 1.23 to 2.52 on other slice counts |
| Infinite-slope family (B12) | +0.15 to +0.25 % | +0.11 to +0.15 % | closed form | a twentieth of a slice width deep, the end ramps slices of their own |

Searches without a published answer moved both ways in the unit tests: the
random, PSO and weak-layer-biased minima mostly rose (old minima were
slicing errors: e.g. PSO 1.703 -> 1.735, weak-layer-biased two-layer 0.957
-> 1.126, random 2.148 -> 2.527), others fell as the search took another
path (DE on a weak layer 1.060 -> 1.004; a cut-short DE 10.27 -> 5.45).
The three spurious rigorous states of "Rigorous noncircular solutions must
hold" (B14) no longer occur: two surfaces have no solution at any count and
V-043's solves at 1.715 at any count. The check stays as the guard.

The w11 judgment-call geometry (3.5 m clay over sandstone; F61): the circle
in the clay 0.9354 unchanged; `weak_layer` 7.523 -> 7.397; `noncircular`
seed 1 7.811 -> 7.740; `noncircular_de` seed 1 7.021 -> 6.678. The DE given
windows 20-30 / 28-60 (5 points, seed 3) reported 6.847; it now finds no
surface that holds and says so (2,325 evaluated: 398 refused for geometry,
1,774 not converged, 134 for m_alpha). The old surface reads 6.80 at any
slice count now; the population takes another path. Its depth fractions
reach the model base 30 m down, where about 1 surface in 300 has a solution
(GM-117).

---

## B16 — Circles cut at layer, water, surcharge and crack edges; chord bases (GM-134, 2026-10-17)
[Duncan, Wright & Brandon (2014) Ch. 6 (straight slice bases; the base
normal of a circle through its centre); Slide2 help: "Slice at piezo-surface
intersection", slice data ("Slide2 uses the mid-point of a slice base to
calculate the length of the moment arm"), error -116. `slices.slice_breaks`,
`slices.slice_boundaries`; DESIGN.md "Where a circle is sliced". Tests:
`tests/test_circle_slicing.py`.]

A circle now has a slice boundary at each break in the ground and each
crossing of a layer boundary, the water table or a tension crack's base, and
every surface (circle or polyline) one at each surcharge edge; each slice,
circle or polyline, takes its base angle and length from the chord between
its boundaries. Cut evenly, a circle's slice across a crossing took one
material, pore pressure or load for its whole base, so a fixed circle's FOS
jumped with the slice count. The tangent at each slice's midpoint (the old
circle rule) leaves the last slice's base short of the arc where the arc
turns vertical, and converges only as one over the square root of the count.

Every move was checked against the OLD code at fine slicing (a worktree at
972c937): of 69 distinct surfaces whose FOS moved in the slope suite and
the validation examples, 66 moved toward the old code's own value on 3,840
slices (error now median 0.09 %, at most 0.90 %; before median 1.76 %, at
most 26.8 %); the other three are two surfaces, below.

| Case (fixed surface unless "search") | Before | After | Old code, fine slicing | Published |
|---|---:|---:|---:|---:|
| V-043 sand fill, critical circle, Spencer, 30 slices | 1.180 (1.190-1.246 on 25-31) | 1.210 (1.208-1.211 on 25-45) | 1.213 on 960 and 3,840 | Slide2 1.209, Tandjiria 1.219 |
| V-043 geosynthetic force to raise that circle to 1.35 | ~55 kN/m | 40.9 kN/m (39.8 on 240 slices) | | 44-45 (circular) |
| V-043, T = 150 kN/m | 1.773 | 1.902 | 1.909 | |
| V-031 Slide2 #57 (layers + water), Bishop / Spencer / GLE, 50 slices | 1.411 / 1.415 / 1.398 (1.399-1.446 on 25-45) | 1.421 / 1.425 / 1.409 (1.420-1.425) | Bishop 1.419, Spencer 1.425 on 480 | Slide2 1.417 / 1.422 |
| V-031 Bishop search, 30 slices | 1.397 | 1.416 | | 1.42 |
| V-026 ACADS 1(b), water-filled crack (strength model), B / S / GLE, 60 | 1.497 / 1.498 / 1.494 (4 crack slices) | 1.485 / 1.486 / 1.482 (5) | 1.485 / 1.490 / 1.483 on 3,840 | Slide2 (truncation) 1.596 / 1.592 / 1.592 |
| V-026 dry crack, B / S / GLE | 1.553 / 1.555 / 1.551 | 1.540 / 1.543 / 1.538 | 1.540 / 1.543 / 1.539 | |
| V-026 E4 mass truncation, B / S / GLE | 1.597 / 1.593 / 1.594 | 1.603 / 1.599 / 1.600 | 1.603 / 1.603 / 1.600 | Slide2 1.596 / 1.592 / 1.592 |
| V-045 #34 correlated pair, FOSM mean (circle), 30 | 3.445 | 3.482 | 3.484 on 3,840 | (capability check) |
| V-054 steep phi = 0 slope, toe circle (2.5, 4, 4.72), Bishop, 40 | 2.287 | 2.379 | 2.374 on 3,840, 2.378 on 10,240 | 2.16 (FLAC 2.18) |
| V-054 the same with level ground past the toe, base circle (new pin) | | 2.222 | | 2.16 (FLAC 2.18) |
| V-054 Bakklandet ratios on the toe circle / anisotropic critical | 1.850 / 1.796 | 1.942 / 1.886 | 1.937 / 1.881 on 3,840 | reduction ~24 % (now 18 % / 21 %) |
| GM-13 steep slope, default search, 40 slices | 2.287 (toe circle) | 2.346 (another toe circle; grid 2.348) | | |
| Slide2 #85 + support, Bishop on Slide2's GLE circle | 1.564 | 1.566 | | GLE 1.575 |
| #85 + support, Bishop search, default (toe circle closed by a crack) | 1.304 (1.325 on 60) | 1.372 (1.373 on 60) | 1.362 on 960, 1.368 on 3,840 (new 1.374) | Slide2 1.531, D&W 1.51 |
| #85, `reverse_curvature='discard'`, active, default / `'every_slice'` | 1.432 / 1.467 | 1.508 / 1.511 | 1.503-1.513 | D&W 1.51 |
| #85, passive, Bishop search default / discard / discard + every_slice | 1.208 / 1.274 / 1.296 | 1.254 / 1.316 / 1.316 | 1.318-1.320 (discard) | D&W 1.32, Slide2 1.324 |
| #85, Spencer / GLE default search | 1.542 / 1.584 | 1.551 / 1.573 | | Slide2 1.884 / 1.575 |
| #85 clay face without the support, Bishop search | 0.882 | 0.910 | | |
| Vertical cut (Taylor), Bishop search | 0.800 | 0.842 | 0.839 (new 0.843) | Taylor 0.851 |
| Duncan Ex 4, Bishop / Fellenius / m_alpha-test circle | 1.249 / 1.260 / 1.259 | 1.271 / 1.250 / 1.288 | 1.270 / 1.253 / 1.287 | |
| Duncan Ex 4, Janbu search | 1.199 | 1.203 (1.212 on 60) | | |
| Two-layer clay over sandstone (w10), circles in the sandstone | 6.31-6.35 | 6.44-6.47 | 6.44-6.47 on 3,840 | |
| Weak layer over rock, four circles | 0.884 / 0.885 / 0.895 / 0.889 | 1.051 / 1.157 / 1.124 / 1.150 | 1.051 / 1.159 / 1.122 / 1.150 | |
| Entry/exit circle passing above the toe; reversed-window direct search | 0.968; 1.003 | 1.322; 1.282 | 1.322; 1.284 | |
| Circles passing above the toe (clay over sandstone), Bishop; Janbu / Spencer | 0.936; 0.938 / 0.936 | 0.983; 0.990 / 0.997 | 0.982; 0.995 / 0.996 | |
| Tension-crack Bishop cases | 1.944 / 1.916 | 1.961 / 1.933 | 1.9625 / 1.9343 | |
| SHANSEP GLE; ten layers; ru multi-layer; calc-steps figure | 1.684; 2.707; 1.457; 1.454 | 1.694; 2.720; 1.445; 1.462 | 1.695; 2.719; 1.445; 1.462 | |
| WE-SLOPE-4 (V-043 worked example): unreinforced; required T | 1.180; 55.5 kN/m | 1.210; 40.9 kN/m | | 1.209; 44-45 |

Two moves went slightly the other way, both inside 0.5 %: a small circle
in reversed search windows 1.199 -> 1.208 against 1.202 converged (the
chord is 0.47 % above at 30 slices, the tangent was 0.27 % below), and a
Janbu surface 1.224 -> 1.203 against 1.214 (0.9 % below against 0.8 %
above). A circle under the deepest layer is now refused at every slice
count (a slice boundary falls where it crosses the base); before, one 8 mm
under it lost two toe slices at 30 slices and scored, and was refused at 90.
On a phi = 0 circle Spencer and the GLE now equal Bishop to about 0.1 %
rather than exactly: the chord's normal passes a hair off the centre and
`gle.py` takes its arm as it is. Search times (default search, laptop, best
of 2-4): Bishop 1.8-2.7 s before and after (-9 % to +14 %, the scatter);
Spencer -15 % (V-031) and -5 % (V-043), but +22 % on the steep phi = 0
slope and +25 % on #85 (17 -> 21 s; GLE 16 -> 20 s): 12 % more rigorous
solves in the centre refinement and about 9 % more solver work each. The
slicing itself costs 0.03-0.05 ms more per surface (0.17 s of a 47 s
profiled search).

---

## B17 — The refinement walk moves on the method's own FOS (GM-138, 2026-10-18)
[DESIGN.md "The refinement walk moves on the method's own FOS". Tests:
`validation_examples/test_published_v043_slope_reinforced.py`,
`validation_examples/test_published_v031_slope.py`.]

Spencer / Morgenstern-Price / GLE searches refine their lowest grid centres
by a walk on which Bishop picks each centre's circle and the method scores
it; before, the walk moved on Bishop's FOS. Default searches (automatic
ranges, refinement and supplementary families on) on the published circular
cases, before and after. "Best known" is an independent brute force: a
20 x 20 grid of centres over the whole profile with 40 radii at each, then
Nelder-Mead in (xc, yc, R) from the five best minima and from both searches'
circles, scored with the same per-surface checks and limited to circles with
no reverse curvature; on phi = 0 slopes Spencer's is taken on Bishop's best
circle (the brute force's Spencer found only the legacy solver's spurious
states there). 480 = that circle on 480 slices.

| Case (slices) | Method | Before | After | Best known (480) | Published |
|---|---|---:|---:|---:|---:|
| V-043 sand fill, Slide2 #39 (30) | Bishop | 1.2182 | 1.2182 | 1.2180 (1.2172) | |
| | Spencer | 1.2375 | 1.1859 | 1.1858 (1.1901) | Slide2 1.209 (its own circle, close to the pinned one), Tandjiria 1.219; noncircular 1.188 |
| | M-P / GLE | 1.2464 | 1.1958 | | |
| V-031 Slide2 #57 (30) | Bishop | 1.4138 | 1.4138 | 1.4134 (1.4136) | 1.417 |
| | Spencer | 1.4177 | 1.4174 | 1.4173 (1.4200) | 1.422 |
| ACADS 1(a) (30) | Bishop | 0.9851 toe circle | 0.9851 | 0.9850 (0.9850) | 1.00 (Slide2 0.987) |
| | Spencer / GLE | 0.9840 / 0.9840 toe circle | same | 0.9839 / 0.9838 | |
| Slide2 #85 + support (30) | Bishop | 1.3715 toe circle closed by a crack | same | 1.5509 (1.5530) | 1.531, D&W 1.51 |
| | Spencer | 1.5505 | same | 1.5493 (1.5527) | 1.884 |
| | GLE | 1.5726 toe circle | same | 1.5843 (1.5865) | 1.575 |
| Steep phi = 0 slope from its toe, GM-13 / V-054's face (40) | Bishop / Spencer | 2.3464 / 2.3466 toe circle | same | 2.3464 / 2.3466 (2.3481) | |
| V-054 with level ground past the toe (40) | Bishop / Spencer | 2.2222 / 2.2212 | same | 2.2222 / 2.2210 (2.2262) | 2.16 (FLAC 2.18) |
| Duncan Ex 1, 4 H each side (30) | Bishop / Spencer | 0.7049 / 0.7044 | same | 0.7047 / 0.7041 (0.7065) | Taylor 0.706 |
| V-042 Hassan & Wolff (30) | Bishop | 1.3334 | 1.3334 | 1.3333 (1.3335) | 1.334 (Slide2 1.340) |
| | Spencer | 1.3464 | 1.3461 | 1.3454 (1.3467) | |
| Duncan Ex 4 sand over clay (30) | Bishop | 1.2709 | 1.2709 | 1.2704 (1.2691) | |
| | Spencer | 1.2586 | 1.2584 | 1.2584 (1.2634) | |

Every move is down. V-043's critical circle is tangent to the soft clay's
base and centred over the face's mid-length, (14.98, 9.76) R 9.76; the circle
the V-043 tests pin (1.210, from an old entry/exit search) is not it. It
reads 1.186 on 30 slices (1.190 on 480), 1.6 % below Slide2's circular
1.209, because Slide2's circular search stopped short of a circle tangent to
the model's base. The published values pair up by surface: Slide2's circular
1.209 and 44-45 kN/m match the pinned circle (1.2096 on 30 slices, 41 kN/m);
its noncircular 1.188 and 56 kN/m match the critical circle (1.190 converged,
57 kN/m). A dry crack cannot explain the lower value: truncated by a dry
crack 1, 2 or 3 m deep (the mass ending at the crack face, as Slide2 models
it), the critical circle reads 1.189 / 1.198 / 1.221 on 30 slices, higher.
The geosynthetic force to raise the critical circle to 1.35 is 57 kN/m (55
on 240 slices); it matches the published noncircular 56 kN/m (the same
slide); the published 44-45 belongs to Slide2's own circle, close to the
pinned one (41 kN/m). On #85 the
brute force without the no-reverse-curvature limit finds Bishop 0.922 on a
circle passing above the toe closed by a crack at the crest; the circles
passing above the toe leave such circles out by design (GM-122, GM-127).
Nothing pinned in the slope suite, the slope validation examples or
`funhouse_agent -k slope` moved.

---

## Notes / known deviations

- Ordinary method with ru: formulation-sensitive (F&K 1.607 vs Slide2
  1.687 on B1 case 2); ours matches F&K's formulation.
- B2 published values themselves disagree at the 1-2% level (F&K vs
  Slide2 vs Zhu); ours sits 0.8-1.9% above F&K and 0.3-1.3% above Slide2
  since GM-131 (1-2% below F&K before, on even slices across the composite
  polyline's corners) — within the documented gate.
- Fully-submerged slope: Bishop matches the buoyant-equivalent analysis
  to 0.1%, Spencer/M-P to ~0.02%; OMS has the textbook N' = W cos(a) - u l
  pathology (conservative), documented in test_water.py.
- Pre-existing bugs fixed on this branch that affect validation:
  SS-4 (m_alpha direction error: Bishop overestimated FOS ~18% on
  crest-on-the-left geometries) and SS-5 (build_slices silently dropped
  below-base fragments, letting absurd FOS ~0.1 win searches).
- Rapid drawdown: IMPLEMENTED (v5.3 B2a + v5.4 E1/E2) — see section B7 above.
  The earlier "designed stub only" status is superseded: `rapid_drawdown_fos`
  (single surface) and `search_rapid_drawdown` (critical-surface search) are
  validated against Slide2 #95/#96/#98.
