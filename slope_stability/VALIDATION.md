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
+ TWO external loads (Table 9.2): a bench load 20 kPa (x 23-43) and a crest load
20->40 kPa (x 70-80). Printed, no optimisation (Table 9.4, manual p. 52): Spencer
0.760 / GLE 0.720 / Janbu corrected 0.734; referee 0.78.]

`SlopeGeometry.surcharges` = list of (pressure, x_start, x_end) zones, summed by
`surcharge_at` on top of the single surcharge. slices + search route through
`surcharge_at`, so no change there.

**Validates on surfaces along the seam (GM-140 step 1 fix round, 2026-11-13).**
A family of 81 three-part surfaces is declared in
`validation_examples/test_published_v028_e8.py`: the seam's mid-line reached at
x = 45, 47, 49 by a wedge from the bench side at 25, 30, 35 degrees, and left at
x = 72, 76, 80 by a wedge up to the crest at 50, 57.5, 65 degrees; 30 slices; no
search routine of the module. With Table 9.2's loads on (the crest's 20->40 kPa
as its mean, 30 kPa), all 81 solve by all three methods:

| Method | Lowest of the family, loads on | Printed, Table 9.4 (no optimisation) | Difference |
|---|---:|---:|---:|
| Spencer | 0.753 | 0.760 | -0.9 % |
| Morgenstern-Price, half-sine | 0.722 | 0.720 (GLE) | +0.3 % |
| Janbu, corrected | 0.728 | 0.734 | -0.8 % |

Each method's lowest is on (42.91, 28.0), (47.0, 26.09), (76.0, 34.25), (79.66,
40.0). The tests assert each within 3 % of print. Without the loads the same
family reads 0.787 / 0.753 / 0.752. Nothing was fitted: the family was declared
before it was scored.

Not checkable from the repository: the ground, the seam and the base are read
off Figure 9.1, and the printed critical surfaces are in Figures 9.2 to 9.4, none
of which is here (the comparison is of each method's lowest value, not of one
surface); the bench is entered at el. 28.0 where Table 9.2 puts the bench load at
y = 27.75; the crest load is entered as its mean. The piezometric line is Table
9.3 as printed.

**The module's weak-layer search does not reach these values.** As V-028 calls it
(seed 3, 700 trials, 6 points, 30 slices, the loads on) it returns Spencer 0.830
on a surface along the seam, x = 41.59 to 77.19, 9.2 % above the printed 0.760.
Since GM-140 step 2 (B20) that is Spencer's own solution on the surface it
reports: `analyze_slope` gives the same 0.830 (theta 8.0 degrees), Janbu
(corrected) 0.830 and Morgenstern-Price 0.816. Before that step it returned
0.515 on a short surface leaving the seam at x = 60, a state with the interslice
force at -43.5 degrees and tension across the back wedge, accepted about the
search's second moment axis where the single-surface solver had no solution. A
strict xfail against the printed 0.760
(`test_v028_weak_layer_search_against_print`), nothing pinned to the module's own
number. What is left is the search's generator (277 of the 700 trials have no
Spencer solution; its best surface is not the declared family's best, 0.753):
step 7 of the slope rebuild (the account's D12).

**The kept surface, a record only.** (41.987, 28.0) ... (77.528, 40.0) was the
minimum of the weak-layer search while the soil was weighed three times (B19); it
is not a critical surface. Janbu (corrected) reads 1.073 on it with no load and
1.034 with both zones (1.023 with the crest zone alone); Spencer and
Morgenstern-Price do not converge on it. Before the step, on the tripled weights,
it read Spencer 0.808 / GLE 0.802 (0.806 / 0.799 with both zones; 0.792 / 0.786
and 0.788 / 0.782 before GM-131, B15): those values are not current.

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
| V-028 Slide2 #9 / ACADS 4, the kept surface, Spencer / GLE / Janbu (on the tripled weights; not current, B19) | 0.792 / 0.786 / 0.804 | 0.808 / 0.802 / 0.819 | referee 0.78; Slide2 no-optimization Spencer 0.760 | old: 50 even slices (0.830 on 51; 0.806-0.815 on 100 to 1,600); now 0.808 on 30 to 1,600 |
| V-028 E8, both loads, the kept surface, Spencer / GLE (on the tripled weights; not current, B19) | 0.788 / 0.782 | 0.806 / 0.799 | | the same |
| V-028 weak-layer search, 30 / 40 slices (on the tripled weights; not current, B19) | 0.792 / 0.820 | 0.808 / 0.808 | referee 0.78 | the same surface at both counts now |
| V-028 `noncircular_de` seeds 1 / 2, 6 points, 30 slices, the seam windows (not pinned; on the tripled weights, not re-measured since B19) | 0.715 / 0.738 | 0.715 / 0.714 | Slide2 optimized Spencer 0.707 | |
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
(Since GM-128, B18, a given circle within a thousandth of the model's height
of the base touches it and is kept at every count; a search keeps to the
model.)
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

## B18 — A surface touching the model's base (GM-128, 2026-10-18)
[DESIGN.md "A surface touching the model's base". Tests:
`tests/test_duncan_verification.py::TestDuncanExample1::test_taylor_circle_on_the_base_at_any_slice_count`,
`tests/test_search_robustness.py::test_dips_below_base`,
`validation_examples/test_published_v043_slope_reinforced.py::test_v043_clay_circle_as_recovered_touches_the_base`.]

The depth of a surface under the bottom of the deepest layer is measured on
the surface, not on its slices. A given surface within a thousandth of the
model's height of the base touches it and keeps every slice; one deeper is
refused at any count; a search keeps to surfaces inside the model.

Taylor's critical circle, D = 2 (Duncan Ex 1 profile, 3 H of level ground
each side; centre 1 H past the crest, 1.908 H high; Taylor 0.706), Bishop:

| Slices | Tangent, as filed (before GM-134) | Tangent | 1 cm under | 5 / 10 cm under |
|---:|---|---:|---:|---|
| 25 | | 0.7045 | 0.7045 | refused |
| 30 | kept | 0.7052 | 0.7052 | refused |
| 40 | refused | 0.7055 | 0.7055 | refused |
| 50 | kept | 0.7059 | 0.7059 | refused |
| 60 | refused | 0.7060 | 0.7060 | refused |
| 80 | refused | 0.7062 | 0.7062 | refused |
| 100 | | 0.7063 | 0.7063 | refused |

Tolerance 0.024 m (the model 24.4 m high). Within it the FOS falls with
depth by no more than the slice count moves it: Taylor 0.02 %, V-043 0.06 %,
Slide2 #85's 'discard' circles 0.14-0.26 %.

What moved, every recorded result of the slope suite (`-m "not slow"`), the
slope validation examples, `funhouse_agent -k slope` and the worked
examples, before and after:

| Case | Before | After | Why |
|---|---:|---:|---|
| V-043 clay circle as the search recovered it (3.5 mm under the base), Spencer | refused | 1.0408 | touches the base (tolerance 0.009 m); the pinned circle, 0.1 mm above, reads 1.0409 |
| Slide2 #85 toe circle given, Bishop (and the agent's two tests of it) | 1.37156 | 1.37152 | the base's crossing at its toe is no longer a slice boundary |
| Taylor D = 3, default Bishop search | 0.677377 | 0.677376 | another centre in the same flat valley |
| Slide2 #85 'discard' searches, the toe family's candidate (not governing) | 1.5584 | 1.5583 | as the toe circle above |

Nothing else moved, and no search result rose. Granted the thousandth, the
searches would have ended that far under the base: Taylor 0.7049 -> 0.7048,
V-031 Spencer 1.4174 -> 1.4160, V-043 Spencer 1.1859 -> 1.1853, Slide2 #85
'discard' 1.511 -> 1.506 and 1.316 -> 1.314, and #85 with its support
1.372 -> 1.01 (a toe circle pivoting about the toe into the base, its
support uncounted); they keep to the model instead.

---

## B19 — One material per point (GM-140 step 1, 2026-11-13)
[The account's departure D1 (`module_work/slope_integral_review/ACCOUNT.md`
section 2). A slice's weight summed every layer overlapping its height, each
from its own flat top; the strength came from the first listed layer
containing the point. Now both come from one lookup, `SlopeGeometry.column_at`
(DESIGN.md "Layers: one material per point"). Tests:
`tests/test_one_material_per_point.py`. Full record:
`module_work/gm140/STEP1_CHECK.md`.]

**What moved.** Only models in which two layers were given the same ground.
Across `slope_stability/tests`, `validation_examples`, `dxf_import`,
`geo_project` and `calc_package/tests` (1,476 tests before), every call of
the slice weight was computed by the old rule and the new one side by side:
they differ in V-028 and its E8 variant and in one unit test with a sloping
boundary (`test_polyline_layers`), and nowhere else beyond rounding (V-030's
sub-layers, 2e-15). The solvers, the searches and the slicing are untouched,
so every other pinned value in this file is the same number as before.

| Case | Before | After | Printed |
|------|-------:|------:|---------|
| V-028 one slice, x = 47.97 to 48.53, weight (kN/m) | 116.0 | 39.13 | by hand 18.84 x 3.7073 x 0.5602 = 39.13 |
| V-028 sliding mass on the kept surface, weight (kN/m) | 8,768 | 3,003 | 3,003 from its area |
| V-028 kept surface (the old search minimum; not a critical surface), Janbu (corrected) | 0.819 | 1.073 | none for this surface: a record of the weight change |
| V-028 kept surface, Spencer | 0.808 | no solution | none for this surface |
| V-028 kept surface, GLE half-sine | 0.802 | no solution | none for this surface |
| V-028 kept surface, E8, both loads | Spencer 0.806 | Janbu 1.034 | none for this surface |
| V-028 family of 81 surfaces along the seam, both loads: lowest Spencer / Morgenstern-Price / Janbu corrected (B11) | not measured | 0.753 / 0.722 / 0.728 | Slide2 Table 9.4, no optimisation: 0.760 / 0.720 / 0.734 |
| V-028 weak-layer search, Spencer, 30 / 40 slices, no loads | 0.808 / 0.808 | 1.435 / 1.435 with the step (Table 9.3's line as printed; 1.430 / 1.431 with the line as entered before); 0.515 / 0.515 after the fix round, on another surface (B11: not confirmed by the direct solution; strict xfail) | Slide2 Spencer 0.760 |
| Two-layer test geometry of `test_polyline_layers`, circle (15, 20) R 18: Bishop; mass (kN/m) | 2.488; 2,017 | 2.849; 1,757 | none (made-up values); 1,757 from its areas, 37.763 m2 of sand at 18 and 63.379 m2 of clay at 17 |
| EM 1110-2-1902 Figs. F-10 and F-11, two soils on a level boundary, Bishop on 30 slices | not re-measured (a level boundary with no overlap: the step does not change it) | 1.329 | 1.33 (Fig. F-11, p. F-21) |

(The piezometric line is now Table 9.3 as the problem text prints it; with the
line as entered before, the after values are Janbu 1.070, fallback Spencer
1.038, as the account measured.)

**V-028 / B11 validate on surfaces along the seam (the fix round,
2026-11-13).** The step's first reading, that V-028 no longer validated, came
from the kept surface: the minimum of a search run while the soil was weighed
three times, which is not a critical surface once each point is weighed once,
scored without the problem's loads. With Table 9.2's loads on, a declared family
of surfaces along the seam reads within 1 % of Slide2's printed no-optimisation
values by Spencer, Morgenstern-Price and Janbu corrected (B11). The problem text
gives the soils, the loads and the piezometric line; the ground, the seam and the
slip surfaces are in figures only, which are not in the repository, so the
entered geometry cannot be checked (Table 9.2 puts the bench load at y = 27.75
where the ground is entered at 28.0). The module's weak-layer search does not
reach the printed values (B11): a strict xfail against print, nothing pinned to
the module's own number.

**The weak-layer search follows a sloping layer (the fix round).** It aimed its
trial surfaces at the target layer's flat `top_elevation` and
`bottom_elevation`; in V-028 every layer reads 40 and 15, so the trials centred
on el. 27.5, not on the seam. It now takes the ground the layer holds at each x
(`search._layer_band_at`: `top_at`, `bottom_at`, less what the layers listed
before it hold). Nothing else in the search changed. For a layer with a flat top
and bottom nothing moves: the two-layer flat section of `tests/test_new_search.py`
(seed 42, 200 trials) reads 1.125632861657882 before and after, on the same
surface to the last digit. A made-up section with a 1 m weak layer between two
sloping polylines (`tests/test_one_material_per_point.py`): the critical surface
had 20 % of its length in the layer before (FOS 2.87) and 75 % after (1.23).

**Printed checks of a sloping boundary.** No printed benchmark with different
unit weights across a sloping boundary was found in the references held (EM
1110-2-1902 Appendix F, FHWA NHI-06-088 chapter 6, UFS 3-220-10 and the Slide2
text were searched). The sloping-boundary checks are:

- ACADS 4 (B11): a sloping seam, equal unit weights, within 1 % of print;
- made-up sections checked by hand: a fill over a clay on a 1-in-5 boundary
  (every slice's weight against a hand formula; one slice by hand, 164.07 kN/m,
  213.67 by the old rule; the mass against its two areas, 3,253.4 kN/m), and the
  sand-over-clay section of `test_polyline_layers` (the mass 1,757 kN/m from its
  areas and Bishop 2.849; 2,017 and 2.488 before the step);
- for two soils on a LEVEL boundary, EM 1110-2-1902 Figs. F-10 and F-11 (pp.
  F-20, F-21; F-5d, p. F-19, where slices in both soils "were divided into two
  portions for calculating the slice weights"): Simplified Bishop 1.329 on 30
  slices against the printed 1.33 (-0.1 %), and the weight in each soil against
  its exact area, embankment 1,724.8 kips against 1,724.1 and foundation 327.7
  against 327.3. It does not exercise a sloping boundary and reads the same as
  before the step. The printed total weight (2,022 kips) is not asserted: column
  3 of Fig. F-11 prints the widths against the slice numbers in reverse order,
  the printed slices end about 5 ft short of where the stated circle leaves the
  crest, and the page takes each area from the slice's mean height.

## B20 — A method's own solution, or no number (GM-140 step 2, 2026-11-13)

The rule, its print and what it replaced: DESIGN.md, "A method's own solution,
or no number". The record with every count: `module_work/gm140/STEP2_CHECK.md`;
the measurements are `module_work/gm140/step2_section11.py`, run on master and
on the branch.

**What moved.** Two benchmark values, and nothing that is pinned to print:

| Case | Before | After | Printed | Why |
|---|---|---|---|---|
| Slide2 #85 without its support, Spencer search, 6 x 6 | 0.9095 (a toe circle scored by the old approximate solver; 267 circles went to it) | 1.047 (a toe circle, Spencer's own solution; 329 trials without one, counted and warned of) | none for this case (Bishop's simplified method: 0.9095, unchanged) | No circle is scored by another solver. On this steep phi = 0 slope Spencer's method has no solution on the circles that govern, so its search cannot reach them; the result says a lower surface cannot be ruled out. |
| ACADS 4 (V-028), weak-layer search, Spencer, loads on | 0.515 | 0.830 | 0.760 (Slide2 Table 9.4) | One solver and one moment axis in a search and on one surface; an iteration that does not settle is no state (B11). |
| ACADS 4, the calc-package report of that search (no loads) | 1.161 printed (the approximate solver on the search's 0.515 surface) | 0.879, the search's own value on its surface | none | The report's re-analysis and the search are the same solve. |

**What did not move** (each measured before and after): #85 with its support
(Bishop 1.372, Spencer 1.551, GLE 1.573, passive GLE 1.359), #86 with its
tiebacks (1.613 / 1.607 / 1.607) and without (0.942), ACADS 1(a) (0.985 / 0.984 /
0.984 / 0.983), V-043 (Spencer 1.1859, the pinned circles 1.2096 and 1.1858),
V-031 (1.4205 / 1.4247 / 1.4091; Spencer search 1.4174), V-044 (1.5974 / 1.5962 /
1.5933), the yield coefficient of V-039's demonstration (Spencer 0.565), the
FOSM and Monte Carlo cases (B4, V-030, V-042, V-045) and ACADS 4's declared
family (0.753 / 0.722 / 0.728). On none of them did the approximate solver, an
unsettled iteration or the second moment axis supply the governing value. Over
1,432 solves of made-up circles (three slopes, the constant and half-sine
functions) no converged GLE value moved by more than 0.002; 3 circles gained a
solution (the lambda search) and 2 lost one (their "solution" was an iteration
that had not settled).

**What a result says now.** #85 with its support, Spencer search: 357 circles of
the main search have no Spencer solution (counted under `no_solution`); the
governing circle is one Spencer solves. #85 with its support, Bishop: 178 + 28 +
92 trials have no net driving moment left (the support's force exceeds it),
counted as `no_driving` and not warned of. A single surface the method cannot
solve: `FOS` None, never another solver's value (GM-139's kind of circle; V-054's
circle by Spencer and GLE; a stabilizing-pile row whose force exceeds the driving
moment, which returned 999.9).

## B21 — Tension cracks and reverse-curvature circles as printed (GM-140 step 4, 2026-11-13)

The rule and its print are in DESIGN.md ("Tension cracks and
reverse-curvature circles as printed"); the record, with every reading and
the tests run, is `module_work/gm140/STEP4_CHECK.md` (its last section is
the fix round after the independent review). "Before" is master before
step 4 (2cc2c45); "first built" is the step as built before the review;
"after" is after the review's fixes, with the crack a condition of the
analysis that the engineer sets and the same on every trial surface (the
orchestrator's ruling on the review's F1). A given surface with no crack and
no reverse curvature runs the same code throughout.

| Benchmark | Before | First built | After | Printed | What moved it |
|---|---|---|---|---|---|
| ACADS 1(b), Slide2 #2 (V-026): given 3.81 m crack full of water, Slide2's circle, Bishop / Spencer / GLE | 1.485 / 1.486 / 1.482 | 1.603 / 1.599 / 1.600 | the same | 1.596 / 1.592 / 1.592 (Table 2.2, p. 25; referee 1.65) | the default model: the surface ends at the crack |
| same, searched on Slide2's centre grid (x 31 to 47, z 34 to 49, 16 by 16, 40 slices), Bishop / Spencer | 0.050 (`'strength'`, then the default) / 0.052 with the surface ended at the crack: a 0.6 m sliver on the face | 0.052 (sliver) | 1.590 / 1.585 (toe circles; 456 surfaces wholly within the cracked depth refused) | 1.596 / 1.592 | a surface that nowhere passes below the crack's depth is no slip surface (the review's D1) |
| same, `tension_crack_model='strength'` | 1.485 / 1.486 / 1.482 | the same | the same | | the stated option |
| same, Eq. C-36 with the developed strengths, full of water (2.19 m) | not offered | 1.631 / 1.627 / 1.628 | the same | the problem uses the full strengths (3.81 m) | reported, not pinned |
| V-031, Slide2 #57, its dry crack (the text prints no depth: Rankine's with the developed strengths, Eq. C-36), Bishop / Spencer, 50 slices | 1.421 / 1.425 (no crack) | not entered | 1.4170 / 1.4211 | 1.417 / 1.422 (Table 57.3, p. 205; crack in the text, p. 203) | the crack the problem has, now entered (the review's D4) |
| V-043 clay embankment, water-filled crest crack, Spencer, recovered circle | 1.041 (no crack) | 0.964 (Eq. C-36, 2.14 m, full) | the same (the crack is asked for) | 0.975 (Table 39.2, pp. 147-150) | the Rankine depth, which the problem does not print |
| V-054 steep phi = 0 slope from the toe, Bishop search | 2.346 | 2.113 | 2.346 (56 reverse-curvature circles discarded) | 2.16 (FLAC 2.18) for the slope; this profile ends at the toe | no crack asked for: the overhanging circles are discarded |
| V-054 toe circle (2.5, 4.0), R 4.717 | 2.379 (closed at its centre's level) | 2.218 | no number without a crack; 2.218 with Rankine's asked for | 2.16 | a reverse-curvature circle; the centre-level crack, which no print gives, stays removed |
| V-054 Bakklandet ratios on that circle, Rankine's crack asked for | 1.942 (-18.4 %) | 1.693 (-23.7 %) | the same | -24 % | the same, at the active strength |
| V-054 Bakklandet ratios on circle (3, 3), Rankine's crack asked for | 1.886 (-20.7 %) | 1.848 (-16.7 %) | the same | -24 % | **away from print, strict xfail** |
| Vertical cut in clay, Bishop search | 0.842 (toe circle closed at its centre's level, 2.30 m) | 0.752 | 0.935 (12 discarded); 0.605 with Rankine's crack asked for | Taylor 0.851 | no crack asked for: the overhanging toe circles are discarded; **away (+9.9 %, the unsafe side), strict xfail** |
| Taylor 45 deg, D = 2.25, 4 H each side | 1.942 (base circle) | 1.835 | 1.942 (38 discarded); 1.811 with Rankine's crack asked for | Taylor 1.947 | no crack asked for |
| Taylor 45 deg, short profile | 2.062 | 1.833 | 2.065 (89 discarded) | Taylor 1.947 | the same |
| Slide2 #85, one support, active, Bishop search | 1.372 (closed at a centre's level, 5.89 ft) | 1.361 | 1.508 (28 discarded); 1.331 with Rankine's crack asked for | 1.531 (D&W 1.51) | no crack asked for; reported, not pinned (the step's brief sets no tolerance on #85) |
| #85 passive, Bishop search | 1.254 | 1.206 | 1.316 (41 discarded); 1.188 with Rankine's crack asked for | 1.324 (D&W 1.32) | the same |
| #85 active, Spencer / GLE search | 1.551 / 1.573 | 1.551 / 1.573 | the same | 1.884 / 1.575 | not moved |
| #85 toe family, Spencer | 1.564 | 1.627 | 1.564 | spread 1.51 to 1.884 | no crack asked for |
| Caltrans T&S Ex. 10-3 / 10-4 circle (worked example WE-SLOPE-1) | OMS 0.839, Bishop 0.977 | no number by default; the same with a 6.096 m crack given | the same | 0.87, 0.96 | the printed circle's centre is 20 ft below the crest: no surface without a crack deep enough |

Every other benchmark of the published set reads as before (the
`validation_examples` slope files and the benchmark files of
`slope_stability/tests`: `module_work/gm140/step4_section11.py`).

As first built, a reverse-curvature circle was closed by Rankine's crack
whenever no crack was given, so a circle's shape switched the crack on: a
circle whose centre was a hair below the crest took the whole crack and one
a hair above took none, and on steep phi = 0 slopes the searches settled on
barely reverse-curved circles, below the chart values. With the crack a
condition of the analysis, F is continuous as the centre passes the
ground's level (`test_cracks_as_printed.py`), and the cases with no crack
read as before step 4. The vertical cut stays away: before step 4 its 0.842
came from a circle closed at its centre's level, which no print gives.

## B22 — The printed acceptability checks (GM-140 step 3, 2026-11-13)

The rule, its print and what it replaced: DESIGN.md, "The printed
acceptability checks". The record, with the whole of section 11 before and
after: `module_work/gm140/STEP3_CHECK.md`; the measurements are
`module_work/gm140/step2_section11.py` (unchanged), `step3_section11.py` and
`step3_screens.py`, run on master and on the branch.

**Against print.** No value pinned to a printed answer moved outside its
tolerance.

| Case | Before | After | Printed | Why |
|---|---|---|---|---|
| Slide2 #85, one support, GLE search | 1.5726 | 1.5732 | 1.575 | the toe-circle family uses the common below-base test (D19) |
| Slide2 #85, one support, Bishop, active / passive | 1.3715 / 1.2538 | 1.3713 / 1.2534 | 1.531 / 1.324 | the same; both are the reverse-curvature toe circles of D24 |
| #85, reverse curvature discarded, Bishop, every slice looked at | 1.511 (157 circles left out) | 1.508, flagged for m_alpha, with 1.511 as the lowest circle without a flag | 1.531 (Duncan and Wright 1.51) | a circle with m_alpha below 0.2 is kept and flagged (OWN-102) |
| Vertical cut in undrained clay, Bishop, every slice looked at | 0.935 | 0.842, flagged, with 0.935 as the lowest unflagged | Taylor 0.851 | the same |

Unchanged, each measured before and after: ACADS 1(a) (0.985 / 0.984 / 0.984 /
0.983), #86 (1.613 / 1.607 / 1.607; 0.942), #85 Spencer (1.551), V-054's toe
circle (2.346), V-032 (1.315), V-042 (1.3335; FOSM and Monte Carlo), Duncan's
Example 1 (0.7049, 0.6774), V-043 (1.2096, 1.1859; DE 1.234), V-031, V-044,
V-039's yield coefficients, ACADS 4's declared family (0.753 / 0.722 / 0.728)
and the weak layer over rock's default search (0.886). On each the circles
formerly left out for m_alpha are now kept and flagged, far above the
minimum.

**What a result says now.** ACADS 4's weak-layer search (0.830 against the
printed 0.760; strict xfail): the surface carries tensile interslice forces
and a line of thrust outside the mass; the lowest scored surface without a
flag is 1.667. V-042's Bishop search: on 537 of 3,942 scored circles Bishop's
value is below the Ordinary Method of Slices value, which is taken there
(Duncan 1996, p. 580); none governs.

**Not pinned to print, and moved.** Each is flagged, with the lowest
unflagged surface beside it:

| Case | Before | After | Printed |
|---|---|---|---|
| Duncan's Example 4 by Janbu, 30 slices | 1.203 (78 circles left out) | 0.843 flagged; 1.215 unflagged; 1.203 with `low_m_alpha='refuse'` | none in hand for Janbu |
| A made-up pore-pressure grid on the ACADS 5 slope (V-029's demonstration), Bishop | 1.247 | 1.175, the frictional term as printed where u b exceeds W; 1.247 with `negative_friction_term='zero'` | none |
| ACADS 1(a), random noncircular, 200 trials, Spencer | 1.092 | 0.633 flagged; 1.096 unflagged | 1.00 |
| Weak layer over rock, DE directed | no surface | 0.612 flagged; no scored trial unflagged | none |

The last two follow from removing the noncircular screens on a jagged surface
and on agreement between moment axes and slicings: every surface those
screens refused in the measured cases carried a flag of the printed checks
(88, 38 and 1), so the surfaces are now scored and flagged.

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
