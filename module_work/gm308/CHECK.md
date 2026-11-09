# GM-308 (b) (fem2d): does the default footing domain cut off the mechanism on frictional soil?

Branch `claude/gm-308-fem2d-domain`, from master 8daed85. All values are
made up. **Verdict: no truncation found; no code changed.** Part (a), a
faster nonlinear iteration, is out of scope (section 5 says how far the
frictional answers still are from levelling off).

## 1. What the print says

- **GEC-6 (FHWA-SA-02-054) §5.2.1.1, pp. 49-50, Fig. 5-3**, and
  **NHI-06-089 (Soils and Foundations Vol. II) §8.4.1.1, p. 8-16,
  Fig. 8-13**: the Prandtl (1920) / Reissner (1924) zones of plastic
  equilibrium under a strip footing in a uniform, rigid-plastic,
  *weightless* soil: an elastic wedge (Zone I), radial shear on a
  logarithmic spiral (Zone II) and a passive zone at 45 - phi'/2 from the
  horizontal (Zone III), the shear planes reaching the surface. **Neither
  prints the zone's extent as a number, nor any guidance on a finite-element
  domain size.** A search of every manual's JSON text in geotech-references
  (912a99d: "log spiral", "Prandtl", "radial shear", "failure zone/surface
  ... extend/width/distance", "mesh/model/domain ... boundary/extent", "zone
  of influence") found nothing more: the other hits are pile-group and
  settlement zones of influence (GEC-6 §5.2.3.2 1.5 B for the water table,
  §5.4 2-4 B for borings), not the shear mechanism.
- So the extents quoted in the item (4.79 B at phi 30, 5.04 B at phi 31)
  are geometry worked from that printed mechanism with the Prandtl wedge at
  45 + phi/2, not a printed number: from the footing centre
  L = B/2 + B e^((pi/2) tan phi) cos(45 - phi/2) / cos(45 + phi/2):
  **4.79 B (phi 30), 6.27 B (35), 8.51 B (40)**; the spiral's deepest
  point is about 1.6, 1.9 and 2.4 B down, well inside the 5 B depth, so
  only the width was studied.
- That mechanism is for an *associated*, *weightless* soil. fem2d's runs
  are non-associated (psi = 0 by default) and, here, with weight; both make
  the real mechanism narrower. The measurement below therefore stands on its
  own as a convergence study.

## 2. The measurement

Strip footing B = 2 m, flexible (the default), T6, E = 1e5 kPa, nu = 0.3,
psi = 0, c = 5 kPa, gamma = 18 kN/m3, surcharge 0 (and 18 kPa where
shown), domain depth 5 B (default). **Element size held at 0.5 m in every
run**: nx = 40, 60, 80, 120 at half-widths 5, 7.5, 10, 15 B (ny = 20), so
only the width changes; nx left at 40 on a wider domain would coarsen the
mesh and confound the comparison. **q_max and n_load_steps are the same
across widths of a row** (q_max = 0.7 x the closed-form estimate, given
explicitly), so every width walks the same load grid; one load step is
given beside each row. **max_iter fixed** (given), not the default.
Scripts: `module_work/gm308/run_width.py` (q_ult) and `run_heave.py` (q_ult
plus the surface heave at the stopping step), run from the repo root,
single-threaded (`OMP_NUM_THREADS=1`; the first batches ran threaded and
oversubscribed, which changes only their times). Every run's output line
is in `module_work/gm308/results.jsonl` (argument order: phi, half-width in
B, max_iter, surcharge, load steps, q_max as a multiple of the estimate).

q_ult, kPa:

| phi | q (kPa) | steps (one step) | max_iter | 5 B | 7.5 B | 10 B | 15 B | 5 B vs widest |
|---|---|---|---|---|---|---|---|---|
| 30 | 0 | 100 (3.9) | 8000 | 248.2 | – | 248.2 | – | 0 |
| 35 | 0 | 100 (7.7) | 8000 | 414.0 | 406.3 | 406.3 | – | +1.9 % (one step) |
| 35 | 0 | 300 (2.6) | 8000 | 431.8 | – | 434.4 | – | -0.6 % (one step) |
| 35 | 0 | 100 (7.7) | 16000 | 429.3 | – | 429.3 | – | 0 |
| 40 | 0 | 100 (16.4) | 8000 | 739.0 | 739.0 | 739.0 | 739.0 | 0 |
| 40 | 0 | 100 (16.4) | 16000 | 788.2 | – | 804.7 | – | -2.1 % (one step) |
| 35 | 18 | 150 (7.8) | 8000 | 1038.2 | – | 1030.4 | – | +0.8 % (one step) |
| 40 | 18 | 150 (16.3) | 8000 | 1850.9 | 1915.8 | 1948.2 | 1526.5 (no mechanism) | -5.0 % against 10 B |
| 40 | 18 | 150 (16.3) | 16000 | 1883.3 | – | – | 1964.4 | -4.1 % |

Every run above stopped at max_iter on its last step (the GM-306 warning
fired on all of them; one step was accepted as settled, in the 15 B,
16000-iteration run only), so each is a
lower bound limited by the iteration count, as GM-306 found. Comparing like
with like (the same max_iter, load grid and element size), the question is
only whether 5 B reads *higher* than a wider domain.

**Reading.**

- **phi 30 and 35, and phi 40 without a surcharge:** 5 B and the widest
  domain agree to within one load step (at most 1.9 %, in both directions),
  and phi 40 reads 739.0 at 5, 7.5, 10 and 15 B. No change over 2 %.
- **phi 40 with q = 18 kPa** is the only case that moves by more than 2 %,
  and it moves the *other* way: 5 B reads **lower** than every wider
  domain that reached a mechanism (8000: 1850.9 against 1915.8 and 1948.2
  at 7.5 and 10 B; 16000: 1883.3 against 1964.4 at 15 B), by 3.5 to 5 %.
  That is the safe side for the default, and it cannot be the side-roller
  truncation the item feared: a roller only adds restraint, which in a
  converged limit load can raise q_ult, never lower it. It is the same
  iteration-limited stop (the gap is two to five load steps, and the
  answers are still rising with max_iter). The 15 B run at 8000 (1526.5)
  is the clearest case: its stopping step had **no mechanism at all**
  (`run_heave.py`: every surface node settling, peak "heave" -6.8 mm, the
  side boundary settling as much as anywhere), so it stopped on the
  ceiling well before collapse; at 16000 it reached 1964.4.
- **Where the mechanism is.** At the stopping step of the 5 B and 10 B
  phi 40, q = 18 runs the surface heave (uy > 5 % of the peak heave) lies
  within 1.0 B of the footing centre on both meshes, and the side boundary
  is settling (uy / peak heave = -0.13 at 5 B, -0.06 at 10 B). The
  non-associated (psi = 0) mechanism sits well inside the 5 B boundary,
  far narrower than the associated, weightless Prandtl extent of 8.51 B.

**At max_iter 1000** (the default on 40 x 20), 200 steps, q = 0: 5 B read
224.3, 355.9 and 574.8 at phi 30, 35 and 40; 7.5 B 221.6, 345.0, 527.8;
10 B 221.6, 339.5, 527.8 (5 B +1.2, +4.8, +8.9 % over 10 B). That looked
like truncation, but at 8000 and 16000 the gaps close to within a step
(table above): at 1000 iterations the stop comes so early, and so far from
any mechanism, that the meshes' differing convergence decides it. The
default's own answers are dominated by the iteration limit, not the width.

## 3. Decision

**No truncation of the mechanism was found, so the code is unchanged**: the
default half-width stays 5 B, no phi-dependent default, no new warning, no
`choices` or `defaults_used` change. The brief's 2 % test is met in every
row compared at a fixed, large max_iter except one (phi 40, q = 18),
where 5 B reads 3.5 to 5 % *below* the wider domains (conservative); the
error the item feared, 5 B overstating q_ult, is not seen anywhere at
max_iter 8000 or 16000 beyond one load step (1.9 %, phi 35). Widening the
default for phi 40 on that row would raise saved answers on evidence that
is itself iteration-limited, which I have not done; it is for the lead
once part (a) gives converged frictional answers (section 7).

Because the code is unchanged, no saved answer moves and there is no
release note. Run time a wider default would have cost (for the record,
single-threaded, phi 40, q = 0, max_iter 8000, 100 steps): 5 B 539 s,
7.5 B 326 s, 10 B 431 s, 15 B 660 s (the 5 B figure was measured under
thread oversubscription; at phi 30, 5 B 148 s against 10 B 281 s, about
twice the time for twice the elements).

## 4. Before / after

No code change: before = after. Worked example (made-up), B = 2 m,
c = 5 kPa, phi = 35, gamma = 18, defaults otherwise but q_max = 766.6 kPa,
100 steps, max_iter 8000: 414.0 kPa on today's default 5 B domain, 406.3
kPa at 10 B with the same element size.

## 5. How far the frictional answers are from levelling off (part (a), not in scope)

5 B, q = 0, 100 steps, the same q_max:

| phi | max_iter 1000 | 8000 | 16000 | last doubling |
|---|---|---|---|---|
| 30 | 213.3 | 248.2 | 255.9 | +3.1 % |
| 35 | 329.6 | 414.0 | 429.3 | +3.7 % |
| 40 | 509.1 | 739.0 | 788.2 | +6.7 % |

Still rising at 16000 on every phi, faster the higher phi (phi 40,
q = 18: 1850.9 and 1883.3 at 8000 and 16000, +1.8 %). The answer also
depends on the load-step size at a fixed max_iter (phi 35: 414.0 at 100
steps, 431.8 at 300), because each step's increment is what the
constant-stiffness iteration must absorb. Today's default (1000) reads
17-35 % below the 16000 values. For comparison the closed-form estimate
(Prandtl-Reissner Nc, Nq with Vesic's N_gamma, associated) is 554, 1095
and 2346 kPa; a psi = 0 FE answer is expected below it.

## 6. Tests (venv Python 3.12, judged by pytest's own exit code)

| Command (`python -m pytest -q -p no:cacheprovider ...`) | Result | Exit |
|---|---|---|
| `webapp/tests/test_page_fingerprints.py webapp/tests/test_public_text.py` | 19 passed | 0 |
| `funhouse_agent/tests/test_defaults_used.py -k fem2d` | 46 passed, 799 deselected | 0 |
| `-n 2 fem2d/tests/test_footing_capacity.py funhouse_agent/tests/test_footing_capacity_adapter.py` | 70 passed | 0 |

No validation example or calc package uses the footing capacity (grep of
`validation_examples/` and `calc_package/`). The fem2d module suite was not
rerun in full: no fem2d file changed (GM-306 measured it at 644 passed).
No new test: no default changed.

## 7. Noted, not fixed

- The width study is at most 15 B and max_iter 16000; with a converging
  iteration (part (a)) the frictional answers rise and the mechanism could
  widen. Repeat the 5 B / 10 B / 15 B comparison at phi 40, q = 18 once
  part (a) lands (a re-run of `run_width.py`): today 5 B reads 3.5-5 %
  below the wider domains there (conservative), and if that gap survives a
  converged iteration a wider default for high phi with a surcharge would
  raise saved answers by about that much.
- The flexible footing's answer depends on the number of load steps at a
  fixed max_iter (section 5): part of the same iteration limit (GM-308 (a)).
- The iteration-limited stop can land on a step with no mechanism formed
  (phi 40, q = 18, 15 B, section 2), a result the GM-306 warning already
  calls a lower bound.
