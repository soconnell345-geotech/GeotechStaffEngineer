# GSTools Agent — Design Notes

## Purpose

Wraps the GSTools library (Müller, Schüler, Zech & Heße 2022, "GSTools v1.3:
a toolbox for geostatistical modelling in Python", *Geoscientific Model
Development* 15, 3161–3182) for spatially variable ground. Four methods, told
apart by the question each answers:

| Method | Answers | Does NOT answer |
|---|---|---|
| `kriging` (`analyze_kriging`) | ONE smoothed best estimate between points: the mean surface and its kriging variance; optional region totals of that surface and Gaussian pointwise expectations | the probability of a total over an area (the surface is smoother than the ground, so a total on it understates the spread) |
| `variogram` (`analyze_variogram`) | the empirical variogram and a fitted covariance model | — |
| `random_field` (`generate_random_field`) | one UNconditioned correlated field | anything tied to data: it does not honour observations |
| `conditional_simulation` (`analyze_conditional_simulation`) | many realizations that honour the data, each integrated over plan polygons: the distribution of a TOTAL (volume above / below a level, area, any encounter), percentiles, exceedance | — (see Limits) |

The conditional simulation was built from a field session (2026-10-15, app on
GPT-5.6 Sol): an engineer asked for the probability of exceedance of ROCK
EXCAVATION VOLUME under a footprint from a few borings. The wrapper then had
only deterministic kriging (returning the whole grid as JSON, which was
offloaded and unusable) and an unconditioned field, so the agent could not do
it and ended with a script it could not run.

## Architecture

```
gstools_agent/
  __init__.py              # exports the 4 analyze functions + result classes
  gstools_utils.py         # has_gstools(), import_gstools()
  kriging.py               # analyze_kriging()  (+ regions on the kriged mean)
  variogram.py             # analyze_variogram(), empirical_variogram(), fit_warnings()
  random_field.py          # generate_random_field()
  conditional.py           # analyze_conditional_simulation()
  conditional_figures.py   # exceedance curve + probability map (PNG + Plotly twin)
  plan_map.py              # the plan maps (probability, kriging): one drawing path
  models.py                # model_class(): GSTools' models, 2-D Exponential ppf fixed
  regions.py               # polygons, lattice, point-in-polygon, integration,
                           # Gaussian pointwise expectation (shared)
  results.py               # KrigingResult, VariogramResult, RandomFieldResult,
                           # ConditionalSimulationResult (+ CSV writers)
  tests/
    test_gstools_agent.py
    test_conditional_simulation.py
    test_conditioning_by_definition.py
    test_variogram_nugget.py
    test_kriging_map.py
    test_models.py
```

The adapter (`funhouse_agent/adapters/gstools_adapter.py`) writes every array
to the conversation's working folder (`_fileio.resolve_output_path` /
`save_verified`, figures with a `.plotly.json` twin via
`write_plotly_sidecar`) and returns paths plus small statistics. A kriging grid
over 2,500 points goes to `<prefix>_grid.csv` instead of inline arrays, and
kriging with `regions` writes `<prefix>_regions.csv` (a row per cell, region
and level);
`KrigingResult.to_dict()` itself is unchanged for library callers. A
conditional simulation writes `<prefix>_realizations.csv`,
`<prefix>_probability.csv` and `<prefix>_kriging.csv`.

**Figures** (`figures=True`, the default, for a conditional simulation): the
exceedance curve P(V > v) per region and level (up to 12 curves, P50 and P90
marked and named) and a map of the probability that the value is above
(below) the level in each cell, with the region outlines, the named
observations and the bounds. The map is drawn as a plan (owner, 2026-10-15):
to scale, with no axes or grid, the title at the left, and the scale bar and
north arrow of `geotech_common/plan_furniture.py`, which
`profile_figure.plot_data(plan_view=True)` shares. `north_deg` (degrees
clockwise from up, as the source plan shows it) turns the arrow; without it
the arrow points up and the result's `notes` say so. `value_name` (what the
values are, in the logs' words, e.g. "top of hard rock elevation") names z
in the exceedance curve's axis label and in the map's title and hover text,
with `value_unit` after the level on the map; without it the figures say
"value". `outlines` draws what the source plan shows besides the regions
(`[{name, polygon | line, kind}]`, kind `property_line` dash-dot as drafted,
or `outline`); the view takes them in.

**The kriging map** (`figures=True`, the default, for `kriging`; GM-96):
`<prefix>_kriging_map.png` and its Plotly twin, the kriged mean and the
kriging standard deviation of the value at a point, side by side (stacked
for a site more than about twice as long as it is wide) at the same scale,
each as filled bands between contour lines. Each level is labelled once,
written level, at the spot on its line farthest from the observations,
their names, the outlines, the field's edge and the other labels
(`plan_map.label_spots`); a level with no clear spot is left unlabelled
rather than written over a boring's name or a building line. The mean's
labels are its scale. The standard deviation rises from the observations,
so its lowest levels ring them too tightly to label: its bands also get a
colour scale beside the plan. The observations are named when `names` is
given (else marked only) and called by `observations_name` in the legend
('Borings'; 'Observations' without it); the regions and the `outlines` are
drawn, with one scale bar and one north arrow. The title names z
(`value_name`), each plan names its quantity and unit, and a caption under
the plans states the kriging type, the covariance model and its parameters,
whether they were fitted, and whether the standard deviation is zero at an
observation (nugget 0) or not (the nugget is in it). A grid of one line or
one point has no map. With `figures=false` the inputs that only label the
map are listed in a note, and `fit_nugget=true` with `fit_variogram=false`
is refused (nothing is fitted).
Both maps go through `plan_map.plan_map_figure`, one layout in printed
points (the margins, bar and arrow of `profile_figure`'s plan view) and the
furniture of `geotech_common/plan_furniture.py`; nothing in it is drawn
twice. Set `figures=false` in a sweep of many calls.

## Conditional simulation: the method and its basis

**Realizations.** Default `simulation_method="kriging_residual"`:
conditioning by kriging (Journel & Huijbregts 1978, *Mining Geostatistics*,
ch. V; Chilès & Delfiner 2012, *Geostatistics*, §7.3.1). Draw an unconditional
Gaussian realization S of the covariance model at the cells AND the
observations, krige the residual (observation − S there, with simulated
measurement error), and add it: `z = S + K(z_obs − S_obs − e)`. For a Gaussian
field this draws the conditional distribution the chosen model implies — mean
= the kriged surface, pointwise variance = the kriging variance, and the right
covariance between cells, which is what decides the spread of a total.

**What "right" and "exact" mean here.** Relative to the ASSUMED model only:
a Gaussian field with the chosen covariance, mean and the observations used.
None of that is known about the ground, and those choices usually move the
answer more than the method does, so run cases. Given the assumptions,
"conditioned on the observations" has one meaning: of all the fields the
model can produce, only those that pass through the observations.
`tests/test_conditioning_by_definition.py` draws that with no kriging at all
(plain numpy: unconditioned fields, keeping those within 0.1 standard
deviation of every observation) and sets the spread of a total over a
footprint against both methods and the textbook Gaussian-conditioning
formula. Measured (3 observations, 12 cells, exponential, variance 4,
len_scale 15 m): by definition 145–149, formula 144, `kriging_residual` 142,
`CondSRF` 178. A larger run (3 observations, 120 cells, 159 million fields,
4,000 kept): by definition 579, formula 587, `kriging_residual` 581,
`CondSRF` 802. Both methods get the mean right.

- The kriging weights K come from GSTools' own `Krige` of the chosen type
  (kriging is linear in the data, so kriging each unit vector gives the
  weights).
- S is drawn by Cholesky factoring the model covariance over the points
  (Davis 1987, the "LU" method), using GSTools' `covariance` on its
  `isometrize`d positions (anisotropy included). The draw has exactly the
  model's covariance for every GSTools model and is fast at the allowed size: 1,000 realizations over ~750–3,000
  cells take 0.1–1 s. A diagonal jitter (smallest of 1e-10…1e-3 × variance
  that factors) handles numerically singular smooth models; above 1e-6 it is
  reported. The cell cap (`MAX_SIM_CELLS` = 5,000; the matrix is n² doubles)
  refuses with the count and a cell size that fits.

`simulation_method="condsrf"`: GSTools `CondSRF` exactly as GSTools builds it
— the kriged field plus a randomization-method field scaled pointwise by
sqrt(kriging variance / variance), plus the nugget as white noise per cell.
It honours the data and the pointwise kriging variance, but the correlation
between two points is the UNconditioned one. Measured here: two cells 5 m
either side of a datum, exponential model, len_scale 10 m — the model's
conditional covariance 0 (the datum screens), `kriging_residual` 0.006,
`CondSRF` 0.33. On a 60 × 50 m footprint with seven borings the volume std
came out 1.8× (exponential) to 2.5× (Gaussian) the model's own and the P90
26 % higher, at the same mean. It is also 100–200× slower (randomization method per realization;
models without an analytic spectral inverse — Spherical, Matern, Stable, … —
sample their spectra by MCMC, ~0.1–0.2 s each). Offered to reproduce a GSTools
CondSRF notebook; a run projected over 600 s is refused with the options. The
CondSRF positions are set ONCE and the raw kriging field is stored, so the
kriging is solved on the first realization and reused.

**Nugget and measurement error** (GSTools' `Krige(exact, cond_err)`):
the kriging is GSTools' `exact=False` form, the observation-error variance
`cond_err` = `measurement_error` (one number or one per point) when given,
else the nugget (GSTools' default). GSTools treats the measurement error as
part of the nugget (`cond_err ≤ nugget`): a larger error is refused, and with
the nugget not given it is set to the largest error. With nugget 0 and no
error the field passes exactly through the data. The realizations are the
CONTINUOUS part of the field: the nugget is not added cell by cell (a total
over cells would then depend on the cell size); the continuous kriging
variance is GSTools' variance less the nugget.

**Totals.** On a cell-centred lattice aligned to the regions' lowest x, y
(so a footprint drawn on round coordinates gets whole cells), cells whose
CENTRE is inside the polygon (matplotlib `Path.contains_points`, concave
shapes fine; a centre exactly on an edge decided by a half-open rule):
`volume_above = Σ max(z − level, 0)·A_cell`, `area_above = Σ A_cell [z >
level]`, and the `below` mirror. Every region and level is evaluated on the
SAME realizations (common random numbers): a region nested in another never
has the larger volume in any realization. Reported per region/level: polygon
area vs summed cell area, mean/std/P5…P95 of volume and area, `p_any`
(= P(volume > 0)), an exceedance table with Monte Carlo standard errors, a
convergence table (mean and P90 at 100/250/500/1000/… realizations), the
KRIGED-surface total, and the Gaussian pointwise expectation of the total
(below). P90 means a 90 % chance of NOT being exceeded.

**Pointwise expectations** (`regions.pointwise_gaussian`): from the kriging
mean μ and continuous std σ per cell, P(z > L) = Φ(d) and E[max(z − L, 0)] =
σφ(d) + (μ − L)Φ(d), d = (μ − L)/σ. Summed they give the EXPECTED total and
the expected area (expectation is linear), so the realizations' mean must
match them — a Monte Carlo check that the tests pin. They say nothing about
the probability that the total exceeds a value; that needs the joint
distribution. `kriging` returns them with `regions` and says so.

**Inequality data** (`bounds`, e.g. a boring that ended above rock: the rock
surface is deeper than its bottom). `bounds_mode="report"` counts, per bound,
the fraction of realizations violating it (each realization is evaluated
exactly at the bound's point). `bounds_mode="reject"` keeps only realizations
satisfying every bound, drawing more up to 10 × `n_realizations`, and reports
the acceptance rate (warning below 10 %). This is REJECTION SAMPLING, not
inequality kriging: in the limit the kept set is what the chosen model
implies given the bounds; it is slow, and possibly unrepresentative when the
bounds sit in the tail of the model.

**Data-dependent defaults** are computed at run time from the observations of
THIS call and echoed in `defaults_from_data` with their basis (variance = the
sample variance of the n observations, len_scale = a third of the data
extent, the simple-kriging mean). A field session kept a variance worked out
before a boring was added. `fit_variogram` defaults to False here: with a
handful of borings a fitted variogram is unstable (in `kriging` it defaults
to True and then overrides given variance, len_scale and nugget, which the
notes say).

**The variogram's nugget** (GM-160, 2026-10-19): left out, it is fitted with
the sill and range; a number (0 or more) fixes it and only the sill and range
are fitted (GSTools `fit_variogram` with a fixed value); `nugget_fitted` says
which. Before, the value was never used: 0 or more fitted it, a negative value
fixed it at 0 (now refused).

**Non-finite realizations.** The field session's own CondSRF script
(ordinary kriging → CondSRF, 2 m grid) produced ONE all-NaN realization in a
thousand and its statistics printed NaN. Any realization with a non-finite
value is redrawn with a fresh seed (seed + 1,000,000,000 + k), counted and
reported (`non_finite_redrawn`, a warning); five failures in a row refuse the
run; NaN never reaches a percentile. The guard stays.

*The cause* (found 2026-10-20, GM-96): GSTools 1.7.0's two-dimensional
`Exponential.spectral_rad_ppf(u) = sqrt(1/u² − 1) / len` returns infinity
wherever `np.isclose(u, 0)`, i.e. for EVERY u ≤ 1e-8 (numpy's absolute
tolerance), not only at u = 0. One infinite wavenumber makes cos(k·x) NaN
at every point, so the whole randomization-method field is NaN. A field's
1,000 spectral radii come from one of GSTools' 65,535 random substreams (the
seeds 1 to 2¹⁶ its MasterRNG hands out); only substream 12837 draws such a
u (8.99e-9, mode 310). Master seeds below 200,000 that reach it: 47863 and
137080. A run seeded 47000, as the session's was, reaches it at realization
863, the session's NaN realization; `gs.SRF(Exponential, seed=47863)`
alone is NaN too. Gaussian's inverse has no such test and is finite for
every draw; the other seven models sample the spectrum by emcee MCMC.
*Fixed at the cause*: `gstools_agent.models.model_class("Exponential")` is
GSTools' class with the exact 2-D inverse for every u > 0 (u = 0, possible
but at odds of 2⁻⁵³, is clipped to 2⁻⁵³), bit-identical to GSTools for
u > 1e-8; `conditional` and `random_field` draw with it (kriging and the
variogram fit never sample the spectrum). Pinned in `tests/test_models.py`.
GSTools' ppf inverts the radial CDF F(k) = 1 − 1/√(1 + (k·len)²) at 1 − u
(the same distribution as u); the fix keeps that, so draws match. If GSTools
fixes its test, the override still draws the same fields and can go.

*Upstream issue, drafted for the owner to post (not posted):*

> **Exponential (2-D): `spectral_rad_ppf` returns inf for every u ≤ 1e-8,
> so `SRF` can return an all-NaN field.** In GSTools 1.7.0,
> `Exponential.spectral_rad_ppf` for `dim == 2` computes
> `np.divide(1, u**2, out=np.full_like(u, np.inf),
> where=np.logical_not(np.isclose(u, 0)))`. `np.isclose` uses an absolute
> tolerance of 1e-8, so every u ≤ 1e-8 maps to an infinite wavenumber, not
> only u = 0; `sqrt(1/u² − 1)/len` is finite there (about 1e8/len at
> u = 1e-8). One infinite mode makes `cos(k·x)` NaN at every point.
> Reproduce: `gs.SRF(gs.Exponential(dim=2, var=2, len_scale=40),
> seed=47863)((np.linspace(-15, 75, 46), np.linspace(0, 50, 26)),
> mesh_type="structured")` is all NaN (also seed 137080); of the 65,535
> substreams `MasterRNG` hands out, `RandomState(12837).uniform(size=1000)`
> is the one whose minimum is ≤ 1e-8. Suggested fix: test `u > 0` (or clip u
> to the smallest positive uniform draw, 2⁻⁵³) instead of `np.isclose(u, 0)`.

*The scan* (2026-10-15 and 2026-10-20): GSTools' generator alone over seeds
0–999 (1,000 modes, 2,500 points) for Spherical, Matern, Exponential and
Gaussian (2026-10-15); then the five models not scanned before -- Linear,
Stable, Rational, Cubic and HyperSpherical -- through this module's own
conditional simulation on the test fixture shaped like the field session
(seven borings on a plan, the kriged surface, the footprint total; variance
2, len_scale 40 m, 2 m cells), seeds 0–999 with both `condsrf` (the
randomization method; run in chunks of 250 seeds so none meets the 600 s
refusal) and `kriging_residual` (2026-10-20). No realization of the
five came out non-finite on either path: 1,000 kept, none redrawn, no
error, for each model and method (condsrf 7 to 17 minutes a model).

## Limits

- **Gaussian, stationary field**: one covariance model everywhere about a
  constant mean (ordinary), a known mean (simple) or a linear drift
  (universal; the variance is then the residual's). No transform (a skewed
  property should be transformed by the caller).
- **The covariance model drives the answer.** With few observations the
  variance and len_scale cannot be checked against the data; the result warns
  below 10 observations and when a region's centroid is farther than len_scale
  from every observation. Vary them to see the effect.
- **Discretisation**: cells of `cell_size` (default the smallest round size
  giving ≤ ~3,000 cells); polygon vs cell area is reported and a > 5 %
  difference warned. A cell size over len_scale / 3 is warned.
- **2D only** — plan-view fields; depth handled by separate analyses.
- **Inequality data by rejection**, as above.

## Kriging notes

- Covariance models: Gaussian, Exponential, Matern, Spherical, Linear,
  Stable, Rational, Cubic, HyperSpherical.
- Auto variogram fitting (default True for `kriging`) uses data-based lag
  classes and refuses fewer than three populated classes (live smoke GS-1).
  It fits the nugget with the variance and len_scale (unchanged by GM-96);
  `fit_nugget=false` holds the nugget at the value given and fits only those
  two (GSTools `fit_variogram(..., nugget=False)`), e.g. a nugget set from
  the measurement error or twinned borings. The result's notes and the map's
  caption say which.
- Simple kriging uses the sample mean (GSTools defaults it to 0).
- Universal kriging uses a linear drift in x and y.
- Grid auto-bounds: data bounds with a 10 % buffer.
- Kriging variance (`krige_variance`) is GSTools': the variance of the value
  AT a point, nugget included.
