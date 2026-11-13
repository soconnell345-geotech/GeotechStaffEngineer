# Geotechnical judgement calls made by the lead, 2026-11-12

The owner, 2026-11-12: "Go through backlog: if anything is waiting on a
geotechnical-engineering-judgement decision from me and is holding up coding
work downstream, go with your recommendation but make a 'Later' backlog item
with the decision for me to revisit in future rounds."

This file records each call, its basis and what it changes. The backlog item
that asks the owner to revisit them is named in `BACKLOG.md` (search for this
file's path). Every call can be reversed: each one is a default or a rule
that the module states in its result, with the other reading offered.

## The rule applied

1. Where only one printed form exists for the method in question, the module
   follows the print.
2. Where two held references print different values or rules and neither is
   named as governing, the default is the conservative one and the other is
   offered and stated.
3. Where the code already follows a printed source and states the
   alternatives, it stays as built.
4. A module never fills a missing input that the reference treats as a
   finding about the site (a rock quality, a tested value): it refuses and
   names the input.

Two calls depart from the rule, and say so: OWN-104 (the conservative
reading kept as the default although two manuals print the correction) and
OWN-131 (the print followed although it is the less conservative reading).

## A. Calls that unblock work

### OWN-101: the slope plan (GM-140)

**Call:** agree the ten steps of `module_work/slope_integral_review/ACCOUNT.md`
section 12. Priority goes to the steps that change a factor of safety:
1 (one material per point), 2 (a method's own solution or no number),
3 (the printed acceptability checks), 4 (tension cracks and reverse
curvature), 5 (water and loads), 6 (one method across the search) and
9 (rapid drawdown as EM 1110-2-1902 App. G prints it). Step 5 is split
into 5a (the pore-pressure rules, this week) and 5b (new load types: line,
inclined and varying loads, the vertical seismic coefficient; after).
Step 10 (reporting) is small and is built alongside, its method-per-family
part after step 6. Steps 8 (supports) and 7 (search coverage) follow.

**Build dependency:** step 1 first and alone; then 2, 3, 6 and 7 in series
(all edit the search and the surface screening); 4, 5a, 8, 9 and 10 in
parallel once step 1 is merged.

**Basis:** step 1 is a weight error on every model with a sloping layer
boundary (one slice 39.1 kN/m by hand against the module's 116.0); steps 2
to 6 and 9 each remove a departure from print that moves a pinned value.
Steps 7, 8 and 10 widen coverage and reporting.

**Still the owner's:** the parts of step 8 that need Duncan, Wright and
Brandon (2014) section 6.5.3 and chapter 8 wait for the owner's photographs
of those pages (local only). The rest of step 8 follows the references held.
**2026-11-13:** the owner photographed those pages (section 6.5.3, pp. 91 to
93, and chapter 8, pp. 159 to 168; kept local, cited by page and equation
number), so step 8 no longer waits on the owner.

### OWN-102: surfaces with m_alpha below 0.2

**Call:** option (b). The search keeps such a surface, flags it, and reports
where the low m_alpha was (surface and slice) with Bishop's factor of safety
beside the Ordinary Method of Slices value for the same circle. Where
Bishop's value is below the OMS value the OMS value is taken for that
circle, as Duncan (1996, p. 580) prints. Today's refusal stays as a stated
option. The exemption for phi' = 0 slices stays, stated as the module's own
reasoning.

**Limit, added 2026-11-13:** Duncan, Wright and Brandon (2014, p. 93) print
no Ordinary Method of Slices equation with known forces; the book says the
method is generally not used with seismic loading or reinforcement. So the
comparison of Bishop's value with the OMS value applies only to circles
without supports or a seismic coefficient, and the result says when the
comparison was not made and why.

**Basis:** Whitman and Bailey (1967, p. 487) print a warning and an
instruction to examine the surface, not a rejection; EM 1110-2-1902 C-10b
(pp. C-34 to C-36) prints remedies; Duncan (1996, p. 580) prints the check.
On a phi' = 0 slice the strength does not depend on the normal force, so a
small m_alpha cannot change F there; no printed source for the exemption was
found, so the result says it is the module's.

**Changes:** which circles a search counts. On the one case measured in the
account (#85, Bishop, discard) 1.508 against 1.511.

### OWN-103: reverse-curvature circles

**Call:** option (b), with a limit. A reverse-curvature circle is closed by
a vertical crack whose depth is the one EM 1110-2-1902 C-10c (p. C-36)
prints (Rankine's depth with the developed strengths, or the depth of
minimum F). A circle whose overhanging part runs deeper than that crack is
discarded, counted and located in the result. "Discard all" is a stated
option.

**Basis:** the crack the module draws today runs from the centre's level to
the ground, a depth no printed source gives. A crack is justified only to
the depth at which the soil can be in tension; below it an overhanging arc
is not a mechanism the method describes.

**Changes:** #85 with its support reads 1.372 today against the printed
1.531; with all such circles discarded 1.508. The value under this call is
measured when step 4 is built.

### OWN-104: the cos-squared correction for a sloping water table

**Call:** build it (plan step 5); default off. The module keeps the full
vertical head below the water table by default, offers
u = h_w gamma_w cos^2(alpha_w) as a stated choice, and names in every result
with a sloping water table which one it used.

**Basis:** FHWA NHI-06-088 Eq. 6-16 (p. 6-14) and UFC 3-220-10 p. 355 print
the correction. It holds for seepage parallel to a straight water table; the
full head is the lower factor of safety. A line entered as measured heads
(a piezometric line) must not be corrected at all, and the module cannot
tell which the caller means unless told.

**Departs from the rule:** two held manuals print the correction and the
default does not apply it. Reason: the uncorrected head is the safe side and
the correction's premise is a statement about the seepage that the caller
has to make.

### OWN-105: downdrag's headline neutral plane

**Call:** keep the intersection of the load and resistance curves as the
headline (UFC 3-220-20 Sec. 6-7.4 step 3, p. 477). Where the curves meet
below the depth at which the soil stops settling more than the pile, the
result's summary gives the settlement-compatibility plane and drag (Sec.
6-5.8.4.3, p. 449) beside the headline, not only in the comparison table.

**Basis:** shaft resistance is mobilised by a few millimetres of relative
movement, and a settlement profile is not known to that precision, so the
depth where "the soil stops settling more than the pile" is the weaker of
the two numbers to rest a structural load on. The printed conservative step
stays the headline; the compatible value is in view for the engineer who can
defend the settlement profile.

**Changes:** the summary text only.

### OWN-130: the recompute check on a printout that gives only gamma'

**Call:** option (a). Run it as the submerged case (water at or above the
footing base) and say so on the row.

**Basis:** a printout that gives only an effective unit weight is describing
the submerged case; this is a check of a report's own arithmetic, and the
row states the assumption. On the saved reports (a) reproduces 167 of 183
claimed values within 10 %; refusing recomputes 1.

## B. Defaults that change

### OWN-84: a drilled shaft's base in rock with no RQD given

**Call:** refuse, naming the three inputs that settle it (RQD,
`rock_base_conditions_met`, or GSI with m_i).

**Basis:** GEC-10 (2018) p. 10-22's conditions are a finding about the rock
below the base. A default RQD of 100 selects Eq. 10-23 and can double the
base resistance (89,722 kN against 47,378 kN in the probe).

### OWN-85: MSE backfill with Cu below 4 and no tested F*

**Call:** refuse and ask for `F_star_tested`.

**Basis:** GEC-11 Vol. I Table 3-6 note (i), p. 3-17: "For fill outside
these limits, tests must be run."

### OWN-82: passive wall friction when the user gives none

**Call:** (a) `sheet_pile` under log-spiral takes 17 degrees only for a
layer the user marks clean sand or gravel, otherwise 0, and says so; (b)
delta stays 0 under Coulomb; (c) `soe` keeps Rankine; (d) the coefficient
tool keeps refusing a log-spiral Kp with no wall friction given.

**Basis:** UFC 3-220-20 Fig. 4-37 step 1 (p. 237) includes wall friction
only for clean coarse-grained soil. Coulomb's passive coefficient with wall
friction is warned against at UFC 3-220-20 p. 190, USS p. 9 and Caltrans
p. 4-16.

**Changes:** (a) lowers the passive resistance of layers without cohesion
that are not marked clean.

### OWN-68: from what undrained strength a clay takes the stiff-clay p-y curve

**Call:** 48 kPa (option (b)); `stiff_clay_cu` stays a caller setting.

**Basis:** GEC-9 calls 1,000 to 2,000 psf (48 to 96 kPa) stiff (p. 219), and
COM624P's stiff-clay tables start at 47.9 kPa.

**Changes:** a stiffer response for clays of 48 to 100 kPa. Soft-clay curves
there gave the larger deflection, so this moves off the conservative side;
the print is followed because the criterion's own tables begin at 48 kPa.

### OWN-108: the aggregate-pier stress ratio n

**Call:** ask for the column type when n is not given. Stone column: 2.5.
Rammed aggregate pier: 5.

**Basis:** GEC-13 prints 2.0 to 5.0 measured for stone columns and 2.5
"often conservatively used" for preliminary design (pp. 5-31, 5-32), and 5
to 10 for rammed aggregate piers (p. 5-43). One default of 5.0 is the top of
one range and the bottom of the other.

**Changes:** at an area ratio of 0.2 a stone column's settlement reduction
factor goes from 0.56 to 0.77.

### OWN-112: the SPT overburden correction

**Call:** Liao and Whitman stays the default form, capped at 2.0 as GEC-6
Fig. 5-18 draws it. Peck's correction (GEC-12 p. 108, NHI-06-088 Eq. 3-3),
capped at 2.0, is offered and stated.

**Basis:** the cap of 1.7 comes from Youd et al. (2001), which the library
does not hold. The liquefaction procedures keep the correction their own
source prints.

**Changes:** values only where the effective overburden is below about
35 kPa.

### OWN-114: seismic passive wall friction when none is given

**Call:** phi/3.

**Basis:** NHI-06-089 p. 10-19 finds a planar passive wedge unreliable above
phi/3; UFC 3-220-20 Fig. 4-8 allows up to 0.4 phi. The lower friction gives
the lower passive resistance (at phi 30 degrees and kh 0.2, 3.500 against
3.731).

### OWN-118: where the small-group cut starts for driven piles in uplift

**Call:** UFC 3-220-20's reading (fewer than 5 piles) is the default;
GEC-12's (fewer than 3, p. 225) is given beside it.

**Basis:** rule 2. It matters only for groups of 3 or 4 driven piles
(resistance factor 0.40 against 0.50). The lead recalls AASHTO LRFD
10.5.5.2.3 as cutting by 20 % for groups of fewer than five piles; that
specification is not held, so this is a recollection to check, not a cited
basis.

### OWN-131: Meyerhof's depth factors beyond D_f/B = 1

**Call:** option (b). With Meyerhof's method the depth factors follow his
own Table 5-6 (UFC 3-220-20 p. 294, k = D_f/B at any depth). The arctangent
form stays with Brinch Hansen's method, where the manual prints it. Beyond
D_f/B = 1 the result warns that the factor grows without limit and gives
Brinch Hansen's value beside it.

**Basis:** rule 1. The arctangent form is printed only for Brinch Hansen
(p. 295, Table 5-8 note B). All 24 saved Meyerhof claims beyond D_f/B = 1
match the linear form to 0.25 % or better.

**Departs from the rule's safe side:** bearing answers for footings deeper
than they are wide rise by 6 to 16 %.

### OWN-19: moment-curvature EI for lateral piles

**Call:** build it, as GEC-9 Sec. 11.3.10 prints it, when the week's plan
has room; it becomes a build item, not a question.

**Basis:** for a reinforced concrete shaft the cracked stiffness sets the
deflection, and the two fixed choices offered today bracket LPILE's answer
on GEC-9 Example B.1 (735.6 and 691.5 kip-ft against 727).

## C. Confirmed as built (no change to any answer)

| Item | What stays | Printed basis |
|---|---|---|
| OWN-73 | Cantilever wall sliding follows GEC-6: friction for drained loading, adhesion for undrained, not both; UFC's rule offered | GEC-6 p. 100, Eqs. 5-38, 5-39 |
| OWN-79 | The step between Eq. 4-12 and UFC's charts is accepted; each result names its route | UFC 3-220-20 prints both |
| OWN-83 | A clay's alpha in downdrag from UFC's chart, with the alpha 1.0 drag beside it | UFC 3-220-20 Table 6-16, Fig. 6-14, App. B-5.4 |
| OWN-106 | Jeanjean (2009) soft clay stays offered and marked unchecked; getting the paper stays the owner's | none held |
| OWN-107 | An underconsolidated layer runs as given, with the warning | UFS 3-220-10 Sec. 5-2.3 |
| OWN-109 | Braced excavation support loads by the hinge method; the other two offered | Caltrans T&S Fig. 8-7 |
| OWN-110 | Allowable bending stress 0.60 Fy; 0.55 and 0.66 stated | Caltrans T&S examples; GEC-4 p. 81 |
| OWN-111 | `soe` and `sheet_pile` each keep their own reference's moment basis and give the other | Caltrans Example 7-3; UFC Fig. 4-41 step 5 |
| OWN-113 | Idriss and Boulanger residual strength on the lower curve | GEC-12 Fig. 7-78 |
| OWN-115 | A rigid slab rounds up to the next 1/2 inch; the Guide's own rule stated | AASHTO 1993 p. II-44 |
| OWN-117 | The base uplift's moment counts as overturning | no page held prints it; the lower factor of safety |

## Not decided here

These stay with the owner: the report-question conventions (OWN-07, OWN-88,
OWN-125, OWN-127), which define the owner's own question lists; and every
release, hosting, licensing and data item.
