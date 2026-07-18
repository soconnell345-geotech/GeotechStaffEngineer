# LE Method Equations vs Original Papers — Wave-2 Verification

**Summary: All four sources hydrated and verified against the printed originals — Bishop (1955) m_alpha/FOS (p. 10 Eqs. 12/13, p. 11 Xn-Xn+1=0), Spencer (1967) Eq. 5 + the Eq. 7/8 Ff=Fm crossing (pp. 14-15), Morgenstern-Price (1965) X = lambda f(x) E (p. 85 Eq. 18) + strength-reduction F (p. 82 Eq. 5), and EM 1110-2-1902 App. G Eq. G-12 (p. G-8) + the min(R,S) composite-envelope rule (p. G-2): provenance upgradeable SECONDARY -> PRIMARY on every core kernel; zero equation discrepancies; one flagged refinement gap (module uses Eq. G-7 Kf even when c' > 0 instead of Eq. G-8) and the known, in-code-documented items (legacy Spencer/M-P solvers are approximations; rigorous engine is the Fredlund-Krahn discrete restatement of the M-P system).**

Compared: `slope_stability/methods.py` (Bishop simplified m_alpha/FOS; legacy Spencer/M-P),
`slope_stability/gle.py` (rigorous GLE: X = lambda*f(x)*E, F_m/F_f crossing),
`slope_stability/rapid_drawdown.py` (Corps 2-stage envelope, Eq. G-12 interpolation)
against the original sources. All four source PDFs hydrated (nonzero size, opened cleanly).

---

## 1. Morgenstern & Price (1965), Geotechnique 15(1), 79-93

**Source:** `C:\Users\socon\OneDrive\Lib\_To Be Sorted\2022-11-28 Schnabel Backup\_Research Etc\VT\VT\Courses\Seepage\Resources\Papers\M_P.pdf` — HYDRATED (1,073,598 bytes); text layer extracts cleanly.

### 1a. FOS definition (strength-reduction)
Printed, **p. 82, Eq. (5)** (continues p. 83):
> dS = (1/F) [c' dx sec a + (dN') tan phi']
> "It should be noted that equation (5) also constitutes a definition of the factor of safety. The factor of safety with respect to shear strength has been adopted here. It is that value by which the shear strength parameters must be reduced in order to bring the potential sliding mass into a state of limiting equilibrium." (p. 83)

Coded (`gle.py`, `_GLESystem.solve_for_lambda` / reporting): mobilized shear
`Sm = (c*l + max(N - u*l, 0)*tan_phi) / F` — same strength-reduction FOS, per-slice.
**VERDICT: CONFIRMED-PRIMARY** (the max(., 0) tension clamp is a documented numerical guard, not in the paper).

### 1b. Interslice force function f(x)
Printed, **p. 85, Eq. (15)**: `X = lambda f(x) E'` (effective), and after defining E = E' + Pw (Eq. 16),
**p. 85, Eq. (18)**: `X = lambda f(x) E`
> "If f(x) is specified the problem is statically determinate ... The function f(x) can take any prescribed form in principle." (p. 85)

Coded (`gle.py` line 7 and `solve_for_lambda`): `X = lam * f(x) * E` with E the TOTAL interslice
normal (interslice water pressure not tracked separately) — exactly the paper's Eq. (18) form.
f(x) menu (`INTERSLICE_FUNCTIONS`: constant / half_sine / clipped_sine / trapezoidal) is
consistent with the paper's "any prescribed form" statement; the specific menu shapes follow
SLOPE/W convention (constant f(x)=1 is discussed by M-P themselves as one of the arbitrary
distributive assumptions, pp. 89-90).
**VERDICT: CONFIRMED-PRIMARY**

### 1c. General FOS system (governing equations)
Printed, **p. 84**: "the two governing differential equations are:" Eq. (2)
`X = d/dx(E' y_t') - y dE'/dx + d/dx(Pw h) - y dPw/dx` (moment) and Eq. (10)
(force, combining Eqs. (3)-(9) — vertical/normal equilibrium + Mohr-Coulomb with F).
M-P then discretize into finite slices with linear variation within each slice
(p. 85, Eqs. (19)-(21)) and solve for F and lambda satisfying boundary conditions
(E, X -> 0 at the ends).

Coded (`gle.py`): the Fredlund-Krahn (1977) slice-discrete restatement of the same system —
base normal N from vertical slice equilibrium including interslice shear difference dV,
E marched slice-to-slice from horizontal equilibrium, F_m (moment) and F_f (force) iterated,
lambda found where F_m = F_f; end conditions E[0] = 0, V[0] = V[n] = 0 enforced
(module docstring cites F&K 1977 explicitly).
The differential-equation vs finite-slice difference is a discretization choice, not a
different mechanical model: identical statics (both equilibrium conditions), identical
FOS definition, identical closure assumption X = lambda f(x) E, identical boundary
conditions. M-P themselves solve by dividing into finite slices (p. 85).
**VERDICT: EQUIVALENT-NOTATION** (M-P system solved in the Fredlund-Krahn discrete form;
upgrade provenance to PRIMARY for the assumption/definition, with F&K 1977 as the stated
solution-scheme reference — already cited in the docstring).

---

## 2. Spencer (1967), Geotechnique 17(1), 11-26

**Source:** `C:\Users\socon\OneDrive\Lib\_To Be Sorted\2022-11-28 Schnabel Backup\_Research Etc\VT\VT\Courses\Seepage\Resources\Papers\Spencer.pdf` — HYDRATED (1,152,808 bytes); text layer extracts; equation page render checked (see 2d).

### 2a. Interslice resultant Q — the "shifted m_alpha"
Printed, **p. 14, Eq. (5)**:
> Q = [ c'b/F sec a + (tan phi'/F)(W cos a - u b sec a) - W sin a ] / [ cos(a - theta) * (1 + (tan phi'/F) tan(a - theta)) ]

Note the denominator: `cos(a-theta)*[1 + (tan phi'/F) tan(a-theta)] = cos(a-theta) + sin(a-theta)*tan(phi')/F`.

Coded (`methods.py`, `spencer_fos_legacy`): `m_alpha = cos(alpha - theta) + sin(alpha - theta)*tan_phi/F`
— algebraically identical to Spencer's printed denominator. The legacy resisting/driving split
(docstring SS-1) is an engineering rearrangement of Eq. (5), validated vs Duncan-Wright-Brandon
examples, and is honestly labeled "APPROXIMATION" in the docstring.
**VERDICT: CONFIRMED-PRIMARY for the m_alpha(a-theta) kernel; the legacy solver's
resist/drive split remains an approximation of Spencer's Q-recursion (as documented in-code).**

### 2b. Parallel-interslice two-equation formulation
Printed, **p. 14-15**:
- Force: Eq. (7a) `Sum[Q cos theta] = 0`, Eq. (7b) `Sum[Q sin theta] = 0`;
  with theta constant (parallel forces) these collapse to **Eq. (7)** `Sum Q = 0` (p. 15).
- Moment: **Eq. (8)** `Sum[Q cos(a - theta)] = 0` (p. 15; from Sum[Q R cos(a-theta)] = 0, R constant for a circle).
- Solution: "Values of F obtained using the force equilibrium equation (7) are designated F_f, and
  those obtained using the moment equilibrium equation (8) as F_m ... The intersection of the two
  curves gives the value of the factor of safety (F_i) which satisfies both equations (7) and (8)
  and the corresponding slope (theta_i) of the inter-slice forces." (p. 15)

Coded (rigorous path, `gle.py` via `spencer_fos`): f(x) = constant (theta uniform), F_m and F_f
computed separately, lambda (= tan theta) found where F_m = F_f by bracketed root find — exactly
Spencer's crossing-point construction, generalized to noncircular surfaces by the GLE moment axis.
Coded (legacy path, `spencer_fos_legacy`): secant iteration on theta until FOS_moment = FOS_force —
same two-equation crossing idea.
**VERDICT: CONFIRMED-PRIMARY** (Spencer states it for circular surfaces; the noncircular
extension via fitted moment axis is Fredlund-Krahn/GLE, cited in-code).

### 2c. Zero-interslice special case
Printed, **p. 14, Eq. (4)**: `F = Sum[c'b sec a + tan phi'(W cos a - u b sec a)] / Sum[W sin a]`
(inter-slice forces ignored) — the paper's statement of the Fellenius/OMS-type baseline used to
seed the iteration. Coded `fellenius_fos` uses the equivalent `c'*dl + (W cos a - u*dl) tan phi'`
resisting sum over `Sum W sin a` (dl = b sec a). Same expression.
**VERDICT: CONFIRMED-PRIMARY** (supporting detail).

### 2d. Render check of the Eq. (5) page
`spencer_p04.png` (dpi 200 render of the p. 14 page, prior agent) read visually. Printed Eq. (5)
confirmed character-for-character:
> Q = [ (c'b/F) sec a + (tan phi'/F)(W cos a - u b sec a) - W sin a ] / { cos(a-theta) [ 1 + (tan phi'/F) tan(a-theta) ] }
plus Eq. (2) `P = P' + u b sec a`, Eq. (3) `S_m = (c'b/F) sec a + P' tan phi'/F` (mobilized
shear = strength/F, same FOS definition as coded), Eq. (4), Eqs. (7a)/(7b) — all as quoted
in 2a-2c from the text layer. No discrepancy between text-layer and printed page.

---

## 3. Bishop (1955) — "The Use of the Slip Circle in the Stability Analysis of Slopes"

**Source:** `C:\Users\socon\OneDrive\Lib\01\1. Geotechnical\8. Soil Mechanics\_Unsorted papers\1954 The Use of The Slip Circle in The Stability Analysis of Slopes - Bishop.pdf` — HYDRATED (443,946 bytes); scanned copy of the Geotechnique 5(1) 7-17 paper (printed page numbers 7-17 visible), no usable text layer; verified from page renders `bishop_p04.png` (printed p. 10) and `bishop_p05.png` (printed p. 11).

### 3a. m_alpha (base-normal denominator)
Printed, **p. 10, Eq. (12)**:
> P' = [ W + X_n - X_{n+1} - l ( u cos a + (c'/F) sin a ) ] / [ cos a + (tan phi' sin a) / F ]

Denominator `cos a + tan phi' sin a / F` is exactly the coded
`m_alpha = cos(alpha) + sin(alpha)*tan_phi/fos` (`methods.py` bishop_fos, line 264;
`gle.py` `_normal`, line 330). The numerator is exactly the coded GLE base-normal equation
(`gle.py` `_normal`): `[W - dV - (c*l - u*l*tan_phi)*sin(a)/F] / m_alpha` — identical after
noting Bishop's P' is the EFFECTIVE normal (u l already removed): Bishop's
`W + (Xn - Xn+1) - l*u*cos a - l*(c'/F) sin a` over m_alpha equals N' where the coded N is
TOTAL normal (N' = N - u l); substituting N = N' + u l reproduces the coded numerator
term-for-term (u*l*tan_phi*sin(a)/F appears from u l moved through m_alpha).
**VERDICT: CONFIRMED-PRIMARY** (dV = V_i - V_{i+1} is Bishop's X_n - X_{n+1}).

### 3b. Bishop rigorous FOS and the Simplified assumption
Printed, **p. 10, Eq. (13)**:
> F = (1/[Sum W sin a]) * Sum[ { c'b + tan phi' (W(1 - B-bar) + (X_n - X_{n+1})) } * sec a / (1 + tan phi' tan a / F) ]
(with u = B-bar (W/b), Eq. (10), so W(1 - B-bar) = W - u b) and printed, **p. 11**:
> "In practice, an initial value of F is obtained by solving equation (13) on the assumption that (X_n - X_{n+1}) = 0 throughout. ... although there are a number of different distributions of (X_n - X_{n+1}) which satisfy equation (18), the corresponding variations in the value of F are found to be insignificant (less than 1% in a typical case)."

Coded (`methods.py` bishop_fos):
`FOS = Sum[(c'*b + max(W - u*b, 0)*tan_phi) / m_alpha] / Sum[W sin a]` with
`sec a / (1 + tan a tan phi'/F) = 1/(cos a + sin a tan phi'/F) = 1/m_alpha` — the same
identity, i.e. coded Bishop = printed Eq. (13) with X_n - X_{n+1} = 0 (the Simplified
assumption stated on p. 11) and general u*b in place of the B-bar pore-pressure
parameterization. The driving sum uses the moment-arm form `W*(x_mid - xc)/R`, identical to
`W sin a` for a circle (Bishop's own substitution `x = R sin a`, p. 10 above Eq. (9)).
Differences vs print: (i) the `max(..., 0)` artesian clamp (documented SS-2 numerical guard);
(ii) the d_sign slope-direction mirror (SS-4, geometry convention); (iii) seismic/crack/pond
terms (modern extensions, not in the 1955 paper) — all documented in-code, none alters the
printed kernel.
**VERDICT: CONFIRMED-PRIMARY**

### 3c. Fellenius/conventional baseline
Printed, **p. 10, Eq. (9)**: `F = (1/[Sum W sin a]) * Sum[c'l + tan phi' (W cos a - u l)]`.
Coded (`methods.py` fellenius_fos): resisting `c*dl + max(W cos a - u*dl, 0)*tan phi'` over
driving `Sum W sin a` (moment-arm form for circles) — identical kernel.
**VERDICT: CONFIRMED-PRIMARY** (supporting detail).

---

## 4. EM 1110-2-1902 (31 Oct 03), Appendix G — Rapid Drawdown

**Source:** `C:\Users\socon\OneDrive\Lib\02 Tech\Military\ACoE\EM_1110-2-1902 Slope Stability.pdf` — HYDRATED (6,397,896 bytes); text layer extracts cleanly.

### 4a. Corps 1970 composite (min R, S) envelope — 2-stage procedure
Printed, **p. G-2, para G-2b**:
> "The shear strengths are estimated from a 'composite,' bilinear shear strength envelope. The envelope represents the lower bound of the R and S strength envelopes."
and **para G-2b(2)**: "The composite envelope ... represents the lower bound of the empirical R envelope described above, and the effective stress S envelope. Shear strengths are determined for the second-stage computations using the effective normal stress calculated for the first stage ... Shear strengths are determined in this manner for each slice whose base lies in material that does not drain freely."
**para G-2c** (p. G-2): second stage assigns these strengths "as values of cohesion, c, with phi equal to zero"; free-draining materials keep c', phi' with post-drawdown pore pressures; still-submerged face carries external water loads.

Coded (`rapid_drawdown.py`, `method='corps_2stage'`, lines 404-408):
`s_R = R_c + sigma_fc*tan(R_phi)`; `s_D = c' + sigma_fc*tan(phi')`; `s_und = min(s_R, s_D)`,
applied at the stage-1 effective normal stress, assigned as c = tau_ff with phi = 0
(free-draining layers `R_phi is None` keep drained strength; drawn-down pool applied as
external load + hydrostatic u). This is exactly the printed lower-bound-of-R-and-S rule
evaluated at sigma'_fc, per slice, undrained slices only.
**VERDICT: CONFIRMED-PRIMARY**

### 4b. Eq. G-12 (Kc interpolation, improved / Lowe-Karafiath / DWW procedure)
Printed, **p. G-8, Eq. (G-12)** (garbled text layer un-scrambled; structure unambiguous):
> tau_ff = [ (K_f - K_c) * tau_ff(Kc=1) + (K_c - 1) * tau_ff(Kc=Kf) ] / (K_f - 1)
with tau_ff(Kc=1) = d + sigma'_c tan(psi) from the R envelope (Eq. G-9) and
tau_ff(Kc=Kf) = c' + sigma'_c tan(phi') from the effective-stress envelope (Eq. G-10);
K_f from Eq. G-7 `(1+sin phi')/(1-sin phi')` (c'=0) or G-8 (c' != 0);
K_c from Eq. G-11 assuming principal-stress orientation at consolidation = at failure
(attributed to Lowe & Karafiath 1960).

Coded (`rapid_drawdown.py`, lines 416-421):
`w = (Kc-1)/(Kf-1)` (clamped to [0,1]); `s_und = s_R + w*(s_D - s_R)`.
Algebraic identity: `s_R + (Kc-1)/(Kf-1)*(s_D - s_R) = [(Kf-Kc)*s_R + (Kc-1)*s_D]/(Kf-1)` = Eq. G-12 exactly.
Kf coded as `(1+sin phi)/(1-sin phi)` = Eq. G-7 (module uses the c'=0 form; a substantive-but-
conservative-direction simplification vs Eq. G-8 for c' > 0 soils — flagged below).
Kc coded in `_consolidation_Kc` from the base-plane (sigma'_fc, tau_fc) pair assuming the major
principal consolidation stress is vertical — same physical assumption as Eq. G-11
("orientation of principal stresses during consolidation is the same as at failure" per
Lowe-Karafiath; the EM's G-11 closed form and the module's Mohr-circle back-solve agree for
the same assumed orientation). Stage-3 drained substitution (Eqs. G-13/G-14/G-15 + the
"if drained < undrained, substitute and recompute" rule, pp. G-8/G-9) matches
`duncan_3stage` stage 3 (`sigma'_post = N/l - u`, `s_d = c' + sigma'_post tan phi'`).
**VERDICT: EQUIVALENT-NOTATION** (Eq. G-12 itself: exact algebraic identity ->
CONFIRMED-PRIMARY; Kf uses the G-7 (c'=0) closed form rather than the G-8
stress-dependent form, and Kc is obtained by Mohr-circle back-solve rather than the G-11
closed form — same assumption, same result where c'=0; classify the Kf simplification as a
minor substantive difference for c' > 0 low-perm layers: the G-7 Kf is SMALLER than the G-8
value when c' > 0, which raises the interpolation weight w and moves tau_ff toward the
drained (upper) envelope — slightly unconservative in the usual s_D > s_R case, bounded by
the drained envelope via the w <= 1 clamp and, for `duncan_3stage`, caught by the stage-3
drained-substitution check. Recommend a follow-up ticket to add the Eq. G-8
stress-dependent Kf for c' > 0 layers; not a wrong equation, an omitted refinement).

---
