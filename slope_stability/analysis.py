"""
Top-level slope stability analysis orchestrator.

Provides analyze_slope() and search_critical_surface() as the
primary entry points, following the project pattern of analyze_*()
functions returning result dataclasses.

References:
    Duncan, Wright & Brandon (2014) — Soil Strength and Slope Stability
"""

import math
from typing import Optional, Tuple

from slope_stability.cracks import (
    counts_reverse_discards as _counts_reverse_discards,
    crack_choices as _crack_choices, discards_so_far as _discards_so_far,
)
from slope_stability.geometry import SlopeGeometry
from slope_stability.slip_surface import CircularSlipSurface
from slope_stability.slices import build_slices, compute_slice_forces
from slope_stability.methods import (
    M_ALPHA_MIN, m_alpha_values, m_alpha_checked, sliding_sign,
    check_negative_term, negative_term, negative_term_in_force,
)
from slope_stability.solution import (
    solve_method, check_method, check_interslice, METHOD_TEXT,
)
from slope_stability.search import (
    grid_search, search_noncircular,
    search_pso, search_weak_layer_biased, search_entry_exit, search_de,
    _counts_refusals, _ordered_windows, _applies_m_alpha_rule,
    m_alpha_rule_in_force, search_budget,
)
from slope_stability.acceptability import (
    surface_findings, findings_warnings, C10B_REMEDIES,
)
from slope_stability.results import (
    SlopeStabilityResult, SliceData, SearchResult, InfiniteSlopeResult,
)


def analyze_slope(geom: SlopeGeometry,
                  xc: float = None,
                  yc: float = None,
                  radius: float = None,
                  slip_surface=None,
                  method: str = "bishop",
                  n_slices: int = 30,
                  tol: float = 1e-4,
                  f_interslice: str = "half_sine",
                  include_slice_data: bool = False,
                  compare_methods: bool = False,
                  m_alpha_rule: str = "frictional",
                  negative_friction_term: Optional[str] = None,
                  ) -> SlopeStabilityResult:
    """Run slope stability analysis for a specified slip surface.

    Parameters
    ----------
    geom : SlopeGeometry
        Complete slope definition.
    xc : float, optional
        Circle center x-coordinate (m). Used with yc/radius for circular.
    yc : float, optional
        Circle center z-coordinate / elevation (m).
    radius : float, optional
        Circle radius (m).
    slip_surface : CircularSlipSurface or PolylineSlipSurface, optional
        Explicit slip surface object. If provided, xc/yc/radius are ignored.
    method : str
        'fellenius', 'bishop', 'janbu', 'spencer', 'morgenstern_price'
        or 'gle'. 'gle' is an alias for the rigorous Morgenstern-Price
        solution with a selectable interslice force function. 'janbu'
        reports the f0-corrected FOS (uncorrected value in
        FOS_janbu_uncorrected). Default 'bishop'.
    n_slices : int
        Number of slices. Default 30.
    tol : float
        Convergence tolerance for iterative methods. Default 1e-4.
    f_interslice : str
        Interslice force function for 'morgenstern_price' / 'gle':
        'constant', 'half_sine' (default), 'clipped_sine', 'trapezoidal'.
    include_slice_data : bool
        If True, include per-slice breakdown in results.
    compare_methods : bool
        If True, compute FOS for all four methods (Fellenius, Bishop,
        Spencer, Morgenstern-Price). For noncircular surfaces, Bishop
        is skipped (requires circular).
    m_alpha_rule : str
        Which slices the m_alpha < 0.2 check looks at (Whitman and Bailey
        1967, p. 487): 'frictional' (default), those with phi' > 0 (the
        module's own reasoning, no printed source: on a phi' = 0 slice
        m_alpha does not depend on F and the strength does not depend on
        the normal force); 'every_slice', every slice.
    negative_friction_term : str, optional
        Bishop's simplified method and the Ordinary Method of Slices, on a
        slice where the pore-water force exceeds the weight: 'as_printed'
        (default), W - u b (Bishop 1955 Eq. 13, p. 10) and W cos(alpha) -
        u l (Eq. 9) used as printed, negative; 'zero', the term set to zero
        there (the module's rule before GM-140 step 3; no printed source).
        Either way the result names the slices
        (``methods.NEGATIVE_TERM_RULES``). None (default) takes the rule in
        force, which is 'as_printed' unless a search was given another.

    Returns
    -------
    SlopeStabilityResult
        ``warnings`` says what the engineer must check on this surface
        (GM-81): a reverse-curvature circle closed by a crack, with the
        crack's depth; supports crossing a Bishop / OMS circle with a short
        arm, with Spencer's FOS on the circle; a method with no solution;
        and each printed acceptability check the surface does not meet
        (``acceptability``, below).

    The printed acceptability checks (GM-140 step 3;
    ``slope_stability.acceptability``). The surface is analysed whatever
    they find, and ``acceptability`` lists each finding with its check, its
    source, the slices or slice sides and the values; ``flags`` names the
    checks not met: m_alpha below 0.2 on a slice (Whitman and Bailey 1967,
    p. 487; the warning names EM 1110-2-1902 C-10b's remedies, none is
    applied); a negative frictional term in Bishop's method or the
    Ordinary Method of Slices; and, for Spencer, Morgenstern-Price and GLE,
    a negative effective normal force on a slice base, a tensile
    interslice force on a slice side and the line of thrust outside the
    sliding mass (EM 1110-2-1902 C-10a, pp. C-33 to C-34). For Bishop's
    method on a circle with no supports and no seismic coefficient the
    Ordinary Method of Slices value for the same circle is computed
    (``FOS_fellenius``; Bishop's own value in ``FOS_bishop``); where
    Bishop's is the lower, ``FOS`` is the Ordinary Method of Slices value
    and ``method_used`` says so (Duncan 1996, p. 580). With supports or a
    seismic coefficient the comparison is not made and the result says
    why. ``thrust_line`` is clipped to the sliding mass for display;
    ``thrust_line_unclipped`` is the line as solved.

    A method's own solution, or no number (GM-140 step 2; UFC 3-220-10
    p. 362). ``FOS`` is the named method's own converged solution on the
    surface (``solution.solve_method``, the solver the searches use too).
    Where the method has none (Spencer / Morgenstern-Price / GLE find no
    lambda at which the moment and force factors of safety agree; Bishop's
    or Janbu's iteration does not converge; the surface has no driving
    moment or force) ``FOS`` is None, ``method_solved`` is False,
    ``method_used`` is None and ``warnings`` says why. Nothing stands in
    for the method: the old approximate Spencer and Morgenstern-Price
    solvers are gone. A compare_methods value with no solution is None and
    named in ``methods_without_solution``.

    An unknown ``method`` or ``f_interslice`` is refused with the list of
    names (it used to run Bishop's simplified method, or to be accepted and
    not used).
    """
    check_method(method)
    check_interslice(f_interslice)      # whatever the method (item 7)
    if negative_friction_term is not None:
        check_negative_term(negative_friction_term)
        if negative_friction_term != negative_term_in_force():
            with negative_term(negative_friction_term):
                return analyze_slope(
                    geom, xc=xc, yc=yc, radius=radius,
                    slip_surface=slip_surface, method=method,
                    n_slices=n_slices, tol=tol, f_interslice=f_interslice,
                    include_slice_data=include_slice_data,
                    compare_methods=compare_methods,
                    m_alpha_rule=m_alpha_rule)
    # Build slip surface from either explicit object or xc/yc/radius
    if slip_surface is not None:
        slip = slip_surface
    else:
        if xc is None or yc is None or radius is None:
            raise ValueError(
                "Either provide slip_surface or all of xc, yc, radius"
            )
        slip = CircularSlipSurface(xc, yc, radius)

    is_circular = getattr(slip, 'is_circular', True)
    x_entry, x_exit = slip.find_entry_exit(geom)

    # The surface's crack as printed (GM-140 step 4, ``cracks``). A depth
    # that depends on the surface's own factor of safety (Rankine's with
    # the developed strengths, the depth of minimum F, or the crack closing
    # a reverse-curvature circle) is solved for on copies of the geometry
    # with the crack fixed; a circle the printed crack does not close has
    # no factor of safety.
    from slope_stability import cracks
    crack_out = None
    if cracks.may_need_solution(geom, slip):
        plan = cracks.plan_crack(geom, slip, x_entry, x_exit)
        if plan is not None and (plan.needs_solution or plan.discard):
            result, crack_out = cracks.solve_with_crack(
                geom, slip,
                lambda g: analyze_slope(
                    g, slip_surface=slip, method=method, n_slices=n_slices,
                    tol=tol, f_interslice=f_interslice,
                    include_slice_data=include_slice_data,
                    compare_methods=compare_methods,
                    m_alpha_rule=m_alpha_rule),
                fos_of=lambda r: r.FOS, plan=plan)
            if crack_out.discarded:
                raise cracks.refusal_error(crack_out)
            cracks.apply_to_result(result, crack_out)
            return result
    if getattr(geom, "_crack_fixed", None) is None:
        crack_out = cracks.given_outcome(geom, slip, x_entry, x_exit)
    slices = build_slices(geom, slip, n_slices)

    # Reinforcement crossings (nails / anchors / geosynthetics)
    from slope_stability.reinforcement import compute_reinforcement_forces
    reinf_forces = compute_reinforcement_forces(geom, slip, x_entry, x_exit)
    if not reinf_forces:
        reinf_forces = None

    # For result: store circle params if circular, zeros otherwise
    r_xc = slip.xc if is_circular else 0.0
    r_yc = slip.yc if is_circular else 0.0
    r_radius = slip.radius if is_circular else 0.0
    r_slip_points = None if is_circular else list(slip.points)

    # Statements the engineer must check on this surface (GM-81)
    warnings_out = []
    no_solution = []       # rigorous methods whose FOS field is not solved
    method_solved = True
    method_used = ""       # "" = the method itself

    # For noncircular, force Spencer (Bishop doesn't work)
    if not is_circular and method == "bishop":
        method = "spencer"
        warnings_out.append(
            "Bishop's simplified method takes moments about the centre of a "
            "circle; this noncircular surface is analysed with Spencer's "
            "method instead.")

    # Primary FOS: the method's own solution, or no number (GM-140 step 2)
    labels = {"fellenius": "Fellenius", "janbu": "Janbu",
              "spencer": "Spencer", "morgenstern_price": "Morgenstern-Price",
              "gle": "GLE", "bishop": "Bishop"}
    method_name = labels[method]
    sols = {}

    def _sol(m):
        if m not in sols:
            sols[m] = solve_method(slices, slip, m, f_interslice, tol,
                                   reinf_forces)
        return sols[m]

    def _missing(m, field=""):
        # the name methods_without_solution carries (GLE is M-P's solve)
        name = "Morgenstern-Price" if m == "gle" else labels[m]
        if name not in no_solution:
            no_solution.append(name)
        warnings_out.append(_no_solution_warning(
            labels[m], _sol(m), reinf_forces, field=field, method=m))

    primary = _sol(method)
    fos = primary.fos
    gle_result = primary.gle   # rich GLE state for rigorous methods
    if not primary.solved:
        method_solved = False
        method_used = None
        _missing(method)
    theta_spencer = primary.param if method == "spencer" else None
    fos_spencer = fos if method == "spencer" else None
    rig = method in ("morgenstern_price", "gle")
    fos_mp = fos if rig else None
    lambda_mp = primary.param if rig else None
    fos_janbu = fos if method == "janbu" else None
    fos_janbu_uncorr = primary.fos_uncorrected if method == "janbu" else None
    f0_janbu = primary.f0 if method == "janbu" else None

    # Comparison FOS values (each its method's own solution, or None)
    fos_fellenius = None
    fos_bishop = None
    if method == "bishop" and primary.solved:
        # Bishop's own value beside the Ordinary Method of Slices value for
        # the same circle (Duncan 1996, p. 580; GM-140 step 3). Where
        # Bishop's is the lower, ``fos`` is already the latter.
        fos_bishop = primary.fos_bishop
        fos_fellenius = primary.fos_oms
        if primary.oms_taken:
            method_used = ("Ordinary Method of Slices value (Bishop's "
                           "value is below it; Duncan 1996, p. 580)")
    if compare_methods:
        fos_fellenius = _sol("fellenius").fos
        if not _sol("fellenius").solved and method != "fellenius":
            _missing("fellenius", "FOS_fellenius")
        if is_circular:
            # (Bishop's own value: the comparison column is each method's)
            fos_bishop = (_sol("bishop").fos_bishop
                          if _sol("bishop").solved else None)
            if not _sol("bishop").solved and method != "bishop":
                _missing("bishop", "FOS_bishop")
        if method != "spencer":
            fos_spencer = _sol("spencer").fos
            theta_spencer = _sol("spencer").param
            if not _sol("spencer").solved:
                _missing("spencer", "FOS_spencer")
        if not rig:
            mp = _sol("morgenstern_price")
            fos_mp, lambda_mp = mp.fos, mp.param
            if not mp.solved:
                _missing("morgenstern_price", "FOS_morgenstern_price")
        if method != "janbu":
            jb = _sol("janbu")
            fos_janbu, fos_janbu_uncorr, f0_janbu = (
                jb.fos, jb.fos_uncorrected, jb.f0)
            if not jb.solved:
                _missing("janbu", "FOS_janbu")

    # Slice data (per-slice force table) and thrust line
    slice_data = None
    thrust_line = None
    thrust_raw = None
    if gle_result is not None and gle_result.boundary_x:
        # the line as solved, not clipped (EM 1110-2-1902 C-10a; D50)
        thrust_raw = [(bx, tz) for bx, tz in zip(
            gle_result.boundary_x, gle_result.thrust_elevation)]
    if include_slice_data:
        slice_data = []
        rich = gle_result  # rigorous per-slice/boundary forces, if any
        for i, s in enumerate(slices):
            sf = compute_slice_forces(s)
            dl = s.base_length
            U = s.pore_pressure * dl
            if rich is not None:
                # rigorous base forces at the converged GLE state
                N_eff = rich.base_normal_eff[i]
                S_mob = rich.shear_mobilized[i]
                T_avail = S_mob * rich.fos
                E_l, E_r = rich.interslice_E[i], rich.interslice_E[i + 1]
                X_l, X_r = rich.interslice_X[i], rich.interslice_X[i + 1]
            else:
                # Fellenius decomposition (no interslice forces)
                N_eff = sf.N_prime
                S_mob = sf.S_mobilized
                T_avail = sf.T_available
                E_l = E_r = X_l = X_r = None
            sigma_n = N_eff / dl if dl > 0 else 0.0
            tau_mob = S_mob / dl if dl > 0 else 0.0
            tau_avail = T_avail / dl if dl > 0 else 0.0
            slice_data.append(SliceData(
                x_mid=s.x_mid,
                z_top=s.z_top,
                z_base=s.z_base,
                width=s.width,
                height=s.height,
                alpha_deg=math.degrees(s.alpha),
                weight=s.weight,
                pore_pressure=s.pore_pressure,
                c=s.c,
                phi=s.phi,
                base_length=s.base_length,
                normal_stress_kPa=sigma_n,
                shear_stress_kPa=tau_mob,
                shear_resistance_kPa=tau_avail,
                in_tension_crack=s.in_tension_crack,
                N_eff_kN=N_eff,
                S_mob_kN=S_mob,
                U_base_kN=U,
                E_left_kN=E_l,
                E_right_kN=E_r,
                X_left_kN=X_l,
                X_right_kN=X_r,
            ))
        if rich is not None and rich.boundary_x:
            # Clamp the line of thrust to the physical section (between the slip
            # surface and the ground) for display. Near the slip exit the
            # interslice normal E -> 0, so the point of application
            # z = z_base - m/E (gle.py) can spike to a non-physical elevation.
            # The line as solved is ``thrust_line_unclipped``, and the
            # ``thrust_outside`` finding says where it left the sliding
            # mass (GM-140 step 3, D50).
            thrust_line = []
            for bx, tz in zip(rich.boundary_x, rich.thrust_elevation):
                z_slip = slip.slip_elevation_at(bx)
                z_ground = geom.ground_elevation_at(bx)
                if z_slip is not None and z_ground is not None:
                    lo, hi = ((z_slip, z_ground) if z_slip <= z_ground
                              else (z_ground, z_slip))
                    tz = min(max(tz, lo), hi)
                thrust_line.append((bx, tz))

    result = SlopeStabilityResult(
        FOS=fos,
        method=method_name,
        method_solved=method_solved,
        method_used=method_used,
        methods_without_solution=no_solution,
        warnings=warnings_out,
        layer_overlaps=geom.layer_overlaps(),
        xc=r_xc,
        yc=r_yc,
        radius=r_radius,
        x_entry=x_entry,
        x_exit=x_exit,
        theta_spencer=theta_spencer,
        FOS_fellenius=fos_fellenius,
        FOS_bishop=fos_bishop,
        FOS_spencer=fos_spencer,
        FOS_morgenstern_price=fos_mp,
        lambda_mp=lambda_mp,
        FOS_janbu=fos_janbu,
        FOS_janbu_uncorrected=fos_janbu_uncorr,
        janbu_f0=f0_janbu,
        reinforcements=[
            {
                "kind": f.kind, "index": f.index,
                "x_m": round(f.x, 3), "z_m": round(f.z, 3),
                "T_kN_per_m": round(f.T, 2),
                "controlled_by": f.controlled_by,
                "support_convention": "passive" if f.passive else "active",
            } for f in (reinf_forces or [])
        ] or None,
        n_slices=len(slices),
        has_seismic=geom.kh > 0,
        kh=geom.kh,
        slice_data=slice_data,
        thrust_line=thrust_line,
        thrust_line_unclipped=thrust_raw,
        slip_points=r_slip_points,
        tension_crack_depth=geom.tension_crack_depth,
        tension_crack_water_depth=geom.tension_crack_water_depth,
    )

    # The printed acceptability checks (GM-140 step 3): every finding goes
    # on the result, with the slice or side and the value; a check not met
    # is a warning. The surface is analysed whatever they find.
    result.acceptability = surface_findings(geom, slip, slices, primary,
                                            rule=m_alpha_rule)
    result.warnings.extend(findings_warnings(result.acceptability))

    # What the searches say about their governing circle, said about this
    # one too (GM-59, GM-81): a reverse-curvature circle closed by a crack,
    # and supports Bishop / the OMS count little of.
    cracks.apply_to_result(result, crack_out)
    if is_circular:
        from slope_stability.search_supplements import _support_arm_warning
        if reinf_forces and method in ("bishop", "fellenius"):
            def _spencer_here():
                return _sol("spencer").fos
            w = _support_arm_warning(geom, result, method, n_slices, tol,
                                     spencer_fn=_spencer_here,
                                     in_search=False)
            if w:
                result.warnings.append(w)
    return result


# Plain names of the rigorous methods, for the warnings.
_METHOD_PHRASE = {
    "Spencer": "Spencer's method",
    "Morgenstern-Price": "The Morgenstern-Price method",
    "GLE": "The GLE (Morgenstern-Price) method",
}


def m_alpha_statement(slices, slip, method: str, fos,
                      fos_janbu_uncorrected=None,
                      given: Optional[str] = "analyze_slope",
                      rule: Optional[str] = None,
                      ) -> Optional[str]:
    """Say so when m_alpha = cos(a) (1 + tan(a) tan(phi) / F) is below 0.2 on
    a slice of this surface at the method's own F (GM-127), or None.

    m_alpha is the denominator of the normal force on a slice base in
    Bishop's simplified method, Janbu's simplified method and the GLE
    behind Spencer, Morgenstern-Price and GLE (`methods.m_alpha_values`);
    the Ordinary Method of Slices has none. Only the slices ``rule`` looks
    at count (`methods.m_alpha_checked`; default the rule in force, which
    leaves out slices with phi' = 0). Whitman and Bailey (1967, p. 487)
    print a warning for such a surface; a search keeps it and flags it, or
    leaves it out under ``low_m_alpha='refuse'`` (GM-140 step 3). A surface
    given to a single-surface analysis (``given``, its name) is analysed
    and this says so. `acceptability.low_m_alpha_finding` is the finding
    `analyze_slope` reports; this statement is the rapid-drawdown
    result's.
    """
    if method == "fellenius" or fos is None or not (0.0 < fos < 900.0):
        return None
    rule = rule or m_alpha_rule_in_force()
    which = ("gle" if method in ("spencer", "morgenstern_price", "gle")
             else "janbu" if method == "janbu" else "bishop")
    f = (fos_janbu_uncorrected if method == "janbu"
         and fos_janbu_uncorrected else fos)
    vals = m_alpha_values(slices, f, slip, which)
    checked = [i for i, s in enumerate(slices) if m_alpha_checked(s, rule)]
    low = [i for i in checked if vals[i] < M_ALPHA_MIN]
    if not low:
        return None
    i = min(checked, key=vals.__getitem__)
    s = slices[i]
    passive = sliding_sign(slices, slip, which) * s.alpha < 0
    where = (f"{vals[i]:.2f} on the slice at x = {s.x_mid:.2f} (base "
             f"inclined {abs(math.degrees(s.alpha)):.0f} deg "
             + ("against" if passive else "with")
             + f" the sliding, phi = {s.phi:g} deg)")
    if len(low) > 1:
        where += f", and below 0.2 on {len(low)} of {len(vals)} slices"
    if rule != "every_slice" and len(checked) < len(vals):
        where += " (slices with phi = 0 are not checked)"
    zone = ("there the surface comes up to the ground steeply in the "
            "passive zone" if passive else
            "there the base is nearly vertical where the surface enters")
    what = (f"This surface, given to {given}, is analysed and its factor "
            f"of safety reported."
            if given else "A search keeps such a surface and flags it.")
    return (f"m_alpha = cos(a) + sin(a) tan(phi) / F, the denominator "
            f"of the normal force on a slice base, is {where} at F = "
            f"{f:.3f}: {zone}. Where it is below 0.2 on any slice the user "
            f"must examine the result in detail (Whitman and Bailey 1967, "
            f"p. 487). {what} A search keeps such surfaces and flags them, "
            f"or leaves them out under low_m_alpha='refuse'. Finer slices "
            f"lower m_alpha at a steep end rather than clear it. "
            + C10B_REMEDIES + ".")


def _no_solution_warning(method_name: str, sol, reinf_forces,
                         field: str = "", method: str = "") -> str:
    """The plain statement that a method has no solution on this surface,
    and that no number is reported for it (GM-81, GM-140 step 2).
    ``field`` names the compare_methods field it concerns ('' = the
    headline FOS)."""
    phrase = METHOD_TEXT.get(method) or _METHOD_PHRASE.get(method_name,
                                                           method_name)
    if field:
        phrase += " (method comparison)"
    rigorous = method in ("spencer", "morgenstern_price", "gle")
    head = (f"{phrase} finds no equilibrium solution on this surface: "
            f"{sol.reason}." if rigorous else
            f"{phrase} has no solution on this surface: {sol.reason}.")
    what = f"{field} is not" if field else "No factor of safety is"
    tail = (f" {what} reported: a factor of safety under a method's name "
            f"is that method's own converged solution, and nothing stands "
            f"in for it (UFC 3-220-10 p. 362).")
    n = len(reinf_forces or [])
    if n:
        t = sum(f.T for f in reinf_forces)
        tail += f" {n} support(s) cross the surface ({t:.1f} kN/m)."
    if not field:
        tail += (" Analyse another surface, or search for the critical "
                 "surface (a search passes over surfaces with no solution "
                 "and counts them).")
    return head + tail


def compare_methods_table(geom: SlopeGeometry,
                          xc: float = None,
                          yc: float = None,
                          radius: float = None,
                          slip_surface=None,
                          n_slices: int = 30,
                          tol: float = 1e-4,
                          f_interslice: str = "half_sine") -> dict:
    """Run ALL limit-equilibrium methods on one slip surface.

    The classic Fredlund & Krahn (1977) style comparison: one surface,
    every method, side by side — useful for judging method sensitivity
    and for validation against published tables.

    Parameters
    ----------
    geom : SlopeGeometry
    xc, yc, radius : float, optional
        Trial circle (or pass slip_surface).
    slip_surface : CircularSlipSurface or PolylineSlipSurface, optional
    n_slices : int
    tol : float
    f_interslice : str
        Interslice function for the Morgenstern-Price row.

    Returns
    -------
    dict with keys:
        'rows' : list of dicts {method, FOS, detail}
        'surface' : surface description dict
        'summary' : formatted text table
    """
    res = analyze_slope(geom, xc=xc, yc=yc, radius=radius,
                        slip_surface=slip_surface, method="bishop"
                        if (slip_surface is None or
                            getattr(slip_surface, "is_circular", True))
                        else "spencer",
                        n_slices=n_slices, tol=tol,
                        f_interslice=f_interslice, compare_methods=True)

    rows = []
    if res.FOS_fellenius is not None:
        rows.append({"method": "Fellenius (OMS)",
                     "FOS": round(res.FOS_fellenius, 4), "detail": ""})
    if res.FOS_bishop is not None:
        rows.append({"method": "Bishop simplified",
                     "FOS": round(res.FOS_bishop, 4), "detail": ""})
    if res.FOS_janbu_uncorrected is not None:
        rows.append({"method": "Janbu simplified (uncorrected)",
                     "FOS": round(res.FOS_janbu_uncorrected, 4),
                     "detail": ""})
    if res.FOS_janbu is not None:
        rows.append({"method": "Janbu simplified (corrected)",
                     "FOS": round(res.FOS_janbu, 4),
                     "detail": f"f0={res.janbu_f0:.3f}"
                     if res.janbu_f0 else ""})
    if res.FOS_spencer is not None:
        rows.append({"method": "Spencer",
                     "FOS": round(res.FOS_spencer, 4),
                     "detail": f"theta={res.theta_spencer:.2f} deg"
                     if res.theta_spencer is not None else ""})
    if res.FOS_morgenstern_price is not None:
        rows.append({"method": f"Morgenstern-Price ({f_interslice})",
                     "FOS": round(res.FOS_morgenstern_price, 4),
                     "detail": f"lambda={res.lambda_mp:.3f}"
                     if res.lambda_mp is not None else ""})

    surface = {"type": "circular" if res.is_circular else "noncircular",
               "xc": res.xc, "yc": res.yc, "radius": res.radius,
               "x_entry": round(res.x_entry, 3),
               "x_exit": round(res.x_exit, 3)}

    lines = ["Method comparison (one surface, all methods)",
             "-" * 52]
    for r in rows:
        lines.append(f"  {r['method']:<34s} {r['FOS']:>7.3f}  {r['detail']}")
    for name in res.methods_without_solution:
        # a method with no solution has no row and no number (GM-140 step 2)
        lines.append(f"  {name:<34s} {'none':>7s}  no solution on this "
                     f"surface (see WARNING)")
    for o in res.layer_overlaps:
        lines.append(f"  LAYERS: {o['statement']}")
    for w in res.warnings:
        lines.append(f"  WARNING: {w}")
    summary = "\n".join(lines)

    out = {"rows": rows, "surface": surface, "summary": summary}
    if res.methods_without_solution:
        out["methods_without_solution"] = list(res.methods_without_solution)
    if res.layer_overlaps:
        out["layer_overlaps"] = [dict(o) for o in res.layer_overlaps]
    if res.warnings:
        out["warnings"] = list(res.warnings)
    return out


@_counts_refusals
@_counts_reverse_discards
@_applies_m_alpha_rule
def search_critical_surface(
    geom: SlopeGeometry,
    x_range: Tuple[float, float] = None,
    y_range: Tuple[float, float] = None,
    nx: int = 10,
    ny: int = 10,
    method: str = "bishop",
    n_slices: int = 30,
    tol: float = 1e-4,
    surface_type: str = "circular",
    x_entry_range: Tuple[float, float] = None,
    x_exit_range: Tuple[float, float] = None,
    n_trials: int = 500,
    n_points: int = 5,
    seed: Optional[int] = None,
    toe_circles: bool = True,
    infinite_slope: bool = True,
    refine: bool = True,
    time_budget_s: Optional[float] = None,
    bishop_screen: bool = False,
    m_alpha_rule: str = "frictional",
    low_m_alpha: str = "flag",
    negative_friction_term: str = "as_printed",
) -> SearchResult:
    """Search for the critical slip surface (minimum FOS).

    Ranges not given are set from the slope itself, not from the model's
    width (GM-57, ``slope_stability.search_ranges``): the toe and crest of
    its faces, its height H and the depth to the base below the toe. The
    end of a surface on the toe side lies from the reach beyond the toe
    to the face's mid-length, the end on the crest side from there to the
    reach behind the crest; circle centres span those windows, from 0.1 H
    to max(2 H, depth to base) above the ground. A range the model's end
    cuts short is recorded in ``search_ranges`` and, where the governing
    surface ends there, in ``warnings``.

    A circular (centre-grid) search then refines its lowest grid centres by
    a local compass search on the centre, each centre with its own optimised
    radius, inside the same centre ranges (GM-87): the automatic grid spans
    the whole profile, so on a long profile its spacing alone can miss the
    critical circle by several per cent.

    After the main search, two supplementary surface families are scored
    with the same method and the lowest surface is reported (see
    ``slope_stability.search_supplements``): circles through the toe of
    each slope face (circular / entry_exit searches), and shallow planar
    surfaces parallel to any face where cohesionless soil (c' = 0) is at
    the surface — the infinite-slope mechanism. The result's
    ``governing_surface`` says which kind governs, ``candidates`` gives each
    family's minimum and ``warnings`` says where the search cannot be sure.
    Caller-given windows and centre ranges are honoured; the automatic
    ones do not restrict the supplementary families.

    A trial surface below the ground over more than one stretch (it leaves
    the ground and comes back into it) is not one sliding mass and is
    refused (GM-122, ``slip_surface.GroundCrossingError``);
    ``n_rejected_ground_crossings`` counts them, in every family.

    Given a time budget (GM-123, ``time_budget_s``; none by default), the
    main search scores its centres or entry/exit pairs coarse first and
    stops at 80 % of the budget; the toe-circle families stop at the
    budget. A search cut short reports the lowest surface it found, and
    ``warnings`` and ``search_ranges['time_budget']`` say what it scored,
    of how many, at what spacing, the run time projected for the whole
    search, and what it did not run. A search that finishes reports what
    it did without a budget. ``choices`` names the budget.

    Parameters
    ----------
    geom : SlopeGeometry
        Slope definition.
    x_range : (float, float), optional
        Circle center x search range (circular only).
    y_range : (float, float), optional
        Circle center y search range (circular only).
    nx, ny : int
        Grid resolution. Default 10x10.
    method : str
        FOS method. Default 'bishop'.
    n_slices : int
        Number of slices per analysis.
    tol : float
        Convergence tolerance for iterative methods. Default 1e-4.
    surface_type : str
        'circular' (default, centre-grid search), 'entry_exit' (circular
        arcs between entry/exit windows), 'noncircular' (random
        polylines), 'noncircular_de' (differential-evolution refinement
        seeded by a random search), 'pso', or 'weak_layer'.
    x_entry_range : (float, float), optional
        Allowed entry x-coordinate range. The entry is the surface's low-x
        end, whichever way the slope faces.
    x_exit_range : (float, float), optional
        Allowed exit x-coordinate range: the surface's high-x end. Windows
        given the other way round (the entry window wholly right of the
        exit window) are swapped, and ``warnings`` says so (GM-124).
    n_trials : int
        Number of random trials for noncircular search. Default 500.
    n_points : int
        Number of polyline vertices for noncircular search. Default 5.
    seed : int, optional
        Random seed for noncircular search reproducibility.
    toe_circles : bool
        Score circles through the toe of each slope face and circles
        passing above it, their lower end on the face (circular and
        entry_exit searches). Default True.
    infinite_slope : bool
        Score shallow planar surfaces on faces of cohesionless soil.
        Default True.
    refine : bool
        Refine the lowest centres of the circular grid by a local compass
        search on the centre (circular only). Default True.
    time_budget_s : float or None
        Time budget of the whole search, in seconds (GM-123). Default None,
        no limit, so a library search never depends on the machine's
        speed; the agent's tools give ``search.SEARCH_TIME_BUDGET_S``
        (180 s).
    bishop_screen : bool
        Circular and entry_exit searches with Spencer / Morgenstern-Price /
        GLE: score every centre or pair with Bishop's simplified method,
        then re-score the lowest of them with the method itself (at least
        10, every one within 5 % of the screen's minimum, at most 30); the
        reported FOS is the method's own. Not applied where supports are
        modelled. Default False (GM-123).
    m_alpha_rule : str
        Which slices the m_alpha < 0.2 check looks at, in every family.
        'frictional' (default): the slices with phi' > 0, since on a
        phi' = 0 slice m_alpha = cos(a) does not change the FOS (the
        module's own reasoning, no printed source); 'every_slice': every
        slice.
    low_m_alpha : str
        What every family does with a surface that has m_alpha < 0.2 on a
        slice (GM-140 step 3, OWN-102). 'flag' (default): the surface is
        kept and scored, and the result says so, with the surface and the
        slice (Whitman and Bailey 1967, p. 487, print a warning and an
        instruction to examine the result, not a rejection). 'refuse': it
        is left out and counted (``n_rejected_m_alpha``), as the module did
        before (GM-127; Slide2's 'Check malpha < 0.2').
    negative_friction_term : str
        Bishop's simplified method and the Ordinary Method of Slices on a
        slice where the pore-water force exceeds the weight: 'as_printed'
        (default; Bishop 1955 Eqs. 9 and 13, p. 10, no lower limit) or
        'zero' (the module's earlier rule, no printed source). See
        ``analyze_slope``.

    Returns
    -------
    SearchResult

    The printed acceptability checks (GM-140 step 3,
    ``slope_stability.acceptability``). Every surface the method solves
    that is one sliding mass is scored, whatever the checks find; nothing
    is left out silently. ``trial_flags`` counts, by family and method, the
    trials scored, those that carried a flag and each flag;
    ``critical_flags`` and ``critical.acceptability`` give what the
    governing surface carries (which check, which slice, what value); and
    when it carries a flag, ``lowest_unflagged`` gives the lowest factor of
    safety among the scored surfaces of the same method that carry none.
    The headline stays the lowest solved surface. With Bishop's method,
    the Ordinary Method of Slices value is computed for every circle
    without supports or a seismic coefficient and taken where Bishop's is
    the lower (Duncan 1996, p. 580); ``critical.FOS_bishop`` and
    ``critical.FOS_fellenius`` give both. The noncircular screens that
    are left (slice count and span, applied before any solve) are the
    module's own, not found in print; ``screen_refusals`` counts each. A
    low factor of safety on a jagged surface, or a solution that held about
    one moment axis or one slicing only, is no longer refused
    (``search.REMOVED_SCREENS``): such a surface is scored, kept and
    flagged, and can be the headline.
    """
    from slope_stability.search_supplements import (
        apply_supplements, search_choices, note_budget, CIRCULAR,
        NONCIRCULAR, m_alpha_confirmation, finer_slices_warning,
        m_alpha_none_warning, critical_acceptability,
        acceptability_warning,
    )
    from slope_stability.search import (
        _screen_method, _m_alpha_refusals_so_far, _no_solution_so_far,
        no_solution_records, _scoring_family, _flag_tally_so_far,
        flag_records,
    )
    from slope_stability.search_supplements import no_solution_warning
    check_method(method)
    from slope_stability.search_ranges import resolve_ranges
    # Entry is the low-x end, exit the high-x end: windows given the other
    # way round are swapped, and the result says so (GM-124).
    x_entry_range, x_exit_range, window_note = _ordered_windows(
        x_entry_range, x_exit_range)
    given = dict(x_entry_range=x_entry_range, x_exit_range=x_exit_range,
                 x_range=x_range, y_range=y_range)
    windows_auto = (x_entry_range is None, x_exit_range is None)
    grid_auto = x_range is None and y_range is None

    # Ranges the caller left out are set from the slope itself: its toe,
    # crest, height and the depth to the base (GM-57, search_ranges.py).
    x_entry_range, x_exit_range, x_range, y_range, auto = resolve_ranges(
        geom, x_entry_range, x_exit_range, x_range, y_range)

    noncirc_method = method if method != "bishop" else "spencer"
    screened = (bishop_screen and surface_type in ("circular", "entry_exit")
                and _screen_method(geom, method, None, True) != method)
    screen_not_applied = None       # asked for, not applied: why (GM-123)
    if bishop_screen and not screened:
        if surface_type not in ("circular", "entry_exit"):
            screen_not_applied = (
                "noncircular searches are not screened: Bishop's simplified "
                "method is for circles, and a Bishop-like method and a "
                "rigorous one can rank noncircular surfaces differently")
        elif _screen_method(geom, "spencer", None, True) == "spencer" \
                and method in ("spencer", "morgenstern_price", "gle"):
            screen_not_applied = (
                f"the model has supports, which Bishop's simplified method "
                f"counts little of; every circle was scored with {method}")
        else:
            screen_not_applied = (
                f"the method is {method}; the screen is for Spencer, "
                f"Morgenstern-Price and GLE")

    def _finish(result, search_kind, grid_used=None):
        budget.supplements()      # the rest of the budget (GM-123)
        # (the families count their own trials without a solution; the
        # checks made on the governing surface are not trials)
        with _scoring_family(None):
            return _finish_counted(result, search_kind, grid_used)

    def _finish_counted(result, search_kind, grid_used):
        result = apply_supplements(
            geom, result, search_kind=search_kind, method=method,
            n_slices=n_slices, tol=tol,
            windows_used=(x_entry_range, x_exit_range),
            windows_auto=windows_auto, grid_used=grid_used,
            grid_auto=grid_auto, toe_circles=toe_circles,
            infinite_slope=infinite_slope, auto_ranges=auto, **given)
        # What the search chose (GM-115): PSO and the weak-layer search
        # score with Spencer whatever is named, a noncircular search with
        # Spencer for Bishop.
        scored = ("spencer" if surface_type in ("pso", "weak_layer")
                  else noncirc_method if search_kind == NONCIRCULAR
                  else method)
        # Surfaces left out for m_alpha < 0.2 so far, the supplementary
        # families' included, and the governing surface on finer slices
        # (GM-127).
        result.n_rejected_m_alpha = _m_alpha_refusals_so_far()
        # Trials left without a solution, by family and method, and a
        # warning when the governing family lost trials (GM-140 step 2).
        result.no_solution = no_solution_records(
            _no_solution_so_far(), rename={"main": search_kind})
        w = no_solution_warning(result)
        if w:
            result.warnings.append(w)
        # The printed acceptability checks (GM-140 step 3): the flags of
        # the scored trials by family, what the governing surface carries,
        # and the lowest surface without a flag beside it.
        result.trial_flags = flag_records(
            _flag_tally_so_far(), rename={"main": search_kind})
        critical_acceptability(geom, result, scored, n_slices, tol,
                               m_alpha_rule=m_alpha_rule)
        w = acceptability_warning(result)
        if w:
            result.warnings.append(w)
        if result.critical is not None:
            # (a comparison with the Ordinary Method of Slices value that
            # was not made on the governing circle, and why)
            result.warnings.extend(
                f["text"] for f in result.critical.acceptability
                if f["check"] == "bishop_against_oms"
                and not f.get("compared"))
        confirm = m_alpha_confirmation(geom, result.critical, scored,
                                       n_slices, tol)
        result.layer_overlaps = geom.layer_overlaps()
        result.choices = search_choices(
            result, surface_type=surface_type, method=method,
            scored_with=scored, toe_circles=toe_circles,
            infinite_slope=infinite_slope, budget=budget,
            screened=screened, screen_not_applied=screen_not_applied,
            m_alpha=confirm, m_alpha_rule=m_alpha_rule,
            low_m_alpha=low_m_alpha,
            negative_friction_term=negative_friction_term)
        # the crack rule (GM-140 step 4): the crack asked for, and what
        # was done with reverse-curvature circles
        result.choices += _crack_choices(
            geom, _discards_so_far(),
            circles=surface_type in ("circular", "entry_exit")
            or toe_circles)
        w = finer_slices_warning(confirm) or m_alpha_none_warning(result)
        if w:
            result.warnings.append(w)
        if window_note:
            result.warnings.insert(0, window_note)
        # A search cut short says so first (GM-123).
        note_budget(result, budget, method=method, screened=screened,
                    can_screen=surface_type in ("circular", "entry_exit"))
        return result

    with search_budget(time_budget_s) as budget:
        if surface_type in ("noncircular", "noncircular_de", "pso",
                            "weak_layer"):
            result = _search_noncircular_kind(
                geom, surface_type, x_entry_range, x_exit_range, nx, ny,
                n_slices, tol, n_trials, n_points, seed, noncirc_method)
            return _finish(result, NONCIRCULAR)

        if surface_type == "entry_exit":
            result = search_entry_exit(
                geom,
                x_entry_range=x_entry_range,
                x_exit_range=x_exit_range,
                n_entry=nx,
                n_exit=ny,
                method=method,
                n_slices=n_slices,
                tol=tol,
                bishop_screen=bishop_screen,
            )
            return _finish(result, CIRCULAR)

        # Circular search
        result = grid_search(geom, x_range, y_range, nx, ny, method,
                             n_slices, tol=tol, x_entry_range=x_entry_range,
                             x_exit_range=x_exit_range, refine=refine,
                             bishop_screen=bishop_screen)

        return _finish(result, CIRCULAR, grid_used=(x_range, y_range))


def _search_noncircular_kind(geom, surface_type, x_entry_range, x_exit_range,
                             nx, ny, n_slices, tol, n_trials, n_points, seed,
                             noncirc_method) -> SearchResult:
    """Dispatch the noncircular search types of `search_critical_surface`."""
    if surface_type == "noncircular":
        return search_noncircular(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_trials=n_trials,
            n_points=n_points,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
            method=noncirc_method,
        )

    if surface_type == "noncircular_de":
        return search_de(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_points=max(n_points, 5),
            method=noncirc_method,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
        )

    if surface_type == "pso":
        return search_pso(
            geom,
            x_entry_range=x_entry_range,
            x_exit_range=x_exit_range,
            n_particles=max(nx * ny, 30),
            n_iterations=n_trials // max(nx * ny, 30) if n_trials > 30 else 50,
            n_points=n_points,
            n_slices=n_slices,
            tol=tol,
            seed=seed,
        )

    # weak_layer
    return search_weak_layer_biased(
        geom,
        x_entry_range=x_entry_range,
        x_exit_range=x_exit_range,
        n_trials=n_trials,
        n_points=n_points,
        n_slices=n_slices,
        tol=tol,
        seed=seed,
    )


def rapid_drawdown_fos(geom: SlopeGeometry,
                       drawdown_from_elevation: float,
                       drawdown_to_elevation: float,
                       xc: float = None, yc: float = None, radius: float = None,
                       slip_surface=None,
                       method: str = "duncan_3stage",
                       f_interslice: str = "constant",
                       n_slices: int = 50,
                       tol: float = 1e-4,
                       stage1_phreatic_points=None,
                       stage3_effective_normal: str = "fellenius"):
    """Rapid-drawdown factor of safety (USACE 2-stage / Duncan-Wright-Wong
    3-stage) on a specified slip surface.

    Low-permeability layers must carry the R-envelope (`R_c`, `R_phi`) in
    addition to the effective `c_prime`/`phi`; `R_phi is None` marks a
    free-draining layer. See `slope_stability.rapid_drawdown` for the full
    theory and `RapidDrawdownResult`.

    Parameters
    ----------
    geom : SlopeGeometry
    drawdown_from_elevation, drawdown_to_elevation : float
        Reservoir surface elevation before / after drawdown (m).
    xc, yc, radius / slip_surface :
        The specified slip surface (circle or explicit surface).
    method : str
        'duncan_3stage' (default) or 'corps_2stage'.
    f_interslice : str
        Interslice function ('constant' = Spencer). Default 'constant'.
    n_slices, tol : usual meanings.
    stage1_phreatic_points : list of (x, z), optional
        Steady-seepage phreatic surface for the stage-1 consolidation stresses
        only (default None = flat full-pool, the conservative no-through-seepage
        bound). See `slope_stability.rapid_drawdown` for details.
    stage3_effective_normal : str, optional
        Stage-3 (3-stage) drained-substitution effective-normal basis:
        'fellenius' (default, preserved) or 'gle' (rigorous). See
        `slope_stability.rapid_drawdown` for details.

    Returns
    -------
    RapidDrawdownResult
    """
    from slope_stability.rapid_drawdown import rapid_drawdown_fos as _rd
    return _rd(geom, drawdown_from_elevation, drawdown_to_elevation,
               xc=xc, yc=yc, radius=radius, slip_surface=slip_surface,
               method=method, f_interslice=f_interslice, n_slices=n_slices,
               tol=tol, stage1_phreatic_points=stage1_phreatic_points,
               stage3_effective_normal=stage3_effective_normal)


# Re-export the rapid-drawdown critical-surface search (defined in
# rapid_drawdown.py, which composes it from search.py) for parity with
# rapid_drawdown_fos and the rest of the analysis-orchestrator surface.
from slope_stability.rapid_drawdown import (  # noqa: E402
    search_rapid_drawdown, RapidDrawdownSearchResult,
)


def infinite_slope_fos(slope_angle: float,
                       phi: float,
                       gamma: float,
                       c: float = 0.0,
                       depth: float = 1.0,
                       water_condition: str = "dry",
                       gamma_w: float = 9.81,
                       ru: float = 0.0,
                       water_depth: float = 0.0) -> InfiniteSlopeResult:
    """Infinite-slope (planar translational) factor of safety — closed form.

    Effective-stress limit equilibrium of a slip plane parallel to an
    infinitely long slope at depth ``z`` below the surface (Duncan, Wright &
    Brandon 2014, Ch. 14; Skempton & DeLory 1957):

        sigma_n = gamma * z * cos^2(beta)             (total normal on plane)
        tau     = gamma * z * sin(beta) * cos(beta)   (driving shear)
        FOS     = [ c' + (sigma_n - u) * tan(phi') ] / tau

    Pore pressure ``u`` on the slip plane by ``water_condition``:

    * ``'dry'``              : u = 0.
    * ``'seepage_parallel'`` : steady seepage parallel to the slope, phreatic
      surface a depth ``water_depth`` below the ground;
      u = gamma_w * (z - water_depth) * cos^2(beta)  (0 if z <= water_depth).
      ``water_depth = 0`` -> water table at the surface (the usual case).
    * ``'ru'``               : u = ru * gamma * z  (pore-pressure ratio).

    For a cohesionless soil (c' = 0) the depth cancels:
        dry:                FOS = tan(phi') / tan(beta)
        seepage (d_w = 0):  FOS = (gamma'/gamma) * tan(phi') / tan(beta).

    Parameters
    ----------
    slope_angle : float
        Slope inclination beta (degrees from horizontal), in (0, 90).
    phi : float
        Effective friction angle (degrees).
    gamma : float
        Total (moist / saturated) unit weight (kN/m3).
    c : float
        Effective cohesion (kPa). Default 0 (cohesionless).
    depth : float
        Slip-plane depth z below the surface (m). Cancels for c' = 0; must be
        positive. Default 1.0.
    water_condition : str
        'dry' (default), 'seepage_parallel', or 'ru'.
    gamma_w : float
        Unit weight of water (kN/m3). Default 9.81.
    ru : float
        Pore-pressure ratio, used when ``water_condition='ru'``. Default 0.
    water_depth : float
        Depth of the phreatic surface below the ground (m), used when
        ``water_condition='seepage_parallel'``. Default 0 (table at surface).

    Returns
    -------
    InfiniteSlopeResult
    """
    if not (0.0 < slope_angle < 90.0):
        raise ValueError(
            f"slope_angle must be in (0, 90) deg, got {slope_angle}")
    if gamma <= 0:
        raise ValueError(f"gamma must be positive, got {gamma}")
    if depth <= 0:
        raise ValueError(f"depth must be positive, got {depth}")
    if phi < 0 or c < 0:
        raise ValueError("phi and c must be non-negative")
    if water_condition not in ("dry", "seepage_parallel", "ru"):
        raise ValueError(
            "water_condition must be 'dry', 'seepage_parallel' or 'ru', "
            f"got '{water_condition}'")

    beta = math.radians(slope_angle)
    cosb, sinb = math.cos(beta), math.sin(beta)
    sigma_n = gamma * depth * cosb * cosb
    tau = gamma * depth * sinb * cosb

    if water_condition == "seepage_parallel":
        head = max(depth - water_depth, 0.0)
        u = gamma_w * head * cosb * cosb
    elif water_condition == "ru":
        u = ru * gamma * depth
    else:
        u = 0.0

    sigma_eff = max(sigma_n - u, 0.0)
    resist = c + sigma_eff * math.tan(math.radians(phi))
    fos = resist / tau if tau > 0 else float("inf")

    return InfiniteSlopeResult(
        FOS=fos, slope_angle=slope_angle, depth=depth, phi=phi, c=c,
        gamma=gamma, water_condition=water_condition,
        normal_stress=sigma_n, shear_stress=tau, pore_pressure=u, ru=ru,
    )
