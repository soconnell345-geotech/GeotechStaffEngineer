"""
A method's own solution on one slip surface, or no number (GM-140 step 2).

The rule. A factor of safety reported under a method's name is that
method's own converged solution on that surface. Where the method has no
solution there is no number: the solution's ``fos`` is None, ``solved`` is
False and ``reason`` says why. `solve_method` is the one solver every caller
uses for a method: one given surface (``analyze_slope``), every search
family (``search._compute_fos``), the yield acceleration, FOSM and Monte
Carlo. So a search's score and a single-surface result are the same solve:
the same moment axis, the same tolerance, the same test of convergence.

Print. UFC 3-220-10 p. 362: "Critical failure surfaces can be missed as a
result of non-convergence. The engineer should be able to identify when
convergence issues are present." EM 1110-2-1902 C-10d (pp. C-37 to C-38)
lists the ways to avoid non-convergence. The methods' own equations: Bishop
(1955) Eq. 13, p. 10; Spencer (1967) Eqs. 7 and 8, p. 15; Morgenstern and
Price (1965) Eq. 18, p. 85.

What "no solution" covers:

* Ordinary Method of Slices: no driving moment or force on the surface (the
  ratio has no meaning), or, with passive supports, an iteration on F that
  does not settle.
* Bishop's simplified method: no driving moment, or the iteration on F
  (Eq. 13) not within its tolerance after its iteration limit.
* Janbu's simplified method: no state of horizontal force equilibrium with
  horizontal interslice forces (the iteration cannot take a step, or does
  not settle).
* Spencer, Morgenstern-Price and GLE: no lambda in range at which the
  moment and force factors of safety agree within the solver's tolerance
  (``gle.gle_fos``: ``converged`` False).

No approximate solver stands in for a method anywhere (the old approximate
Spencer and Morgenstern-Price solvers are removed), and the solvers'
internal 999.9 marker is never returned as a factor of safety.
"""

import math
from dataclasses import dataclass
from typing import List, Optional

#: The limit-equilibrium methods, as every entry point names them.
METHODS = ("fellenius", "bishop", "janbu", "spencer", "morgenstern_price",
           "gle")

#: Plain names, for messages.
METHOD_TEXT = {
    "fellenius": "The Ordinary Method of Slices",
    "bishop": "Bishop's simplified method",
    "janbu": "Janbu's simplified method",
    "spencer": "Spencer's method",
    "morgenstern_price": "The Morgenstern-Price method",
    "gle": "The GLE (Morgenstern-Price) method",
}

#: Why a method has no solution on a surface (`NoSolution.kind`).
NO_SOLUTION_KINDS = ("not_converged", "no_driving")

#: The internal marker of the solvers and the searches' "not scored" score.
#: Not a factor of safety (`solve_method` never returns it).
_FOS_MAX = 999.9


class NoSolution(ValueError):
    """The method has no solution on this surface: there is no factor of
    safety to report (GM-140 step 2). ``reason`` says why, in plain words.

    A ValueError, so a caller that already treats a surface it cannot
    analyse as refused treats this one the same way; `solve_method` returns
    it as a `MethodSolution` instead of raising.

    ``kind`` is one of `NO_SOLUTION_KINDS`: 'not_converged' (the method's
    equations were not solved: a critical surface can be missed this way,
    UFC 3-220-10 p. 362) or 'no_driving' (the surface has no net driving
    moment or force, as where a support's force exceeds it: the ratio has
    no meaning, and such a surface is not a candidate critical surface)."""

    def __init__(self, reason: str, kind: str = "not_converged"):
        super().__init__(reason)
        self.reason = reason
        self.kind = kind


def check_method(method, allowed=METHODS) -> str:
    """``method`` if it names a method, else a ValueError listing the names
    (GM-140 step 2, D49: an unknown name used to run Bishop's simplified
    method without saying so)."""
    if method not in allowed:
        raise ValueError(
            f"Unknown method {method!r}. Choose from: "
            + ", ".join(repr(m) for m in allowed) + ".")
    return method


def check_interslice(f_interslice) -> str:
    """``f_interslice`` if it names an interslice force function of the
    Morgenstern-Price / GLE solver, else a ValueError listing the names."""
    from slope_stability.gle import INTERSLICE_FUNCTIONS
    if f_interslice not in INTERSLICE_FUNCTIONS:
        raise ValueError(
            f"Unknown interslice function {f_interslice!r}. Choose from: "
            + ", ".join(repr(f) for f in sorted(INTERSLICE_FUNCTIONS)) + ".")
    return f_interslice


@dataclass
class MethodSolution:
    """One method on one slip surface.

    method : str
        The method solved (one of `METHODS`).
    fos : float or None
        The method's own converged factor of safety; None when it has no
        solution (never a marker value, never another solver's value).
    solved : bool
        The method's own solution was found.
    reason : str
        Why there is no solution, in plain words ('' when solved).
    kind : str
        '' when solved, else one of `NO_SOLUTION_KINDS`.
    param : float or None
        theta in degrees (Spencer) or lambda (Morgenstern-Price / GLE).
    gle : GLEResult or None
        The converged GLE state (per-slice and interslice forces) of
        Spencer / Morgenstern-Price / GLE.
    fos_uncorrected, f0 : float or None
        Janbu's simplified method: the force-equilibrium F before the f0
        correction, and f0 (``fos`` is the corrected value).
    """
    method: str
    fos: Optional[float]
    solved: bool
    reason: str = ""
    param: Optional[float] = None
    gle: Optional[object] = None
    fos_uncorrected: Optional[float] = None
    f0: Optional[float] = None
    kind: str = ""


def _usable(fos) -> bool:
    return (fos is not None and not math.isnan(fos) and not math.isinf(fos)
            and 0.0 < fos < _FOS_MAX)


def solve_method(slices: List, slip, method: str,
                 f_interslice: str = "half_sine",
                 tol: float = 1e-4,
                 reinf_forces=None) -> MethodSolution:
    """The method's own solution on the sliced surface, or no number.

    Parameters
    ----------
    slices : list of Slice
        From ``build_slices``.
    slip : CircularSlipSurface or PolylineSlipSurface
    method : str
        One of `METHODS`. 'gle' is Morgenstern-Price with ``f_interslice``.
        An unknown name is refused (ValueError, with the names).
    f_interslice : str
        Interslice force function of 'morgenstern_price' / 'gle'
        ('constant', 'half_sine', 'clipped_sine', 'trapezoidal'); an
        unknown name is refused. Spencer is the constant function.
    tol : float
        Convergence tolerance. Bishop's simplified method and the Ordinary
        Method of Slices take it as given; Janbu's simplified method and
        the GLE behind Spencer / Morgenstern-Price / GLE run at
        ``min(tol, 1e-4) / 10``, as they did in ``analyze_slope``.
    reinf_forces : list of ReinforcementForce, optional
        Support forces crossing the surface.

    Returns
    -------
    MethodSolution
        ``solved`` False with ``fos`` None and the ``reason`` where the
        method has no solution.

    Raises
    ------
    ValueError
        Unknown ``method`` or ``f_interslice``; Bishop's simplified method
        on a noncircular surface (it takes moments about a circle's
        centre); fewer than 3 slices for Janbu or the GLE.

    Notes
    -----
    On a noncircular surface the GLE takes moments about one axis, its own
    least-squares circle through the slice base midpoints
    (``gle._fit_axis_point``), in a search and on one given surface alike.
    Before GM-140 step 2 a search tried a second, fixed axis where that
    solve did not converge, and so could score a surface that
    ``analyze_slope`` reported as having no solution.
    """
    check_method(method)
    check_interslice(f_interslice)      # whatever the method (item 7)
    from slope_stability.methods import fellenius_fos, bishop_fos
    from slope_stability.gle import janbu_fos, gle_fos
    try:
        if method == "fellenius":
            fos = fellenius_fos(slices, slip, reinf_forces=reinf_forces)
            return _number(method, fos)
        if method == "bishop":
            fos = bishop_fos(slices, slip, tol=tol,
                             reinf_forces=reinf_forces)
            return _number(method, fos)
        if method == "janbu":
            fos, fos_u, f0 = janbu_fos(slices, slip,
                                       tol=min(tol, 1e-4) * 0.1,
                                       reinf_forces=reinf_forces)
            out = _number(method, fos)
            if out.solved:
                out.fos_uncorrected, out.f0 = fos_u, f0
            return out
    except NoSolution as exc:
        return MethodSolution(method=method, fos=None, solved=False,
                              reason=exc.reason, kind=exc.kind)

    spencer = method == "spencer"
    f_name = "constant" if spencer else f_interslice
    res, err = None, None
    try:
        res = gle_fos(slices, slip, f_interslice=f_name,
                      tol=min(tol, 1e-4) * 0.1, reinf_forces=reinf_forces)
    except (ZeroDivisionError, OverflowError) as exc:
        err = exc
    if res is not None and res.converged and _usable(res.fos):
        param = math.degrees(math.atan(res.lam)) if spencer else res.lam
        return MethodSolution(method=method, fos=res.fos, solved=True,
                              param=param, gle=res)
    return MethodSolution(method=method, fos=None, solved=False,
                          reason=gle_failure_reason(res, err),
                          kind="not_converged")


def _number(method: str, fos) -> MethodSolution:
    """A solver's value as a solution: a marker or a non-number is none."""
    if _usable(fos):
        return MethodSolution(method=method, fos=fos, solved=True)
    kind = "not_converged"
    if fos is not None and not math.isnan(fos) and fos >= _FOS_MAX:
        # (an astronomical ratio: the driving sum is zero to rounding)
        reason = (f"the surface has all but no net driving moment or force "
                  f"(the ratio of resisting to driving comes to "
                  f"{fos:.3g}, which is no factor of safety)")
        kind = "no_driving"
    elif fos is not None:
        reason = f"no usable factor of safety (F = {fos:.4g})"
    else:
        reason = "no solution"
    return MethodSolution(method=method, fos=None, solved=False,
                          reason=reason, kind=kind)


def gle_failure_reason(res, err) -> str:
    """Plain words for why ``gle_fos`` gave no solution."""
    if err is not None:
        return f"the solution stopped with an error ({err})"
    if res is None:
        return "no solution"
    for w in res.warnings:
        if w.startswith("no F_m = F_f crossing"):
            return ("no lambda from -1.5 to 1.5 gives moment and force "
                    "factors of safety that agree")
        if w.startswith("no equilibrium state at any lambda"):
            return "no lambda from -1.5 to 1.5 gives an equilibrium state"
        if w.startswith("F_m - F_f residual"):
            # gle_fos: "F_m - F_f residual {:.2e} at lambda={:.3f}"
            parts = w.replace("=", " ").split()
            try:
                gap, lam = float(parts[4]), float(parts[-1])
            except (IndexError, ValueError):
                return ("the moment and force factors of safety do not "
                        f"agree at any lambda found ({w})")
            return ("the moment and force factors of safety still differ by "
                    f"{gap:.3g} at the closest lambda found ({lam:.3f})")
    if res.converged:
        return f"no usable factor of safety (F = {res.fos:.4g})"
    return "the solution does not converge"


def no_solution_text(method: str, reason: str) -> str:
    """'<Method> has no solution on this surface: <reason>.'"""
    return (f"{METHOD_TEXT.get(method, method)} has no solution on this "
            f"surface: {reason}.")
