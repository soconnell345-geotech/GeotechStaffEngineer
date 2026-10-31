"""
Top-level analysis runner for lateral pile analysis.

Provides the LateralPileAnalysis class that ties together the pile definition,
soil layers, p-y curve models, and finite difference solver into a clean API
suitable for use by both human engineers and LLM agents.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
- FHWA GEC 9: FHWA-HIF-18-031
"""

import copy
import dataclasses
import math
import warnings
from typing import List, Optional

import numpy as np

from lateral_pile.pile import Pile
from lateral_pile.py_curves import (
    PCI_TO_KNM3, SAND_K_TABLE_A3, STIFF_CLAY_K_TABLE, SandAPI, SandReese,
    StiffClayBelowWT, WeakRock,
)
from lateral_pile.soil import SoilLayer, get_layer_at_depth, validate_layers
from lateral_pile.solver import solve_lateral_pile
from lateral_pile.results import Results


#: Each p-y model's source, as its class docstring cites it, for a result's
#: ``choices`` (GM-115), and the models offered for the same ground.
_PY_SOURCE = {
    "SoftClayMatlock": "Matlock (1970), OTC 1204, soft clay, as printed "
                       "in COM624P (FHWA-SA-91-048) pp. 323-327 and GEC 9 "
                       "App. A.1",
    "StiffClayBelowWT": "Reese, Cox & Koop (1975), OTC 2312, stiff clay "
                        "below the water table, as printed in COM624P "
                        "(FHWA-SA-91-048) pp. 328-335 and GEC 9 App. A.2; "
                        "GEC 9 p. 219: for stiff clay with free water at "
                        "or near the ground surface; with the groundwater "
                        "at depth, Welch & Reese (StiffClayAboveWT) is "
                        "more appropriate, even below it",
    "StiffClayAboveWT": "Welch & Reese (1972), Report 89-10, stiff clay "
                        "above the water table, as printed in COM624P "
                        "(FHWA-SA-91-048) pp. 336-340 and GEC 9 App. A.3 "
                        "(\"stiff clay with no free water\"); GEC 9 p. 219: "
                        "more appropriate than Reese et al. (1975) even "
                        "below the water table where the groundwater is at "
                        "depth and free water is not at or near the ground "
                        "surface",
    "SoftClayJeanjean": "Jeanjean (2009), OTC-20158-MS, soft clay; not "
                        "checked against a print (neither the paper nor "
                        "API RP 2GEO is in the reference library); GEC 9's "
                        "soft clay criterion is Matlock (SoftClayMatlock, "
                        "App. A.1)",
    "SandReese": "Reese, Cox & Koop (1974), OTC 2080, sand, as printed in "
                 "GEC 9 App. A.4: p_s the smaller of Eqs. A-23 and A-24, "
                 "A and B from Figs. A-10 and A-11",
    "SandAPI": "API RP 2A-WSD (21st ed., 2007) Secs. 6.8.6-6.8.7 / O'Neill "
               "& Murchison (1983), sand: C1, C2, C3 from Fig. 6.8.6-1",
    "WeakRock": "Reese (1997), JGGE 123(11), weak rock, as printed in "
                "FHWA GEC 9 App. A.5",
    "SandLiquefied": "Rollins et al. (2005), liquefied sand, as printed in "
                     "GEC 9 App. A.6 (Eqs. A-43, A-44), for p up to about "
                     "14.9 kN/m, y up to 152 mm, depths to 6.1 m and sand "
                     "of about 50 % relative density; held at its 150 mm "
                     "value beyond, where no page prints it",
}
#: SandReese's construction other than GEC 9's printed curve.
_REESE_SIMPLIFIED = ("construction 'simplified' is the module's own "
                     "three-part curve, which no held page prints; GEC 9's "
                     "curve is 'reese1974'")
_PY_FAMILIES = (
    ("SoftClayMatlock", "StiffClayBelowWT", "StiffClayAboveWT",
     "SoftClayJeanjean"),
    ("SandReese", "SandAPI"),
    ("WeakRock",),
)


def _field_default(obj, name):
    """The dataclass default of ``obj.name``, or None."""
    import dataclasses
    for f in dataclasses.fields(obj) if dataclasses.is_dataclass(obj) else ():
        if f.name == name:
            return f.default
    return None


def py_choices(layers, within: str = "layers") -> list:
    """What a lateral analysis used (GM-115): each layer's p-y model with
    its source and any option left at its default, and the loading (static
    or cyclic) over the layers. ``within`` is the input the layers came in,
    so a calc package's call that says it is not the user's makes each
    model and loading read "assumed" (GM-180)."""
    from geotech_common.choices import DEFAULT, choice, how_given
    out, loading, all_default = [], [], True
    for layer in layers:
        model = layer.py_model
        name = type(model).__name__
        notes = []
        for opt in ("J", "construction"):
            if hasattr(model, opt):
                value = getattr(model, opt)
                if how_given(value, _field_default(model, opt)) == DEFAULT:
                    notes.append(f"{opt} = {value:g}" if isinstance(
                        value, (int, float)) else f"{opt} {value}")
        basis = _PY_SOURCE.get(name, "")
        if notes:
            basis += "; default " + ", ".join(notes)
        if getattr(model, "construction", None) == "simplified":
            basis += "; " + _REESE_SIMPLIFIED
        if isinstance(model, SandAPI) and not 20.0 <= model.phi <= 40.0:
            basis += (f"; phi' {model.phi:g} deg is off the chart's 20-40 "
                      "deg, so C1, C2, C3 are its end values")
        if isinstance(model, WeakRock):
            basis += _weak_rock_note(model)
        family = next((f for f in _PY_FAMILIES if name in f), ())
        out.append(choice(f"p-y curves {layer.top:g}-{layer.bottom:g} m",
                          name, how_given(name, None, within=within),
                          basis, family))
        if isinstance(model, (SandReese, SandAPI)):
            out.append(_sand_k_choice(model, layer, within))
        if isinstance(model, WeakRock):
            out += _weak_rock_choices(model, layer, within)
        if getattr(model, "n_cycles", None) is not None:
            out.append(_cycles_choice(model, layer, within))
        if isinstance(model, StiffClayBelowWT):
            out.append(_stiff_clay_k_choice(model, layer, within))
        load = getattr(model, "loading", None)
        if load is not None:
            if how_given(load, _field_default(model, "loading")) != DEFAULT:
                all_default = False
            if loading and loading[-1][0] == load:
                loading[-1][2] = layer.bottom
            else:
                loading.append([load, layer.top, layer.bottom])
    if loading:
        one = loading[0][0] if len(loading) == 1 else None
        chosen = (f"{one}, every layer" if one else
                  "; ".join(f"{l} {a:g}-{b:g} m" for l, a, b in loading))
        out.append(choice(
            "loading", chosen,
            DEFAULT if all_default else how_given(chosen, None,
                                                  within=within),
            "each model's static or cyclic p-y curve (COM624P, "
            "FHWA-SA-91-048)",
            [x for x in ("static", "cyclic") if one and x != one]))
    return out


def _weak_rock_note(model) -> str:
    """Where a weak-rock layer's depths are measured from, and what its
    criterion covers (GEC 9 App. A.5 and p. 92)."""
    note = (f"; z_r below the rock surface at {model.rock_top or 0.0:g} m")
    if not 500.0 <= model.qu <= 5000.0:
        note += (f"; qu {model.qu:g} kPa is outside the 0.5-5 MPa the "
                 "criterion was proposed for")
    if model.loading == "cyclic":
        note += ("; the criterion is for static loading (GEC 9 p. 92), so "
                 "the cyclic curve is the static one")
    return note


def _cycles_choice(model, layer, within: str = "layers") -> dict:
    """The number of cycles of load application N of a cyclic stiff-clay
    layer above the water table (GM-202), always given: the references set
    no default."""
    from geotech_common.choices import choice, how_given
    n = model.n_cycles
    others = [(n_alt, f"N = {n_alt:g}: y50 x {1 + 0.6 * math.log10(n_alt):.2g}")
              for n_alt in (10.0, 100.0, 1000.0)]
    return choice(
        f"number of cycles of load application N "
        f"{layer.top:g}-{layer.bottom:g} m",
        f"{n:g} (y50 x {model.cyclic_factor:.3g})",
        how_given(n, None, within=within),
        "y_c = y_s + y50 C log N, C = 9.6 (p/pu)^4 (Welch & Reese 1972; "
        "COM624P Eqs. 3.28-3.29, p. 338; GEC 9 Eqs. A-20, A-21): the "
        "static curve with y50 times 1 + 0.6 log N, reaching pu at "
        "16 y50 + 9.6 y50 log N; the references give no default N",
        ["N = 1: the static curve, the stiffest"]
        + [text for n_alt, text in others if abs(n_alt - n) > 1e-9]
        + ["more cycles: a softer curve, more deflection and moment"])


def _sand_k_choice(model, layer, within: str = "layers") -> dict:
    """The k of a sand layer's initial line (GM-287), always given: GEC 9
    Table A-3 by relative density for Reese sand, API RP 2A Fig. 6.8.7-1 by
    phi' for API sand."""
    from geotech_common.choices import choice, how_given
    span = f"{layer.top:g}-{layer.bottom:g} m"
    if isinstance(model, SandAPI):
        from geotech_common.soil_profile import _estimate_k_sand
        below, above = (_estimate_k_sand(model.phi, w) for w in (True, False))
        basis = ("the initial slope k H of API's tanh curve (API RP 2A-WSD "
                 "Eq. 6.8.7-1), k from Fig. 6.8.7-1 by phi' (relative "
                 "density on the same chart)")
        others = [(below, f"Fig. 6.8.7-1 at phi' {model.phi:g} deg, below "
                          "the water table"),
                  (above, f"Fig. 6.8.7-1 at phi' {model.phi:g} deg, above "
                          "it")]
    else:
        basis = ("the initial line p = (k z) y (GEC 9 Eq. A-27), k from GEC "
                 "9 Table A-3 by relative density, not by phi")
        others = [(v * PCI_TO_KNM3, f"{density} sand, {where}")
                  for where, key in (("submerged", "submerged"),
                                     ("above the water", "above"))
                  for density, v in SAND_K_TABLE_A3[key].items()]
    return choice(
        f"k of the initial line {span}", f"{model.k:g} kN/m3",
        how_given(model.k, None, within=within), basis,
        [f"{k:,.0f} kN/m3: {text}" for k, text in others
         if abs(k - model.k) > 0.5 * PCI_TO_KNM3]
        + ["a smaller k: a softer start to the curve, more deflection at "
           "small loads"])


def _stiff_clay_k_choice(model, layer, within: str = "layers") -> dict:
    """The k of a stiff-clay layer below the water table (GM-208): ks for
    static loading, kc for cyclic, always given (COM624P Table 3.2)."""
    from geotech_common.choices import choice, how_given
    name = "ks" if model.loading == "static" else "kc"
    rows = STIFF_CLAY_K_TABLE[name]
    return choice(
        f"k of the initial line, {name}, {layer.top:g}-{layer.bottom:g} m",
        f"{model.k:g} kN/m3",
        how_given(model.k, None, within=within),
        f"the initial line p = ({name} x) y up to the "
        f"{model.loading} curve (Reese, Cox & Koop 1975; COM624P Eq. 3.16 "
        "and Table 3.2, GEC 9 Eq. A-9 and Table A-1), by the average "
        "undrained shear strength to five pile diameters",
        [f"{k:,.0f} kN/m3 for {lo:g}-{hi:g} kPa" for lo, hi, k in rows
         if abs(k - model.k) > 0.5]
        + ["a smaller k: a softer start to the curve, more deflection at "
           "small loads"])


def _weak_rock_choices(model, layer, within: str = "layers") -> list:
    """A weak-rock layer's alpha_r (from RQD) and k_rm (GM-159), each
    read from a field of the input ``within``."""
    from geotech_common.choices import choice, how_given
    span = f"{layer.top:g}-{layer.bottom:g} m"
    krm_options = (
        (0.0005, "0.0005, the softer end: more deflection"),
        (0.00005, "0.00005, the stiffer end: less deflection (a 1.2 m "
                  "shaft in rock: 0.23 to 0.10 mm at 500 kN, 5 % less at "
                  "8,000 kN)"),
        (0.0004, "0.0004 (GEC 10 p. 9-31: decomposed rock): slightly less "
                 "deflection than 0.0005"))
    return [
        choice(f"strength reduction alpha_r {span}",
               f"{model.alpha_r:.3g} (RQD {model.RQD:g} %)",
               how_given(model.RQD, _field_default(model, "RQD"),
                         within=within),
               "alpha_r = 1/3 at RQD 100 %, rising linearly to 1.0 at RQD 0 "
               "(Reese 1997, GEC 9 p. 231); RQD 100 % gives the printed lower "
               "bound, alpha_r = 1/3",
               ["a measured RQD (RQD): RQD below 100 % raises alpha_r and "
                "p_ur, so less deflection",
                "RQD 0: alpha_r = 1.0, p_ur three times as large, so less "
                "deflection"]),
        choice(f"k_rm {span}", model.krm,
               how_given(model.krm, _field_default(model, "krm"),
                         within=within),
               "strain at 50 % of the ultimate, 0.00005 to 0.0005 (GEC 9 "
               "p. 232); the lower value gives a stiffer initial response "
               "(GEC 10 p. 9-31), so 0.0005 is the softer end",
               [text for value, text in krm_options
                if abs(value - model.krm) > 1e-12]),
    ]


def _with_rock_surface(layers: list) -> list:
    """The layers, each weak-rock model measuring z_r from the rock
    surface (GEC 9 App. A.5): a model given no ``rock_top`` takes the one
    of the run of weak-rock layers that holds it, which is the first
    layer's ``rock_top`` where given, else its top. Such a model is
    copied, so the caller's object is left as it was."""
    out, top = [], None
    for layer in layers:
        model = layer.py_model
        if not isinstance(model, WeakRock):
            top = None
            out.append(layer)
            continue
        if top is None:
            top = model.rock_top if model.rock_top is not None else layer.top
        if model.rock_top is None:
            model = copy.copy(model)
            model.rock_top = top
            layer = dataclasses.replace(layer, py_model=model)
        out.append(layer)
    return out


class LateralPileAnalysis:
    """Lateral pile analysis using p-y curve method.

    Accepts a Pile object and a list of SoilLayer objects, then runs the
    iterative finite difference solver to compute deflection, moment, shear,
    slope, and soil reaction profiles along the pile.

    Parameters
    ----------
    pile : Pile
        Pile definition with geometry and material properties.
    layers : list of SoilLayer
        Soil layers with associated p-y curve models.

    Examples
    --------
    >>> from lateral_pile import Pile, SoilLayer, LateralPileAnalysis
    >>> from lateral_pile.py_curves import SoftClayMatlock, SandAPI
    >>>
    >>> pile = Pile(length=20.0, diameter=0.6, thickness=0.012, E=200e6)
    >>> layers = [
    ...     SoilLayer(top=0.0, bottom=5.0,
    ...               py_model=SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5)),
    ...     SoilLayer(top=5.0, bottom=20.0,
    ...               py_model=SandAPI(phi=35.0, gamma=10.0, k=16000)),
    ... ]
    >>> analysis = LateralPileAnalysis(pile, layers)
    >>> results = analysis.solve(Vt=100.0, Mt=0.0, Q=500.0, head_condition='free')
    >>> print(f"Pile head deflection: {results.y_top:.4f} m")
    """

    def __init__(self, pile: Pile, layers: List[SoilLayer]):
        self.pile = pile
        self.layers = sorted(layers, key=lambda l: l.top)
        validate_layers(self.layers, self.pile.length)
        self.layers = _with_rock_surface(self.layers)

    def solve(
        self,
        Vt: float = 0.0,
        Mt: float = 0.0,
        Q: float = 0.0,
        head_condition: str = 'free',
        rotational_stiffness: float = 0.0,
        n_elements: int = 100,
        tolerance: float = 1e-5,
        max_iterations: int = 500,
        ei_tolerance: float = 1e-3,
        max_ei_iterations: int = 20,
        stickup: float = 0.0,
    ) -> Results:
        """Run the lateral pile analysis.

        Parameters
        ----------
        Vt : float
            Lateral load at pile head (kN). Default 0. When stickup > 0
            the head load is applied at the top of the stickup.
        Mt : float
            Moment at pile head (kN-m). Default 0.
        Q : float
            Axial load on pile (kN), positive in compression. Default 0.
        head_condition : str
            'free': specified shear and moment (default).
            'fixed': specified shear and zero rotation.
            'partial': specified shear and rotational stiffness.
            Applied at the top of the stickup when stickup > 0.
        rotational_stiffness : float
            Rotational stiffness at pile head (kN-m/rad). Only used when
            head_condition='partial'. Default 0.
        n_elements : int
            Number of pile segments for finite difference, over the total
            (stickup + embedded) length. Default 100.
        tolerance : float
            Convergence tolerance on deflection. Default 1e-5.
        max_iterations : int
            Maximum number of solver iterations. Default 500 (100 before
            GM-208; softening p-y curves take more near their peak). When
            the iteration does not converge and the soil cannot resist the
            load, ``lateral_pile.solver.SoilResistanceExceeded`` is raised
            (no solution); when it can, the result has ``converged``
            False and a ``message``.
        ei_tolerance : float
            Convergence tolerance on EI for cracked concrete sections.
            Default 1e-3.
        max_ei_iterations : int
            Maximum number of outer EI iterations. Default 20.
        stickup : float
            Above-ground free (unsupported) pile length (m). Default 0
            (head at the ground surface — identical to previous behavior).
            When > 0, the mesh extends above grade with zero soil
            resistance (p = 0) over the stickup and the head load /
            boundary conditions act at the top of the stickup
            (pile-bent / column applications). ``pile.length`` remains
            the EMBEDDED length; soil layer depths are unchanged
            (measured from the ground surface). Above-grade EI is taken
            as the EI at grade when variable sections are defined.

        Returns
        -------
        Results
            Analysis results with deflection, moment, shear, slope,
            soil reaction profiles, and summary properties. The depth
            array spans [-stickup, pile.length]; node depths above grade
            are negative.
        """
        if stickup < 0:
            raise ValueError(f"stickup must be >= 0, got {stickup}")

        n_nodes = n_elements + 1
        z_top = -stickup if stickup > 0 else 0.0
        z_nodes = np.linspace(z_top, self.pile.length, n_nodes)

        # Get initial EI at each node. Section depths are measured from
        # grade; above-grade nodes use the EI at grade (z clamped to 0).
        EI_values = self.pile.get_EI_profile(np.maximum(z_nodes, 0.0))

        # Build list of p-y functions for each node (no soil above grade).
        # Tolerance matches the solver's grade tolerance so the node that
        # lands exactly at grade (within floating point) keeps its soil.
        grade_tol = 1e-9 * max(self.pile.length + stickup, 1.0)
        zero_p = lambda y, z, b: 0.0  # noqa: E731
        py_functions = []
        for i, z in enumerate(z_nodes):
            if z < -grade_tol:
                py_functions.append(zero_p)
            else:
                py_functions.append(self._make_py_function(max(z, 0.0)))

        # Check if pile has an RC section for cracked-EI iteration
        rc_section = getattr(self.pile, 'rc_section', None)

        if rc_section is None:
            # Standard solve: single pass with constant EI
            solver_result = solve_lateral_pile(
                pile_length=self.pile.length,
                EI_values=EI_values,
                py_functions=py_functions,
                Vt=Vt, Mt=Mt, Q=Q,
                head_condition=head_condition,
                rotational_stiffness=rotational_stiffness,
                n_elements=n_elements,
                tolerance=tolerance,
                max_iterations=max_iterations,
                pile_diameter=self.pile.diameter,
                stickup=stickup,
            )

            return Results(
                z=solver_result.z,
                deflection=solver_result.y,
                slope=solver_result.slope,
                moment=solver_result.moment,
                shear=solver_result.shear,
                soil_reaction=solver_result.soil_reaction,
                Es=solver_result.Es,
                iterations=solver_result.iterations,
                converged=solver_result.converged,
                message=solver_result.message,
                pile_length=self.pile.length,
                pile_diameter=self.pile.diameter,
                Vt=Vt, Mt=Mt, Q=Q,
                stickup=stickup,
                choices=py_choices(self.layers),
            )

        # Cracked-EI outer iteration loop for RC sections
        ei_converged = False
        ei_iter = 0

        for ei_iter in range(1, max_ei_iterations + 1):
            solver_result = solve_lateral_pile(
                pile_length=self.pile.length,
                EI_values=EI_values,
                py_functions=py_functions,
                Vt=Vt, Mt=Mt, Q=Q,
                head_condition=head_condition,
                rotational_stiffness=rotational_stiffness,
                n_elements=n_elements,
                tolerance=tolerance,
                max_iterations=max_iterations,
                pile_diameter=self.pile.diameter,
                stickup=stickup,
            )

            # Update EI based on computed moments
            EI_new = np.array([
                rc_section.get_effective_EI(solver_result.moment[i])
                for i in range(n_nodes)
            ])

            # Check EI convergence
            EI_max = np.max(EI_values)
            ei_diff = np.max(np.abs(EI_new - EI_values)) / EI_max
            if ei_diff < ei_tolerance:
                ei_converged = True
                EI_values = EI_new
                break

            # Relaxation for stability: always blend to prevent oscillation.
            # Branson's cubic ratio amplifies small moment changes, so the
            # EI update is inherently oscillatory without damping.
            EI_values = 0.5 * EI_values + 0.5 * EI_new

        if not ei_converged:
            warnings.warn(
                f"Cracked-EI iteration did not converge after "
                f"{max_ei_iterations} iterations. Last EI change: {ei_diff:.2e}"
            )

        # Final solve with converged EI
        solver_result = solve_lateral_pile(
            pile_length=self.pile.length,
            EI_values=EI_values,
            py_functions=py_functions,
            Vt=Vt, Mt=Mt, Q=Q,
            head_condition=head_condition,
            rotational_stiffness=rotational_stiffness,
            n_elements=n_elements,
            tolerance=tolerance,
            max_iterations=max_iterations,
            pile_diameter=self.pile.diameter,
            stickup=stickup,
        )

        return Results(
            z=solver_result.z,
            deflection=solver_result.y,
            slope=solver_result.slope,
            moment=solver_result.moment,
            shear=solver_result.shear,
            soil_reaction=solver_result.soil_reaction,
            Es=solver_result.Es,
            iterations=solver_result.iterations,
            converged=solver_result.converged,
            message=solver_result.message,
            pile_length=self.pile.length,
            pile_diameter=self.pile.diameter,
            Vt=Vt, Mt=Mt, Q=Q,
            EI_profile=EI_values,
            ei_iterations=ei_iter,
            stickup=stickup,
            choices=py_choices(self.layers),
        )

    def _make_py_function(self, z: float):
        """Create a p-y function for a specific depth.

        Returns a callable that takes (y, z, b) and returns p.
        If the depth is outside all soil layers, returns a function
        that always returns 0 (no soil resistance).
        """
        try:
            layer = get_layer_at_depth(self.layers, z)
        except ValueError:
            # No soil layer at this depth — no resistance
            return lambda y, z, b: 0.0

        py_model = layer.py_model

        def py_func(y, z, b):
            return py_model.get_p(y, z, b)

        return py_func
