"""
Top-level analysis runner for lateral pile analysis.

Provides the LateralPileAnalysis class that ties together the pile definition,
soil layers, p-y curve models, and finite difference solver into a clean API
suitable for use by both human engineers and LLM agents.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).

References
----------
- COM624P Manual: FHWA-SA-91-048 (Wang & Reese, 1993)
- FHWA GEC-13: FHWA-HIF-18-031
"""

import warnings
from typing import List, Optional

import numpy as np

from lateral_pile.pile import Pile
from lateral_pile.soil import SoilLayer, get_layer_at_depth, validate_layers
from lateral_pile.solver import solve_lateral_pile
from lateral_pile.results import Results


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

    def solve(
        self,
        Vt: float = 0.0,
        Mt: float = 0.0,
        Q: float = 0.0,
        head_condition: str = 'free',
        rotational_stiffness: float = 0.0,
        n_elements: int = 100,
        tolerance: float = 1e-5,
        max_iterations: int = 100,
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
            Maximum number of solver iterations. Default 100.
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
                pile_length=self.pile.length,
                pile_diameter=self.pile.diameter,
                Vt=Vt, Mt=Mt, Q=Q,
                stickup=stickup,
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
            pile_length=self.pile.length,
            pile_diameter=self.pile.diameter,
            Vt=Vt, Mt=Mt, Q=Q,
            EI_profile=EI_values,
            ei_iterations=ei_iter,
            stickup=stickup,
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
