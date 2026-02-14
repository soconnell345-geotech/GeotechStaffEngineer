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
    ) -> Results:
        """Run the lateral pile analysis.

        Parameters
        ----------
        Vt : float
            Lateral load at pile head (kN). Default 0.
        Mt : float
            Moment at pile head (kN-m). Default 0.
        Q : float
            Axial load on pile (kN), positive in compression. Default 0.
        head_condition : str
            'free': specified shear and moment (default).
            'fixed': specified shear and zero rotation.
            'partial': specified shear and rotational stiffness.
        rotational_stiffness : float
            Rotational stiffness at pile head (kN-m/rad). Only used when
            head_condition='partial'. Default 0.
        n_elements : int
            Number of pile segments for finite difference. Default 100.
        tolerance : float
            Convergence tolerance on deflection. Default 1e-5.
        max_iterations : int
            Maximum number of solver iterations. Default 100.

        Returns
        -------
        Results
            Analysis results with deflection, moment, shear, slope,
            soil reaction profiles, and summary properties.
        """
        n_nodes = n_elements + 1
        z_nodes = np.linspace(0, self.pile.length, n_nodes)

        # Get EI at each node
        EI_values = self.pile.get_EI_profile(z_nodes)

        # Build list of p-y functions for each node
        py_functions = []
        for i, z in enumerate(z_nodes):
            py_func = self._make_py_function(z)
            py_functions.append(py_func)

        # Run the solver
        solver_result = solve_lateral_pile(
            pile_length=self.pile.length,
            EI_values=EI_values,
            py_functions=py_functions,
            Vt=Vt,
            Mt=Mt,
            Q=Q,
            head_condition=head_condition,
            rotational_stiffness=rotational_stiffness,
            n_elements=n_elements,
            tolerance=tolerance,
            max_iterations=max_iterations,
            pile_diameter=self.pile.diameter,
        )

        # Package into Results
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
            Vt=Vt,
            Mt=Mt,
            Q=Q,
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
