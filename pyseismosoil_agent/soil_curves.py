"""
Nonlinear soil curve generation and Vs profile analysis using PySeismoSoil.
"""

import numpy as np

from pyseismosoil_agent.pyseismosoil_utils import import_pyseismosoil
from pyseismosoil_agent.results import CurveResult, VsProfileResult


_VALID_MODELS = {"MKZ", "HH"}

_MKZ_REQUIRED_KEYS = {"gamma_ref", "beta", "s", "Gmax"}
_HH_REQUIRED_KEYS = {"gamma_t", "a", "gamma_ref", "beta", "s", "Gmax", "mu", "Tmax", "d"}


def _validate_curve_inputs(model, params, n_points):
    """Validate curve generation inputs."""
    if model not in _VALID_MODELS:
        raise ValueError(f"model must be one of {sorted(_VALID_MODELS)}, got '{model}'")
    if n_points < 2:
        raise ValueError(f"n_points must be >= 2, got {n_points}")

    if model == "MKZ":
        missing = _MKZ_REQUIRED_KEYS - set(params.keys())
        if missing:
            raise ValueError(f"MKZ model requires parameters: {sorted(_MKZ_REQUIRED_KEYS)}. Missing: {sorted(missing)}")
    elif model == "HH":
        missing = _HH_REQUIRED_KEYS - set(params.keys())
        if missing:
            raise ValueError(f"HH model requires parameters: {sorted(_HH_REQUIRED_KEYS)}. Missing: {sorted(missing)}")


def generate_curves(
    model="MKZ",
    params=None,
    strain_min=1e-4,
    strain_max=10.0,
    n_points=50,
) -> CurveResult:
    """Generate G/Gmax and damping curves from constitutive model.

    Parameters
    ----------
    model : str
        Model type: 'MKZ' (Modified Kodner-Zelasko) or 'HH' (Hybrid Hyperbolic).
    params : dict
        Model parameters. MKZ requires: gamma_ref, beta, s, Gmax.
        HH requires: gamma_t, a, gamma_ref, beta, s, Gmax, mu, Tmax, d.
    strain_min : float
        Minimum shear strain in percent. Default 1e-4.
    strain_max : float
        Maximum shear strain in percent. Default 10.0.
    n_points : int
        Number of log-spaced strain points. Default 50.

    Returns
    -------
    CurveResult
        G/Gmax and damping curves.
    """
    if params is None:
        raise ValueError("params must be provided")

    _validate_curve_inputs(model, params, n_points)

    MKZ_Param, HH_Param, _ = import_pyseismosoil()

    if model == "MKZ":
        param_obj = MKZ_Param(params)
    else:
        param_obj = HH_Param(params)

    strain = np.geomspace(strain_min, strain_max, n_points)
    G_Gmax = param_obj.get_GGmax(strain)
    damping = param_obj.get_damping(strain)

    return CurveResult(
        model=model,
        params=dict(params),
        n_points=n_points,
        strain_pct=strain,
        G_Gmax=G_Gmax,
        damping_pct=damping,
    )


def _validate_profile_inputs(thicknesses, vs_values):
    """Validate Vs profile inputs."""
    if len(thicknesses) < 2:
        raise ValueError(f"Need at least 2 layers, got {len(thicknesses)}")
    if len(thicknesses) != len(vs_values):
        raise ValueError(
            f"thicknesses and vs_values must have same length: "
            f"{len(thicknesses)} vs {len(vs_values)}"
        )
    if thicknesses[-1] != 0:
        raise ValueError("Last layer thickness must be 0 (halfspace)")
    for i, (t, v) in enumerate(zip(thicknesses, vs_values)):
        if i < len(thicknesses) - 1 and t <= 0:
            raise ValueError(f"Layer {i} thickness must be > 0, got {t}")
        if v <= 0:
            raise ValueError(f"Layer {i} Vs must be > 0, got {v}")


def analyze_vs_profile(
    thicknesses,
    vs_values,
) -> VsProfileResult:
    """Analyze a Vs profile for site characterization parameters.

    Parameters
    ----------
    thicknesses : list of float
        Layer thicknesses in meters. Last layer must be 0 (halfspace).
    vs_values : list of float
        Shear wave velocity for each layer in m/s.

    Returns
    -------
    VsProfileResult
        Site characterization: Vs30, f0, z1, etc.
    """
    thk = list(thicknesses)
    vs = list(vs_values)
    _validate_profile_inputs(thk, vs)

    _, _, Vs_Profile = import_pyseismosoil()

    data = np.column_stack([thk, vs])
    vsp = Vs_Profile(data)

    return VsProfileResult(
        n_layers=vsp.n_layer,
        vs30=float(vsp.vs30),
        f0_bh=float(vsp.get_f0_BH()),
        f0_ro=float(vsp.get_f0_RO()),
        z1=float(vsp.get_z1()),
        z_max=float(vsp.z_max),
        thicknesses=thk,
        vs_values=vs,
        depth_array=[float(x) for x in vsp.get_depth_array()],
    )
