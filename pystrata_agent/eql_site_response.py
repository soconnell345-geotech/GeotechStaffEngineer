"""
1D equivalent-linear and linear elastic site response using pystrata.

Wraps the pystrata library to provide SHAKE-type 1D site response analysis
with dict-based I/O for LLM agents. Supports Darendeli (2001), Menq (2003),
linear elastic, and custom G/Gmax + damping curve soil models.

References:
    Schnabel, P.B., Lysmer, J. & Seed, H.B. (1972). "SHAKE: A Computer
    Program for Earthquake Response Analysis of Horizontally Layered Sites."
    Report EERC 72-12, UC Berkeley.

    Darendeli, M.B. (2001). "Development of a New Family of Normalized
    Modulus Reduction and Material Damping Curves." PhD Dissertation,
    UT Austin.

    Menq, F.Y. (2003). "Dynamic Properties of Sandy and Gravelly Soils."
    PhD Dissertation, UT Austin.
"""

import numpy as np

from pystrata_agent.results import EQLSiteResponseResult


# Valid soil model names
_SOIL_MODELS = ("darendeli", "menq", "linear", "custom")


# ===========================================================================
# Input validation
# ===========================================================================

def _validate_eql_inputs(layers, strain_ratio=0.65, tolerance=0.01,
                         max_iterations=15, max_freq_hz=25.0,
                         wave_frac=0.2):
    """Validate all site response inputs, raising ValueError for bad values.

    Parameters
    ----------
    layers : list of dict
        Soil layers from surface to bedrock. Last layer must be bedrock
        half-space (thickness=0).
    strain_ratio : float
        EQL strain ratio (0.5 to 1.0).
    tolerance : float
        Convergence tolerance (> 0).
    max_iterations : int
        Maximum EQL iterations (>= 1).
    max_freq_hz : float
        Max frequency for auto-discretization (> 0).
    wave_frac : float
        Wavelength fraction for auto-discretization (> 0).
    """
    # -- layers --
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list of dicts")

    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{i}] must be a dict")

        # Required keys
        for key in ("thickness", "Vs", "unit_wt", "soil_model"):
            if key not in layer:
                raise ValueError(
                    f"layers[{i}] missing required key '{key}'")

        # Numeric range checks
        if i < len(layers) - 1:
            # Non-bedrock layers must have positive thickness
            if layer["thickness"] <= 0:
                raise ValueError(
                    f"layers[{i}]['thickness'] must be positive, "
                    f"got {layer['thickness']}")
        else:
            # Last layer must be bedrock half-space
            if layer["thickness"] != 0:
                raise ValueError(
                    "Last layer must be the bedrock half-space "
                    "(thickness=0). Got thickness="
                    f"{layer['thickness']}")

        if layer["Vs"] <= 0:
            raise ValueError(
                f"layers[{i}]['Vs'] must be positive, got {layer['Vs']}")
        if layer["unit_wt"] <= 0:
            raise ValueError(
                f"layers[{i}]['unit_wt'] must be positive, "
                f"got {layer['unit_wt']}")

        # Soil model validation
        model = layer["soil_model"].lower().strip()
        if model not in _SOIL_MODELS:
            raise ValueError(
                f"layers[{i}]['soil_model'] = '{layer['soil_model']}' "
                f"not recognized. Valid: {', '.join(_SOIL_MODELS)}")

        # Model-specific required params
        if model == "darendeli":
            if "plas_index" not in layer:
                raise ValueError(
                    f"layers[{i}] (darendeli) requires 'plas_index'")
            if layer["plas_index"] < 0:
                raise ValueError(
                    f"layers[{i}]['plas_index'] must be >= 0, "
                    f"got {layer['plas_index']}")

        elif model == "linear":
            if "damping" not in layer:
                raise ValueError(
                    f"layers[{i}] (linear) requires 'damping'")
            if not (0 <= layer["damping"] < 1):
                raise ValueError(
                    f"layers[{i}]['damping'] must be in [0, 1), "
                    f"got {layer['damping']}")

        elif model == "custom":
            for key in ("strains", "mod_reduc", "damping_values"):
                if key not in layer:
                    raise ValueError(
                        f"layers[{i}] (custom) requires '{key}'")
            n_s = len(layer["strains"])
            if len(layer["mod_reduc"]) != n_s:
                raise ValueError(
                    f"layers[{i}] mod_reduc length ({len(layer['mod_reduc'])}) "
                    f"must match strains length ({n_s})")
            if len(layer["damping_values"]) != n_s:
                raise ValueError(
                    f"layers[{i}] damping_values length "
                    f"({len(layer['damping_values'])}) "
                    f"must match strains length ({n_s})")

    # At least 2 layers needed (1 soil + 1 bedrock)
    if len(layers) < 2:
        raise ValueError(
            "At least 2 layers required (1 soil layer + 1 bedrock half-space)")

    # -- scalar params --
    if not (0.5 <= strain_ratio <= 1.0):
        raise ValueError(
            f"strain_ratio must be in [0.5, 1.0], got {strain_ratio}")
    if tolerance <= 0:
        raise ValueError(f"tolerance must be > 0, got {tolerance}")
    if max_iterations < 1:
        raise ValueError(
            f"max_iterations must be >= 1, got {max_iterations}")
    if max_freq_hz <= 0:
        raise ValueError(f"max_freq_hz must be > 0, got {max_freq_hz}")
    if wave_frac <= 0:
        raise ValueError(f"wave_frac must be > 0, got {wave_frac}")


# ===========================================================================
# Soil type builders
# ===========================================================================

def _compute_stress_mean(layers, layer_idx):
    """Auto-compute mean effective stress at layer midpoint.

    Uses simplified calculation: sigma_v = sum(unit_wt * thickness) above,
    then stress_mean = sigma_v_mid * (1 + 2*K0) / 3 with K0=0.5.
    """
    K0 = 0.5
    cumulative_depth = 0.0
    sigma_v_top = 0.0

    for i in range(layer_idx):
        h = layers[i]["thickness"]
        gamma = layers[i]["unit_wt"]
        sigma_v_top += gamma * h
        cumulative_depth += h

    # Stress at midpoint of target layer
    h_target = layers[layer_idx]["thickness"]
    gamma_target = layers[layer_idx]["unit_wt"]
    sigma_v_mid = sigma_v_top + gamma_target * (h_target / 2.0)

    stress_mean = sigma_v_mid * (1.0 + 2.0 * K0) / 3.0
    return max(stress_mean, 5.0)  # Floor at 5 kPa


def _build_soil_type(pystrata, layer, layers, layer_idx):
    """Build a pystrata SoilType from a layer dict.

    Parameters
    ----------
    pystrata : module
        The imported pystrata module.
    layer : dict
        Layer definition dict.
    layers : list of dict
        All layers (for stress_mean computation).
    layer_idx : int
        Index of this layer in the layers list.

    Returns
    -------
    pystrata.site.SoilType or subclass
    """
    model = layer["soil_model"].lower().strip()
    unit_wt = layer["unit_wt"]

    if model == "darendeli":
        plas_index = layer["plas_index"]
        ocr = layer.get("ocr", 1.0)
        if "stress_mean" in layer:
            stress_mean = layer["stress_mean"]
        else:
            stress_mean = _compute_stress_mean(layers, layer_idx)
        return pystrata.site.DarendeliSoilType(
            unit_wt=unit_wt,
            plas_index=plas_index,
            ocr=ocr,
            stress_mean=stress_mean,
        )

    elif model == "menq":
        uniformity_coeff = layer.get("uniformity_coeff", 10.0)
        diam_mean = layer.get("diam_mean", 5.0)
        if "stress_mean" in layer:
            stress_mean = layer["stress_mean"]
        else:
            stress_mean = _compute_stress_mean(layers, layer_idx)
        return pystrata.site.MenqSoilType(
            unit_wt=unit_wt,
            uniformity_coeff=uniformity_coeff,
            diam_mean=diam_mean,
            stress_mean=stress_mean,
        )

    elif model == "linear":
        damping = layer["damping"]
        return pystrata.site.SoilType(
            name="Linear",
            unit_wt=unit_wt,
            mod_reduc=None,
            damping=damping,
        )

    elif model == "custom":
        strains = np.array(layer["strains"])
        mod_reduc_vals = np.array(layer["mod_reduc"])
        damping_vals = np.array(layer["damping_values"])

        mod_reduc = pystrata.site.NonlinearProperty(
            name="Custom G/Gmax",
            strains=strains,
            values=mod_reduc_vals,
        )
        damping_prop = pystrata.site.NonlinearProperty(
            name="Custom Damping",
            strains=strains,
            values=damping_vals,
        )
        return pystrata.site.SoilType(
            name="Custom",
            unit_wt=unit_wt,
            mod_reduc=mod_reduc,
            damping=damping_prop,
        )


# ===========================================================================
# Public API
# ===========================================================================

def analyze_eql_site_response(
    layers,
    motion=None,
    accel_history=None,
    dt=None,
    strain_ratio=0.65,
    tolerance=0.01,
    max_iterations=15,
    max_freq_hz=25.0,
    wave_frac=0.2,
):
    """Run 1D equivalent-linear site response analysis using pystrata.

    Parameters
    ----------
    layers : list of dict
        Soil layers from surface to bedrock. Each dict requires:
        ``thickness`` (m), ``Vs`` (m/s), ``unit_wt`` (kN/m3),
        ``soil_model`` ('darendeli', 'menq', 'linear', 'custom').
        Darendeli requires: ``plas_index``. Optional: ``ocr``, ``stress_mean``.
        Menq optional: ``uniformity_coeff``, ``diam_mean``, ``stress_mean``.
        Linear requires: ``damping`` (decimal).
        Custom requires: ``strains``, ``mod_reduc``, ``damping_values``.
        Last layer must be bedrock half-space (thickness=0).
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse').
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    strain_ratio : float
        Ratio of effective to maximum shear strain. Default 0.65.
    tolerance : float
        Convergence tolerance for strain-compatible iteration. Default 0.01.
    max_iterations : int
        Maximum number of EQL iterations. Default 15.
    max_freq_hz : float
        Maximum frequency for profile auto-discretization (Hz). Default 25.
    wave_frac : float
        Wavelength fraction for auto-discretization. Default 0.2 (1/5).

    Returns
    -------
    EQLSiteResponseResult
        Analysis results with surface motion, spectra, and depth profiles.

    Raises
    ------
    ValueError
        For invalid input parameters.
    ImportError
        If pystrata is not installed.
    """
    _validate_eql_inputs(layers, strain_ratio, tolerance,
                         max_iterations, max_freq_hz, wave_frac)

    from opensees_agent.ground_motions import validate_motion_input
    accel_g, dt_motion = validate_motion_input(motion, accel_history, dt)
    motion_name = motion if motion else "custom"

    from pystrata_agent.pystrata_utils import import_pystrata
    pystrata = import_pystrata()

    return _run_eql_analysis(
        pystrata, layers, accel_g, dt_motion, motion_name,
        strain_ratio, tolerance, max_iterations, max_freq_hz, wave_frac,
        analysis_type="equivalent_linear",
    )


def analyze_linear_site_response(
    layers,
    motion=None,
    accel_history=None,
    dt=None,
    max_freq_hz=25.0,
    wave_frac=0.2,
):
    """Run 1D linear elastic site response analysis using pystrata.

    Same as ``analyze_eql_site_response`` but with no strain-compatible
    iteration. Uses initial (small-strain) properties throughout.

    Parameters
    ----------
    layers : list of dict
        Same format as ``analyze_eql_site_response``.
    motion : str, optional
        Built-in motion name.
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    max_freq_hz : float
        Max frequency for auto-discretization (Hz). Default 25.
    wave_frac : float
        Wavelength fraction for auto-discretization. Default 0.2.

    Returns
    -------
    EQLSiteResponseResult
        Results with analysis_type="linear_elastic", n_iterations=0.
    """
    # Validate with default EQL params (they won't be used)
    _validate_eql_inputs(layers, max_freq_hz=max_freq_hz, wave_frac=wave_frac)

    from opensees_agent.ground_motions import validate_motion_input
    accel_g, dt_motion = validate_motion_input(motion, accel_history, dt)
    motion_name = motion if motion else "custom"

    from pystrata_agent.pystrata_utils import import_pystrata
    pystrata = import_pystrata()

    return _run_eql_analysis(
        pystrata, layers, accel_g, dt_motion, motion_name,
        strain_ratio=0.65, tolerance=0.01, max_iterations=1,
        max_freq_hz=max_freq_hz, wave_frac=wave_frac,
        analysis_type="linear_elastic",
    )


# ===========================================================================
# Internal analysis runner
# ===========================================================================

def _run_eql_analysis(pystrata, layers, accel_g, dt_motion, motion_name,
                      strain_ratio, tolerance, max_iterations,
                      max_freq_hz, wave_frac, analysis_type):
    """Build pystrata model, run analysis, extract results."""

    # 1. Build pystrata layers
    pystrata_layers = []
    for i, layer in enumerate(layers):
        soil_type = _build_soil_type(pystrata, layer, layers, i)
        pystrata_layers.append(
            pystrata.site.Layer(soil_type, layer["thickness"], layer["Vs"])
        )

    # 2. Build profile and auto-discretize
    profile = pystrata.site.Profile(pystrata_layers)
    profile = profile.auto_discretize(
        max_freq=max_freq_hz, wave_frac=wave_frac)

    # 3. Build motion (accels must be in g)
    ts_motion = pystrata.motion.TimeSeriesMotion(
        filename="",
        description=motion_name,
        time_step=dt_motion,
        accels=accel_g,
    )

    # 4. Create calculator
    if analysis_type == "linear_elastic":
        calc = pystrata.propagation.LinearElasticCalculator()
    else:
        calc = pystrata.propagation.EquivalentLinearCalculator(
            strain_ratio=strain_ratio,
            tolerance=tolerance,
            max_iterations=max_iterations,
        )

    # 5. Define outputs
    freqs = np.logspace(-1, 2, num=500)
    loc_surface = pystrata.output.OutputLocation("outcrop", index=0)
    loc_bedrock = pystrata.output.OutputLocation("outcrop", index=-1)

    outputs = pystrata.output.OutputCollection([
        pystrata.output.AccelerationTSOutput(loc_surface),
        pystrata.output.ResponseSpectrumOutput(freqs, loc_surface, 0.05),
        pystrata.output.ResponseSpectrumOutput(freqs, loc_bedrock, 0.05),
        pystrata.output.MaxStrainProfile(),
        pystrata.output.MaxAccelProfile(),
        pystrata.output.InitialVelProfile(),
        pystrata.output.CompatVelProfile(),
    ])

    # 6. Run analysis
    loc_input = profile.location("outcrop", index=-1)
    calc(ts_motion, profile, loc_input)

    # 7. Compute outputs
    outputs(calc)

    # 8. Extract results
    # Surface acceleration time history
    accel_ts_output = outputs[0]
    surface_accel = np.array(accel_ts_output.values).flatten()
    time_arr = np.array(accel_ts_output.refs).flatten()

    # Surface response spectrum (pystrata outputs Sa vs frequency)
    sa_surface_output = outputs[1]
    Sa_surface = np.array(sa_surface_output.values).flatten()
    output_freqs = np.array(sa_surface_output.refs).flatten()
    periods = 1.0 / output_freqs

    # Input response spectrum (at bedrock outcrop)
    sa_input_output = outputs[2]
    Sa_input = np.array(sa_input_output.values).flatten()

    # Sort by ascending period (freqs are ascending, so periods descending)
    sort_idx = np.argsort(periods)
    periods = periods[sort_idx]
    Sa_surface = Sa_surface[sort_idx]
    Sa_input = Sa_input[sort_idx]

    # Depth profiles
    strain_output = outputs[3]
    accel_output = outputs[4]
    init_vel_output = outputs[5]
    compat_vel_output = outputs[6]

    depths_strain = np.array(strain_output.refs).flatten()
    # Linear elastic calculator returns None for strain values; replace with 0
    raw_strain = strain_output.values
    max_strain = np.array(
        [0.0 if v is None else float(v) for v in raw_strain])

    depths_accel = np.array(accel_output.refs).flatten()
    max_accel = np.array(accel_output.values).flatten()

    depths_vs = np.array(init_vel_output.refs).flatten()
    initial_Vs = np.array(init_vel_output.values, dtype=float).flatten()
    compatible_Vs = np.array(compat_vel_output.values, dtype=float).flatten()

    # Use strain profile depths as canonical (they should all match)
    depths = depths_strain

    # 9. Compute scalars
    pga_input = float(np.max(np.abs(accel_g)))
    pga_surface = float(np.max(np.abs(surface_accel)))
    amplification = pga_surface / pga_input if pga_input > 0 else 0.0

    # Number of soil layers (exclude bedrock half-space)
    n_soil_layers = sum(1 for L in layers if L["thickness"] > 0)
    total_depth = sum(L["thickness"] for L in layers)

    # Convergence info
    if analysis_type == "linear_elastic":
        n_iterations = 0
        converged = True
    else:
        # pystrata does not expose iteration count; check convergence via
        # max_error on the profile (set by the EQL iteration loop)
        max_error = float(max(calc.profile.max_error))
        converged = bool(max_error < calc.tolerance)
        # Estimate iteration count: not available from pystrata, report 0
        # if converged, max_iterations if not
        n_iterations = 0 if converged else max_iterations

    return EQLSiteResponseResult(
        analysis_type=analysis_type,
        total_depth_m=total_depth,
        n_layers=n_soil_layers,
        motion_name=motion_name,
        pga_input_g=pga_input,
        pga_surface_g=pga_surface,
        amplification_factor=amplification,
        n_iterations=n_iterations,
        converged=converged,
        time=time_arr,
        surface_accel_g=surface_accel,
        depths=depths,
        max_strain_pct=max_strain * 100.0,
        max_accel_g=max_accel,
        initial_Vs=initial_Vs,
        compatible_Vs=compatible_Vs,
        periods=periods,
        Sa_surface_g=Sa_surface,
        Sa_input_g=Sa_input,
    )
