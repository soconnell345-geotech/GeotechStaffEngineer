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

def _max_abs_error_percent(profile):
    """The largest change, in percent, of any soil layer's shear modulus or
    damping over the last iteration. pystrata's own ``max_error`` keeps the
    sign of the change, so a layer that stiffened could read as converged;
    this takes its size. Falls back to ``max_error`` if the layer
    properties cannot be read."""
    errs = []
    for layer in list(profile)[:-1]:
        for prop in (getattr(layer, "_shear_mod", None),
                     getattr(layer, "_damping", None)):
            prev = getattr(prop, "previous", None)
            val = getattr(prop, "value", None)
            if prev is None or val is None:
                continue
            with np.errstate(divide="ignore", invalid="ignore"):
                e = 100.0 * np.max(np.abs(
                    (np.asarray(prev, float) - np.asarray(val, float))
                    / np.asarray(val, float)))
            if np.isfinite(e):
                errs.append(float(e))
    if errs:
        return max(errs)
    return float(max(profile.max_error))


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

#: Unit weight of water in the pore pressure below the water table (kN/m3).
GAMMA_W = 9.81

#: The at-rest coefficient taken when ``k0`` is not given. 0.5 is Jaky's
#: 1 - sin(phi') at phi' = 30 deg, the K0 NAVFAC DM7.2 Eq. 6-22 puts in the
#: mean effective stress.
K0_DEFAULT = 0.5

#: The lowest mean effective stress handed to a curve (kPa): the curves'
#: reference-strain terms take a power of the stress, and a layer at the
#: ground surface would otherwise start from about zero.
STRESS_MEAN_FLOOR = 5.0


def _validate_stress_inputs(water_table_depth, k0):
    if water_table_depth is not None and water_table_depth < 0:
        raise ValueError(
            "water_table_depth must be >= 0 (m below the ground surface), "
            f"got {water_table_depth}")
    if k0 is None or not (0 < k0 <= 3.0):
        raise ValueError(
            f"k0 must be above 0 and at most 3 (at-rest earth pressure "
            f"coefficient), got {k0}")


def layer_stress(layers, layer_idx, water_table_depth=None, k0=K0_DEFAULT):
    """Stresses at a layer's mid-depth, for the Darendeli and Menq curves.

    The total vertical stress is the sum of unit_wt x thickness over the
    layers above plus half this layer. The pore pressure is hydrostatic
    below ``water_table_depth`` (none when it is None: the profile is taken
    to have no water table, so effective = total). The mean effective
    stress is sigma'm = sigma'v (1 + 2 K0) / 3 (NAVFAC DM7.2 Eq. 6-22, with
    K0 for its 1 - sin(phi')), not below :data:`STRESS_MEAN_FLOOR`.

    Until GM-325 the "mean effective stress" was this formula on the TOTAL
    stress with K0 fixed at 0.5, whatever the water table.

    Returns
    -------
    dict
        ``mid_depth_m``, ``sigma_v_kPa``, ``u_kPa``, ``sigma_v_eff_kPa``,
        ``K0``, ``stress_mean_kPa`` and ``floored`` (True when the floor
        set the value).
    """
    z = 0.0
    sigma_v = 0.0
    for i in range(layer_idx):
        sigma_v += layers[i]["unit_wt"] * layers[i]["thickness"]
        z += layers[i]["thickness"]
    half = layers[layer_idx]["thickness"] / 2.0
    sigma_v += layers[layer_idx]["unit_wt"] * half
    z += half
    u = 0.0
    if water_table_depth is not None and z > water_table_depth:
        u = GAMMA_W * (z - water_table_depth)
    sigma_v_eff = sigma_v - u
    mean = sigma_v_eff * (1.0 + 2.0 * k0) / 3.0
    return {
        "mid_depth_m": z,
        "sigma_v_kPa": sigma_v,
        "u_kPa": u,
        "sigma_v_eff_kPa": sigma_v_eff,
        "K0": k0,
        "stress_mean_kPa": max(mean, STRESS_MEAN_FLOOR),
        "floored": bool(mean < STRESS_MEAN_FLOOR),
    }


def _compute_stress_mean(layers, layer_idx, water_table_depth=None,
                         k0=K0_DEFAULT):
    """Mean effective stress at a layer's mid-depth (kPa);
    :func:`layer_stress` has the rule."""
    return layer_stress(layers, layer_idx, water_table_depth,
                        k0)["stress_mean_kPa"]


def curve_layer_stresses(layers, water_table_depth=None, k0=K0_DEFAULT):
    """One row per Darendeli or Menq layer: the confining stress its
    curves were built for and where it came from (GM-325).

    A layer's own ``stress_mean`` is used as given (``stress_mean_from``
    "given"); otherwise the value is :func:`layer_stress`'s ("computed").
    ``layer`` counts from 1 at the ground surface.
    """
    rows = []
    for i, layer in enumerate(layers):
        model = str(layer["soil_model"]).lower().strip()
        if model not in ("darendeli", "menq"):
            continue
        st = layer_stress(layers, i, water_table_depth, k0)
        row = {
            "layer": i + 1,
            "mid_depth_m": round(st["mid_depth_m"], 3),
            "sigma_v_kPa": round(st["sigma_v_kPa"], 2),
            "u_kPa": round(st["u_kPa"], 2),
            "sigma_v_eff_kPa": round(st["sigma_v_eff_kPa"], 2),
        }
        if "stress_mean" in layer:
            row["stress_mean_kPa"] = round(float(layer["stress_mean"]), 2)
            row["stress_mean_from"] = "given"
        else:
            row["stress_mean_kPa"] = round(st["stress_mean_kPa"], 2)
            row["stress_mean_from"] = (
                f"computed (floor {STRESS_MEAN_FLOOR:g} kPa)"
                if st["floored"] else "computed")
        rows.append(row)
    return rows


def stress_choices(layers, water_table_depth=None, k0=K0_DEFAULT):
    """What the run chose (``geotech_common.choices``) and what to warn
    of: the motion's placement, and, when a Darendeli or Menq layer's
    stress was computed, the water table and K0 behind it.

    Returns
    -------
    (choices, warnings) : (list of dict, list of str)
    """
    from geotech_common.choices import MODULE, choice, how_given
    out = [choice(
        "input motion", "a rock outcrop motion at the top of the half-space",
        MODULE,
        "the record is taken as recorded on outcropping rock, and the "
        "half-space's own Vs and unit weight carry the radiation into it; "
        "it is not applied as a within (borehole) motion")]
    warns = []
    rows = curve_layer_stresses(layers, water_table_depth, k0)
    computed = [r["layer"] for r in rows
                if r["stress_mean_from"] != "given"]
    if not computed:
        return out, warns
    which = ", ".join(str(n) for n in computed)
    if water_table_depth is None:
        out.append(choice(
            "water table", "none in the profile: effective stress = total "
            "stress", how_given(None, None, name="water_table_depth"),
            "no water_table_depth was given. Below a real water table the "
            "confining stress is lower than this, and the Darendeli and "
            "Menq curves then give more modulus reduction and damping at "
            "a strain (give water_table_depth)"))
        warns.append(
            "No water_table_depth was given, so the mean effective stress "
            f"behind the Darendeli/Menq curves of layer(s) {which} was "
            "computed with no pore pressure (effective = total stress). "
            "If the water table is within the profile, give "
            "water_table_depth: the stress below it is lower, and the "
            "curves soften.")
    else:
        out.append(choice(
            "water table", f"{water_table_depth:g} m below the ground "
            "surface",
            how_given(water_table_depth, None, name="water_table_depth"),
            f"hydrostatic pore pressure below it at {GAMMA_W:g} kN/m3"))
    out.append(choice(
        "mean effective stress for the Darendeli and Menq curves",
        f"sigma'm = sigma'v (1 + 2 K0) / 3 with K0 = {k0:g}",
        how_given(k0, K0_DEFAULT, name="k0"),
        "NAVFAC DM7.2 Eq. 6-22 (its K0 is 1 - sin(phi'); 0.5 is phi' = 30 "
        "deg), at each layer's mid-depth, sigma'v from the layers' unit "
        f"weights and the water table; layer(s) {which} (layer_stresses "
        "lists the values). A layer's own stress_mean replaces it",
        alternatives=["a layer's own stress_mean (kPa)"]))
    floored = [r["layer"] for r in rows if "floor" in r["stress_mean_from"]]
    if floored:
        warns.append(
            "The computed mean effective stress of layer(s) "
            + ", ".join(str(n) for n in floored)
            + f" was below {STRESS_MEAN_FLOOR:g} kPa and was set to it.")
    return out, warns


def _build_soil_type(pystrata, layer, layers, layer_idx,
                     water_table_depth=None, k0=K0_DEFAULT):
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
    from pystrata_agent import compat

    model = layer["soil_model"].lower().strip()
    unit_wt = layer["unit_wt"]

    if model in ("darendeli", "menq"):
        if "stress_mean" in layer:
            stress_mean = layer["stress_mean"]
        else:
            stress_mean = _compute_stress_mean(
                layers, layer_idx, water_table_depth, k0)

    if model == "darendeli":
        return compat.darendeli_soil_type(
            pystrata,
            unit_wt=unit_wt,
            plas_index=layer["plas_index"],
            ocr=layer.get("ocr", 1.0),
            stress_mean=stress_mean,
        )

    elif model == "menq":
        return compat.menq_soil_type(
            pystrata,
            unit_wt=unit_wt,
            uniformity_coeff=layer.get("uniformity_coeff", 10.0),
            diam_mean=layer.get("diam_mean", 5.0),
            stress_mean=stress_mean,
        )

    elif model == "linear":
        return compat.linear_soil_type(
            pystrata, unit_wt=unit_wt, damping=layer["damping"])

    elif model == "custom":
        return compat.custom_soil_type(
            pystrata,
            unit_wt=unit_wt,
            strains=layer["strains"],
            mod_reduc=layer["mod_reduc"],
            damping=layer["damping_values"],
        )


# ===========================================================================
# Public API
# ===========================================================================

def analyze_eql_site_response(
    layers,
    motion=None,
    accel_history=None,
    dt=None,
    target_pga_g=None,
    strain_ratio=0.65,
    tolerance=0.01,
    max_iterations=15,
    max_freq_hz=25.0,
    wave_frac=0.2,
    water_table_depth=None,
    k0=K0_DEFAULT,
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
    target_pga_g : float, optional
        Scale the motion so its PGA equals this value (g) before the
        analysis (built-in motions have fixed PGAs). Default None.
    strain_ratio : float
        Ratio of effective to maximum shear strain. Default 0.65.
    tolerance : float
        Convergence tolerance for strain-compatible iteration, as a
        FRACTION: the iteration stops once no layer's shear modulus or
        damping changes by more than ``tolerance`` (0.01 = 1 %) between
        iterations. Default 0.01. (pystrata 0.5 works in percent and 0.6
        in decimal; ``compat`` passes each its own unit.)
    max_iterations : int
        Maximum number of EQL iterations. Default 15.
    max_freq_hz : float
        Maximum frequency for profile auto-discretization (Hz). Default 25.
    wave_frac : float
        Wavelength fraction for auto-discretization. Default 0.2 (1/5).
    water_table_depth : float, optional
        Depth to the water table (m below the ground surface), for the
        mean effective stress behind the Darendeli and Menq curves
        (:func:`layer_stress`). None (default): no water table in the
        profile, so effective stress = total stress; the result says so.
        It changes nothing else: a total-stress analysis has no pore
        pressure.
    k0 : float, optional
        At-rest earth pressure coefficient in that mean effective stress,
        sigma'm = sigma'v (1 + 2 K0) / 3. Default 0.5.

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
    _validate_stress_inputs(water_table_depth, k0)

    from opensees_agent.ground_motions import (
        validate_motion_input, motion_label)
    accel_g, dt_motion = validate_motion_input(
        motion, accel_history, dt, target_pga_g=target_pga_g)
    motion_name = motion_label(motion, target_pga_g)

    from pystrata_agent.pystrata_utils import import_pystrata
    pystrata = import_pystrata()

    return _run_eql_analysis(
        pystrata, layers, accel_g, dt_motion, motion_name,
        strain_ratio, tolerance, max_iterations, max_freq_hz, wave_frac,
        analysis_type="equivalent_linear",
        water_table_depth=water_table_depth, k0=k0,
    )


def analyze_linear_site_response(
    layers,
    motion=None,
    accel_history=None,
    dt=None,
    target_pga_g=None,
    max_freq_hz=25.0,
    wave_frac=0.2,
    water_table_depth=None,
    k0=K0_DEFAULT,
):
    """Run 1D linear elastic site response analysis using pystrata.

    Same as ``analyze_eql_site_response`` but with no strain-compatible
    iteration. Uses initial (small-strain) properties throughout. The peak
    shear strain at each sub-layer's middle is the peak of its strain
    history (pystrata's ``StrainTSOutput``), the quantity the
    equivalent-linear calculator stores as ``strain_max``; pystrata's
    linear calculator computes none itself.

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
    target_pga_g : float, optional
        Scale the motion so its PGA equals this value (g) before the
        analysis (built-in motions have fixed PGAs). Default None.
    max_freq_hz : float
        Max frequency for auto-discretization (Hz). Default 25.
    wave_frac : float
        Wavelength fraction for auto-discretization. Default 0.2.
    water_table_depth : float, optional
        Depth to the water table (m below the ground surface), for the
        mean effective stress behind the Darendeli and Menq curves
        (:func:`layer_stress`). None (default): no water table in the
        profile, so effective stress = total stress; the result says so.
        It changes nothing else: a total-stress analysis has no pore
        pressure.
    k0 : float, optional
        At-rest earth pressure coefficient in that mean effective stress,
        sigma'm = sigma'v (1 + 2 K0) / 3. Default 0.5.

    Returns
    -------
    EQLSiteResponseResult
        Results with analysis_type="linear_elastic", n_iterations=0.
    """
    # Validate with default EQL params (they won't be used)
    _validate_eql_inputs(layers, max_freq_hz=max_freq_hz, wave_frac=wave_frac)
    _validate_stress_inputs(water_table_depth, k0)

    from opensees_agent.ground_motions import (
        validate_motion_input, motion_label)
    accel_g, dt_motion = validate_motion_input(
        motion, accel_history, dt, target_pga_g=target_pga_g)
    motion_name = motion_label(motion, target_pga_g)

    from pystrata_agent.pystrata_utils import import_pystrata
    pystrata = import_pystrata()

    return _run_eql_analysis(
        pystrata, layers, accel_g, dt_motion, motion_name,
        strain_ratio=0.65, tolerance=0.01, max_iterations=1,
        max_freq_hz=max_freq_hz, wave_frac=wave_frac,
        analysis_type="linear_elastic",
        water_table_depth=water_table_depth, k0=k0,
    )


# ===========================================================================
# Internal analysis runner
# ===========================================================================

def _run_eql_analysis(pystrata, layers, accel_g, dt_motion, motion_name,
                      strain_ratio, tolerance, max_iterations,
                      max_freq_hz, wave_frac, analysis_type,
                      water_table_depth=None, k0=K0_DEFAULT):
    """Build pystrata model, run analysis, extract results."""
    from pystrata_agent import compat

    # 1. Build pystrata layers
    pystrata_layers = []
    for i, layer in enumerate(layers):
        soil_type = _build_soil_type(pystrata, layer, layers, i,
                                     water_table_depth, k0)
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
            # pystrata 0.5's relative error is in PERCENT (passing the
            # fraction straight through asked for 0.01 %, BACKLOG GM-78);
            # 0.6's is in DECIMAL (INF-68). compat picks the unit.
            tolerance=compat.eql_tolerance(pystrata, tolerance),
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

    # 6. Run analysis. pystrata keeps its EQL iteration count in a local
    # variable; count the wave solves instead (one per iteration) so the
    # result reports the real number, not a placeholder (live smoke G6, PST-1).
    wave_solves = [0]
    orig_calc_waves = getattr(calc, "_calc_waves", None)
    if analysis_type != "linear_elastic" and callable(orig_calc_waves):
        def _counted_calc_waves(*a, **k):
            wave_solves[0] += 1
            return orig_calc_waves(*a, **k)
        calc._calc_waves = _counted_calc_waves
    loc_input = profile.location("outcrop", index=-1)
    calc(ts_motion, profile, loc_input)
    # The first wave solve is pystrata's linear-elastic start (base-class
    # __call__); each EQL iteration adds one more.
    eql_passes = wave_solves[0] - 1

    # 7. Compute outputs
    outputs(calc)

    # 7b. A linear run's peak shear strains. pystrata's
    # LinearElasticCalculator never sets a layer's strain_max (only the
    # equivalent-linear calculator does, after its last iteration), so
    # MaxStrainProfile reads None for every sub-layer; until the GM-325
    # review the module turned that into 0.0. pystrata's StrainTSOutput
    # gives the strain history at a depth (the strain transfer function on
    # the input motion); its peak at each sub-layer's middle is what the
    # equivalent-linear calculator stores as strain_max.
    linear_peaks = None
    if analysis_type == "linear_elastic":
        z_mid = np.asarray(calc.profile.depth_mid, dtype=float)[:-1]
        strain_ts = pystrata.output.OutputCollection([
            pystrata.output.StrainTSOutput(
                pystrata.output.OutputLocation("within", depth=float(z)))
            for z in z_mid])
        strain_ts(calc)
        linear_peaks = np.array([
            float(np.max(np.abs(np.asarray(o.values, dtype=float))))
            for o in strain_ts])

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

    depths_strain = np.array(strain_output.refs, dtype=float).flatten()
    if linear_peaks is not None:
        # Laid out on MaxStrainProfile's own depths, which differ by
        # version: pystrata 0.5 gives the surface and each sub-layer's
        # middle; 0.6 adds the base of the soil column, carrying the last
        # sub-layer's value (output._LayerMidProfile). Each depth takes the
        # strain of the sub-layer it lies in; the surface takes 0.0, as
        # MaxStrainProfile writes it.
        tops = np.asarray(calc.profile.depth, dtype=float)[:len(linear_peaks)]
        where = np.clip(np.searchsorted(tops, depths_strain, side="right") - 1,
                        0, len(linear_peaks) - 1)
        max_strain = np.where(depths_strain > 0, linear_peaks[where], 0.0)
    else:
        # A sub-layer pystrata gave no strain for stays missing (NaN, null
        # in the result), never 0.0.
        max_strain = np.array(
            [np.nan if v is None else float(v)
             for v in strain_output.values])

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
        max_error = _max_abs_error_percent(calc.profile)
        converged = bool(max_error < tolerance * 100.0)
        # The counted wave solves (one per EQL iteration); None when the
        # count could not be taken (pystrata internals changed), never a
        # made-up number.
        n_iterations = (eql_passes if callable(orig_calc_waves)
                        and eql_passes >= 1 else None)

    run_choices, run_warnings = stress_choices(layers, water_table_depth, k0)

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
        depths_accel=depths_accel,
        depths_vs=depths_vs,
        water_table_depth_m=water_table_depth,
        k0=k0,
        layer_stresses=curve_layer_stresses(layers, water_table_depth, k0),
        choices=run_choices,
        warnings=run_warnings,
    )
