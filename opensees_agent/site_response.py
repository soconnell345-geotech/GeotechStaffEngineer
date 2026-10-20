"""
1D effective-stress site response analysis using OpenSees.

Builds a 1D soil column with SSPquadUP elements, applies earthquake
ground motion through a Lysmer-Kuhlmeyer viscous dashpot at the base,
and extracts surface motion, depth profiles, and response spectra.

Materials:
  Sand: PressureDependMultiYield02
  Clay: PressureIndependMultiYield

The model follows the OpenSees example "Effective Stress Site Response
Analysis of a Layered Soil Column" (McGann & Arduino, University of
Washington; OpenSeesPy docs section 14.4.2, freeFieldEffective.py):
gravity as element body forces, an elastic then an elastoplastic gravity
stage at a permeability of 1 m/s, each layer's own permeability set
through element parameters, the clock set back to zero, then the motion
as a force history on the base dashpot. The dynamic stage runs Newton, as
the authors' total stress example does ("Site Response Analysis of a
Layered Soil Column (Total Stress Analysis)"), with KrylovNewton and
modified Newton as fallbacks.

References:
    Lysmer, J. & Kuhlemeyer, R.L. (1969). "Finite Dynamic Model for
    Infinite Media." J. Eng. Mech. Div., ASCE, 95(4), 859-877.

    Joyner, W.B. & Chen, A.T.F. (1975). "Calculation of Nonlinear Ground
    Response in Earthquakes." Bull. Seismol. Soc. Am., 65(5), 1315-1336.

    Yang, Z., Elgamal, A., & Parra, E. (2003). "Computational Model
    for Cyclic Mobility and Associated Shear Deformation." J. Geotech.
    Geoenviron. Eng., 129(12), 1119-1127.

    Kramer, S.L. (1996). Geotechnical Earthquake Engineering. Prentice
    Hall. Section 7.2 (uniform damped soil on elastic rock).

    Kwok, A.O.L., Stewart, J.P., Hashash, Y.M.A., et al. (2007). "Use of
    Exact Solutions of Wave Propagation Problems to Guide Implementation
    of Nonlinear Seismic Ground Response Analysis Procedures." J. Geotech.
    Geoenviron. Eng., 133(11), 1385-1398.
"""

import json
import math
import os
import pickle
import subprocess
import sys
import tempfile

import numpy as np

from opensees_agent.results import SiteResponseResult


# The analysis runs in a child process so a solver that never returns can
# be stopped: openseespy holds the Python GIL inside ops.analyze(), so a
# stuck step would freeze every thread of the calling process (an app
# server, a test run). GM-77, 2026-10-14.
DEFAULT_TIME_LIMIT_S = 600.0

# Gravity stages (McGann & Arduino's example: 10 elastic and 40
# elastoplastic steps of 500 s, long enough for the pore pressures to
# settle to hydrostatic under a low-permeability clay).
_GRAVITY_DT_S = 500.0
_GRAVITY_ELASTIC_STEPS = 10
_GRAVITY_PLASTIC_STEPS = 40

# Permeability (m/s) of every element during the gravity stages. As in the
# OpenSees effective stress example (freeFieldEffective.py: "permeabilities
# 1.0 m/s for the gravity analysis", then each layer's own value set
# through element parameters before the shaking). Gravity run at the
# shaking permeability (1e-4 m/s) failed at the first elastoplastic step
# on sand columns with elements of about 2 m and less (GM-85). The update
# sets the same quantity the element takes (checked: updating to the
# constructor's own value changes nothing, to machine zero).
_GRAVITY_PERMEABILITY = 1.0

# Dynamic stage (GM-84). Full Newton, as McGann & Arduino's total stress
# example: it converges quadratically, so once a correction is under the
# examples' 1 mm test the step is in balance whatever the element size.
# KrylovNewton (the effective stress example's choice) and modified Newton
# converge linearly, slowly while the soil yields, so the same 1 mm test
# passed them with the step still out of balance on small elements (1 mm
# is 0.4 % shear strain on a 0.25 m element; Sa off by up to -35 %, GM-83).
# They are kept as fallbacks with a test 100 times tighter, at which
# KrylovNewton gives the converged answer on 0.25 m elements.
_DYNAMIC_TEST = ("NormDispIncr", 1.0e-3, 35, 0)
_FALLBACK_TEST = ("NormDispIncr", 1.0e-5, 50, 0)
_DYNAMIC_ALGORITHMS = (
    ("Newton", _DYNAMIC_TEST),
    ("KrylovNewton", _FALLBACK_TEST),
    ("ModifiedNewton", _FALLBACK_TEST),
)
_DYNAMIC_SUBSTEPS = 10


# ===========================================================================
# Input validation
# ===========================================================================

def _validate_site_response_inputs(layers, gwt_depth, bedrock_Vs,
                                   bedrock_density, damping,
                                   scale_factor, n_elem_per_layer,
                                   rayleigh_freqs_hz=None):
    """Validate all site response inputs, raising ValueError for bad values."""
    # -- layers --
    if not isinstance(layers, list) or len(layers) == 0:
        raise ValueError("layers must be a non-empty list of dicts")

    for i, layer in enumerate(layers):
        if not isinstance(layer, dict):
            raise ValueError(f"layers[{i}] must be a dict")

        for key in ("thickness", "Vs", "density", "material_type"):
            if key not in layer:
                raise ValueError(
                    f"layers[{i}] missing required key '{key}'")

        if layer["thickness"] <= 0:
            raise ValueError(
                f"layers[{i}]['thickness'] must be positive, "
                f"got {layer['thickness']}")
        if layer["Vs"] <= 0:
            raise ValueError(
                f"layers[{i}]['Vs'] must be positive, got {layer['Vs']}")
        if layer["density"] <= 0:
            raise ValueError(
                f"layers[{i}]['density'] must be positive, "
                f"got {layer['density']}")

        mat = layer["material_type"].lower().strip()
        if mat not in ("sand", "clay"):
            raise ValueError(
                f"layers[{i}]['material_type'] must be 'sand' or 'clay', "
                f"got '{layer['material_type']}'")

        if mat == "sand":
            if "phi" not in layer:
                raise ValueError(
                    f"layers[{i}] (sand) requires 'phi' (friction angle)")
            if layer["phi"] <= 0:
                raise ValueError(
                    f"layers[{i}]['phi'] must be positive, got {layer['phi']}")
        if mat == "clay":
            if "su" not in layer:
                raise ValueError(
                    f"layers[{i}] (clay) requires 'su' "
                    f"(undrained shear strength)")
            if layer["su"] <= 0:
                raise ValueError(
                    f"layers[{i}]['su'] must be positive, got {layer['su']}")
        _validate_layer_options(i, layer, mat)

    # -- scalar params --
    if gwt_depth < 0:
        raise ValueError(f"gwt_depth must be >= 0, got {gwt_depth}")
    if bedrock_Vs <= 0:
        raise ValueError(f"bedrock_Vs must be positive, got {bedrock_Vs}")
    if bedrock_density <= 0:
        raise ValueError(
            f"bedrock_density must be positive, got {bedrock_density}")
    if not (0 <= damping < 1):
        raise ValueError(f"damping must be in [0, 1), got {damping}")
    if scale_factor <= 0:
        raise ValueError(
            f"scale_factor must be positive, got {scale_factor}")
    if n_elem_per_layer < 1:
        raise ValueError(
            f"n_elem_per_layer must be >= 1, got {n_elem_per_layer}")
    if rayleigh_freqs_hz is not None:
        try:
            f1, f2 = (float(f) for f in rayleigh_freqs_hz)
        except (TypeError, ValueError):
            raise ValueError(
                f"rayleigh_freqs_hz must be two frequencies [f1, f2] in Hz, "
                f"got {rayleigh_freqs_hz!r}") from None
        if not (0 < f1 < f2):
            raise ValueError(
                f"rayleigh_freqs_hz must be two positive frequencies with "
                f"f1 < f2, got {rayleigh_freqs_hz!r}")


# Sand constants a layer may set (GM-79), each defaulting to the Dr = 40 %
# column of PDMY02_DR40, and the keys each curve reads (GM-80).
SAND_CONSTANTS = ("contrac1", "contrac2", "contrac3", "dilat1", "dilat2",
                  "dilat3", "liquefac1", "liquefac2")
CURVE_KEYS = {
    "opensees": (),
    "darendeli": ("plas_index", "ocr", "freq_hz", "n_cycles"),
    "menq": ("uniformity_coeff", "diam_mean", "n_cycles"),
    "user": ("G_Gmax_curve", "damping_curve"),
}
# Poisson's ratio of each material's skeleton when a layer gives none:
# drained sand 0.3, and clay 0.495 (nearly incompressible, undrained).
DEFAULT_NU = {"sand": 0.3, "clay": 0.495}


def _validate_layer_options(i, layer, mat):
    """The GM-79 / GM-80 layer options: curve and its keys, n_elem, nu,
    the sand constants."""
    label = f"layers[{i}]"
    from opensees_agent.curves import CURVES, table, table_ggmax
    curve = str(layer.get("curve", "opensees")).lower().strip()
    if curve not in CURVES:
        raise ValueError(f"{label}['curve'] must be one of {CURVES}, got "
                         f"{layer.get('curve')!r}")
    for other, keys in CURVE_KEYS.items():
        if other == curve:
            continue
        stray = [k for k in keys if k in layer
                 and k not in CURVE_KEYS[curve] and k != "n_cycles"]
        if stray:
            raise ValueError(
                f"{label} gives {stray}, which the '{other}' curve reads, "
                f"but its curve is '{curve}': set curve='{other}' or "
                f"leave them out")
    if curve == "menq":
        for key in ("uniformity_coeff", "diam_mean"):
            if key not in layer:
                raise ValueError(
                    f"{label} (curve 'menq') requires '{key}' "
                    f"({'Cu' if key == 'uniformity_coeff' else 'D50, mm'})")
    if curve == "user":
        if "G_Gmax_curve" not in layer:
            raise ValueError(
                f"{label} (curve 'user') requires 'G_Gmax_curve': "
                f"{{'strain_pct': [...], 'G_Gmax': [...]}}")
        table_ggmax(layer["G_Gmax_curve"], label + "['G_Gmax_curve']")
        if layer.get("damping_curve") is not None:
            table(layer["damping_curve"], "damping_pct",
                  label + "['damping_curve']")
    for key in ("plas_index",):
        if key in layer and layer[key] < 0:
            raise ValueError(f"{label}['{key}'] must be >= 0")
    for key in ("ocr", "freq_hz", "n_cycles", "uniformity_coeff",
                "diam_mean"):
        if key in layer and not layer[key] > 0:
            raise ValueError(f"{label}['{key}'] must be positive, got "
                             f"{layer[key]!r}")
    if "n_elem" in layer:
        n = layer["n_elem"]
        if isinstance(n, bool) or not isinstance(n, int) or n < 1:
            raise ValueError(f"{label}['n_elem'] must be a whole number "
                             f">= 1, got {n!r}")
    if "nu" in layer and not (0.0 <= layer["nu"] < 0.5):
        raise ValueError(f"{label}['nu'] must be in [0, 0.5), got "
                         f"{layer['nu']!r}")
    if mat == "clay":
        given = [k for k in SAND_CONSTANTS if k in layer]
        if given:
            raise ValueError(f"{label} is clay: {given} are sand "
                             f"(PressureDependMultiYield02) constants")
    for key in SAND_CONSTANTS:
        if key in layer and (not isinstance(layer[key], (int, float))
                             or layer[key] < 0):
            raise ValueError(f"{label}['{key}'] must be a number >= 0, got "
                             f"{layer[key]!r}")


# ===========================================================================
# Public API
# ===========================================================================

def analyze_site_response(
    layers,
    motion=None,
    accel_history=None,
    dt=None,
    target_pga_g=None,
    gwt_depth=0.0,
    bedrock_Vs=760.0,
    bedrock_density=2.4,
    damping=0.02,
    scale_factor=1.0,
    n_elem_per_layer=4,
    linear_elastic=False,
    time_limit_s=DEFAULT_TIME_LIMIT_S,
    rayleigh_freqs_hz=None,
):
    """Run 1D effective-stress site response analysis.

    Builds a layered soil column in OpenSees using SSPquadUP elements
    with PressureDependMultiYield02 (sand) and PressureIndependMultiYield
    (clay) constitutive models. Input motion is applied at the base
    through a Lysmer-Kuhlemeyer viscous dashpot.

    Parameters
    ----------
    layers : list of dict
        Soil layers from top to bottom. Each dict must contain:
        ``thickness`` (m), ``Vs`` (m/s), ``density`` (Mg/m3),
        ``material_type`` ('sand' or 'clay').
        Sand requires: ``phi`` (degrees). Optional: ``K0``, ``PTAng``
        (phase-transformation angle, degrees; default 26, capped at phi).
        Clay requires: ``su`` (kPa), the undrained strength in simple
        shear (the column shears in simple shear; the model's peak
        stress there equals su).
        Optional for all: ``n_surf`` (default 20), ``e_init`` (void
        ratio; default 0.77 sand, 0.85 clay), ``n_elem`` (this layer's
        element count; default ``n_elem_per_layer``), ``nu`` (Poisson's
        ratio of the skeleton; default 0.3 sand, 0.495 clay).
        Sand optional: the PressureDependMultiYield02 constants
        ``contrac1``, ``contrac2``, ``contrac3``, ``dilat1``, ``dilat2``,
        ``dilat3``, ``liquefac1``, ``liquefac2`` (default the OpenSees
        wiki's Dr = 40 % column, PDMY02_DR40, and the command's defaults).
        Stress-strain curve (GM-80), ``curve``: 'opensees' (default: the
        material's own hyperbolic backbone to its strength at 10 % strain),
        'darendeli' (Darendeli 2001; optional ``plas_index`` default 0,
        ``ocr`` 1, ``freq_hz`` 1, ``n_cycles`` 10), 'menq' (Menq 2003;
        requires ``uniformity_coeff`` Cu and ``diam_mean`` D50 in mm;
        optional ``n_cycles`` 10) or 'user' (requires ``G_Gmax_curve``
        ``{"strain_pct": [...], "G_Gmax": [...]}``; optional
        ``damping_curve`` ``{"strain_pct": [...], "damping_pct": [...]}``).
        A chosen curve is evaluated at the layer's mid-depth mean
        effective stress and becomes the material's user-defined backbone,
        followed to the layer's strength (phi or su) and cut there; its
        damping is reported, not imposed (the material damps on Masing's
        rules; see opensees_agent/curves.py).
    motion : str, optional
        Built-in motion name (e.g. 'synthetic_pulse'). The motion is
        taken as a ROCK OUTCROP motion.
    accel_history : array_like, optional
        Custom acceleration time history (g).
    dt : float, optional
        Time step for custom motion (s).
    target_pga_g : float, optional
        Scale the motion so its PGA equals this value (g) before the
        analysis (built-in motions have fixed PGAs). Default None.
    gwt_depth : float
        Groundwater table depth from surface (m). Default 0.0 (at surface).
        The model puts it at the deepest node at or above this depth
        (exact when it falls on a node, e.g. a layer boundary).
    bedrock_Vs : float
        Bedrock shear wave velocity (m/s). Default 760.
    bedrock_density : float
        Bedrock mass density (Mg/m3). Default 2.4.
    damping : float
        Target Rayleigh damping ratio, matched at the two Rayleigh
        frequencies (a little less between). Default 0.02.
    scale_factor : float
        Scale factor applied to input acceleration. Default 1.0.
    n_elem_per_layer : int
        Number of elements per soil layer. Default 4.
    linear_elastic : bool
        Keep the soil in its linear elastic stage through the shaking
        (shear modulus rho*Vs^2, no yielding, damping from the Rayleigh
        term only). A check of the column against the closed-form
        response of a uniform damped layer on elastic rock, not a design
        analysis. Default False (nonlinear effective-stress analysis).
    time_limit_s : float or None
        Wall-clock limit (s). The analysis runs in a child process that
        is stopped at this limit with a RuntimeError, because a stuck
        OpenSees step cannot be interrupted any other way (the OpenSees
        wiki example's 160-element column through 4,000 steps takes
        about 9 s). None runs it in this process with no limit. Default
        600.
    rayleigh_freqs_hz : [f1, f2] or None
        The two frequencies (Hz) at which the Rayleigh damping equals
        ``damping``. Default None: the site frequency Vs/(4H) and five
        times it.

    Returns
    -------
    SiteResponseResult
        Analysis results with surface motion, spectra, and depth profiles.

    Raises
    ------
    ValueError
        For invalid input parameters.
    ImportError
        If openseespy is not installed.
    RuntimeError
        If a gravity stage or a dynamic step does not converge, or the
        analysis does not finish within ``time_limit_s``.
    """
    _validate_site_response_inputs(
        layers, gwt_depth, bedrock_Vs, bedrock_density,
        damping, scale_factor, n_elem_per_layer, rayleigh_freqs_hz)
    if time_limit_s is not None and not (
            isinstance(time_limit_s, (int, float))
            and math.isfinite(time_limit_s) and time_limit_s > 0):
        raise ValueError(
            f"time_limit_s must be a positive number of seconds or None, "
            f"got {time_limit_s!r}")

    from opensees_agent.ground_motions import (
        validate_motion_input, motion_label)
    accel_g, dt_motion = validate_motion_input(
        motion, accel_history, dt, target_pga_g=target_pga_g)

    motion_name = motion_label(motion, target_pga_g)
    plans = _layer_plans(layers, gwt_depth, n_elem_per_layer)
    choices = _site_response_choices(
        plans, damping, rayleigh_freqs_hz, n_elem_per_layer,
        linear_elastic)

    job = dict(
        layers=[dict(L) for L in layers],
        plans=plans,
        rayleigh_freqs_hz=(None if rayleigh_freqs_hz is None
                           else [float(f) for f in rayleigh_freqs_hz]),
        accel_g=np.asarray(accel_g, dtype=float),
        dt_motion=float(dt_motion),
        gwt_depth=float(gwt_depth),
        bedrock_Vs=float(bedrock_Vs),
        bedrock_density=float(bedrock_density),
        damping=float(damping),
        scale_factor=float(scale_factor),
        n_elem_per_layer=int(n_elem_per_layer),
        motion_name=motion_name,
        linear_elastic=bool(linear_elastic),
    )
    if time_limit_s is None:
        result = _run_job(job)
    else:
        result = _run_job_in_child(job, float(time_limit_s))
    result.choices = choices
    result.layer_curves = [p["report"] for p in plans]
    return result


# ===========================================================================
# Running the job (in this process, or in a child process with a limit)
# ===========================================================================

def _run_job(job):
    """Build and run the OpenSees model for one job dict (this process)."""
    from opensees_agent.opensees_utils import fresh_model
    ops = fresh_model(ndm=2, ndf=3)
    try:
        return _run_site_response_model(ops, **job)
    finally:
        ops.wipe()


# The child imports this module from the parent's own sys.path (passed as
# JSON), so it finds the same package whether it is installed, editable or
# only on a test run's path.
_CHILD_CODE = (
    "import json, sys\n"
    "sys.path[:0] = json.loads(sys.argv[3])\n"
    "from opensees_agent.site_response import _child_main\n"
    "_child_main(sys.argv[1], sys.argv[2])\n"
)


def _child_main(job_path, out_path):
    """Child-process entry: run the job, pickle ('ok', result) or the error."""
    with open(job_path, "rb") as f:
        job = pickle.load(f)
    try:
        payload = ("ok", _run_job(job))
    except Exception as exc:  # reported to the parent, re-raised there
        payload = ("error", (type(exc).__name__, str(exc)))
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)


_ERROR_TYPES = {"ValueError": ValueError, "ImportError": ImportError,
                "ModuleNotFoundError": ImportError}


def _run_job_in_child(job, time_limit_s):
    """Run the job in a child Python process stopped at ``time_limit_s``."""
    path = [os.path.abspath(p) if p else os.getcwd() for p in sys.path]
    with tempfile.TemporaryDirectory(prefix="gse_site_response_",
                                     ignore_cleanup_errors=True) as tmp:
        job_path = os.path.join(tmp, "job.pkl")
        out_path = os.path.join(tmp, "result.pkl")
        with open(job_path, "wb") as f:
            pickle.dump(job, f)
        cmd = [sys.executable, "-c", _CHILD_CODE, job_path, out_path,
               json.dumps(path)]
        try:
            proc = subprocess.run(cmd, cwd=tmp, capture_output=True,
                                  text=True, errors="replace",
                                  timeout=time_limit_s)
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"The OpenSees site response analysis did not finish within "
                f"time_limit_s = {time_limit_s:g} s and was stopped. A "
                f"solver step that never returns is the usual cause; try "
                f"more elements per layer, a shorter or weaker motion, or "
                f"linear_elastic=True to check the column.") from None
        except OSError as exc:
            raise RuntimeError(
                f"Could not start the child process for the OpenSees site "
                f"response analysis ({exc}). time_limit_s=None runs it in "
                f"this process, without the time limit.") from exc
        if not os.path.exists(out_path):
            tail = (proc.stderr or proc.stdout or "").strip()[-1500:]
            raise RuntimeError(
                f"The OpenSees site response child process ended without a "
                f"result (exit code {proc.returncode}). Its last output:\n"
                f"{tail}")
        with open(out_path, "rb") as f:
            status, payload = pickle.load(f)
    if status == "ok":
        return payload
    name, message = payload
    raise _ERROR_TYPES.get(name, RuntimeError)(message)


# ===========================================================================
# Model building & analysis
# ===========================================================================

def _run_site_response_model(ops, layers, accel_g, dt_motion,
                             gwt_depth, bedrock_Vs, bedrock_density,
                             damping, scale_factor, n_elem_per_layer,
                             motion_name, linear_elastic=False,
                             plans=None, rayleigh_freqs_hz=None):
    """Build model, run gravity + dynamic, extract results."""
    g_accel = 9.81  # m/s^2
    col_width = 1.0  # m (plane-strain unit width)
    total_depth = sum(L["thickness"] for L in layers)

    # ------------------------------------------------------------------
    # 1. Mesh generation: compute element layout bottom-to-top
    # ------------------------------------------------------------------
    # Layers are given top-to-bottom; we reverse for building bottom-up
    if plans is None:
        plans = _layer_plans(layers, gwt_depth, n_elem_per_layer)
    layer_elems = []  # (layer_dict, elem_height, n_elems) per layer
    for L, plan in zip(reversed(layers), reversed(plans)):
        n_e = plan["n_elem"]
        h = L["thickness"] / n_e
        layer_elems.append((L, h, n_e))
    plans_bottom_up = list(reversed(plans))

    n_elem_total = sum(n_e for _, _, n_e in layer_elems)
    n_nodes_col = n_elem_total + 1  # nodes per column (left or right)

    # Node tag helpers (1-indexed)
    def left_node(i):
        return i + 1

    def right_node(i):
        return n_nodes_col + i + 1

    # Compute node elevations (from base = 0 to surface = total_depth)
    elevations = [0.0]
    for L_dict, h_elem, n_e in layer_elems:
        for _ in range(n_e):
            elevations.append(elevations[-1] + h_elem)

    # ------------------------------------------------------------------
    # 2. Create nodes
    # ------------------------------------------------------------------
    for i in range(n_nodes_col):
        ops.node(left_node(i), 0.0, elevations[i])
        ops.node(right_node(i), col_width, elevations[i])

    # ------------------------------------------------------------------
    # 3. Boundary conditions
    # ------------------------------------------------------------------
    # Base nodes: fix vertical, free horizontal, free pore pressure
    ops.fix(left_node(0), 0, 1, 0)
    ops.fix(right_node(0), 0, 1, 0)

    # Tie left-right node pairs (periodic BCs for 1D column)
    for i in range(n_nodes_col):
        if i == 0:
            # Base pair: tie x only (y already fixed)
            ops.equalDOF(left_node(0), right_node(0), 1)
        else:
            ops.equalDOF(left_node(i), right_node(i), 1, 2)

    # Surface nodes: fix pore pressure (drained surface)
    top_idx = n_nodes_col - 1
    ops.fix(left_node(top_idx), 0, 0, 1)
    ops.fix(right_node(top_idx), 0, 0, 1)

    # Nodes at or above the GWT: fix pore pressure at zero. The water
    # table then sits at the deepest node at or above gwt_depth (exactly
    # at gwt_depth when it falls on a node, e.g. a layer boundary).
    # (Until GM-77 a node AT the water table was left free, which put the
    # water table one element too shallow.)
    drained = [i == top_idx for i in range(n_nodes_col)]
    for i in range(n_nodes_col):
        depth_from_surface = total_depth - elevations[i]
        if depth_from_surface <= gwt_depth + 1.0e-9 and i != top_idx:
            ops.fix(left_node(i), 0, 0, 1)
            ops.fix(right_node(i), 0, 0, 1)
            drained[i] = True
    # An element with zero pore pressure at both its node rows holds no
    # pore water: its ru is 0 (its vertical effective stress still swings
    # with the column's vertical motion, which is not pore pressure).
    elem_dry = [drained[e] and drained[e + 1]
                for e in range(n_nodes_col - 1)]

    # ------------------------------------------------------------------
    # 4. Create materials (one per layer)
    # ------------------------------------------------------------------
    mat_tags = []
    perms = []  # permeability per layer
    e_inits = []  # void ratio per layer
    mat_tag = 1
    for (L_dict, h_elem, n_e), plan in zip(layer_elems, plans_bottom_up):
        mat_type = plan["material"]
        if mat_type == "sand":
            perm = _create_sand_material(ops, mat_tag, L_dict, plan)
        else:
            perm = _create_clay_material(ops, mat_tag, L_dict, plan)

        e_init = L_dict.get(
            "e_init", PDMY02_DR40["e"] if mat_type == "sand" else 0.85)

        mat_tags.append(mat_tag)
        perms.append(perm)
        e_inits.append(e_init)
        mat_tag += 1

    # ------------------------------------------------------------------
    # 5. Create elements (SSPquadUP)
    # ------------------------------------------------------------------
    elem_tag = 1
    elem_mat_idx = []  # which layer index each element belongs to
    ei = 0  # element counter within current layer
    layer_idx = 0

    for layer_idx_local, (L_dict, h_elem, n_e) in enumerate(layer_elems):
        for _ in range(n_e):
            n1 = left_node(ei)
            n2 = right_node(ei)
            n3 = right_node(ei + 1)
            n4 = left_node(ei + 1)

            e_val = e_inits[layer_idx_local]

            # Gravity is the element body force: SSPquadUP takes the
            # COMPONENTS OF GRAVITY (0, -g), not unit weights (OpenSees
            # SSPquadUP note 3). It loads the soil with its saturated
            # weight and the pore fluid with its own, so the pore
            # pressure settles hydrostatic and the effective stress
            # buoyant below the water table. (Until GM-77 the weight was
            # nodal loads on a Linear time series, factor = time, so at
            # t = 10,000 s the column carried 10,000 times its weight.)
            # Permeability: 1 m/s for the gravity stages, the layer's own
            # value set after them (step 8; GM-85).
            ops.element('SSPquadUP', elem_tag,
                        n1, n2, n3, n4,
                        mat_tags[layer_idx_local],
                        1.0,       # thickness (plane strain)
                        2.2e6,     # fluid bulk modulus (water, kPa)
                        1.0,       # fluid mass density (Mg/m3)
                        _GRAVITY_PERMEABILITY,  # horizontal permeability
                        _GRAVITY_PERMEABILITY,  # vertical permeability
                        e_val,     # void ratio
                        1.0e-5,    # alpha (pressure stabilization)
                        0.0,       # b1: horizontal gravity component
                        -g_accel)  # b2: vertical gravity component

            elem_mat_idx.append(layer_idx_local)
            elem_tag += 1
            ei += 1

    # ------------------------------------------------------------------
    # 6. Lysmer-Kuhlemeyer dashpot at base
    # ------------------------------------------------------------------
    total_nodes = 2 * n_nodes_col
    dashpot_node = total_nodes + 1
    ops.node(dashpot_node, 0.0, 0.0)
    ops.fix(dashpot_node, 1, 1, 1)

    # Dashpot coefficient: c = rho_bedrock * Vs_bedrock * area
    c_dashpot = bedrock_density * bedrock_Vs * col_width  # kN*s/m

    dashpot_mat = mat_tag + 100
    ops.uniaxialMaterial('Viscous', dashpot_mat, c_dashpot, 1.0)

    dashpot_elem = elem_tag + 100
    ops.element('zeroLength', dashpot_elem,
                dashpot_node, left_node(0),
                '-mat', dashpot_mat, '-dir', 1)

    # ------------------------------------------------------------------
    # 7. Gravity analysis (elastic stage). The weight is the element body
    #    force (step 5), constant from the first step; the transient
    #    steps of 500 s let the pore pressures settle to hydrostatic.
    # ------------------------------------------------------------------
    for mt in mat_tags:
        ops.updateMaterialStage('-material', mt, '-stage', 0)

    ops.constraints('Penalty', 1.0e14, 1.0e14)
    ops.test('NormDispIncr', 1.0e-4, 35, 0)
    ops.algorithm('KrylovNewton')
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    _gravity_stage(ops, _GRAVITY_ELASTIC_STEPS, "elastic")

    # ------------------------------------------------------------------
    # 8. Switch to elastoplastic (stage 1) -- unless the column is kept
    #    linear elastic through the shaking.
    # ------------------------------------------------------------------
    if not linear_elastic:
        for mt in mat_tags:
            ops.updateMaterialStage('-material', mt, '-stage', 1)
        _gravity_stage(ops, _GRAVITY_PLASTIC_STEPS, "elastoplastic")

    # Each element's own permeability for the shaking (sand 1e-4, clay
    # 1e-7 m/s), set through element parameters as the OpenSees effective
    # stress example does after its gravity stages (GM-85). Without it the
    # column would shake drained (ru about 0.2 where it liquefies).
    _set_shaking_permeability(
        ops, [perms[i] for i in elem_mat_idx])

    # Set the clock back to zero: the gravity stages end at t = 25,000 s
    # (or 5,000 s), and the input motion's force history starts at
    # t = 0, so without this the shaking stage reads the history past its
    # end and applies NO input (GM-77: 8.5e-7 g at the surface from a
    # 0.30 g motion). As in the OpenSees example (setTime 0.0).
    ops.setTime(0.0)
    ops.wipeAnalysis()

    # ------------------------------------------------------------------
    # 9. Record initial stresses for ru computation
    # ------------------------------------------------------------------
    initial_sigma_v = np.zeros(n_elem_total)
    for e in range(n_elem_total):
        stress = ops.eleResponse(e + 1, 'stress')
        # stress[1] = sigma_yy (negative = compression in OpenSees)
        initial_sigma_v[e] = -stress[1] if len(stress) > 1 else 1.0

    # ------------------------------------------------------------------
    # 10. Set up dynamic loading
    # ------------------------------------------------------------------
    accel_scaled = np.asarray(accel_g, dtype=float) * scale_factor

    # Integrate acceleration to velocity (trapezoid, at rest at t = 0)
    a_ms2 = accel_scaled * g_accel
    velocity = np.concatenate(
        ([0.0], np.cumsum(0.5 * (a_ms2[1:] + a_ms2[:-1]) * dt_motion)))

    # Force at the base (Joyner & Chen 1975): F = 2 * rho_r * Vs_r * A *
    # v_incident, and a rock OUTCROP motion is twice the incident
    # (upgoing) wave, so F = rho_r * Vs_r * A * v_outcrop = c * v_outcrop.
    # (Until GM-77 the outcrop velocity was doubled again, which doubled
    # the input.) Same as the OpenSees example (-factor = c * area).
    force_vals = c_dashpot * velocity

    # Path time series for the force
    ts_dyn = 100
    pat_dyn = 100

    ops.timeSeries('Path', ts_dyn, '-dt', dt_motion,
                   '-values', *force_vals.tolist(), '-factor', 1.0)
    ops.pattern('Plain', pat_dyn, ts_dyn)
    ops.load(left_node(0), 1.0, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 11. Rayleigh damping
    # ------------------------------------------------------------------
    # Average Vs (travel-time weighted)
    Vs_avg = total_depth / sum(
        L["thickness"] / L["Vs"] for L in layers)
    T1 = 4.0 * total_depth / Vs_avg  # site period 4H/Vs
    if rayleigh_freqs_hz is None:
        f1 = 1.0 / T1
        f2 = 5.0 * f1  # second target: five times the site frequency
    else:
        f1, f2 = rayleigh_freqs_hz
    omega1 = 2.0 * math.pi * f1
    omega2 = 2.0 * math.pi * f2
    a0_ray = 2.0 * damping * omega1 * omega2 / (omega1 + omega2)
    a1_ray = 2.0 * damping / (omega1 + omega2)

    ops.rayleigh(a0_ray, a1_ray, 0.0, 0.0)

    # ------------------------------------------------------------------
    # 12. Dynamic analysis: Newton, with KrylovNewton and modified Newton
    #     as fallbacks at a tighter test (see _DYNAMIC_ALGORITHMS; GM-84)
    # ------------------------------------------------------------------
    ops.wipeAnalysis()
    ops.constraints('Penalty', 1.0e14, 1.0e14)
    first_algorithm, first_test = _DYNAMIC_ALGORITHMS[0]
    ops.test(*first_test)
    ops.algorithm(first_algorithm)
    ops.numberer('RCM')
    ops.system('ProfileSPD')
    ops.integrator('Newmark', 0.5, 0.25)
    ops.analysis('Transient')

    n_steps = len(accel_scaled)
    surface_accel = np.zeros(n_steps)  # index k is t = k * dt; at rest at 0
    max_accel = np.zeros(n_elem_total)
    max_strain = np.zeros(n_elem_total)
    max_ru = np.zeros(n_elem_total)

    surface_node = left_node(top_idx)
    # How each step converged (GM-79): Newton, a fallback algorithm at the
    # full step, or substeps (which can move the peak, -6 to -20 % on the
    # OpenSees wiki column's PGA spike).
    algo_counts = {name: 0 for name, _ in _DYNAMIC_ALGORITHMS}
    substep_steps = []

    for step in range(1, n_steps):
        t_target = step * dt_motion
        ok, used = _dynamic_step(ops, dt_motion)
        if ok == 0:
            algo_counts[used] += 1
        else:
            substep_steps.append(step)
            # Fallback: substeps up to the same target time, each tried
            # with every algorithm in turn
            while ops.getTime() < t_target - 1.0e-9 * dt_motion:
                dt_sub = min(dt_motion / _DYNAMIC_SUBSTEPS,
                             t_target - ops.getTime())
                ok, _ = _dynamic_step(ops, dt_sub)
                if ok != 0:
                    break
        if ok != 0:
            # A bounded failure, not a hang and not silently skipped: the
            # record would no longer line up with the input motion.
            raise RuntimeError(
                f"The dynamic analysis did not converge at t = "
                f"{ops.getTime():.3f} s (step {step} of {n_steps - 1}, "
                f"after Newton, Krylov-Newton and modified Newton, then "
                f"{_DYNAMIC_SUBSTEPS} substeps). The soil may have failed "
                f"or liquefied under this motion; try more elements per "
                f"layer or a weaker motion.")

        # Surface acceleration
        a_surf = ops.nodeAccel(surface_node, 1)
        surface_accel[step] = a_surf / g_accel

        # Element profiles
        for e in range(n_elem_total):
            e_tag = e + 1
            stress = ops.eleResponse(e_tag, 'stress')
            strain = ops.eleResponse(e_tag, 'strain')

            # Max acceleration (average of top and bottom node accels)
            a_bot = abs(ops.nodeAccel(left_node(e), 1))
            a_top = abs(ops.nodeAccel(left_node(e + 1), 1))
            a_avg = (a_bot + a_top) / 2.0 / g_accel
            if a_avg > max_accel[e]:
                max_accel[e] = a_avg

            # Max shear strain
            if len(strain) > 2:
                gamma = abs(strain[2])  # engineering shear strain
                if gamma > max_strain[e]:
                    max_strain[e] = gamma

            # Pore pressure ratio
            if (len(stress) > 1 and initial_sigma_v[e] > 1.0
                    and not elem_dry[e]):
                sigma_v_curr = -stress[1]
                ru = 1.0 - sigma_v_curr / initial_sigma_v[e]
                ru = max(0.0, min(ru, 1.0))
                if ru > max_ru[e]:
                    max_ru[e] = ru

    # ------------------------------------------------------------------
    # 13. Post-process
    # ------------------------------------------------------------------
    time_arr = np.arange(n_steps) * dt_motion
    pga_input = float(np.max(np.abs(accel_scaled)))
    pga_surface = float(np.max(np.abs(surface_accel)))
    amplification = pga_surface / pga_input if pga_input > 0 else 0.0

    # Element center depths (from surface)
    depths = np.zeros(n_elem_total)
    for e in range(n_elem_total):
        elev_center = (elevations[e] + elevations[e + 1]) / 2.0
        depths[e] = total_depth - elev_center

    # Response spectra
    from opensees_agent.opensees_utils import compute_response_spectrum
    periods = np.logspace(-2, 1, 100)
    _, Sa_surface = compute_response_spectrum(
        surface_accel, dt_motion, periods)
    _, Sa_input = compute_response_spectrum(
        accel_scaled, dt_motion, periods)

    return SiteResponseResult(
        total_depth_m=total_depth,
        n_layers=len(layers),
        motion_name=motion_name,
        pga_input_g=pga_input,
        pga_surface_g=pga_surface,
        amplification_factor=amplification,
        time=time_arr,
        surface_accel_g=surface_accel,
        depths=depths,
        max_strain_pct=max_strain * 100.0,
        max_accel_g=max_accel,
        max_pore_pressure_ratio=max_ru,
        periods=periods,
        Sa_surface_g=Sa_surface,
        Sa_input_g=Sa_input,
        analysis_type=("linear elastic" if linear_elastic
                       else "nonlinear effective stress"),
        site_period_s=T1,
        initial_sigma_v_eff_kPa=initial_sigma_v,
        solver_steps=_solver_steps(algo_counts, substep_steps, n_steps,
                                   dt_motion),
        rayleigh_freqs_hz=[float(f1), float(f2)],
        n_elements=n_elem_total,
    )


def _gravity_stage(ops, n_steps, stage):
    """Run one gravity stage step by step; raise if a step fails."""
    for i in range(n_steps):
        if ops.analyze(1, _GRAVITY_DT_S) != 0:
            raise RuntimeError(
                f"The {stage} gravity stage did not converge (step {i + 1} "
                f"of {n_steps}). The column could not carry its own weight "
                f"with these properties: check su (clay) and phi (sand) "
                f"against the layers' stresses, or use more elements per "
                f"layer.")


def _set_shaking_permeability(ops, perm_by_element, first_tag=10001):
    """Set each SSPquadUP element's permeability (m/s) for the shaking.

    ``perm_by_element[k]`` is element k + 1's. As the OpenSees effective
    stress example does after gravity: one parameter per element and
    direction ('hPerm', 'vPerm'), then updateParameter.
    """
    tag = first_tag
    for e, perm in enumerate(perm_by_element):
        for name in ('hPerm', 'vPerm'):
            ops.parameter(tag, 'element', e + 1, name)
            ops.updateParameter(tag, perm)
            tag += 1


def _dynamic_step(ops, dt):
    """One dynamic step of ``dt``: Newton first, then each fallback with
    its own test (_DYNAMIC_ALGORITHMS). Newton and its test are restored
    after a fallback. Returns 0 when a step converged, else the last
    nonzero code from ops.analyze, with the algorithm that converged (or
    the last one tried)."""
    ok = -1
    for k, (algorithm, test) in enumerate(_DYNAMIC_ALGORITHMS):
        if k > 0:
            ops.test(*test)
            ops.algorithm(algorithm)
        ok = ops.analyze(1, dt)
        if ok == 0:
            break
    if k > 0:
        first_algorithm, first_test = _DYNAMIC_ALGORITHMS[0]
        ops.test(*first_test)
        ops.algorithm(first_algorithm)
    return ok, algorithm


# ===========================================================================
# Material creation helpers
# ===========================================================================

# PressureDependMultiYield02 calibration constants: the OpenSees wiki's
# table of suggested parameters ("PressureDependMultiYield02 Material",
# Dr = 40 % column; also the medium sand of the OpenSeesPy effective-stress
# site response example): phase-transformation angle 26 deg, contrac1
# 0.067, contrac3 0.23, dilat1 0.06, dilat3 0.27, void ratio 0.77; and the
# command's documented defaults contrac2 5.0, dilat2 3.0, liquefac1 1.0,
# liquefac2 0.0, cs1 0.9, cs2 0.02, cs3 0.7, pa 101 kPa.
PDMY02_DR40 = dict(
    PTAng=26.0,
    contrac1=0.067, contrac3=0.23,
    dilat1=0.06, dilat3=0.27,
    contrac2=5.0, dilat2=3.0,
    liquefac1=1.0, liquefac2=0.0,
    e=0.77,
    cs1=0.9, cs2=0.02, cs3=0.7, pa=101.0,
)


def _sigma_v_eff(layers, depth, gwt_depth, g_accel=9.81):
    """Vertical effective stress (kPa) at ``depth`` (m) below the surface.

    Layers are top to bottom; densities are saturated (Mg/m3); the pore
    pressure is hydrostatic below ``gwt_depth``.
    """
    sigma_v = 0.0
    top = 0.0
    for L in layers:
        bottom = top + L["thickness"]
        if depth <= top:
            break
        sigma_v += L["density"] * g_accel * (min(depth, bottom) - top)
        top = bottom
    u = 1.0 * g_accel * max(0.0, depth - gwt_depth)
    return sigma_v - u


# Strains (%) at which a layer's curve is reported.
_REPORT_STRAINS_PCT = (0.001, 0.01, 0.1, 1.0)


def _layer_plans(layers, gwt_depth, n_elem_per_layer, g_accel=9.81):
    """What each layer's material will be, top to bottom: the stresses,
    moduli, constants and backbone (GM-79, GM-80). Computed before the
    model so the result can report it whichever process runs the model."""
    from opensees_agent import curves as C
    plans = []
    top = 0.0
    for L in layers:
        mat = L["material_type"].lower().strip()
        thickness = L["thickness"]
        # Vertical effective stress at mid-layer: the weight of every
        # layer above plus half this one, less hydrostatic pore pressure
        sigma_v_eff = _sigma_v_eff(layers, top + thickness / 2.0,
                                   gwt_depth, g_accel)
        top += thickness
        g_max = L["density"] * L["Vs"] ** 2
        nu = float(L.get("nu", DEFAULT_NU[mat]))
        b_max = 2.0 * g_max * (1.0 + nu) / (3.0 * (1.0 - 2.0 * nu))
        if mat == "sand":
            k0 = L.get("K0", 1.0 - math.sin(math.radians(L["phi"])))
            # Reference confining pressure: mean effective stress there,
            # so the layer's Vs holds at its mid-depth
            p_ref = max(sigma_v_eff * (1.0 + 2.0 * k0) / 3.0, 5.0)
            tau_f = C.sand_strength(L["phi"], p_ref)
            constants = {k: float(L.get(k, PDMY02_DR40[k]))
                         for k in SAND_CONSTANTS}
        else:
            # Cap to prevent numerical issues
            b_max = min(b_max, g_max * 100.0)
            p_ref = max(sigma_v_eff * 0.67, 5.0)
            tau_f = float(L["su"])
            constants = {}
        curve = str(L.get("curve", "opensees")).lower().strip()
        damping_fn = None
        report = {"curve": curve}
        if curve == "darendeli":
            prm = C.darendeli_params(
                p_ref, L.get("plas_index", 0.0), L.get("ocr", 1.0),
                L.get("freq_hz", 1.0), L.get("n_cycles", 10.0))
            ggmax_fn = (lambda g, prm=prm:
                        C.modified_hyperbolic(g, *prm)[0])
            damping_fn = (lambda g, prm=prm:
                          C.modified_hyperbolic(g, *prm)[1])
            report.update(reference_strain_pct=round(prm[0] * 100, 5),
                          curvature=round(prm[1], 4),
                          min_damping_pct=round(prm[2] * 100, 3))
        elif curve == "menq":
            prm = C.menq_params(p_ref, L["uniformity_coeff"],
                                L["diam_mean"], L.get("n_cycles", 10.0))
            ggmax_fn = (lambda g, prm=prm:
                        C.modified_hyperbolic(g, *prm)[0])
            damping_fn = (lambda g, prm=prm:
                          C.modified_hyperbolic(g, *prm)[1])
            report.update(reference_strain_pct=round(prm[0] * 100, 5),
                          curvature=round(prm[1], 4),
                          min_damping_pct=round(prm[2] * 100, 3))
        elif curve == "user":
            cs, cv = C.table_ggmax(L["G_Gmax_curve"], "G_Gmax_curve")
            ggmax_fn = (lambda g, cs=cs, cv=cv: C.ggmax_at(g, cs, cv))
            if L.get("damping_curve") is not None:
                ds, dv = C.table(L["damping_curve"], "damping_pct",
                                 "damping_curve")
                damping_fn = (lambda g, ds=ds, dv=dv: np.interp(
                    np.log(g), np.log(ds), dv) / 100.0)
        else:
            ggmax_fn = C.hyperbolic_ggmax(g_max, tau_f)
        pairs = None
        if curve != "opensees":
            pairs, info = C.fit_backbone(ggmax_fn, g_max, tau_f)
            report.update(info)
        report["mean_effective_stress_kPa"] = round(p_ref, 2)
        report["strength_kPa"] = round(tau_f, 2)
        strains = np.array(_REPORT_STRAINS_PCT) / 100.0
        report["strain_pct"] = list(_REPORT_STRAINS_PCT)
        report["G_Gmax"] = [round(float(v), 4) for v in ggmax_fn(strains)]
        if damping_fn is not None:
            report["damping_pct"] = [round(float(v) * 100, 3)
                                     for v in damping_fn(strains)]
        if pairs is not None:
            # The damping the material will give: Masing loops on the
            # fitted backbone (it takes no damping curve)
            report["backbone_masing_damping_pct"] = [
                round(float(v) * 100, 3)
                for v in C.masing_damping(pairs, g_max, strains)]
        plans.append(dict(
            material=mat, n_elem=int(L.get("n_elem", n_elem_per_layer)),
            nu=nu, g_max=g_max, b_max=b_max, p_ref=p_ref, tau_f=tau_f,
            constants=constants, curve=curve, pairs=pairs, report=report,
            _given=tuple(sorted(L))))
    return plans


_CURVE_BASIS = {
    "opensees": ("the material's own hyperbolic backbone, reaching the "
                 "layer's strength at the peak shear strain of 10 %; much "
                 "stiffer than laboratory curves in sand (GM-80)"),
    "darendeli": ("Darendeli (2001) modified hyperbolic curve at the "
                  "layer's mid-depth mean effective stress, the material's "
                  "user-defined backbone to the layer's strength"),
    "menq": ("Menq (2003) modified hyperbolic curve for sands and gravels "
             "at the layer's mid-depth mean effective stress, the "
             "material's user-defined backbone to the layer's strength"),
    "user": ("the G/Gmax table given, the material's user-defined "
             "backbone to the layer's strength"),
}


def _site_response_choices(plans, damping, rayleigh_freqs_hz,
                           n_elem_per_layer, linear_elastic):
    """The result's ``choices`` (GM-115): what the column was built with,
    each with its basis and alternatives."""
    from geotech_common.choices import MODULE, DEFAULT, GIVEN, choice, \
        how_given
    from opensees_agent.curves import CURVES
    out = []
    for i, plan in enumerate(plans):
        label = f"layers[{i}] ({plan['material']})"
        curve = plan["curve"]
        how = DEFAULT if curve == "opensees" and not _layer_has(
            plans, i, "curve") else GIVEN
        out.append(choice(
            f"{label} stress-strain curve", curve, how,
            _CURVE_BASIS[curve] + (
                "; not used: the column stays linear elastic"
                if linear_elastic else ""),
            CURVES))
        out.append(choice(
            f"{label} Poisson's ratio", round(plan["nu"], 4),
            GIVEN if _layer_has(plans, i, "nu") else DEFAULT,
            ("drained sand skeleton" if plan["material"] == "sand" else
             "nearly incompressible undrained clay (the OpenSees wiki "
             "total-stress example uses 0.0)"),
            ("any value in [0, 0.5) as layers[i]['nu']",)))
        out.append(choice(
            f"{label} element count", plan["n_elem"],
            GIVEN if _layer_has(plans, i, "n_elem") else how_given(
                n_elem_per_layer, 4, name="n_elem_per_layer"),
            "layers[i]['n_elem'], else n_elem_per_layer",
            ("layers[i]['n_elem']",)))
        if plan["material"] == "sand":
            given = sorted(k for k in plan["constants"]
                           if _layer_has(plans, i, k))
            out.append(choice(
                f"{label} contraction / dilation constants",
                ", ".join(f"{k} {v:g}" for k, v in plan["constants"].items()),
                GIVEN if given else DEFAULT,
                "the OpenSees wiki's suggested PressureDependMultiYield02 "
                "parameters, Dr = 40 % column, and the command's defaults"
                + (f"; given: {', '.join(given)}" if given else
                   "; a dense layer with these liquefies like Dr = 40 %"),
                ("contrac1..3, dilat1..3, liquefac1..2 per layer",)))
    f_how = how_given(rayleigh_freqs_hz, None, name="rayleigh_freqs_hz")
    out.append(choice(
        "Rayleigh damping frequencies",
        ("site frequency Vs/(4H) and 5 times it"
         if rayleigh_freqs_hz is None else
         f"{rayleigh_freqs_hz[0]:g} and {rayleigh_freqs_hz[1]:g} Hz"),
        f_how,
        f"damping {damping:g} at both, a little less between",
        ("rayleigh_freqs_hz=[f1, f2]",)))
    out.append(choice(
        "hysteretic damping", "Masing loops on the backbone", MODULE,
        "the multi-yield materials damp by hysteresis on Masing's rules "
        "and take no damping curve; a curve's damping is reported beside "
        "the backbone's Masing damping (layer_curves)"))
    return out


def _layer_has(plans, i, key):
    """Whether layer i was given ``key`` (read off its plan's source)."""
    return key in plans[i].get("_given", ())


def _solver_steps(algo_counts, substep_steps, n_steps, dt):
    """How the dynamic steps converged (GM-79)."""
    out = {"n_steps": n_steps - 1,
           "by_algorithm": dict(algo_counts),
           "fallback_steps": sum(v for k, v in algo_counts.items()
                                 if k != _DYNAMIC_ALGORITHMS[0][0]),
           "substepped_steps": len(substep_steps)}
    if substep_steps:
        out["substepped_times_s"] = [round(k * dt, 4)
                                     for k in substep_steps[:20]]
        out["note"] = (
            f"{len(substep_steps)} step(s) converged only in "
            f"{_DYNAMIC_SUBSTEPS} substeps; substeps can move a peak "
            f"(-6 to -20 % on the OpenSees wiki column's PGA spike).")
    return out


def _create_sand_material(ops, mat_tag, layer, plan):
    """Create PressureDependMultiYield02 material for a sand layer from
    its plan (_layer_plans): the reference shear modulus rho*Vs^2 at the
    mid-layer mean effective stress, and either the material's own
    backbone (noYieldSurf > 0) or the fitted user-defined one
    (noYieldSurf < 0 followed by strain, G/Gmax pairs).

    Returns permeability (m/s) for use in element creation.
    """
    phi = layer["phi"]
    n_surf = int(layer.get("n_surf", 20))
    e_init = layer.get("e_init", PDMY02_DR40["e"])
    # Phase transformation cannot exceed the friction angle (at phi it
    # means no dilation, the table's loosest column).
    pt_ang = min(float(layer.get("PTAng", PDMY02_DR40["PTAng"])), phi)
    gamma_max = 0.1  # 10% peak shear strain
    p = PDMY02_DR40
    c = plan["constants"]
    perm = 1.0e-4  # m/s (sand)
    if plan["pairs"] is None:
        surfaces = [n_surf]
    else:
        surfaces = [-(len(plan["pairs"]) // 2)] + list(plan["pairs"])

    # Argument order as documented (OpenSees wiki, PressureDependMultiYield02
    # Material; OpenSeesPy docs): tag nd rho refShearModul refBulkModul
    # frictionAng peakShearStra refPress pressDependCoe PTAng contrac1
    # contrac3 dilat1 dilat3 noYieldSurf [r/Gs pairs] contrac2 dilat2
    # liquefac1 liquefac2 e cs1 cs2 cs3 pa. (Until GM-77 PTAng was left
    # out, so every later value sat one slot off.)
    ops.nDMaterial('PressureDependMultiYield02', mat_tag,
                   2,              # nd
                   layer["density"],  # saturated mass density (Mg/m3)
                   plan["g_max"],  # refShearModul at refPress (kPa)
                   plan["b_max"],  # refBulkModul at refPress (kPa)
                   phi,            # frictionAng (degrees)
                   gamma_max,      # peakShearStra
                   plan["p_ref"],  # refPress (kPa)
                   0.5,            # pressDependCoe
                   pt_ang,         # PTAng (degrees)
                   c["contrac1"], c["contrac3"],
                   c["dilat1"], c["dilat3"],
                   *surfaces,      # noYieldSurf (and the user pairs)
                   c["contrac2"], c["dilat2"],
                   c["liquefac1"], c["liquefac2"],
                   e_init,         # e (void ratio)
                   p["cs1"], p["cs2"], p["cs3"], p["pa"])

    return perm


def _create_clay_material(ops, mat_tag, layer, plan):
    """Create PressureIndependMultiYield material for a clay layer from
    its plan (_layer_plans); with a pressure dependence coefficient of 0
    the reference pressure changes nothing in the backbone.

    Returns permeability (m/s) for use in element creation.
    """
    n_surf = int(layer.get("n_surf", 20))
    gamma_max = 0.1  # 10%
    perm = 1.0e-7  # m/s (clay, effectively undrained)

    # PIMY with phi = 0 is von Mises: its strength in simple shear (how a
    # 1D column shears) is 2c/sqrt(3) = 1.155c, not c. The OpenSees wiki
    # (PressureIndependMultiYield Material) gives c = sqrt(3)/2 * sigma_m
    # for the peak shear stress sigma_m of a user-defined backbone; the
    # same relation makes the column's simple-shear strength equal su.
    # Checked on one SSPquad element in simple shear: peak tau = 1.1547c.
    # (Until GM-77 su went in as c, so the clay was 15 % too strong.) With
    # a user-defined backbone the material sets c from its last pair the
    # same way, and the fit ends that pair at su.
    cohesion = math.sqrt(3.0) / 2.0 * layer["su"]
    if plan["pairs"] is None:
        surfaces = [n_surf]
    else:
        surfaces = [-(len(plan["pairs"]) // 2)] + list(plan["pairs"])

    ops.nDMaterial('PressureIndependMultiYield', mat_tag,
                   2,            # ndm
                   layer["density"],  # mass density
                   plan["g_max"],  # reference shear modulus (kPa)
                   plan["b_max"],  # reference bulk modulus (kPa)
                   cohesion,     # cohesi: simple-shear strength = su (kPa)
                   gamma_max,    # peak shear strain
                   0.0,          # friction angle (0 for total stress)
                   plan["p_ref"],  # reference pressure (kPa)
                   0.0,          # pressure dependence coefficient
                   *surfaces)    # number of yield surfaces (and pairs)

    return perm
