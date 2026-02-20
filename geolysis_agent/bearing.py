"""
Bearing capacity — Allowable (SPT-based) and ultimate methods.

Uses the geolysis library for SPT-based allowable bearing capacity
(Bowles, Meyerhof, Terzaghi) and ultimate bearing capacity (Vesic, Terzaghi).
"""

from geolysis_agent.geolysis_utils import import_bearing_capacity
from geolysis_agent.results import BearingCapacityResult


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_common_inputs(depth, width, shape):
    """Validate common bearing capacity inputs."""
    if depth < 0:
        raise ValueError(f"depth must be >= 0, got {depth}")

    if width <= 0:
        raise ValueError(f"width must be > 0, got {width}")

    valid_shapes = {"square", "rectangle", "circle", "strip"}
    if shape not in valid_shapes:
        raise ValueError(
            f"shape must be one of {valid_shapes}, got '{shape}'"
        )


def _validate_allowable_spt_inputs(
    corrected_spt_n_value,
    tol_settlement,
    depth,
    width,
    shape,
    foundation_type,
    abc_method,
):
    """Validate allowable bearing capacity (SPT) inputs."""
    _validate_common_inputs(depth, width, shape)

    if corrected_spt_n_value < 0:
        raise ValueError(
            f"corrected_spt_n_value must be >= 0, got {corrected_spt_n_value}"
        )

    if tol_settlement <= 0:
        raise ValueError(f"tol_settlement must be > 0, got {tol_settlement}")

    valid_types = {"pad", "raft"}
    if foundation_type not in valid_types:
        raise ValueError(
            f"foundation_type must be one of {valid_types}, got '{foundation_type}'"
        )

    valid_methods = {"bowles", "meyerhof", "terzaghi"}
    if abc_method not in valid_methods:
        raise ValueError(
            f"abc_method must be one of {valid_methods}, got '{abc_method}'"
        )


def _validate_ultimate_inputs(
    friction_angle,
    cohesion,
    moist_unit_wgt,
    depth,
    width,
    factor_of_safety,
    shape,
    ubc_method,
):
    """Validate ultimate bearing capacity inputs."""
    _validate_common_inputs(depth, width, shape)

    if not 0 <= friction_angle <= 50:
        raise ValueError(
            f"friction_angle must be 0-50 degrees, got {friction_angle}"
        )

    if cohesion < 0:
        raise ValueError(f"cohesion must be >= 0, got {cohesion}")

    if moist_unit_wgt <= 0:
        raise ValueError(f"moist_unit_wgt must be > 0, got {moist_unit_wgt}")

    if factor_of_safety <= 1.0:
        raise ValueError(f"factor_of_safety must be > 1.0, got {factor_of_safety}")

    valid_methods = {"vesic", "terzaghi"}
    if ubc_method not in valid_methods:
        raise ValueError(
            f"ubc_method must be one of {valid_methods}, got '{ubc_method}'"
        )


# ---------------------------------------------------------------------------
# Bearing capacity functions
# ---------------------------------------------------------------------------

def allowable_bc_spt(
    corrected_spt_n_value,
    tol_settlement=25.0,
    depth=1.5,
    width=2.0,
    shape="square",
    foundation_type="pad",
    abc_method="bowles",
):
    """
    Compute allowable bearing capacity from SPT N-value.

    Uses empirical SPT-based correlations (Bowles, Meyerhof, or Terzaghi)
    for cohesionless soils.

    Parameters
    ----------
    corrected_spt_n_value : float
        Corrected SPT N-value (N1_60 or fully corrected).
    tol_settlement : float, default 25.0
        Tolerable settlement (mm).
    depth : float, default 1.5
        Foundation depth below ground surface (m).
    width : float, default 2.0
        Foundation width (m). For rectangular, use the smaller dimension.
    shape : str, default "square"
        Foundation shape. Choices: 'square', 'rectangle', 'circle', 'strip'.
    foundation_type : str, default "pad"
        Foundation type. Choices: 'pad', 'raft'.
    abc_method : str, default "bowles"
        Method for allowable bearing capacity. Choices: 'bowles', 'meyerhof',
        'terzaghi'.

    Returns
    -------
    BearingCapacityResult
        Contains allowable bearing capacity (kPa), allowable load (kN),
        and input parameters.

    Notes
    -----
    - All lengths in meters, pressures in kPa
    - For rectangular footings, width is the smaller dimension
    - Requires geolysis library (pip install geolysis)
    """
    _validate_allowable_spt_inputs(
        corrected_spt_n_value,
        tol_settlement,
        depth,
        width,
        shape,
        foundation_type,
        abc_method,
    )

    abc, _ = import_bearing_capacity()

    abc_obj = abc.create_abc_4_cohesionless_soils(
        corrected_spt_n_value=corrected_spt_n_value,
        tol_settlement=tol_settlement,
        depth=depth,
        width=width,
        shape=shape,
        foundation_type=foundation_type,
        abc_method=abc_method,
    )

    bc_kpa = abc_obj.allowable_bearing_capacity()
    load_kn = abc_obj.allowable_applied_load()

    return BearingCapacityResult(
        method=abc_method,
        bc_type="allowable_spt",
        bearing_capacity_kpa=bc_kpa,
        allowable_load_kn=load_kn,
        depth_m=depth,
        width_m=width,
        shape=shape,
        n_c=None,
        n_q=None,
        n_gamma=None,
        factor_of_safety=None,
        corrected_spt_n=corrected_spt_n_value,
        settlement_mm=tol_settlement,
    )


def ultimate_bc(
    friction_angle,
    cohesion=0.0,
    moist_unit_wgt=18.0,
    depth=1.5,
    width=2.0,
    factor_of_safety=3.0,
    shape="square",
    ubc_method="vesic",
):
    """
    Compute ultimate bearing capacity using Vesic or Terzaghi method.

    Parameters
    ----------
    friction_angle : float
        Soil friction angle (degrees), 0-50.
    cohesion : float, default 0.0
        Soil cohesion (kPa).
    moist_unit_wgt : float, default 18.0
        Moist unit weight of soil (kN/m³).
    depth : float, default 1.5
        Foundation depth below ground surface (m).
    width : float, default 2.0
        Foundation width (m).
    factor_of_safety : float, default 3.0
        Factor of safety. Must be > 1.0.
    shape : str, default "square"
        Foundation shape. Choices: 'square', 'rectangle', 'circle', 'strip'.
    ubc_method : str, default "vesic"
        Method for ultimate bearing capacity. Choices: 'vesic', 'terzaghi'.

    Returns
    -------
    BearingCapacityResult
        Contains ultimate and allowable bearing capacity (kPa), bearing
        capacity factors (Nc, Nq, Nγ), and input parameters.

    Notes
    -----
    - All lengths in meters, pressures in kPa, angles in degrees
    - Vesic method is recommended for general use
    - Requires geolysis library (pip install geolysis)
    """
    _validate_ultimate_inputs(
        friction_angle,
        cohesion,
        moist_unit_wgt,
        depth,
        width,
        factor_of_safety,
        shape,
        ubc_method,
    )

    _, ubc = import_bearing_capacity()

    ubc_obj = ubc.create_ubc_4_all_soils(
        friction_angle=friction_angle,
        cohesion=cohesion,
        moist_unit_wgt=moist_unit_wgt,
        depth=depth,
        width=width,
        factor_of_safety=factor_of_safety,
        shape=shape,
        ubc_method=ubc_method,
    )

    q_ult = ubc_obj.ultimate_bearing_capacity()
    q_allow = ubc_obj.allowable_bearing_capacity()

    # Extract bearing capacity factors
    # geolysis stores these as properties
    n_c = ubc_obj.n_c
    n_q = ubc_obj.n_q
    n_gamma = ubc_obj.n_gamma

    return BearingCapacityResult(
        method=ubc_method,
        bc_type="ultimate",
        bearing_capacity_kpa=q_ult,
        allowable_load_kn=None,  # Not applicable for ultimate BC
        depth_m=depth,
        width_m=width,
        shape=shape,
        n_c=n_c,
        n_q=n_q,
        n_gamma=n_gamma,
        factor_of_safety=factor_of_safety,
        corrected_spt_n=None,
        settlement_mm=None,
        allowable_bearing_capacity_kpa=q_allow,
    )
