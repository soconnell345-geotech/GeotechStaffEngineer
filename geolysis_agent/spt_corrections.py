"""
SPT corrections — Energy, overburden, and dilatancy corrections.

Uses the geolysis library for Standard Penetration Test (SPT) N-value
corrections and design N-value computation.
"""

from geolysis_agent.geolysis_utils import import_spt
from geolysis_agent.results import SPTCorrectionResult


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------

def _validate_spt_inputs(
    recorded_spt_n_value,
    eop,
    energy_percentage,
    borehole_diameter,
    rod_length,
    hammer_type,
    sampler_type,
    opc_method,
):
    """Validate SPT correction inputs."""
    if recorded_spt_n_value < 0:
        raise ValueError(
            f"recorded_spt_n_value must be >= 0, got {recorded_spt_n_value}"
        )

    if eop <= 0:
        raise ValueError(f"eop (effective overburden pressure) must be > 0, got {eop}")

    if not 0 < energy_percentage <= 1.0:
        raise ValueError(
            f"energy_percentage must be in (0, 1], got {energy_percentage}"
        )

    if borehole_diameter <= 0:
        raise ValueError(f"borehole_diameter must be > 0, got {borehole_diameter}")

    if rod_length <= 0:
        raise ValueError(f"rod_length must be > 0, got {rod_length}")

    valid_hammers = {
        "automatic", "donut_1", "donut_2", "safety", "drop", "pin"
    }
    if hammer_type not in valid_hammers:
        raise ValueError(
            f"hammer_type must be one of {valid_hammers}, got '{hammer_type}'"
        )

    valid_samplers = {
        "standard", "non_standard", "liner_4_dense_sand_and_clay", "liner_4_loose_sand"
    }
    if sampler_type not in valid_samplers:
        raise ValueError(
            f"sampler_type must be one of {valid_samplers}, got '{sampler_type}'"
        )

    valid_opc = {"gibbs", "bazaraa", "peck", "liao", "skempton"}
    if opc_method not in valid_opc:
        raise ValueError(
            f"opc_method must be one of {valid_opc}, got '{opc_method}'"
        )


def _validate_design_n_inputs(corrected_spt_n_values, method):
    """Validate design N-value inputs."""
    if not corrected_spt_n_values:
        raise ValueError("corrected_spt_n_values cannot be empty")

    if any(n < 0 for n in corrected_spt_n_values):
        raise ValueError("All corrected_spt_n_values must be >= 0")

    valid_methods = {"wgt", "min", "avg"}
    if method not in valid_methods:
        raise ValueError(
            f"method must be one of {valid_methods}, got '{method}'"
        )


# ---------------------------------------------------------------------------
# SPT correction functions
# ---------------------------------------------------------------------------

def correct_spt(
    recorded_spt_n_value,
    eop,
    energy_percentage=0.6,
    borehole_diameter=65.0,
    rod_length=10.0,
    hammer_type="safety",
    sampler_type="standard",
    opc_method="gibbs",
    dilatancy_corr_method=None,
):
    """
    Correct SPT N-value for energy, overburden, and optionally dilatancy.

    Parameters
    ----------
    recorded_spt_n_value : int
        Field-recorded SPT N-value (blows per 300 mm).
    eop : float
        Effective overburden pressure (kPa).
    energy_percentage : float, default 0.6
        Energy ratio (decimal, 0-1). 0.6 = 60% energy.
    borehole_diameter : float, default 65.0
        Borehole diameter (mm). Standard: 65-115 mm.
    rod_length : float, default 10.0
        Rod length (m).
    hammer_type : str, default "safety"
        Hammer type. Choices: 'automatic', 'donut_1', 'donut_2', 'safety',
        'drop', 'pin'.
    sampler_type : str, default "standard"
        Sampler type. Choices: 'standard', 'non_standard',
        'liner_4_dense_sand_and_clay', 'liner_4_loose_sand'.
    opc_method : str, default "gibbs"
        Overburden pressure correction method. Choices: 'gibbs', 'bazaraa',
        'peck', 'liao', 'skempton'.
    dilatancy_corr_method : str, optional
        Dilatancy correction method. If None, no dilatancy correction applied.
        See geolysis.spt.DilatancyCorrection for available methods.

    Returns
    -------
    SPTCorrectionResult
        Contains N60, N1_60, final corrected N, and all input parameters.

    Notes
    -----
    - borehole_diameter is in mm (geolysis convention)
    - rod_length is in m
    - eop is in kPa (matches project convention)
    - Requires geolysis library (pip install geolysis)
    """
    _validate_spt_inputs(
        recorded_spt_n_value,
        eop,
        energy_percentage,
        borehole_diameter,
        rod_length,
        hammer_type,
        sampler_type,
        opc_method,
    )

    spt = import_spt()

    # Use geolysis's all-in-one correction function
    n_corrected = spt.correct_spt_n_value(
        recorded_spt_n_value=recorded_spt_n_value,
        eop=eop,
        energy_percentage=energy_percentage,
        borehole_diameter=borehole_diameter,
        rod_length=rod_length,
        hammer_type=hammer_type,
        sampler_type=sampler_type,
        opc_method=opc_method,
        dilatancy_corr_method=dilatancy_corr_method,
    )

    # Also compute intermediate values for transparency
    # Energy correction only
    energy_corr = spt.EnergyCorrection(
        recorded_spt_n_value=recorded_spt_n_value,
        energy_percentage=energy_percentage,
        borehole_diameter=borehole_diameter,
        rod_length=rod_length,
        hammer_type=hammer_type,
        sampler_type=sampler_type,
    )
    n60 = energy_corr.standardized_spt_n_value()

    # Overburden correction
    opc = spt.create_overburden_pressure_correction(
        std_spt_n_value=n60,
        eop=eop,
        opc_method=opc_method,
    )
    n1_60 = opc.corrected_spt_n_value()

    return SPTCorrectionResult(
        recorded_n=recorded_spt_n_value,
        n60=n60,
        n1_60=n1_60,
        n_corrected=n_corrected,
        energy_percentage=energy_percentage,
        hammer_type=hammer_type,
        sampler_type=sampler_type,
        opc_method=opc_method,
        eop_kpa=eop,
        dilatancy_applied=(dilatancy_corr_method is not None),
    )


def design_n_value(corrected_spt_n_values, method="wgt"):
    """
    Compute design N-value from a set of corrected SPT N-values.

    Parameters
    ----------
    corrected_spt_n_values : list of float
        Corrected SPT N-values (N1_60 or fully corrected).
    method : str, default "wgt"
        Design N method. Choices:
        - 'wgt': Weighted average (gives more weight to lower values)
        - 'min': Minimum value
        - 'avg': Arithmetic average

    Returns
    -------
    float
        Design N-value.

    Notes
    -----
    - Weighted average method is recommended for conservative design
    - Requires geolysis library (pip install geolysis)
    """
    _validate_design_n_inputs(corrected_spt_n_values, method)

    spt = import_spt()

    spt_obj = spt.SPT(corrected_spt_n_values=corrected_spt_n_values, method=method)
    return spt_obj.n_design()
