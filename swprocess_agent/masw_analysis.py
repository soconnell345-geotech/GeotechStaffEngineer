"""
MASW dispersion analysis wrapper using swprocess.
"""

import numpy as np

from swprocess_agent.swprocess_utils import import_swprocess
from swprocess_agent.results import DispersionResult


_VALID_TRANSFORMS = {"phase_shift", "fk", "fdbf"}


def _validate_masw_inputs(traces, offsets, dt, transform, fmin, fmax, vmin, vmax, nvel):
    """Validate MASW inputs."""
    if len(traces) < 3:
        raise ValueError(f"Need at least 3 traces, got {len(traces)}")
    if len(traces) != len(offsets):
        raise ValueError(
            f"traces and offsets must have same length: "
            f"{len(traces)} vs {len(offsets)}"
        )
    if dt <= 0:
        raise ValueError(f"dt must be > 0, got {dt}")
    if transform not in _VALID_TRANSFORMS:
        raise ValueError(f"transform must be one of {sorted(_VALID_TRANSFORMS)}, got '{transform}'")
    if fmin >= fmax:
        raise ValueError(f"fmin ({fmin}) must be < fmax ({fmax})")
    if vmin >= vmax:
        raise ValueError(f"vmin ({vmin}) must be < vmax ({vmax})")
    if nvel < 2:
        raise ValueError(f"nvel must be >= 2, got {nvel}")

    # Check all traces have same length
    lengths = set(len(t) for t in traces)
    if len(lengths) > 1:
        raise ValueError(f"All traces must have same length, got {sorted(lengths)}")


def analyze_masw(
    traces,
    offsets,
    dt,
    transform="phase_shift",
    fmin=5.0,
    fmax=100.0,
    vmin=50.0,
    vmax=1000.0,
    nvel=200,
) -> DispersionResult:
    """Run MASW dispersion analysis on multi-channel seismic data.

    Parameters
    ----------
    traces : list of array-like
        Seismograms for each sensor channel. Each is a 1D array.
    offsets : list of float
        Source-receiver offset for each channel (m).
    dt : float
        Sampling interval (seconds).
    transform : str
        Wavefield transform: 'phase_shift', 'fk', or 'fdbf'. Default 'phase_shift'.
    fmin : float
        Minimum frequency (Hz). Default 5.
    fmax : float
        Maximum frequency (Hz). Default 100.
    vmin : float
        Minimum phase velocity (m/s). Default 50.
    vmax : float
        Maximum phase velocity (m/s). Default 1000.
    nvel : int
        Number of velocity bins. Default 200.

    Returns
    -------
    DispersionResult
        Dispersion image and extracted curve.
    """
    traces_arr = [np.asarray(t, dtype=float) for t in traces]
    offsets_arr = [float(o) for o in offsets]

    _validate_masw_inputs(traces_arr, offsets_arr, dt, transform, fmin, fmax, vmin, vmax, nvel)

    sw = import_swprocess()
    ActiveTimeSeries = sw["ActiveTimeSeries"]
    Sensor1C = sw["Sensor1C"]
    Source = sw["Source"]
    Array1D = sw["Array1D"]

    transform_map = {
        "phase_shift": sw["PhaseShift"],
        "fk": sw["FK"],
        "fdbf": sw["FDBF"],
    }

    # Build sensor array
    sensors = []
    for trace, offset in zip(traces_arr, offsets_arr):
        ts = ActiveTimeSeries(trace, dt)
        sensor = Sensor1C.from_activetimeseries(ts, x=offset, y=0, z=0)
        sensors.append(sensor)

    source = Source(x=0, y=0, z=0)
    arr = Array1D(sensors, source)

    # Run wavefield transform — FK requires fdbf-specific block
    settings = {
        "fmin": fmin, "fmax": fmax,
        "vmin": vmin, "vmax": vmax,
        "nvel": nvel, "vspace": "linear",
        "fdbf-specific": {
            "weighting": "sqrt",
            "steering": "cylindrical",
        },
    }
    transform_cls = transform_map[transform]
    wf = transform_cls.from_array(arr, settings)

    # Extract dispersion curve (peak power at each frequency)
    peak_velocities = wf.find_peak_power()

    return DispersionResult(
        n_channels=arr.nchannels,
        spacing_m=float(arr.spacing),
        transform=transform,
        n_freq=len(wf.frequencies),
        n_vel=nvel,
        frequencies=wf.frequencies,
        velocities_grid=wf.velocities,
        power=wf.power,
        disp_freq=wf.frequencies,
        disp_vel=peak_velocities,
    )
