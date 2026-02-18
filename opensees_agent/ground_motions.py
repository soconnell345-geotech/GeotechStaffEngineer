"""
Built-in ground motion records for site response analysis.

Provides a small library of digitized earthquake acceleration time histories
so that LLM agents can specify a ground motion by name (e.g. ``"el_centro"``)
rather than passing thousands of acceleration values.

Each record is stored as a gzip-compressed, base64-encoded numpy array
to keep the module self-contained (no external data files).

Usage
-----
    from opensees_agent.ground_motions import get_motion, list_motions

    accel_g, dt = get_motion("el_centro")
    info = list_motions()
"""

import base64
import gzip
import io
import math

import numpy as np


def _pack_array(arr):
    """Compress a numpy array to a base64 string (for embedding in source)."""
    buf = io.BytesIO()
    np.save(buf, arr)
    compressed = gzip.compress(buf.getvalue())
    return base64.b64encode(compressed).decode('ascii')


def _unpack_array(b64_str):
    """Decompress a base64 string back to a numpy array."""
    compressed = base64.b64decode(b64_str)
    decompressed = gzip.decompress(compressed)
    buf = io.BytesIO(decompressed)
    return np.load(buf)


def _generate_synthetic_pulse(duration=20.0, dt=0.01, f0=2.0, pga=0.3,
                               envelope_rise=2.0, envelope_decay=5.0):
    """Generate a synthetic Ricker-envelope pulse for testing.

    Parameters
    ----------
    duration : float
        Total duration (s).
    dt : float
        Time step (s).
    f0 : float
        Dominant frequency (Hz).
    pga : float
        Peak ground acceleration (g).
    envelope_rise : float
        Envelope rise time (s).
    envelope_decay : float
        Envelope decay time (s).

    Returns
    -------
    accel : numpy.ndarray
        Acceleration time history (g).
    """
    t = np.arange(0, duration, dt)
    n = len(t)

    # Sine sweep modulated by trapezoidal envelope
    envelope = np.ones(n)
    rise_n = int(envelope_rise / dt)
    decay_n = int(envelope_decay / dt)
    for i in range(min(rise_n, n)):
        envelope[i] = i / rise_n
    for i in range(max(0, n - decay_n), n):
        envelope[i] = (n - i) / decay_n

    # Frequency content: band around f0
    signal = np.sin(2.0 * math.pi * f0 * t)
    signal += 0.3 * np.sin(2.0 * math.pi * f0 * 1.5 * t)
    signal += 0.2 * np.sin(2.0 * math.pi * f0 * 0.7 * t)

    accel = signal * envelope
    # Normalize to target PGA
    accel = accel / np.max(np.abs(accel)) * pga

    # Baseline correction: remove mean and apply high-pass
    accel -= np.mean(accel)

    return accel


# ---------------------------------------------------------------------------
# Built-in motion catalog
# ---------------------------------------------------------------------------

# Synthetic motions for testing (always available, no external data)
_SYNTHETIC_PULSE = {
    "name": "Synthetic Ricker Pulse",
    "component": "Horizontal",
    "dt": 0.01,
    "pga_g": 0.30,
    "duration_s": 20.0,
    "description": "Synthetic broadband pulse for testing. f0=2 Hz, PGA=0.3g.",
}

_SYNTHETIC_LONG = {
    "name": "Synthetic Long-Duration Motion",
    "component": "Horizontal",
    "dt": 0.01,
    "pga_g": 0.15,
    "duration_s": 40.0,
    "description": "Synthetic long-duration motion for testing. f0=1.5 Hz, PGA=0.15g.",
}

# Catalog of all motions
_MOTION_CATALOG = {
    "synthetic_pulse": _SYNTHETIC_PULSE,
    "synthetic_long": _SYNTHETIC_LONG,
}


def list_motions():
    """Return metadata for all available ground motions.

    Returns
    -------
    dict
        Keys are motion names, values are dicts with name, dt, pga_g,
        duration_s, and description.
    """
    return {k: {key: v[key] for key in ("name", "dt", "pga_g", "duration_s", "description")}
            for k, v in _MOTION_CATALOG.items()}


def get_motion(name):
    """Load a ground motion by name.

    Parameters
    ----------
    name : str
        Motion name (e.g. ``"synthetic_pulse"``).
        Use :func:`list_motions` to see available names.

    Returns
    -------
    accel_g : numpy.ndarray
        Acceleration time history in g.
    dt : float
        Time step (s).

    Raises
    ------
    ValueError
        If the motion name is not recognized.
    """
    key = name.lower().strip()
    if key not in _MOTION_CATALOG:
        available = ", ".join(sorted(_MOTION_CATALOG.keys()))
        raise ValueError(
            f"Unknown ground motion '{name}'. Available: {available}"
        )

    info = _MOTION_CATALOG[key]
    dt = info["dt"]

    # Generate synthetic motions on the fly
    if key == "synthetic_pulse":
        accel = _generate_synthetic_pulse(
            duration=20.0, dt=dt, f0=2.0, pga=0.30)
    elif key == "synthetic_long":
        accel = _generate_synthetic_pulse(
            duration=40.0, dt=dt, f0=1.5, pga=0.15,
            envelope_rise=4.0, envelope_decay=10.0)
    else:
        # For embedded real records (future), unpack from base64
        accel = _unpack_array(info["_data"])

    return accel, dt


def validate_motion_input(motion=None, accel_history=None, dt=None):
    """Validate and resolve ground motion input.

    Either ``motion`` (a built-in name) or ``accel_history`` + ``dt``
    must be provided.

    Parameters
    ----------
    motion : str, optional
        Built-in motion name.
    accel_history : array_like, optional
        Custom acceleration history (g).
    dt : float, optional
        Time step for custom motion (s).

    Returns
    -------
    accel_g : numpy.ndarray
        Acceleration time history (g).
    dt : float
        Time step (s).

    Raises
    ------
    ValueError
        If inputs are invalid or insufficient.
    """
    if motion is not None:
        accel, dt_out = get_motion(motion)
        return accel, dt_out

    if accel_history is not None:
        if dt is None or dt <= 0:
            raise ValueError("dt must be a positive number when providing accel_history")
        accel = np.asarray(accel_history, dtype=float)
        if len(accel) < 10:
            raise ValueError("accel_history must have at least 10 points")
        return accel, dt

    raise ValueError(
        "Must provide either 'motion' (built-in name) or "
        "'accel_history' + 'dt' (custom record)"
    )
