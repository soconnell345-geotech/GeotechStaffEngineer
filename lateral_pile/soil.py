"""
Soil layer definition module.

Defines the SoilLayer class for representing soil layers with
associated p-y curve models for lateral pile analysis.

All units are SI: meters (m), kilonewtons (kN), kilopascals (kPa).
"""

import warnings
from dataclasses import dataclass
from typing import Optional


@dataclass
class SoilLayer:
    """A soil layer with an associated p-y curve model.

    Parameters
    ----------
    top : float
        Depth to top of layer (m), measured from ground surface.
    bottom : float
        Depth to bottom of layer (m), measured from ground surface.
    py_model : object
        A p-y curve model instance (e.g., SoftClayMatlock, SandAPI).
        Must implement get_p(y, z, b) and get_py_curve(z, b) methods.
    description : str, optional
        Description of the soil layer (e.g., "Soft marine clay").

    Examples
    --------
    >>> from lateral_pile.py_curves import SoftClayMatlock
    >>> layer = SoilLayer(
    ...     top=0.0, bottom=5.0,
    ...     py_model=SoftClayMatlock(c=25.0, gamma=8.0, eps50=0.02, J=0.5),
    ...     description="Soft clay"
    ... )
    """
    top: float
    bottom: float
    py_model: object
    description: Optional[str] = None

    def __post_init__(self):
        if self.bottom <= self.top:
            raise ValueError(
                f"Layer bottom ({self.bottom} m) must be greater than top ({self.top} m)"
            )
        # Verify the py_model has the required interface
        if not hasattr(self.py_model, 'get_p'):
            raise TypeError(
                f"py_model must implement get_p(y, z, b) method, "
                f"got {type(self.py_model).__name__}"
            )
        if not hasattr(self.py_model, 'get_py_curve'):
            raise TypeError(
                f"py_model must implement get_py_curve(z, b) method, "
                f"got {type(self.py_model).__name__}"
            )

    @property
    def thickness(self) -> float:
        """Layer thickness (m)."""
        return self.bottom - self.top

    def contains_depth(self, z: float) -> bool:
        """Check if a depth falls within this layer.

        Parameters
        ----------
        z : float
            Depth from ground surface (m).

        Returns
        -------
        bool
            True if z is within [top, bottom].
        """
        return self.top <= z <= self.bottom


def get_layer_at_depth(layers: list, z: float) -> 'SoilLayer':
    """Find the soil layer containing a given depth.

    Parameters
    ----------
    layers : list of SoilLayer
        Soil layer definitions, should cover the full pile length.
    z : float
        Depth from ground surface (m).

    Returns
    -------
    SoilLayer
        The layer containing depth z.

    Raises
    ------
    ValueError
        If no layer contains the specified depth.
    """
    for layer in layers:
        if layer.contains_depth(z):
            return layer
    raise ValueError(
        f"No soil layer defined at depth {z} m. "
        f"Layers cover depths: {[(l.top, l.bottom) for l in layers]}"
    )


def validate_layers(layers: list, pile_length: float) -> None:
    """Validate that soil layers are consistent and cover the pile length.

    Parameters
    ----------
    layers : list of SoilLayer
        Soil layer definitions.
    pile_length : float
        Embedded pile length (m).

    Raises
    ------
    ValueError
        If layers overlap, have gaps, or don't cover pile length.
    """
    if not layers:
        raise ValueError("At least one soil layer must be defined")

    # Sort by top depth
    sorted_layers = sorted(layers, key=lambda l: l.top)

    # Check for gaps and overlaps
    for i in range(len(sorted_layers) - 1):
        current = sorted_layers[i]
        next_layer = sorted_layers[i + 1]

        if current.bottom < next_layer.top:
            raise ValueError(
                f"Gap in soil layers between {current.bottom} m and {next_layer.top} m"
            )
        if current.bottom > next_layer.top:
            raise ValueError(
                f"Overlap in soil layers: layer ending at {current.bottom} m "
                f"overlaps with layer starting at {next_layer.top} m"
            )

    # Check coverage
    top_depth = sorted_layers[0].top
    bottom_depth = sorted_layers[-1].bottom

    if top_depth > 0:
        warnings.warn(
            f"Soil layers start at {top_depth} m, not at ground surface (0 m). "
            f"No soil resistance will be applied above {top_depth} m."
        )

    if bottom_depth < pile_length:
        warnings.warn(
            f"Soil layers extend to {bottom_depth} m but pile length is {pile_length} m. "
            f"No soil resistance will be applied below {bottom_depth} m."
        )
