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


def validate_layers(layers: list, pile_length: float,
                    extend_last_layer: bool = False) -> None:
    """Validate that soil layers are consistent and cover the pile length.

    Parameters
    ----------
    layers : list of SoilLayer
        Soil layer definitions.
    pile_length : float
        Embedded pile length (m).
    extend_last_layer : bool, optional
        True accepts layers that stop above the pile toe; the caller then
        carries the last layer on to the toe (``LateralPileAnalysis``).
        Default False: refused (GM-318).

    Raises
    ------
    ValueError
        If layers overlap, have gaps, start below the ground surface, or
        stop above the pile toe (unless ``extend_last_layer``). Before
        GM-318 the last two only raised a Python warning and the run went
        on with no soil resistance there.
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

    tol = 1e-9 * max(pile_length, 1.0)
    if top_depth > tol:
        raise ValueError(
            f"The soil layers start at {top_depth:g} m, not at the ground "
            f"surface (0 m): no p-y curve is given for the pile from 0 to "
            f"{top_depth:g} m, and the run would give it no soil "
            f"resistance without saying so. Give layers from the ground "
            f"surface (depths are measured from it). For a pile length "
            f"standing free above the ground give stickup; for soil that "
            f"is gone (scour, an excavation) measure depths from the new "
            f"ground surface and add the lost length to stickup: the p-y "
            f"depths then start at the new surface, soil removed, as GEC 9 "
            f"(FHWA-HIF-18-031) Sec. 6.7.5, p. 91 takes the whole scour "
            f"zone. That differs from soil that stays in place, keeping "
            f"its weight on the soil below, and is to give no resistance "
            f"over the top {top_depth:g} m: for that, give the top "
            f"{top_depth:g} m from the ground surface as its own layer of "
            f"negligible strength.")

    if bottom_depth < pile_length - tol and not extend_last_layer:
        raise ValueError(
            f"The soil layers stop at {bottom_depth:g} m and the pile toe "
            f"is at {pile_length:g} m: no p-y curve is given for the pile "
            f"from {bottom_depth:g} m to the toe, in soil nobody explored, "
            f"and the run would give it no soil resistance without saying "
            f"so. Give layers down to the toe, or set "
            f"extend_last_layer=true to take the last layer as going on to "
            f"the toe; the result then says so.")
