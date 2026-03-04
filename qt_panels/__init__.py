"""Qt panel package — each analysis type in its own module."""

from qt_panels.common import APP_NAME, APP_VERSION, STYLESHEET
from qt_panels.bearing_capacity_panel import BearingCapacityPanel
from qt_panels.slope_stability_panel import SlopeStabilityPanel
from qt_panels.settlement_panel import SettlementPanel
from qt_panels.fem2d_panel import FEM2DPanel
from qt_panels.footing_design_panel import FootingDesignPanel

__all__ = [
    "APP_NAME", "APP_VERSION", "STYLESHEET",
    "BearingCapacityPanel",
    "SlopeStabilityPanel",
    "SettlementPanel",
    "FEM2DPanel",
    "FootingDesignPanel",
]
