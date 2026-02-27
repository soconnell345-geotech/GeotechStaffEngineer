"""
CSV and dict loaders for SiteModel.

Provides three entry points:
- load_site_from_dict : Create SiteModel from nested dict structure
- load_site_from_csv : Create SiteModel from CSV file paths
- load_cpt_to_investigation : Bridge pygef_agent CPTParseResult to Investigation
"""

from __future__ import annotations

import csv
import io
from typing import Optional

import numpy as np

from subsurface_characterization.site_model import (
    SiteModel, Investigation, PointMeasurement, LithologyInterval,
)


def load_site_from_dict(data: dict) -> SiteModel:
    """Create SiteModel from nested dict structure.

    Expected format::

        {
            "project_name": "My Project",
            "coordinate_system": "EPSG:2263",
            "investigations": [
                {
                    "investigation_id": "B-1",
                    "investigation_type": "boring",
                    "x": 100.0, "y": 200.0,
                    "elevation_m": 10.0,
                    "total_depth_m": 30.0,
                    "gwl_depth_m": 5.0,
                    "lithology": [
                        {"top_depth_m": 0, "bottom_depth_m": 3, "description": "Fill", "uscs": "SM"},
                    ],
                    "measurements": [
                        {"depth_m": 1.5, "parameter": "N_spt", "value": 10},
                    ]
                },
            ]
        }
    """
    return SiteModel.from_dict(data)


def load_site_from_csv(
    borings_csv: str,
    measurements_csv: str,
    lithology_csv: Optional[str] = None,
) -> SiteModel:
    """Create SiteModel from CSV file paths or CSV strings.

    Parameters
    ----------
    borings_csv : str
        Path to CSV or CSV string with columns:
        investigation_id, investigation_type, x, y, elevation_m, total_depth_m,
        gwl_depth_m (optional), drilling_method (optional)
    measurements_csv : str
        Path to CSV or CSV string with columns:
        investigation_id, depth_m, parameter, value,
        source (optional), test_type (optional), sample_id (optional)
    lithology_csv : str, optional
        Path to CSV or CSV string with columns:
        investigation_id, top_depth_m, bottom_depth_m, description,
        uscs (optional), color (optional)

    Returns
    -------
    SiteModel
    """
    borings_rows = _read_csv(borings_csv)
    meas_rows = _read_csv(measurements_csv)
    lith_rows = _read_csv(lithology_csv) if lithology_csv else []

    # Build investigations from borings CSV
    investigations = {}
    for row in borings_rows:
        inv_id = row["investigation_id"]
        investigations[inv_id] = Investigation(
            investigation_id=inv_id,
            investigation_type=row.get("investigation_type", "boring"),
            x=float(row.get("x", 0)),
            y=float(row.get("y", 0)),
            elevation_m=float(row.get("elevation_m", 0)),
            total_depth_m=float(row.get("total_depth_m", 0)),
            gwl_depth_m=float(row["gwl_depth_m"]) if row.get("gwl_depth_m") else None,
            drilling_method=row.get("drilling_method", ""),
        )

    # Add measurements
    for row in meas_rows:
        inv_id = row["investigation_id"]
        if inv_id not in investigations:
            investigations[inv_id] = Investigation(investigation_id=inv_id)
        investigations[inv_id].measurements.append(PointMeasurement(
            depth_m=float(row["depth_m"]),
            parameter=row["parameter"],
            value=float(row["value"]),
            source=row.get("source", "field"),
            test_type=row.get("test_type", ""),
            sample_id=row.get("sample_id", ""),
        ))

    # Add lithology
    for row in lith_rows:
        inv_id = row["investigation_id"]
        if inv_id not in investigations:
            investigations[inv_id] = Investigation(investigation_id=inv_id)
        investigations[inv_id].lithology.append(LithologyInterval(
            top_depth_m=float(row["top_depth_m"]),
            bottom_depth_m=float(row["bottom_depth_m"]),
            description=row.get("description", ""),
            uscs=row.get("uscs", ""),
            color=row.get("color", ""),
        ))

    return SiteModel(
        investigations=list(investigations.values()),
    )


def load_cpt_to_investigation(cpt_result) -> Investigation:
    """Bridge pygef_agent CPTParseResult to Investigation.

    Parameters
    ----------
    cpt_result : CPTParseResult
        Parsed CPT data from pygef_agent.

    Returns
    -------
    Investigation
        With CPT measurements as PointMeasurements.
    """
    measurements = []

    depths = cpt_result.depth_m
    if len(depths) == 0:
        return Investigation(
            investigation_id=cpt_result.alias or "CPT-unknown",
            investigation_type="cpt",
        )

    # qc
    if len(cpt_result.q_c_kPa) > 0:
        for d, v in zip(depths, cpt_result.q_c_kPa):
            measurements.append(PointMeasurement(
                depth_m=float(d),
                parameter="qc_kPa",
                value=float(v),
                source="field",
                test_type="CPTu",
            ))

    # fs
    if len(cpt_result.f_s_kPa) > 0:
        for d, v in zip(depths, cpt_result.f_s_kPa):
            measurements.append(PointMeasurement(
                depth_m=float(d),
                parameter="fs_kPa",
                value=float(v),
                source="field",
                test_type="CPTu",
            ))

    # u2
    if len(cpt_result.u_2_kPa) > 0:
        for d, v in zip(depths, cpt_result.u_2_kPa):
            measurements.append(PointMeasurement(
                depth_m=float(d),
                parameter="u2_kPa",
                value=float(v),
                source="field",
                test_type="CPTu",
            ))

    # Rf
    if len(cpt_result.Rf_pct) > 0:
        for d, v in zip(depths, cpt_result.Rf_pct):
            measurements.append(PointMeasurement(
                depth_m=float(d),
                parameter="Rf_pct",
                value=float(v),
                source="field",
                test_type="CPTu",
            ))

    return Investigation(
        investigation_id=cpt_result.alias or "CPT-unknown",
        investigation_type="cpt",
        x=cpt_result.x if cpt_result.x is not None else 0.0,
        y=cpt_result.y if cpt_result.y is not None else 0.0,
        total_depth_m=float(cpt_result.final_depth_m),
        gwl_depth_m=float(cpt_result.gwl_m) if cpt_result.gwl_m is not None else None,
        coordinate_system=cpt_result.srs_name or "",
        measurements=measurements,
    )


def _read_csv(source: str) -> list:
    """Read CSV from file path or string, return list of dicts."""
    try:
        # Try as file path first
        with open(source, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            return list(reader)
    except (FileNotFoundError, OSError):
        # Treat as CSV string
        reader = csv.DictReader(io.StringIO(source))
        return list(reader)
