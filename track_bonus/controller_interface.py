"""Shared high-level controller interface for the track tournament.

The official high-level output is always:

    [vx_mps, vy_mps, yaw_rate_radps]

The command is consumed by the HW1-style low-level Go2 policy.  This module
keeps the command contract explicit so student planners remain compatible with
single-policy evaluation and later multi-policy tournament rendering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.planner import yaw_from_quat_wxyz


MAX_TOURNAMENT_ENTRIES = 10

TRACK_OBS_FEATURE_NAMES = (
    "lap_fraction",
    "lateral_error_norm",
    "boundary_margin_norm",
    "heading_error_rad",
    "curvature_norm",
)


@dataclass(frozen=True)
class TrackControllerObservation:
    """Compact track-coordinate observation for the tournament controller."""

    lap_fraction: float
    lateral_error_norm: float
    boundary_margin_norm: float
    heading_error_rad: float
    curvature_norm: float

    def as_array(self) -> np.ndarray:
        return np.asarray(
            [
                self.lap_fraction,
                self.lateral_error_norm,
                self.boundary_margin_norm,
                self.heading_error_rad,
                self.curvature_norm,
            ],
            dtype=np.float32,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "lap_fraction": float(self.lap_fraction),
            "lateral_error_norm": float(self.lateral_error_norm),
            "boundary_margin_norm": float(self.boundary_margin_norm),
            "heading_error_rad": float(self.heading_error_rad),
            "curvature_norm": float(self.curvature_norm),
        }


def build_track_controller_observation(
    *,
    qpos: np.ndarray,
    track: StandardOvalTrack,
) -> TrackControllerObservation:
    """Build the official 5D high-level observation from the robot's own pose."""
    qpos = np.asarray(qpos, dtype=np.float32)
    base_xy = np.asarray(qpos[:2], dtype=np.float64)
    base_yaw = yaw_from_quat_wxyz(np.asarray(qpos[3:7], dtype=np.float64))
    projection = track.project_xy_to_track(base_xy)
    _, track_heading, curvature = track.centerline_pose(projection.s)
    heading_error = wrap_angle(track_heading - base_yaw)
    half_width = max(float(track.half_width_m), 1e-6)
    turn_radius = max(float(track.turn_radius_m), 1e-6)
    return TrackControllerObservation(
        lap_fraction=float((projection.s % track.length_m) / track.length_m),
        lateral_error_norm=float(projection.signed_lateral_error / half_width),
        boundary_margin_norm=float(projection.distance_to_boundary / half_width),
        heading_error_rad=float(heading_error),
        curvature_norm=float(curvature * turn_radius),
    )


def validate_high_level_command(command: np.ndarray) -> np.ndarray:
    """Validate a high-level command without changing its numeric values."""
    command = np.asarray(command, dtype=np.float32)
    if command.shape != (3,):
        raise ValueError(f"High-level command must have shape (3,), got {command.shape}.")
    if not np.all(np.isfinite(command)):
        raise ValueError(f"High-level command contains non-finite values: {command!r}.")
    return command.astype(np.float32)
