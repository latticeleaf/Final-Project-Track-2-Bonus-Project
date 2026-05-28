"""Starter high-level planner for the 200 m track bonus.

The evaluator passes `qpos` to this wrapper so the starter can construct track
features internally. The student-facing tournament contract is the compact 5D
track observation defined in `track_bonus/controller_interface.py`, mapped to
the local joystick command consumed by the HW1 Go2 locomotion policy:

    qpos + track -> [vx, vy, yaw_rate]

This file is intentionally small.  It is a weak baseline and an interface
example, not a solved full-lap controller.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle


@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "starter_pd"
    speed_mps: float = 0.45
    min_speed_mps: float = 0.12
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.25
    lookahead_m: float = 3.0
    k_heading: float = 0.55
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0
    track_length_m: float = 200.0
    turn_radius_m: float = 18.25
    half_width_m: float = 2.0

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {
            "planner_type": self.planner_type,
            "speed_mps": self.speed_mps,
            "min_speed_mps": self.min_speed_mps,
            "max_lateral_speed_mps": self.max_lateral_speed_mps,
            "max_yaw_rate_radps": self.max_yaw_rate_radps,
            "lookahead_m": self.lookahead_m,
            "k_heading": self.k_heading,
            "k_lateral": self.k_lateral,
            "heading_slowdown": self.heading_slowdown,
            "stand_seconds": self.stand_seconds,
            "track_length_m": self.track_length_m,
            "turn_radius_m": self.turn_radius_m,
            "half_width_m": self.half_width_m,
        }


def yaw_from_quat_wxyz(quat: np.ndarray) -> float:
    w, x, y, z = np.asarray(quat, dtype=np.float64)
    return float(math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


class StarterTrackPlanner:
    """Conservative coordinate-to-command baseline.

    The policy is deliberately simple and conservative. Students should improve
    it by changing this controller, replacing it with an MLP, or training a
    higher-level policy that produces the same command vector.
    """

    def __init__(self, config: StarterPlannerConfig) -> None:
        if config.planner_type != "starter_pd":
            raise ValueError(f"Unsupported planner_type: {config.planner_type!r}")
        self.config = config
        self.track = StandardOvalTrack(
            length_m=float(config.track_length_m),
            turn_radius_m=float(config.turn_radius_m),
            half_width_m=float(config.half_width_m),
        )

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        return cls(StarterPlannerConfig.load(path))

    def command(self, qpos: np.ndarray, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)

        xy = np.asarray(qpos[:2], dtype=np.float64)
        yaw = yaw_from_quat_wxyz(np.asarray(qpos[3:7], dtype=np.float64))
        projection = self.track.project_xy_to_track(xy)
        _, lookahead_heading, lookahead_curvature = self.track.centerline_pose(
            projection.s + float(self.config.lookahead_m)
        )

        lateral_error = projection.signed_lateral_error
        lateral_bias = math.atan2(
            float(self.config.k_lateral) * lateral_error,
            max(float(self.config.speed_mps), 1e-3),
        )
        desired_heading = wrap_angle(lookahead_heading - lateral_bias)
        heading_error = wrap_angle(desired_heading - yaw)

        speed_scale = 1.0 - float(self.config.heading_slowdown) * min(abs(heading_error), math.pi) / math.pi
        vx = np.clip(
            float(self.config.speed_mps) * speed_scale,
            float(self.config.min_speed_mps),
            float(self.config.speed_mps),
        )
        vy = np.clip(
            -float(self.config.k_lateral) * lateral_error,
            -float(self.config.max_lateral_speed_mps),
            float(self.config.max_lateral_speed_mps),
        )
        yaw_rate = np.clip(
            lookahead_curvature * vx + float(self.config.k_heading) * heading_error,
            -float(self.config.max_yaw_rate_radps),
            float(self.config.max_yaw_rate_radps),
        )
        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)
