"""Learned MLP high-level planner for the 200 m track bonus.

Replaces the hand-written PD baseline with a small neural network:
    5D track observation -> MLP(theta) -> [vx, vy, yaw_rate]

Weights are stored in a .npz file and loaded via StarterTrackPlanner.load().
The entry points (StarterTrackPlanner.load, planner.command) are unchanged
so the evaluator works without modification.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from go2_pg_env.track import StandardOvalTrack, wrap_angle
from track_bonus.controller_interface import TrackControllerObservation
from track_bonus.official_track import official_track


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class StarterPlannerConfig:
    planner_type: str = "mlp"
    speed_mps: float = 1.5          # max forward speed
    min_speed_mps: float = 0.6
    max_lateral_speed_mps: float = 0.08
    max_yaw_rate_radps: float = 0.65
    k_heading: float = 1.2
    k_lateral: float = 0.08
    heading_slowdown: float = 0.45
    stand_seconds: float = 1.0
    weights_path: str = ""          # path to .npz weights, relative to planner.py
    hidden_size: int = 32

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "StarterPlannerConfig":
        valid = set(cls.__dataclass_fields__.keys())
        values = {key: payload[key] for key in valid if key in payload}
        return cls(**values)

    @classmethod
    def load(cls, path: Path) -> "StarterPlannerConfig":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def to_dict(self) -> dict[str, Any]:
        return {f: getattr(self, f) for f in self.__dataclass_fields__}


# ---------------------------------------------------------------------------
# Tiny MLP (numpy only, no framework dependency at inference)
# ---------------------------------------------------------------------------

class MLPPolicy:
    """Two-hidden-layer MLP: 5 -> H -> H -> 3, tanh activations."""

    def __init__(self, weights: dict[str, np.ndarray]) -> None:
        self.W1 = weights["W1"]   # (H, 5)
        self.b1 = weights["b1"]   # (H,)
        self.W2 = weights["W2"]   # (H, H)
        self.b2 = weights["b2"]   # (H,)
        self.W3 = weights["W3"]   # (3, H)
        self.b3 = weights["b3"]   # (3,)

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = np.tanh(self.W1 @ x + self.b1)
        x = np.tanh(self.W2 @ x + self.b2)
        x = np.tanh(self.W3 @ x + self.b3)
        return x.astype(np.float32)

    @classmethod
    def random_init(cls, hidden_size: int = 32, seed: int = 0) -> "MLPPolicy":
        rng = np.random.default_rng(seed)
        scale = 0.3
        weights = {
            "W1": rng.normal(0, scale, (hidden_size, 5)),
            "b1": np.zeros(hidden_size),
            "W2": rng.normal(0, scale, (hidden_size, hidden_size)),
            "b2": np.zeros(hidden_size),
            "W3": rng.normal(0, scale, (3, hidden_size)),
            "b3": np.zeros(3),
        }
        return cls(weights)

    @classmethod
    def load(cls, path: Path) -> "MLPPolicy":
        data = np.load(path)
        return cls({k: data[k] for k in data.files})

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, W1=self.W1, b1=self.b1,
                 W2=self.W2, b2=self.b2,
                 W3=self.W3, b3=self.b3)

    def get_flat_params(self) -> np.ndarray:
        return np.concatenate([
            self.W1.ravel(), self.b1,
            self.W2.ravel(), self.b2,
            self.W3.ravel(), self.b3,
        ])

    @classmethod
    def from_flat_params(cls, params: np.ndarray, hidden_size: int = 32) -> "MLPPolicy":
        H = hidden_size
        idx = 0
        def take(shape):
            nonlocal idx
            n = int(np.prod(shape))
            chunk = params[idx:idx+n].reshape(shape)
            idx += n
            return chunk
        weights = {
            "W1": take((H, 5)),
            "b1": take((H,)),
            "W2": take((H, H)),
            "b2": take((H,)),
            "W3": take((3, H)),
            "b3": take((3,)),
        }
        return cls(weights)


# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------

class StarterTrackPlanner:
    """Learned MLP high-level planner.

    Maps the official 5D track observation to [vx, vy, yaw_rate].
    Output is scaled by the config speed/rate limits so the MLP
    only needs to learn a normalized [-1, 1] policy.
    """

    def __init__(self, config: StarterPlannerConfig, mlp: MLPPolicy) -> None:
        self.config = config
        self.mlp = mlp
        self.track: StandardOvalTrack = official_track()

    @classmethod
    def load(cls, path: Path) -> "StarterTrackPlanner":
        config = StarterPlannerConfig.load(path)
        if config.weights_path:
            weights_path = path.parent / config.weights_path
            if weights_path.exists():
                mlp = MLPPolicy.load(weights_path)
                return cls(config, mlp)
        # No weights yet — use random init (will be trained by CEM)
        mlp = MLPPolicy.random_init(hidden_size=config.hidden_size)
        return cls(config, mlp)

    def command(self, obs: TrackControllerObservation, t: float) -> np.ndarray:
        if t < self.config.stand_seconds:
            return np.zeros(3, dtype=np.float32)
        return self.command_from_observation(obs)

    def command_from_observation(self, obs: TrackControllerObservation) -> np.ndarray:
        x = obs.as_array()  # 5D input, already normalized by controller_interface
        raw = self.mlp.forward(x)  # tanh output in [-1, 1]

        # Scale to physical limits
        vx = float(self.config.min_speed_mps +
                   (raw[0] * 0.5 + 0.5) *
                   (self.config.speed_mps - self.config.min_speed_mps))
        vy = float(raw[1]) * float(self.config.max_lateral_speed_mps)
        yaw_rate = float(raw[2]) * float(self.config.max_yaw_rate_radps)

        return np.asarray([vx, vy, yaw_rate], dtype=np.float32)
