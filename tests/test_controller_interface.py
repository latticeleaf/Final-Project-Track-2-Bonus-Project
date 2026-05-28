import numpy as np
import pytest

from competition.race_scene import resolve_go2_asset_model_dir
from competition.track_scene import build_track_model
from go2_pg_env.track import StandardOvalTrack
from track_bonus.controller_interface import (
    MAX_TOURNAMENT_ENTRIES,
    TRACK_OBS_FEATURE_NAMES,
    build_track_controller_observation,
    validate_high_level_command,
)


def test_validate_high_level_command_keeps_values_and_validates_shape() -> None:
    command = validate_high_level_command(np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(command, np.asarray([2.0, -1.0, 2.0], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_high_level_command(np.asarray([0.1, 0.2], dtype=np.float32))
    with pytest.raises(ValueError):
        validate_high_level_command(np.asarray([0.1, np.nan, 0.2], dtype=np.float32))


def test_track_controller_observation_is_compact_track_state() -> None:
    track = StandardOvalTrack()
    xy, heading, _ = track.centerline_pose(0.0)
    qpos = np.zeros(19, dtype=np.float32)
    qpos[:2] = xy
    qpos[2] = 0.31
    qpos[3:7] = np.asarray([np.cos(0.5 * heading), 0.0, 0.0, np.sin(0.5 * heading)], dtype=np.float32)
    obs = build_track_controller_observation(qpos=qpos, track=track)
    assert TRACK_OBS_FEATURE_NAMES == (
        "lap_fraction",
        "lateral_error_norm",
        "boundary_margin_norm",
        "heading_error_rad",
        "curvature_norm",
    )
    assert obs.as_array().shape == (5,)
    assert abs(obs.lateral_error_norm) < 1e-6
    assert obs.boundary_margin_norm == pytest.approx(1.0)
    assert abs(obs.heading_error_rad) < 1e-6
    assert 0.0 <= obs.lap_fraction < 1.0


def test_track_scene_compiles_ten_dogs_when_assets_are_available() -> None:
    try:
        resolve_go2_asset_model_dir()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model = build_track_model(num_dogs=MAX_TOURNAMENT_ENTRIES, colors=["#2563EB"] * MAX_TOURNAMENT_ENTRIES)
    assert model.nq == 19 * MAX_TOURNAMENT_ENTRIES
    assert model.nu == 12 * MAX_TOURNAMENT_ENTRIES
