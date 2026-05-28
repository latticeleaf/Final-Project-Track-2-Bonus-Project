# Track Controller Interface Contract

This contract keeps all Track 2 submissions compatible with the official
evaluator and the later 10-dog tournament renderer.

## 1. System Architecture

Each tournament entry has two layers:

```text
high-level track controller:
  5D track-coordinate observation -> [vx, vy, yaw_rate]

low-level Go2 locomotion policy:
  proprioception + command -> 12 joint actions
```

The official tournament does not simulate dog-dog collisions for scoring. Each
entry is rolled out independently, then the saved `qpos` trajectories are
synchronized in one MuJoCo scene for visualization. This makes ranking fair and
also avoids Python dependency conflicts between teams.

## 2. High-Level Input

The high-level controller should use the compact 5D track-coordinate
observation:

```text
[
  lap_fraction,
  lateral_error_norm,
  boundary_margin_norm,
  heading_error_rad,
  curvature_norm
]
```

Feature definitions:

- `lap_fraction`: projected progress along the 200 m centerline, divided by
  track length.
- `lateral_error_norm`: signed centerline error divided by track half-width.
- `boundary_margin_norm`: distance to the nearest boundary divided by track
  half-width. It becomes negative outside the lane.
- `heading_error_rad`: track tangent heading minus robot yaw.
- `curvature_norm`: local centerline curvature multiplied by turn radius.

The helper implementation is in:

```text
track_bonus/controller_interface.py
```

Students may compute the same features themselves, but the controller should
not depend on other robots, future states, hidden simulator internals, raw
privileged simulator state, or manually edited evaluator outputs. The compact
feature vector is deliberately smaller than full `qpos` so tournament entries
stay comparable.

## 3. High-Level Output

The output must be exactly:

```text
[vx_mps, vy_mps, yaw_rate_radps]
```

with shape `(3,)`.

The official evaluator checks only shape and finite values. It does not clip or
rescale commands. If a controller outputs commands that are too aggressive for
the low-level policy, the resulting fall or boundary violation is part of the
score.

## 4. Low-Level Policy Requirement

The low-level checkpoint must remain a Brax PPO checkpoint compatible with the
HW1-style Go2 joystick environment:

- checkpoint directory contains `ppo_network_config.json`
- actor uses `policy_obs_key = "state"`
- actor does not require `privileged_state`
- action is the standard 12-dimensional Go2 joint target offset

Students can retrain or improve this low-level policy, but the runtime
checkpoint format must stay compatible with `run_track_bonus.py`.

## 5. Submission Compatibility

For the starter repository, the default high-level artifact is:

```text
planner_config.json
```

loaded by `StarterTrackPlanner`. Students can replace the planner logic during
development, but the final submission must still be runnable by the command
listed in `submission.json`.

Recommended tournament manifest for instructors:

```json
{
  "entries": [
    {
      "name": "team_a",
      "rollout_npz": "team_a/track_eval/race_rollouts.npz",
      "color": "#2563EB"
    }
  ]
}
```

The manifest uses rollout files rather than importing 10 teams' Python
controllers into one process. This is the key design choice that prevents
controller conflicts.

## 6. 10-Dog Visualization

The renderer supports at most 10 entries. Internally it attaches prefixed Go2
models into one MuJoCo model:

```text
dog0_..., dog1_..., ..., dog9_...
```

For 10 robots the compiled model should satisfy:

```text
nq = 19 * 10 = 190
nu = 12 * 10 = 120
```

Demo command:

```bash
python scripts/render_track_tournament.py \
  --demo-synthetic \
  --num-dogs 10 \
  --track-half-width-m 2.0 \
  --output-dir artifacts/ten_dog_demo
```

To combine real evaluated submissions:

```bash
python scripts/render_track_tournament.py \
  --entries tournament_entries.json \
  --visual-lane-offsets \
  --output-dir artifacts/tournament_render
```

`--visual-lane-offsets` spreads trajectories only for readability in the video.
It does not change the saved per-team scoring results.
