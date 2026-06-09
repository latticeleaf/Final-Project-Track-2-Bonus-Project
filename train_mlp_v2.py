#!/usr/bin/env python3
"""
Two-phase MLP high-level planner training:

Phase 1 - Supervised pretraining:
  Sample random observations, query the tuned PD planner for commands,
  train the MLP to imitate via MSE. Gives a stable initialization.

Phase 2 - CEM rollout optimization:
  Black-box search over MLP weights using composite_score directly.
  Warm-starts from the supervised pretrained weights.

Usage:
    python train_mlp_v2.py \
        --checkpoint-dir /content/hw1_repo/artifacts/run_baseline/best_checkpoint \
        --output-dir artifacts/highlevel_mlp_v2 \
        --cem-iterations 15 \
        --population 16 \
        --eval-seconds 60
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from track_bonus.planner import MLPPolicy, StarterPlannerConfig
from go2_pg_env.track import wrap_angle

HIDDEN_SIZE = 32


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--config", type=Path, default=ROOT / "configs" / "colab_runtime_config.json")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cem-iterations", type=int, default=15)
    p.add_argument("--population", type=int, default=16)
    p.add_argument("--elite-frac", type=float, default=0.25)
    p.add_argument("--eval-seconds", type=float, default=60.0)
    p.add_argument("--sigma-init", type=float, default=0.3)
    p.add_argument("--sigma-decay", type=float, default=0.92)
    p.add_argument("--sigma-min", type=float, default=0.04)
    p.add_argument("--pretrain-epochs", type=int, default=300)
    p.add_argument("--pretrain-lr", type=float, default=0.005)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--entry-name", type=str, default="mlp_v2")
    p.add_argument("--warm-start-weights", type=Path, default=None,
                   help="Skip pretraining and warm-start CEM from existing weights")
    return p.parse_args()


# ---------------------------------------------------------------------------
# PD planner logic (inline, no file dependency)
# ---------------------------------------------------------------------------

def pd_command(obs_array: np.ndarray) -> np.ndarray:
    """Tuned PD planner — generates supervised pretraining targets."""
    speed_mps = 1.5
    min_speed_mps = 0.6
    max_lateral_speed_mps = 0.08
    max_yaw_rate_radps = 0.65
    k_heading = 1.2
    k_lateral = 0.08
    heading_slowdown = 0.45
    turn_radius_m = 18.25
    half_width_m = 2.0

    _, lateral_error_norm, _, heading_error_rad, curvature_norm = obs_array
    lateral_error = float(lateral_error_norm) * half_width_m
    lateral_bias = math.atan2(k_lateral * lateral_error, max(speed_mps, 1e-3))
    heading_error = wrap_angle(float(heading_error_rad) - lateral_bias)
    speed_scale = 1.0 - heading_slowdown * min(abs(heading_error), math.pi) / math.pi
    vx = float(np.clip(speed_mps * speed_scale, min_speed_mps, speed_mps))
    vy = float(np.clip(-k_lateral * lateral_error, -max_lateral_speed_mps, max_lateral_speed_mps))
    curvature = float(curvature_norm) / max(turn_radius_m, 1e-6)
    yaw_rate = float(np.clip(curvature * vx + k_heading * heading_error,
                              -max_yaw_rate_radps, max_yaw_rate_radps))
    return np.asarray([vx, vy, yaw_rate], dtype=np.float32)


# ---------------------------------------------------------------------------
# Supervised pretraining
# ---------------------------------------------------------------------------

def generate_pd_demos(n_samples: int = 8000, seed: int = 42):
    rng = np.random.default_rng(seed)
    obs = np.zeros((n_samples, 5), dtype=np.float32)
    obs[:, 0] = rng.uniform(0.0, 1.0, n_samples)       # lap_fraction
    obs[:, 1] = rng.uniform(-0.8, 0.8, n_samples)      # lateral_error_norm
    obs[:, 2] = rng.uniform(0.1, 1.0, n_samples)       # boundary_margin_norm
    obs[:, 3] = rng.uniform(-0.8, 0.8, n_samples)      # heading_error_rad
    obs[:, 4] = rng.choice([0.0, 1.0], n_samples)      # curvature_norm
    cmds = np.array([pd_command(o) for o in obs], dtype=np.float32)
    return obs, cmds


def pretrain_supervised(mlp: MLPPolicy, obs: np.ndarray, cmds: np.ndarray,
                        epochs: int = 300, lr: float = 0.005, seed: int = 42) -> MLPPolicy:
    """Imitation learning via finite-difference gradient descent."""
    rng = np.random.default_rng(seed)

    # Normalize targets to [-1, 1] to match MLP tanh output
    targets = np.zeros_like(cmds)
    targets[:, 0] = np.clip(2.0 * (cmds[:, 0] - 0.6) / (1.5 - 0.6) - 1.0, -1, 1)
    targets[:, 1] = np.clip(cmds[:, 1] / 0.08, -1, 1)
    targets[:, 2] = np.clip(cmds[:, 2] / 0.65, -1, 1)

    params = mlp.get_flat_params().copy()
    best_loss = float("inf")
    best_params = params.copy()
    eps = 1e-4
    batch_size = 512
    n = len(obs)

    for epoch in range(epochs):
        idx = rng.permutation(n)[:batch_size]
        X, Y = obs[idx], targets[idx]

        current_mlp = MLPPolicy.from_flat_params(params, HIDDEN_SIZE)
        preds = np.array([current_mlp.forward(x) for x in X])
        loss = float(np.mean((preds - Y) ** 2))

        if loss < best_loss:
            best_loss = loss
            best_params = params.copy()

        # Finite difference gradient on random subset of params
        grad = np.zeros_like(params)
        perturb_idx = rng.choice(len(params), size=min(80, len(params)), replace=False)
        for i in perturb_idx:
            p2 = params.copy(); p2[i] += eps
            m2 = MLPPolicy.from_flat_params(p2, HIDDEN_SIZE)
            preds2 = np.array([m2.forward(x) for x in X])
            grad[i] = (float(np.mean((preds2 - Y) ** 2)) - loss) / eps

        params -= lr * grad

        if epoch % 50 == 0:
            print(f"  pretrain epoch={epoch:03d} loss={loss:.4f} best={best_loss:.4f}", flush=True)

    print(f"  pretraining done. best_loss={best_loss:.4f}")
    return MLPPolicy.from_flat_params(best_params, HIDDEN_SIZE)


# ---------------------------------------------------------------------------
# CEM helpers
# ---------------------------------------------------------------------------

def write_planner_config(output_dir: Path, weights_path: Path) -> Path:
    cfg = {
        "planner_type": "mlp",
        "speed_mps": 1.5,
        "min_speed_mps": 0.6,
        "max_lateral_speed_mps": 0.08,
        "max_yaw_rate_radps": 0.65,
        "k_heading": 1.2,
        "k_lateral": 0.08,
        "heading_slowdown": 0.45,
        "stand_seconds": 1.0,
        "weights_path": str(weights_path),
        "hidden_size": HIDDEN_SIZE,
    }
    cfg_path = output_dir / "planner_config.json"
    cfg_path.write_text(json.dumps(cfg, indent=2))
    return cfg_path


def eval_params(params, *, checkpoint_dir, config, output_dir, weights_path, eval_seconds, entry_name) -> float:
    mlp = MLPPolicy.from_flat_params(params, HIDDEN_SIZE)
    mlp.save(weights_path)
    planner_cfg = write_planner_config(output_dir, weights_path)
    cmd = [
        sys.executable, "run_track_bonus.py",
        "--checkpoint-dir", str(checkpoint_dir),
        "--planner-config", str(planner_cfg),
        "--config", str(config),
        "--output-dir", str(output_dir / "eval"),
        "--entry-name", entry_name,
        "--duration-seconds", str(eval_seconds),
        "--no-render",
    ]
    try:
        subprocess.run(cmd, cwd=ROOT, check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
        results = json.loads((output_dir / "eval" / "results.json").read_text())
        return float(results["scores"]["composite_score"])  # use composite_score directly
    except Exception as e:
        print(f"  [eval error] {e}")
        return -1.0


def run_cem(mean_params, *, checkpoint_dir, config, output_dir, iterations, population,
            elite_frac, eval_seconds, sigma_init, sigma_decay, sigma_min, entry_name, seed):
    rng = np.random.default_rng(seed)
    n_elite = max(1, int(population * elite_frac))
    sigma = sigma_init
    best_score = -999.0
    best_params = mean_params.copy()
    history = []

    for iteration in range(iterations):
        noise = rng.normal(0, sigma, (population, len(mean_params)))
        pop = mean_params[None, :] + noise
        pop[0] = mean_params

        scores = []
        for i, params in enumerate(pop):
            cand_dir = output_dir / f"iter_{iteration:03d}_cand_{i:02d}"
            cand_dir.mkdir(parents=True, exist_ok=True)
            score = eval_params(
                params, checkpoint_dir=checkpoint_dir, config=config,
                output_dir=cand_dir, weights_path=cand_dir / "mlp_weights.npz",
                eval_seconds=eval_seconds, entry_name=entry_name,
            )
            scores.append(score)
            print(f"iter={iteration:03d} cand={i:02d} score={score:.4f} best={max(best_score, max(scores)):.4f}", flush=True)

            if score > best_score:
                best_score = score
                best_params = params.copy()
                MLPPolicy.from_flat_params(best_params, HIDDEN_SIZE).save(output_dir / "best_mlp_weights.npz")
                write_planner_config(output_dir, output_dir / "best_mlp_weights.npz")
                print(f"  *** new best: {best_score:.4f} ***")

        scores_arr = np.array(scores)
        elite_idx = np.argsort(scores_arr)[-n_elite:]
        mean_params = pop[elite_idx].mean(axis=0)
        sigma = max(sigma_min, sigma * sigma_decay)

        history.append({"iteration": iteration, "best_score": float(best_score),
                        "sigma": float(sigma), "scores": [float(s) for s in scores]})
        (output_dir / "history.json").write_text(json.dumps(history, indent=2))
        print(f"--- iter {iteration} done | sigma={sigma:.3f} | best={best_score:.4f} ---\n")

    return best_params


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: supervised pretraining or warm start
    if args.warm_start_weights and Path(args.warm_start_weights).exists():
        print(f"Warm-starting from {args.warm_start_weights}")
        mlp = MLPPolicy.load(Path(args.warm_start_weights))
    else:
        print("Phase 1: generating PD demo data...")
        obs, cmds = generate_pd_demos(n_samples=8000, seed=args.seed)
        print(f"  {len(obs)} demo pairs generated")
        print("Phase 1: supervised pretraining...")
        mlp = MLPPolicy.random_init(HIDDEN_SIZE, seed=args.seed)
        mlp = pretrain_supervised(mlp, obs, cmds,
                                  epochs=args.pretrain_epochs,
                                  lr=args.pretrain_lr,
                                  seed=args.seed)
        mlp.save(output_dir / "pretrained_weights.npz")

    # Quick pretrain eval
    print("\nEvaluating pretrained/warm-start weights...")
    pre_dir = output_dir / "pretrain_eval"
    pre_dir.mkdir(exist_ok=True)
    pre_score = eval_params(
        mlp.get_flat_params(), checkpoint_dir=args.checkpoint_dir, config=args.config,
        output_dir=pre_dir, weights_path=pre_dir / "mlp_weights.npz",
        eval_seconds=args.eval_seconds, entry_name=args.entry_name,
    )
    print(f"Pretrained composite_score: {pre_score:.4f}")

    # Phase 2: CEM
    print("\nPhase 2: CEM rollout optimization (using composite_score)...")
    run_cem(
        mlp.get_flat_params(),
        checkpoint_dir=args.checkpoint_dir, config=args.config,
        output_dir=output_dir, iterations=args.cem_iterations,
        population=args.population, elite_frac=args.elite_frac,
        eval_seconds=args.eval_seconds, sigma_init=args.sigma_init,
        sigma_decay=args.sigma_decay, sigma_min=args.sigma_min,
        entry_name=args.entry_name, seed=args.seed,
    )

    print(f"\nDone.")
    print(f"Best weights: {output_dir / 'best_mlp_weights.npz'}")
    print(f"Planner config: {output_dir / 'planner_config.json'}")


if __name__ == "__main__":
    main()
