import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np


def _cleanup_local_modules():
    for name in ["algo", "uav_env", "eval_seeds"]:
        if name in sys.modules:
            del sys.modules[name]


def _resolve_seed_source(repo_root: Path) -> Path:
    preferred = repo_root / "newnet_6_3" / "eval_seeds.py"
    fallback = repo_root / "newnet_6_3_f" / "eval_seeds.py"
    if preferred.exists():
        return preferred
    if fallback.exists():
        return fallback
    raise FileNotFoundError("无法找到 newnet_6_3/eval_seeds.py 或 newnet_6_3_f/eval_seeds.py")


def load_eval_seeds(repo_root: Path):
    seed_file = _resolve_seed_source(repo_root)
    sys.path.insert(0, str(seed_file.parent))
    try:
        _cleanup_local_modules()
        eval_seeds_module = importlib.import_module("eval_seeds")
        return eval_seeds_module.get_eval_seeds(), seed_file
    finally:
        sys.path.pop(0)
        _cleanup_local_modules()


def step_detection_stats(env, config_cls):
    detected_count = 0
    detected_targets = np.zeros(env.n_targets, dtype=bool)
    for target_idx, target in enumerate(env.targets):
        for agent in env.agents:
            dist = np.sqrt((agent["x"] - target.x) ** 2 + (agent["y"] - target.y) ** 2)
            if dist <= config_cls.SENSOR_RANGE:
                detected_targets[target_idx] = True
                detected_count += 1
                break
    return detected_targets, detected_count


def _coverage_cells(env, config_cls, agent):
    covered_indices = set()
    m_center, n_center = env._pos_to_grid(agent["x"], agent["y"])
    radius_grid = int(config_cls.SENSOR_RANGE / config_cls.GRID_WIDTH)
    for r in range(-radius_grid, radius_grid + 1):
        for c in range(-radius_grid, radius_grid + 1):
            if r ** 2 + c ** 2 <= radius_grid ** 2:
                gm, gn = m_center + r, n_center + c
                if 0 <= gm < config_cls.GRID_ROWS and 0 <= gn < config_cls.GRID_COLS:
                    covered_indices.add((gm, gn))
    return covered_indices


def compute_overlap_rate(env, config_cls):
    coverage_per_agent = []
    for agent in env.agents:
        coverage_per_agent.append(_coverage_cells(env, config_cls, agent))

    union_cells = set().union(*coverage_per_agent) if coverage_per_agent else set()
    if not union_cells:
        return 0.0

    cell_counts = {}
    for cells in coverage_per_agent:
        for cell in cells:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1

    overlap_cells = sum(1 for count in cell_counts.values() if count > 1)
    return overlap_cells / len(union_cells)


def compute_collision_count(env, config_cls):
    collisions = 0
    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            dist = np.sqrt(
                (env.agents[i]["x"] - env.agents[j]["x"]) ** 2
                + (env.agents[i]["y"] - env.agents[j]["y"]) ** 2
            )
            if dist < config_cls.SAFE_DIST_UAV:
                collisions += 1

    for agent in env.agents:
        for ox, oy, radius in env.obstacles:
            dist = np.sqrt((agent["x"] - ox) ** 2 + (agent["y"] - oy) ** 2) - radius
            if dist <= 0:
                collisions += 1
    return collisions


def evaluate_rank_model(env, maddpg, eval_seeds, config_cls, max_steps=200, coverage_steps=100):
    min_all_detect_steps = []
    total_detection_counts = []
    overlap_rates = []
    collision_counts = []
    coverage_efficiencies = []

    for seed in eval_seeds:
        obs_n, _ = env.reset(seed=seed)
        target_seen_once = np.zeros(env.n_targets, dtype=bool)
        min_all_detect_step = max_steps + 1
        total_detect_this_episode = 0
        overlap_accum = []
        collision_accum = 0
        coverage_step_value = None

        for step in range(max_steps):
            actions = maddpg.select_actions(obs_n, noise_std=0.0)
            obs_n, _, terminated, truncated, _ = env.step(actions)
            del terminated, truncated

            detected_targets, detected_count = step_detection_stats(env, config_cls)
            target_seen_once = np.logical_or(target_seen_once, detected_targets)
            total_detect_this_episode += detected_count

            overlap_accum.append(compute_overlap_rate(env, config_cls))
            collision_accum += compute_collision_count(env, config_cls)

            if step + 1 == coverage_steps:
                coverage_cells = int(np.sum(env.global_map_cover))
                coverage_efficiency = coverage_cells / (config_cls.GRID_ROWS * config_cls.GRID_COLS)
                coverage_step_value = coverage_efficiency

            if np.all(target_seen_once) and min_all_detect_step == max_steps + 1:
                min_all_detect_step = step + 1

        if coverage_step_value is None:
            coverage_cells = int(np.sum(env.global_map_cover))
            coverage_step_value = coverage_cells / (config_cls.GRID_ROWS * config_cls.GRID_COLS)

        min_all_detect_steps.append(min_all_detect_step)
        total_detection_counts.append(total_detect_this_episode)
        overlap_rates.append(float(np.mean(overlap_accum)) if overlap_accum else 0.0)
        collision_counts.append(collision_accum)
        coverage_efficiencies.append(coverage_step_value)

    return {
        "min_all_detect_step": float(np.mean(min_all_detect_steps)) if min_all_detect_steps else 0.0,
        "total_detection_count": float(np.mean(total_detection_counts)) if total_detection_counts else 0.0,
        "overlap_rate": float(np.mean(overlap_rates)) if overlap_rates else 0.0,
        "collision_count": float(np.mean(collision_counts)) if collision_counts else 0.0,
        "coverage_efficiency": float(np.mean(coverage_efficiencies)) if coverage_efficiencies else 0.0,
    }


def find_rank01_weights(target_dir: Path) -> Path:
    top_models_dir = target_dir / "models" / "top_models"
    if not top_models_dir.exists():
        raise FileNotFoundError(f"未找到目录: {top_models_dir}")

    rank_dirs = sorted(
        [p for p in top_models_dir.iterdir() if p.is_dir() and p.name.startswith("rank_01")]
    )
    if not rank_dirs:
        raise FileNotFoundError(f"未找到 rank_01 模型目录: {top_models_dir}")

    for rank_dir in rank_dirs:
        weights_dir = rank_dir / "weights"
        if weights_dir.exists():
            return weights_dir
    raise FileNotFoundError(f"rank_01 目录存在，但缺少 weights 子目录: {rank_dirs[0]}")


def evaluate_project(repo_root: Path, project_name: str, eval_seeds, max_steps: int):
    project_dir = repo_root / project_name
    if not project_dir.exists():
        raise FileNotFoundError(f"项目目录不存在: {project_dir}")

    sys.path.insert(0, str(project_dir))
    try:
        _cleanup_local_modules()
        algo = importlib.import_module("algo")
        uav_env = importlib.import_module("uav_env")

        for target_count in (1, 2, 3, 4):
            target_dir = project_dir / "models" / f"target_{target_count}"
            weight_dir = find_rank01_weights(target_dir)

            env = uav_env.UAVSwarmEnv(n_target=target_count)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            global_obs_dim = obs_dim * env.n_agents
            maddpg = algo.MADDPG(env.n_agents, obs_dim, act_dim, global_obs_dim)
            maddpg.load_models(str(weight_dir))

            summary = evaluate_rank_model(
                env,
                maddpg,
                eval_seeds,
                uav_env.Config,
                max_steps=max_steps,
            )

            out_dir = project_dir / "results" / f"target_{target_count}"
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / "metrics_summary.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump({"rank_01": summary}, f, ensure_ascii=False, indent=2)
            print(f"[OK] {project_name} target_{target_count} -> {out_path}")
    finally:
        sys.path.pop(0)
        _cleanup_local_modules()


def main():
    parser = argparse.ArgumentParser(description="评估多个 newnet 工程的 rank_01 模型并输出 metrics_summary.json")
    parser.add_argument(
        "--projects",
        nargs="*",
        default=["newnet_6_3", "newnet_6_4", "newnet_6_5", "newnet_6_6"],
        help="要评估的项目目录名称列表",
    )
    parser.add_argument("--max-steps", type=int, default=200)
    args = parser.parse_args()

    repo_root = Path(__file__).resolve().parent
    eval_seeds, seed_source = load_eval_seeds(repo_root)
    print(f"使用评估随机种子文件: {seed_source}")

    projects = []
    for name in args.projects:
        if name == "newnet_6_3" and not (repo_root / "newnet_6_3").exists():
            projects.append("newnet_6_3_f")
        else:
            projects.append(name)

    for project in projects:
        evaluate_project(repo_root, project, eval_seeds, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
