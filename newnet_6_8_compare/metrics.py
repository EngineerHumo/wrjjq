import numpy as np

from uav_env import Config


def step_detection_stats(env):
    detected_count = 0
    detected_targets = np.zeros(env.n_targets, dtype=bool)
    for target_idx, target in enumerate(env.targets):
        for agent in env.agents:
            dist = np.sqrt((agent["x"] - target.x) ** 2 + (agent["y"] - target.y) ** 2)
            if dist <= Config.SENSOR_RANGE:
                detected_targets[target_idx] = True
                detected_count += 1
                break
    return detected_targets, detected_count


def _coverage_cells(env, agent):
    covered_indices = set()
    m_center, n_center = env._pos_to_grid(agent["x"], agent["y"])
    radius_grid = int(Config.SENSOR_RANGE / Config.GRID_WIDTH)
    for r in range(-radius_grid, radius_grid + 1):
        for c in range(-radius_grid, radius_grid + 1):
            if r ** 2 + c ** 2 <= radius_grid ** 2:
                gm, gn = m_center + r, n_center + c
                if 0 <= gm < Config.GRID_ROWS and 0 <= gn < Config.GRID_COLS:
                    covered_indices.add((gm, gn))
    return covered_indices


def compute_overlap_rate(env):
    coverage_per_agent = []
    for agent in env.agents:
        coverage_per_agent.append(_coverage_cells(env, agent))

    union_cells = set().union(*coverage_per_agent) if coverage_per_agent else set()
    if not union_cells:
        return 0.0

    cell_counts = {}
    for cells in coverage_per_agent:
        for cell in cells:
            cell_counts[cell] = cell_counts.get(cell, 0) + 1

    overlap_cells = sum(1 for count in cell_counts.values() if count > 1)
    return overlap_cells / len(union_cells)


def compute_collision_count(env):
    collisions = 0
    for i in range(env.n_agents):
        for j in range(i + 1, env.n_agents):
            dist = np.sqrt((env.agents[i]["x"] - env.agents[j]["x"]) ** 2 +
                           (env.agents[i]["y"] - env.agents[j]["y"]) ** 2)
            if dist < Config.SAFE_DIST_UAV:
                collisions += 1

    for agent in env.agents:
        for ox, oy, radius in env.obstacles:
            dist = np.sqrt((agent["x"] - ox) ** 2 + (agent["y"] - oy) ** 2) - radius
            if dist <= 0:
                collisions += 1
    return collisions


def evaluate_policy(env, policy, eval_seeds, max_steps=200, coverage_steps=100):
    min_all_detect_steps = []
    total_detection_counts = []
    overlap_rates = []
    collision_counts = []
    coverage_efficiencies = []

    for seed in eval_seeds:
        obs_n, _ = env.reset(seed=seed)
        if hasattr(policy, "reset"):
            policy.reset()
        target_seen_once = np.zeros(env.n_targets, dtype=bool)
        min_all_detect_step = max_steps + 1
        total_detect_this_episode = 0
        overlap_accum = []
        collision_accum = 0
        coverage_step_value = None

        for step in range(max_steps):
            actions = policy.select_actions(obs_n)
            obs_n, _, terminated, truncated, _ = env.step(actions)
            del terminated, truncated

            detected_targets, detected_count = step_detection_stats(env)
            target_seen_once = np.logical_or(target_seen_once, detected_targets)
            total_detect_this_episode += detected_count

            overlap_accum.append(compute_overlap_rate(env))
            collision_accum += compute_collision_count(env)

            if step + 1 == coverage_steps:
                coverage_cells = int(np.sum(env.global_map_cover))
                coverage_efficiency = coverage_cells / (Config.GRID_ROWS * Config.GRID_COLS)
                coverage_step_value = coverage_efficiency

            if np.all(target_seen_once) and min_all_detect_step == max_steps + 1:
                min_all_detect_step = step + 1

        if coverage_step_value is None:
            coverage_cells = int(np.sum(env.global_map_cover))
            coverage_step_value = coverage_cells / (Config.GRID_ROWS * Config.GRID_COLS)

        min_all_detect_steps.append(min_all_detect_step)
        total_detection_counts.append(total_detect_this_episode)
        overlap_rates.append(float(np.mean(overlap_accum)) if overlap_accum else 0.0)
        collision_counts.append(collision_accum)
        coverage_efficiencies.append(coverage_step_value)

    return {
        "eval_seeds": [int(seed) for seed in eval_seeds],
        "min_all_detect_steps": min_all_detect_steps,
        "total_detection_counts": total_detection_counts,
        "overlap_rates": overlap_rates,
        "collision_counts": collision_counts,
        "coverage_efficiencies": coverage_efficiencies,
    }
