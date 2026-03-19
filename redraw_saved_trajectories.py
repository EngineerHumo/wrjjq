import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")

REPO_ROOT = Path(__file__).resolve().parent
STANDARD_DIRS = [f"newnet_6_{idx}" for idx in range(3, 9)]
COMPARE_DIRS = [f"newnet_6_{idx}_compare" for idx in range(3, 9)]
DEFAULT_NETWORK_DIRS = STANDARD_DIRS + COMPARE_DIRS
DEFAULT_OUTPUT_DIR = REPO_ROOT / "redrawn_trajectories"
MAX_STEPS = 200


def parse_n_uav_from_network_name(network_name: str) -> int:
    parts = network_name.split("_")
    if len(parts) < 3:
        raise ValueError(f"无法从目录名解析无人机数量: {network_name}")
    try:
        return int(parts[2])
    except ValueError as exc:
        raise ValueError(f"无法从目录名解析无人机数量: {network_name}") from exc


@dataclass
class ModelEntry:
    network_dir: str
    family: str
    model_name: str
    n_uav: int
    n_target: int
    label: str
    model_path: Path
    use_pf: bool = True
    use_pf_obs: bool = True


class ImportContext:
    def __init__(self, module_dir: Path):
        self.module_dir = module_dir
        self._saved = None

    def __enter__(self):
        self._saved = list(sys.path)
        sys.path.insert(0, str(self.module_dir))
        return self

    def __exit__(self, exc_type, exc, tb):
        sys.path[:] = self._saved


def load_module(module_dir: Path, file_name: str, unique_name: str):
    module_path = module_dir / file_name
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    with ImportContext(module_dir):
        spec.loader.exec_module(module)
    return module


def get_eval_seeds(module_dir: Path) -> List[int]:
    module = load_module(module_dir, "eval_seeds.py", f"eval_seeds_{module_dir.name}")
    return [int(seed) for seed in module.get_eval_seeds()]


def resolve_existing_path(raw_path: str, network_dir: Path) -> Optional[Path]:
    candidates = []
    raw = Path(raw_path)
    candidates.append(raw)
    if raw_path.startswith("/home/wensheng/gjq_workspace/wrjjq/"):
        candidates.append(REPO_ROOT / raw_path.split("/home/wensheng/gjq_workspace/wrjjq/", 1)[1])
    candidates.append(network_dir / raw_path)
    candidates.append(network_dir / raw.name)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def make_output_name(model: ModelEntry, seed_idx: int, seed: int) -> str:
    return (
        f"{model.network_dir}__{model.model_name}__uav{model.n_uav}__"
        f"target{model.n_target}__seedidx{seed_idx:03d}__seed{seed}.png"
    )


def plot_trajectories(map_size, obstacles, uav_trajectories, target_trajectories, detection_points, out_path: Path, title: str):
    plt.figure(figsize=(8, 8))
    ax = plt.gca()
    boundary = plt.Rectangle((0, 0), map_size, map_size, fill=False, edgecolor="black", linestyle="--", linewidth=1.5)
    ax.add_patch(boundary)

    for idx, (ox, oy, radius) in enumerate(obstacles or []):
        circle = plt.Circle((ox, oy), radius, color="black", fill=False, linewidth=1.2, label="Obstacle" if idx == 0 else None)
        ax.add_patch(circle)

    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", "tab:pink", "tab:gray"]
    for idx, traj in enumerate(uav_trajectories):
        arr = np.array(traj)
        if arr.size == 0:
            continue
        color = colors[idx % len(colors)]
        plt.plot(arr[:, 0], arr[:, 1], color=color, label=f"UAV {idx + 1}")
        plt.scatter(arr[0, 0], arr[0, 1], color=color, s=20)

    for idx, traj in enumerate(target_trajectories):
        arr = np.array(traj)
        if arr.size == 0:
            continue
        plt.plot(arr[:, 0], arr[:, 1], linestyle="--", linewidth=1.8, marker="o", markersize=2.5, label=f"Target {idx + 1}")

    if detection_points:
        det = np.array(detection_points)
        plt.scatter(det[:, 0], det[:, 1], s=18, c="red", label="Detection")

    margin = 50.0
    plt.xlim(-margin, map_size + margin)
    plt.ylim(-margin, map_size + margin)
    plt.title(title)
    plt.legend(loc="upper right", fontsize=8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path)
    plt.close()


def run_until_all_targets_detected(env, select_actions: Callable, seed: int, max_steps: int):
    obs_n, _ = env.reset(seed=seed)
    target_seen_once = np.zeros(env.n_targets, dtype=bool)
    stop_step = max_steps
    for step in range(max_steps):
        actions = select_actions(obs_n)
        next_obs_n, _, terminated, truncated, _ = env.step(actions)
        detected_by = getattr(env, "target_detected_by", None) or []
        detected_mask = np.array([agent_idx is not None for agent_idx in detected_by], dtype=bool)
        if detected_mask.size:
            target_seen_once[: detected_mask.size] = np.logical_or(target_seen_once[: detected_mask.size], detected_mask)
        obs_n = next_obs_n
        if np.all(target_seen_once):
            stop_step = step + 1
            break
        if any(terminated) or any(truncated):
            stop_step = step + 1
            break
    return stop_step


def collect_standard_models(network_dir: Path) -> List[ModelEntry]:
    entries: List[ModelEntry] = []
    for summary_path in sorted(network_dir.glob("models_2k/target_*/top_models/top_models_summary.json")):
        n_target = int(summary_path.parent.parent.name.split("_")[-1])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for rec in summary:
            rank = int(rec["rank"])
            episode = int(rec["episode"])
            step_str = f"{float(rec['avg_min_all_detect_step']):.3f}"
            model_path = summary_path.parent / f"rank_{rank:02d}_ep_{episode}_step_{step_str}" / "weights"
            if not model_path.exists():
                continue
            entries.append(ModelEntry(network_dir.name, "standard", f"top_rank{rank:02d}_ep{episode}", parse_n_uav_from_network_name(network_dir.name), n_target, network_dir.name, model_path))
    return entries


def collect_compare_models(network_dir: Path) -> List[ModelEntry]:
    entries: List[ModelEntry] = []

    for summary_path in sorted(network_dir.glob("compare_results_100/maddpg_our_method/models/target_*/top_models/top_models_summary.json")):
        n_target = int(summary_path.parent.parent.name.split("_")[-1])
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        for rec in summary:
            rank = int(rec["rank"])
            episode = int(rec["episode"])
            step_str = f"{float(rec['avg_min_all_detect_step']):.3f}"
            model_path = summary_path.parent / f"rank_{rank:02d}_ep_{episode}_step_{step_str}" / "weights"
            if not model_path.exists():
                continue
            entries.append(ModelEntry(network_dir.name, "compare", f"maddpg_pf_rank{rank:02d}_ep{episode}", parse_n_uav_from_network_name(network_dir.name), n_target, network_dir.name, model_path, True, True))

    baseline_configs = [
        ("maddpg_nopf", False, False, "top_models.json", "episode_4000.json"),
        ("iddpg", True, True, "top_models.json", "episode_4000.json"),
    ]
    for algo_name, use_pf, use_pf_obs, top_name, last_name in baseline_configs:
        pattern = f"compare_results_100/{algo_name}/target_*/models/*/models"
        for models_dir in sorted(network_dir.glob(pattern)):
            n_target = int(models_dir.parents[2].name.split("_")[-1])
            top_path = models_dir / top_name
            if top_path.exists():
                payload = json.loads(top_path.read_text(encoding="utf-8"))
                for idx, rec in enumerate(payload.get("top_models", []), start=1):
                    resolved = resolve_existing_path(rec.get("path", ""), network_dir)
                    if resolved is None:
                        continue
                    episode = int(rec.get("episode", -1))
                    entries.append(ModelEntry(network_dir.name, "compare", f"{algo_name}_top{idx:02d}_ep{episode}", parse_n_uav_from_network_name(network_dir.name), n_target, network_dir.name, resolved, use_pf, use_pf_obs))
            last_path = models_dir / last_name
            if last_path.exists():
                payload = json.loads(last_path.read_text(encoding="utf-8"))
                resolved = resolve_existing_path(payload.get("path", ""), network_dir)
                if resolved is not None:
                    entries.append(ModelEntry(network_dir.name, "compare", f"{algo_name}_episode4000", parse_n_uav_from_network_name(network_dir.name), n_target, network_dir.name, resolved, use_pf, use_pf_obs))
    return entries


def build_standard_policy(network_dir: Path, model: ModelEntry):
    algo_mod = load_module(network_dir, "algo.py", f"algo_{network_dir.name}")
    env_mod = load_module(network_dir, "uav_env.py", f"uav_env_{network_dir.name}")
    env = env_mod.UAVSwarmEnv(n_uav=model.n_uav, n_target=model.n_target)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]
    maddpg = algo_mod.MADDPG(env.n_agents, obs_dim, act_dim, obs_dim * env.n_agents)
    maddpg.load_models(str(model.model_path))
    return env, lambda obs_n: maddpg.select_actions(obs_n, noise_std=0.0), env_mod.Config.MAP_SIZE


def build_compare_policy(network_dir: Path, model: ModelEntry):
    env_mod = load_module(network_dir, "uav_env.py", f"uav_env_{network_dir.name}")
    env = env_mod.UAVSwarmEnv(n_uav=model.n_uav, n_target=model.n_target, use_pf=model.use_pf, use_pf_obs=model.use_pf_obs)
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    if model.model_name.startswith("iddpg"):
        algo_mod = load_module(network_dir, "iddpg.py", f"iddpg_{network_dir.name}")
        policy = algo_mod.IDDPG(env.n_agents, obs_dim, act_dim)
        policy.load(str(model.model_path))
        return env, lambda obs_n: policy.select_actions(obs_n, noise_std=0.0), env_mod.Config.MAP_SIZE

    algo_mod = load_module(network_dir, "maddpg.py", f"maddpg_{network_dir.name}")
    maddpg = algo_mod.MADDPG(env.n_agents, obs_dim, act_dim, obs_dim * env.n_agents)
    maddpg.load_models(str(model.model_path))
    return env, lambda obs_n: maddpg.select_actions(obs_n, noise_std=0.0), env_mod.Config.MAP_SIZE


def collect_models(network_name: str) -> List[ModelEntry]:
    network_dir = REPO_ROOT / network_name
    if not network_dir.exists():
        return []
    if network_name.endswith("_compare"):
        return collect_compare_models(network_dir)
    return collect_standard_models(network_dir)


def build_runner(model: ModelEntry):
    network_dir = REPO_ROOT / model.network_dir
    if model.family == "standard":
        return build_standard_policy(network_dir, model)
    return build_compare_policy(network_dir, model)


def redraw_network(network_name: str, output_root: Path, max_steps: int) -> Dict[str, int]:
    network_dir = REPO_ROOT / network_name
    seeds = get_eval_seeds(network_dir)
    models = collect_models(network_name)
    saved = 0
    skipped = 0

    for model in models:
        try:
            env, select_actions, map_size = build_runner(model)
        except Exception as exc:
            skipped += len(seeds)
            print(f"[WARN] skip model {model.network_dir}/{model.model_name}: load failed: {exc}")
            continue

        for seed_idx, seed in enumerate(seeds, start=1):
            try:
                stop_step = run_until_all_targets_detected(env, select_actions, seed, max_steps)
                title = f"{model.network_dir} | {model.model_name} | UAV={model.n_uav} | Target={model.n_target} | Seed#{seed_idx}={seed} | stop={stop_step}"
                out_dir = output_root / network_name / f"uav_{model.n_uav}" / f"target_{model.n_target}" / model.model_name
                out_path = out_dir / make_output_name(model, seed_idx, seed)
                plot_trajectories(
                    map_size,
                    env.obstacles,
                    env.uav_trajectories,
                    env.target_trajectories,
                    env.detection_points,
                    out_path,
                    title,
                )
                saved += 1
            except Exception as exc:
                skipped += 1
                print(f"[WARN] failed seed {seed} for {model.network_dir}/{model.model_name}: {exc}")
    return {"models": len(models), "saved": saved, "skipped": skipped, "seeds": len(seeds)}


def parse_args():
    parser = argparse.ArgumentParser(description="按固定种子重绘仓库内已保存网络的轨迹图。")
    parser.add_argument("--networks", nargs="*", default=DEFAULT_NETWORK_DIRS, help="要处理的网络目录名列表")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="轨迹图输出目录")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="单回合最大步数")
    return parser.parse_args()


def main():
    args = parse_args()
    summary = {}
    for network_name in args.networks:
        print(f"[INFO] processing {network_name}")
        summary[network_name] = redraw_network(network_name, args.output_dir, args.max_steps)
    print("\n[SUMMARY]")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
