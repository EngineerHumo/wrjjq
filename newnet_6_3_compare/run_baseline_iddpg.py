import argparse
import os

from baseline_utils import (
    IDDPGPolicy,
    ensure_dir,
    load_saved_models,
    sample_eval_seeds,
    save_metrics,
    save_trajectory_plots,
)
from eval_seeds import get_eval_seeds
from iddpg import IDDPG
from metrics import evaluate_policy
from training import train_iddpg
from uav_env import UAVSwarmEnv


def _collect_model_entries(top_models, episode_4000_path):
    entries = []
    top_episodes = set()
    for item in top_models:
        path = item.get("path")
        episode = item.get("episode")
        if path:
            entries.append({"tag": f"top_ep_{episode}", "path": path, "episode": episode})
            top_episodes.add(episode)

    if episode_4000_path and 4000 not in top_episodes:
        entries.append({"tag": "episode_4000", "path": episode_4000_path, "episode": 4000})
    return entries


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-uav", type=int, required=True)
    parser.add_argument("--target-list", type=str, default="1,2,3")
    parser.add_argument("--max-episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./compare_results/iddpg")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    target_counts = [int(item) for item in args.target_list.split(",") if item.strip()]
    if not target_counts:
        raise ValueError("target list is empty")

    output_dir = os.path.abspath(args.output_dir)
    ensure_dir(output_dir)

    eval_seeds = get_eval_seeds()
    sampled_seeds = sample_eval_seeds(eval_seeds, sample_size=10)

    for target_count in target_counts:
        target_dir = os.path.join(output_dir, f"target_{target_count}")
        model_dir = os.path.join(target_dir, "models", "iddpg")
        result_dir = os.path.join(target_dir, "results")
        traj_root = os.path.join(target_dir, "trajectories")
        ensure_dir(model_dir)
        ensure_dir(result_dir)
        ensure_dir(traj_root)

        if not args.skip_train:
            saved_models, _ = train_iddpg(
                model_dir,
                args.n_uav,
                target_count,
                max_episodes=args.max_episodes,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
            )
            top_models = saved_models["top_models"]
            episode_4000_path = saved_models["episode_4000"]
        else:
            top_models, episode_4000_path = load_saved_models(model_dir)

        entries = _collect_model_entries(top_models, episode_4000_path)
        if not entries:
            raise ValueError("No saved IDDPG models found for evaluation.")

        metrics_map = {}
        for entry in entries:
            env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True)
            obs_dim = env.observation_space.shape[0]
            act_dim = env.action_space.shape[0]
            controller = IDDPG(env.n_agents, obs_dim, act_dim)
            controller.load(entry["path"])
            metrics_map[entry["tag"]] = evaluate_policy(
                env,
                IDDPGPolicy(controller),
                eval_seeds,
                max_steps=args.max_steps,
            )

            traj_dir = os.path.join(traj_root, entry["tag"])
            save_trajectory_plots(
                lambda: UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True),
                lambda _env, controller=controller: IDDPGPolicy(controller),
                sampled_seeds,
                args.max_steps,
                traj_dir,
            )

        save_metrics(result_dir, metrics_map)


if __name__ == "__main__":
    main()
