import argparse
import json
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from baselines import GreedyAPFController, LawnmowerController
from eval_seeds import get_eval_seeds
from iddpg import IDDPG
from maddpg import MADDPG
from metrics import evaluate_policy
from training import train_iddpg, train_maddpg
from uav_env import UAVSwarmEnv

matplotlib.use("Agg")


class EnvControllerPolicy:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, _obs_n):
        return self.controller.select_actions()

    def reset(self):
        if hasattr(self.controller, "reset"):
            self.controller.reset()


class MADDPGPolicy:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, obs_n):
        return self.controller.select_actions(obs_n, noise_std=0.0)

    def reset(self):
        return None


class IDDPGPolicy:
    def __init__(self, controller):
        self.controller = controller

    def select_actions(self, obs_n):
        return self.controller.select_actions(obs_n, noise_std=0.0)

    def reset(self):
        return None


def _ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def _avg(values):
    return float(np.mean(values)) if values else 0.0


def _save_json(path, payload):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def plot_bar(values_map, title, ylabel, out_path):
    labels = list(values_map.keys())
    values = [values_map[key] for key in labels]
    plt.figure(figsize=(9, 4))
    plt.bar(labels, values)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_boxplot(data_map, title, ylabel, out_path):
    labels = list(data_map.keys())
    data = [data_map[key] for key in labels]
    plt.figure(figsize=(9, 4))
    plt.boxplot(data, labels=labels, showmeans=True)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.xticks(rotation=20)
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_overlap_curve(histories, out_path):
    plt.figure(figsize=(9, 4))
    for label, history in histories.items():
        episodes = [item["episode"] for item in history]
        overlap = [item["avg_overlap_rate"] for item in history]
        plt.plot(episodes, overlap, label=label)
    plt.xlabel("Episode")
    plt.ylabel("Overlap Rate")
    plt.title("Overlap Rate vs Episode")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-uav", type=int, required=True)
    parser.add_argument("--n-target", type=int)
    parser.add_argument("--target-list", type=str, default="1,2,3,4")
    parser.add_argument("--max-episodes", type=int, default=5000)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-interval", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="./compare_results")
    parser.add_argument("--skip-train", action="store_true")
    args = parser.parse_args()

    if args.n_target is not None and args.target_list:
        target_counts = [args.n_target]
    else:
        target_counts = [int(item) for item in args.target_list.split(",") if item.strip()]
        if not target_counts:
            raise ValueError("target list is empty")

    output_dir = os.path.abspath(args.output_dir)
    _ensure_dir(output_dir)

    for target_count in target_counts:
        target_dir = os.path.join(output_dir, f"target_{target_count}")
        model_dir = os.path.join(target_dir, "models")
        result_dir = os.path.join(target_dir, "results")
        plot_dir = os.path.join(target_dir, "plots")
        _ensure_dir(model_dir)
        _ensure_dir(result_dir)
        _ensure_dir(plot_dir)

        training_histories = {}
        if not args.skip_train:
            maddpg_dir = os.path.join(model_dir, "maddpg_pf")
            _, maddpg_hist = train_maddpg(
                maddpg_dir,
                args.n_uav,
                target_count,
                use_pf=True,
                use_pf_obs=True,
                max_episodes=args.max_episodes,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
            )
            training_histories["MADDPG"] = maddpg_hist

            ablation_dir = os.path.join(model_dir, "maddpg_no_pf")
            _, ablation_hist = train_maddpg(
                ablation_dir,
                args.n_uav,
                target_count,
                use_pf=False,
                use_pf_obs=False,
                max_episodes=args.max_episodes,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
            )
            training_histories["MADDPG-NoPF"] = ablation_hist

            iddpg_dir = os.path.join(model_dir, "iddpg")
            _, iddpg_hist = train_iddpg(
                iddpg_dir,
                args.n_uav,
                target_count,
                max_episodes=args.max_episodes,
                max_steps=args.max_steps,
                eval_interval=args.eval_interval,
            )
            training_histories["IDDPG"] = iddpg_hist

        eval_seeds = get_eval_seeds()
        metrics_all = {}

        env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True)
        controller = LawnmowerController(env)
        metrics_all["Lawnmower"] = evaluate_policy(
            env,
            EnvControllerPolicy(controller),
            eval_seeds,
            max_steps=args.max_steps,
        )

        env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True)
        controller = GreedyAPFController(env)
        metrics_all["Greedy-APF"] = evaluate_policy(
            env,
            EnvControllerPolicy(controller),
            eval_seeds,
            max_steps=args.max_steps,
        )

        env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        global_obs_dim = obs_dim * env.n_agents
        maddpg = MADDPG(env.n_agents, obs_dim, act_dim, global_obs_dim)
        maddpg.load_models(os.path.join(model_dir, "maddpg_pf", "models", "best"))
        metrics_all["MADDPG"] = evaluate_policy(
            env,
            MADDPGPolicy(maddpg),
            eval_seeds,
            max_steps=args.max_steps,
        )

        env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=False, use_pf_obs=False)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        global_obs_dim = obs_dim * env.n_agents
        maddpg_no_pf = MADDPG(env.n_agents, obs_dim, act_dim, global_obs_dim)
        maddpg_no_pf.load_models(os.path.join(model_dir, "maddpg_no_pf", "models", "best"))
        metrics_all["MADDPG-NoPF"] = evaluate_policy(
            env,
            MADDPGPolicy(maddpg_no_pf),
            eval_seeds,
            max_steps=args.max_steps,
        )

        env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True)
        obs_dim = env.observation_space.shape[0]
        act_dim = env.action_space.shape[0]
        iddpg = IDDPG(env.n_agents, obs_dim, act_dim)
        iddpg.load(os.path.join(model_dir, "iddpg", "models", "best"))
        metrics_all["IDDPG"] = evaluate_policy(
            env,
            IDDPGPolicy(iddpg),
            eval_seeds,
            max_steps=args.max_steps,
        )

        _save_json(os.path.join(result_dir, "metrics_raw.json"), metrics_all)

        summary = {}
        for name, metrics in metrics_all.items():
            summary[name] = {
                "min_all_detect_step": _avg(metrics["min_all_detect_steps"]),
                "total_detection_count": _avg(metrics["total_detection_counts"]),
                "overlap_rate": _avg(metrics["overlap_rates"]),
                "collision_count": _avg(metrics["collision_counts"]),
                "coverage_efficiency": _avg(metrics["coverage_efficiencies"]),
            }
        _save_json(os.path.join(result_dir, "metrics_summary.json"), summary)

        plot_bar(
            {k: v["min_all_detect_step"] for k, v in summary.items()},
            "Min All-Detect Step (lower is better)",
            "Steps",
            os.path.join(plot_dir, "min_all_detect_step.png"),
        )

        plot_boxplot(
            {k: v["total_detection_counts"] for k, v in metrics_all.items()},
            "Total Detection Count",
            "Count",
            os.path.join(plot_dir, "total_detection_boxplot.png"),
        )

        plot_bar(
            {k: v["collision_count"] for k, v in summary.items()},
            "Collision Count (lower is better)",
            "Count",
            os.path.join(plot_dir, "collision_count.png"),
        )

        plot_bar(
            {k: v["coverage_efficiency"] for k, v in summary.items()},
            "Coverage Efficiency (within 100 steps)",
            "Coverage %",
            os.path.join(plot_dir, "coverage_efficiency.png"),
        )

        if training_histories:
            plot_overlap_curve(
                training_histories,
                os.path.join(plot_dir, "overlap_rate_curve.png"),
            )


if __name__ == "__main__":
    main()
