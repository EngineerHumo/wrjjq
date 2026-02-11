import argparse
import os

from baselines import GreedyAPFController
from baseline_utils import (
    EnvControllerPolicy,
    ensure_dir,
    sample_eval_seeds,
    save_metrics,
    save_trajectory_plots,
)
from eval_seeds import get_eval_seeds
from metrics import evaluate_policy
from uav_env import UAVSwarmEnv


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-uav", type=int, required=True)
    parser.add_argument("--target-list", type=str, default="1,2,3,4")
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--output-dir", type=str, default="./compare_results/greedy_apf")
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
        result_dir = os.path.join(target_dir, "results")
        traj_dir = os.path.join(target_dir, "trajectories", "policy")
        ensure_dir(result_dir)

        env = UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True)
        controller = GreedyAPFController(env)
        metrics = evaluate_policy(
            env,
            EnvControllerPolicy(controller),
            eval_seeds,
            max_steps=args.max_steps,
        )
        save_metrics(result_dir, {"Greedy-APF": metrics})

        save_trajectory_plots(
            lambda: UAVSwarmEnv(n_uav=args.n_uav, n_target=target_count, use_pf=True, use_pf_obs=True),
            lambda env: EnvControllerPolicy(GreedyAPFController(env)),
            sampled_seeds,
            args.max_steps,
            traj_dir,
        )


if __name__ == "__main__":
    main()
