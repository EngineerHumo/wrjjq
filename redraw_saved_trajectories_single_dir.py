import argparse
import json
from pathlib import Path

from redraw_saved_trajectories import DEFAULT_OUTPUT_DIR, MAX_STEPS, REPO_ROOT, dry_run_network, redraw_network


def resolve_network_name(network_arg: str) -> str:
    candidate = Path(network_arg)
    if candidate.is_absolute():
        try:
            candidate = candidate.relative_to(REPO_ROOT)
        except ValueError as exc:
            raise ValueError(f"路径 {network_arg} 不在仓库根目录 {REPO_ROOT} 下") from exc

    network_name = candidate.as_posix().strip("/")
    if not network_name:
        raise ValueError("network path is empty")

    network_dir = REPO_ROOT / network_name
    if not network_dir.is_dir():
        raise FileNotFoundError(f"目录不存在: {network_dir}")
    return network_name


def parse_args():
    parser = argparse.ArgumentParser(description="只处理单个网络目录下全部已保存模型的轨迹图重绘。")
    parser.add_argument("network_dir", help="单个网络目录，例如 newnet_6_6 或 newnet_6_6_compare")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="轨迹图输出根目录")
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS, help="单回合最大步数")
    parser.add_argument("--dry-run", action="store_true", help="只打印发现到的模型路径与元信息，不执行回放和绘图")
    return parser.parse_args()


def main():
    args = parse_args()
    network_name = resolve_network_name(args.network_dir)
    print(f"[INFO] processing single network: {network_name}")
    summary = dry_run_network(network_name) if args.dry_run else redraw_network(network_name, args.output_dir, args.max_steps)
    print("\n[SUMMARY]")
    print(json.dumps({network_name: summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
