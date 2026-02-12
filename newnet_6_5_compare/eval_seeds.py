"""固定评估种子集，用于跨模型公平对比。"""

import random

EVAL_SEED_GENERATOR = 20250201
EVAL_SEED_COUNT = 2000
_rng = random.Random(EVAL_SEED_GENERATOR)
EVAL_SEEDS = _rng.sample(range(10_000_000), EVAL_SEED_COUNT)


def get_eval_seeds():
    return EVAL_SEEDS.copy()
