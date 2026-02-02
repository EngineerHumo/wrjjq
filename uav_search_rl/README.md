# UAV Search RL

基于论文《空地无人集群路径强化学习规划方法研究》第三章的分层架构实现多无人机搜索不确定目标的强化学习训练工程，包含：

- 目标预测层：改进粒子滤波（先验初始化、CT传播、正向似然/负向信息更新、Sigmoid软约束、归一化与重采样、概率栅格化）。
- 策略学习层：MADDPG（CTDE）多智能体强化学习，Actor分布式执行，Critic集中训练。

## 安装

```bash
pip install -r requirements.txt
```

如果只想快速运行，可直接安装基础依赖（示例）：

```bash
pip install numpy torch pyyaml tensorboard
```

## 训练

```bash
python train.py --config configs/default.yaml
```

支持断点恢复：

```bash
python train.py --config configs/default.yaml --resume checkpoints/last
```

## 评估

```bash
python evaluate.py --ckpt checkpoints/best --episodes 20
```

## 输出指标

- 概率分布熵变化（entropy drop）
- 覆盖率/探索率
- 重叠率、检测率、首次发现时间
- 碰撞率、平滑性、能耗

## 目录结构

```
uav_search_rl/
  configs/default.yaml
  envs/
    uav_search_env.py
    dynamics.py
    target_ct.py
    particle_filter.py
    features.py
  marl/
    maddpg.py
    networks.py
    replay_buffer.py
    noise.py
  eval/evaluator.py
  utils/
    seed.py
    logger.py
    checkpoint.py
  train.py
  evaluate.py
  README.md
```
