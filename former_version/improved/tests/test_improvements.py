import pytest


def test_environment_ready():
    assert True


def _require_numpy_torch():
    np = pytest.importorskip("numpy")
    torch = pytest.importorskip("torch")
    return np, torch


def _import_modules():
    import improved.maddpg_v2 as maddpg
    import improved.targetpre_v2 as targetpre
    return maddpg, targetpre


def test_target_predictor_obstacle_indexing():
    np, _ = _require_numpy_torch()
    _, targetpre = _import_modules()
    map_size = (10, 12)
    obstacles = [(1.2, 2.7), (3.9, 4.1), (9, 11)]
    predictor = targetpre.TargetPredictor(
        map_size=map_size,
        obstacles_map=obstacles,
        v_range=(1, 2),
        theta_range=(0.0, 1.0),
        init_pos=(5.0, 5.0),
        num_particles=100,
    )

    assert predictor.map.shape == map_size
    assert predictor.map[1, 2] == -1
    assert predictor.map[3, 4] == -1
    assert predictor.map[9, 11] == -1


def test_local_entropy_radius_gate():
    np, _ = _require_numpy_torch()
    _, targetpre = _import_modules()
    map_size = (20, 20)
    predictor = targetpre.TargetPredictor(
        map_size=map_size,
        obstacles_map=[],
        v_range=(1, 2),
        theta_range=(0.0, 1.0),
        init_pos=(10.0, 10.0),
        num_particles=50,
    )

    predictor.particles[:, 0:2] = np.array([10.0, 10.0])
    predictor.weights = np.ones(predictor.num_particles) / predictor.num_particles
    assert predictor.get_local_entropy(center=np.array([10.0, 10.0]), radius=1.0) > 0.0
    assert predictor.get_local_entropy(center=np.array([10.0, 10.0]), radius=0.0) == 0.0


def test_observation_length_with_global_belief():
    _require_numpy_torch()
    maddpg, targetpre = _import_modules()
    map_size = (100, 100)
    uav = maddpg.UAVAgent(
        uav_id=1,
        initial_pos=[10.0, 10.0],
        initial_v=30.0,
        initial_phi=0.0,
        step=1.0,
        total_state_dim=17,
        total_action_dim=2,
    )

    predictor = targetpre.TargetPredictor(
        map_size=map_size,
        obstacles_map=[],
        v_range=(1, 2),
        theta_range=(0.0, 1.0),
        init_pos=(50.0, 50.0),
        num_particles=10,
    )

    obs = uav.get_observation(map_size, None, [uav], [predictor])
    assert obs.shape[0] == 17


def test_actor_update_uses_other_agent_actor():
    _, torch = _require_numpy_torch()
    maddpg, _ = _import_modules()
    uav_a = maddpg.UAVAgent(
        uav_id=1,
        initial_pos=[10.0, 10.0],
        initial_v=30.0,
        initial_phi=0.0,
        step=1.0,
        total_state_dim=34,
        total_action_dim=4,
    )
    uav_b = maddpg.UAVAgent(
        uav_id=2,
        initial_pos=[20.0, 20.0],
        initial_v=30.0,
        initial_phi=0.0,
        step=1.0,
        total_state_dim=34,
        total_action_dim=4,
    )

    buffer = maddpg.MultiAgentReplayBuffer(
        max_size=10,
        num_agents=2,
        state_dims=[17, 17],
        action_dims=[2, 2],
    )

    obs = [np.zeros(17), np.ones(17)]
    act = [np.zeros(2), np.ones(2)]
    rew = [0.0, 0.0]
    next_obs = [np.zeros(17), np.ones(17)]
    done = [0.0, 0.0]
    for _ in range(6):
        buffer.add(obs, act, rew, next_obs, done)

    before = [p.detach().clone() for p in uav_a.brain.actor.parameters()]
    maddpg.train_centralized([uav_a, uav_b], buffer, batch_size=4)
    after = [p.detach().clone() for p in uav_a.brain.actor.parameters()]

    assert any(not torch.equal(b, a) for b, a in zip(before, after))
