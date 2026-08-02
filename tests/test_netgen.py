"""
tests/test_netgen.py — Smoke tests for NetGen

Run:
    pytest tests/test_netgen.py -v

Coverage goals:
  - GraphEnv correctness (reset, valid_actions, step, _evaluate)
  - Topology generator correctness for all six types
  - select_action return contract
  - run_episode invariant: len(logps) == len(rewards) (the T1-B fix)
  - calculate_topology_success_rates produces bounded real values
  - run_random_baseline output shape
"""

import os
import sys

# Ensure project root is importable regardless of where pytest is invoked
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pytest
import networkx as nx
import torch

from gen_net_design_ai import (
    Constraints,
    EdgePolicy,
    GraphEnv,
    calculate_topology_success_rates,
    generate_topology,
    make_nodes,
    run_episode,
    run_random_baseline,
    select_action,
    train_policy,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

SIMPLE_DEVICES = [("Server", 2), ("Switch", 1)]
SIMPLE_CON = Constraints(must_be_connected=True)


# ---------------------------------------------------------------------------
# GraphEnv
# ---------------------------------------------------------------------------

class TestGraphEnv:
    def test_reset_gives_clean_graph(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        env.reset()
        assert env.G.number_of_edges() == 0
        assert env.steps == 0
        assert not env.done

    def test_state_shape(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        state = env.reset()
        # 3 nodes (2 Server + 1 Switch), 5 features (degree + 4-dim one-hot type)
        assert state.shape == (3, 5)

    def test_valid_actions_initial_count(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        env.reset()
        # C(3, 2) = 3 possible edges
        assert len(env.valid_actions()) == 3

    def test_step_increments_counter(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        env.reset()
        action = env.valid_actions()[0]
        env.step(action)
        assert env.steps == 1

    def test_evaluate_success_connected(self):
        """Connected graph under must_be_connected → success_connected + terminal."""
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        env.reset()
        # Manually wire a connected graph (n=3, need ≥ 2 edges)
        env.G.add_edge(0, 1)
        env.G.add_edge(0, 2)
        env.steps = 2  # within max_steps=6

        reward, terminal, info = env._evaluate()
        assert nx.is_connected(env.G)
        assert terminal
        assert info.get("status") == "success_connected"

    def test_evaluate_cycle_penalty_in_tree_mode(self):
        """A cycle when must_be_tree is set should yield a negative reward."""
        env = GraphEnv(SIMPLE_DEVICES, Constraints(must_be_tree=True))
        env.reset()
        env.G.add_edge(0, 1)
        env.G.add_edge(1, 2)
        env.G.add_edge(0, 2)  # creates a cycle
        reward, _, _ = env._evaluate()
        assert reward < 0

    def test_step_raises_after_done(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        env.reset()
        env.done = True
        with pytest.raises(RuntimeError, match="already done"):
            env.step((0, 1))


# ---------------------------------------------------------------------------
# generate_topology
# ---------------------------------------------------------------------------

class TestGenerateTopology:
    def test_star_connected_and_edge_count(self):
        devices = [("Server", 1), ("Switch", 2), ("EndDevice", 2)]
        G, _ = generate_topology(devices, "star")
        assert nx.is_connected(G)
        assert G.number_of_edges() == 4  # n - 1 = 5 - 1

    def test_ring_connected_and_edge_count(self):
        devices = [("Router", 4)]
        G, _ = generate_topology(devices, "ring")
        assert nx.is_connected(G)
        assert G.number_of_edges() == 4  # n edges in a ring

    def test_mesh_complete_graph(self):
        devices = [("Switch", 3)]
        G, _ = generate_topology(devices, "mesh")
        assert nx.is_connected(G)
        assert G.number_of_edges() == 3  # K_3

    def test_bus_path_graph(self):
        devices = [("EndDevice", 4)]
        G, _ = generate_topology(devices, "bus")
        assert nx.is_connected(G)
        assert G.number_of_edges() == 3  # n - 1

    def test_node_type_mapping_correct(self):
        devices = [("Server", 2), ("Switch", 1)]
        G, node_types = generate_topology(devices, "star")
        assert G.number_of_nodes() == 3
        assert sum(1 for t in node_types.values() if t == "Server") == 2
        assert sum(1 for t in node_types.values() if t == "Switch") == 1

    def test_all_topologies_produce_correct_node_count(self):
        devices = [("Server", 2), ("Switch", 2), ("Router", 1)]
        for topo in ["star", "ring", "mesh", "tree", "bus", "hybrid"]:
            G, _ = generate_topology(devices, topo)
            assert G.number_of_nodes() == 5, f"{topo}: expected 5 nodes"


# ---------------------------------------------------------------------------
# select_action
# ---------------------------------------------------------------------------

class TestSelectAction:
    def test_returns_action_in_valid_edges(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        state = env.reset()
        policy = EdgePolicy()
        valid_edges = env.valid_actions()
        action, logp, entropy = select_action(policy, state, valid_edges)
        assert action in valid_edges
        assert logp is not None
        assert entropy is not None

    def test_returns_none_tuple_on_empty_edges(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        state = env.reset()
        policy = EdgePolicy()
        action, logp, entropy = select_action(policy, state, [])
        assert action is None
        assert logp is None
        assert entropy is None

    def test_logp_is_scalar_tensor(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        state = env.reset()
        policy = EdgePolicy()
        _, logp, entropy = select_action(policy, state, env.valid_actions())
        assert logp.shape == torch.Size([])
        assert entropy.shape == torch.Size([])


# ---------------------------------------------------------------------------
# run_episode invariants (T1-B regression test)
# ---------------------------------------------------------------------------

class TestRunEpisode:
    def test_logps_rewards_entropies_same_length(self):
        """
        Core correctness invariant after T1-B fix:
        logps, rewards, and entropies must always have equal length.
        The terminal reward is stored in episode.terminal_r, NOT in rewards.
        """
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        policy = EdgePolicy()
        for trial in range(10):
            episode, _, _ = run_episode(env, policy)
            n_lp = len(episode.logps)
            n_r = len(episode.rewards)
            n_e = len(episode.entropies)
            assert n_lp == n_r, (
                f"Trial {trial}: logps ({n_lp}) != rewards ({n_r})"
            )
            assert n_lp == n_e, (
                f"Trial {trial}: logps ({n_lp}) != entropies ({n_e})"
            )

    def test_terminal_r_is_float(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        policy = EdgePolicy()
        episode, _, _ = run_episode(env, policy)
        assert isinstance(episode.terminal_r, float)

    def test_episode_returns_graph(self):
        env = GraphEnv(SIMPLE_DEVICES, SIMPLE_CON)
        policy = EdgePolicy()
        _, graph, info = run_episode(env, policy)
        assert isinstance(graph, nx.Graph)
        assert graph.number_of_nodes() == 3


# ---------------------------------------------------------------------------
# calculate_topology_success_rates (T1-E: real metrics)
# ---------------------------------------------------------------------------

class TestCalculateTopologySuccessRates:
    def test_scores_in_bounds(self):
        scores = calculate_topology_success_rates(2, 1, 1, 2)
        for topo, score in scores.items():
            assert 0 <= score <= 100, f"{topo}: {score} out of [0, 100]"

    def test_all_six_topologies_present(self):
        scores = calculate_topology_success_rates(2, 1, 1, 2)
        expected = {"star", "ring", "mesh", "tree", "bus", "hybrid"}
        assert set(scores.keys()) == expected

    def test_single_device_returns_all_zeros(self):
        """With only 1 device, no topology can be connected → all zeros."""
        scores = calculate_topology_success_rates(1, 0, 0, 0)
        for score in scores.values():
            assert score == 0

    def test_mesh_scores_high_for_small_network(self):
        """Mesh is fully connected — should always score ≥ 50 (connected bonus)."""
        scores = calculate_topology_success_rates(0, 0, 3, 0)
        assert scores["mesh"] >= 50

    def test_scores_are_integers(self):
        scores = calculate_topology_success_rates(2, 2, 0, 2)
        for score in scores.values():
            assert isinstance(score, int)


# ---------------------------------------------------------------------------
# run_random_baseline
# ---------------------------------------------------------------------------

class TestRunRandomBaseline:
    def test_output_keys_and_lengths(self):
        result = run_random_baseline(SIMPLE_DEVICES, SIMPLE_CON, episodes=5)
        assert "episode_rewards" in result
        assert "success" in result
        assert len(result["episode_rewards"]) == 5
        assert len(result["success"]) == 5

    def test_rewards_are_floats(self):
        result = run_random_baseline(SIMPLE_DEVICES, SIMPLE_CON, episodes=3)
        for r in result["episode_rewards"]:
            assert isinstance(r, float)

    def test_success_are_bools(self):
        result = run_random_baseline(SIMPLE_DEVICES, SIMPLE_CON, episodes=3)
        for s in result["success"]:
            assert isinstance(s, bool)


# ---------------------------------------------------------------------------
# train_policy with variable_n (Task 5 generalization test)
# ---------------------------------------------------------------------------

class TestVariableNTraining:
    def test_train_policy_with_variable_n(self):
        """
        Confirm that train_policy runs successfully when variable_n=True.
        """
        policy, best_graph, best_reward, history = train_policy(
            SIMPLE_DEVICES,
            SIMPLE_CON,
            episodes=5,
            variable_n=True
        )
        assert policy is not None
        assert isinstance(best_reward, float)
        assert len(history["episode_rewards"]) == 5
