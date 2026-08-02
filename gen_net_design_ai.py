"""
gen_net_design_ai.py
--------------------
Core RL engine for NetGen.

Algorithm: REINFORCE (Monte-Carlo Policy Gradient) with:
  - Per-episode EMA baseline for variance reduction
  - Entropy bonus for exploration
  - Correct logp / reward alignment (terminal reward stored separately)
"""

from __future__ import annotations
import json
import os
import random
import webbrowser
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from pyvis.network import Network


# ---------------------------------------------------------------------------
# Node helpers
# ---------------------------------------------------------------------------

def make_nodes(devices: List[Tuple[str, int]]) -> List[Tuple[int, str]]:
    """Expand a (type, count) list into a flat list of (node_id, type)."""
    nodes: List[Tuple[int, str]] = []
    nid = 0
    for t, c in devices:
        for _ in range(c):
            nodes.append((nid, t))
            nid += 1
    return nodes


# ---------------------------------------------------------------------------
# Hardware Access/Trunk/WAN Cost Model
# ---------------------------------------------------------------------------

EDGE_COST_TABLE = {
    ("Server", "Switch"): 1.0,      # Standard gigabit local access link
    ("EndDevice", "Switch"): 1.0,   # Standard workstation Ethernet drop
    ("Switch", "Router"): 1.5,      # Medium-distance uplink
    ("Switch", "Switch"): 2.5,      # Backbone trunk link, fiber
    ("Router", "Router"): 3.5,      # Wide area network inter-router transit
}

def get_edge_cost(u_type: str, v_type: str) -> float:
    """Lookup symmetric edge cost from the cost table, penalizing non-standard links."""
    if (u_type, v_type) in EDGE_COST_TABLE:
        return EDGE_COST_TABLE[(u_type, v_type)]
    if (v_type, u_type) in EDGE_COST_TABLE:
        return EDGE_COST_TABLE[(v_type, u_type)]
    return 6.0  # Heavy penalty for direct EndDevice-Server, Server-Router, etc.


# ---------------------------------------------------------------------------
# Constraints
# ---------------------------------------------------------------------------

@dataclass
class Constraints:
    must_be_tree: bool = False
    must_be_connected: bool = True
    servers_connect_only_to_switches: bool = False

    @staticmethod
    def from_strings(constraints: List[str]) -> "Constraints":
        c = Constraints()
        for s in constraints:
            s_low = s.strip().lower()
            if "tree" in s_low:
                c.must_be_tree = True
            if "connected" in s_low:
                c.must_be_connected = True
            if "server" in s_low and "switch" in s_low:
                c.servers_connect_only_to_switches = True
        if c.must_be_tree:
            c.must_be_connected = True
        return c


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------

class GraphEnv:
    def __init__(
        self,
        devices: List[Tuple[str, int]],
        constraints: Constraints,
        max_steps: Optional[int] = None,
        lambda_cost: float = 0.0,
        lambda_resilience: float = 0.0,
    ):
        self.nodes = make_nodes(devices)
        self.type_of: Dict[int, str] = {i: t for i, t in self.nodes}
        self.n = len(self.nodes)
        self.G = nx.Graph()
        self.G.add_nodes_from([i for i, _ in self.nodes])
        self.constraints = constraints
        self.steps = 0
        self.max_steps = max_steps or (self.n * 2)
        self.done = False
        self.lambda_cost = lambda_cost
        self.lambda_resilience = lambda_resilience

    def _edge_respects_type_rules(self, u: int, v: int) -> bool:
        if not self.constraints.servers_connect_only_to_switches:
            return True
        tu, tv = self.type_of[u], self.type_of[v]
        if tu == "Server" and tv != "Switch":
            return False
        if tv == "Server" and tu != "Switch":
            return False
        return True

    def _ensure_server_switch_connections(self) -> None:
        servers = [i for i, t in self.nodes if t == "Server"]
        switches = [i for i, t in self.nodes if t == "Switch"]
        for server in servers:
            for switch in switches:
                if not self.G.has_edge(server, switch):
                    self.G.add_edge(server, switch)

    def _edge_keeps_tree_property(self, u: int, v: int) -> bool:
        if not self.constraints.must_be_tree:
            return True
        return not nx.has_path(self.G, u, v)

    def valid_actions(self) -> List[Tuple[int, int]]:
        opts = []
        for u in range(self.n):
            for v in range(u + 1, self.n):
                if self.G.has_edge(u, v):
                    continue
                if not self._edge_respects_type_rules(u, v):
                    continue
                if not self._edge_keeps_tree_property(u, v):
                    continue
                opts.append((u, v))
        return opts

    def step(self, action: Tuple[int, int]):
        if self.done:
            raise RuntimeError("Episode already done")
        u, v = action
        self.G.add_edge(u, v)
        self.steps += 1
        reward, terminal, info = self._evaluate()
        self.done = terminal
        return self._state(), reward, terminal, info

    def _state(self) -> torch.Tensor:
        degs = {i: self.G.degree(i) for i in self.G.nodes}
        type_map = {"Server": 0, "Router": 1, "Switch": 2, "EndDevice": 3}
        features = []
        for i in range(self.n):
            tvec = [0, 0, 0, 0]
            tvec[type_map.get(self.type_of[i], 1)] = 1
            features.append([degs[i]] + tvec)
        return torch.tensor(features, dtype=torch.float32)

    def _evaluate(self) -> Tuple[float, bool, Dict]:
        """
        Multi-objective Reward breakdown:
          connectivity reward:
            +0.05 / n_components  – per-step connectivity shaping bonus
            +1.0                  – tree completion (must_be_tree)
            -0.5                  – cycle detected when tree required
            +0.7                  – connected graph (must_be_connected)
            -0.2                  – timeout penalty when still disconnected
          cost reward penalty:
            - lambda_cost * normalized_edge_cost
          resilience reward bonus (based on edge redundancy):
            + lambda_resilience * resilience_proxy
        """
        n_components = nx.number_connected_components(self.G)
        reward = 0.05 / n_components  # favour fewer components at every step

        terminal = False
        info: Dict = {}

        if self.constraints.must_be_tree:
            if nx.is_forest(self.G):
                if self.G.number_of_edges() == self.n - 1 and nx.is_connected(self.G):
                    reward += 1.0
                    terminal = True
                    info["status"] = "success_tree"
            else:
                reward -= 0.5  # cycle penalty

        elif self.constraints.must_be_connected:
            if nx.is_connected(self.G) and self.G.number_of_edges() >= self.n - 1:
                reward += 0.7
                terminal = True
                info["status"] = "success_connected"

        if self.steps >= self.max_steps:
            terminal = True
            info["status"] = info.get("status", "max_steps")
            if self.constraints.must_be_connected and not nx.is_connected(self.G):
                reward -= 0.2  # timeout while still disconnected

        # Apply multi-objective parameters: Cost and Resilience Proxy
        if self.n > 1:
            # 1. Normalized Cost Penalty
            total_cost = 0.0
            for u, v in self.G.edges():
                total_cost += get_edge_cost(self.type_of[u], self.type_of[v])
            max_cost = (self.n - 1) * 6.0
            normalized_cost = total_cost / max_cost
            reward -= self.lambda_cost * normalized_cost

            # 2. Resilience Edge Redundancy Proxy
            redundant_edges = max(0, self.G.number_of_edges() - (self.n - 1))
            resilience_proxy = min(redundant_edges, 2) / 2.0
            reward += self.lambda_resilience * resilience_proxy

        return reward, terminal, info

    def reset(self) -> torch.Tensor:
        self.G = nx.Graph()
        self.G.add_nodes_from([i for i, _ in self.nodes])
        if self.constraints.servers_connect_only_to_switches:
            self._ensure_server_switch_connections()
        self.steps = 0
        self.done = False
        return self._state()


# ---------------------------------------------------------------------------
# Policy network
# ---------------------------------------------------------------------------

class EdgePolicy(nn.Module):
    """
    Edge-scoring policy.

    Architecture: each node is embedded by a shared MLP, then candidate
    edge (u,v) is scored by a second MLP over [h_u, h_v, |h_u - h_v|].
    The absolute difference makes the scoring symmetric and sensitive to
    type/degree asymmetry between endpoints.
    """

    def __init__(self, node_feat_dim: int = 5, hidden: int = 64):
        super().__init__()
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(
        self,
        node_feats: torch.Tensor,
        pairs: List[Tuple[int, int]],
    ) -> torch.Tensor:
        H = self.node_mlp(node_feats)
        e_feats = []
        for u, v in pairs:
            hu, hv = H[u], H[v]
            e_feats.append(torch.cat([hu, hv, torch.abs(hu - hv)], dim=-1))
        E = torch.stack(e_feats) if e_feats else torch.zeros((0, H.shape[1] * 3))
        return self.edge_mlp(E).squeeze(-1)


# ---------------------------------------------------------------------------
# Episode data container
# ---------------------------------------------------------------------------

@dataclass
class Episode:
    logps: List[torch.Tensor]
    rewards: List[float]
    entropies: List[torch.Tensor]
    terminal_r: float = 0.0
    """
    Reward from the terminal state where no action was taken.
    Stored separately so len(logps) == len(rewards) always holds.
    The terminal reward still propagates into earlier returns via discounting
    inside train_policy().
    """


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def select_action(
    policy: EdgePolicy,
    state: torch.Tensor,
    valid_edges: List[Tuple[int, int]],
    training: bool = True,
) -> Tuple[Optional[Tuple[int, int]], Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Sample or greedily pick an edge from the policy distribution over valid_edges.

    Returns
    -------
    action  : chosen (u, v) pair, or None if no valid edges
    logp    : log-probability of the chosen action
    entropy : policy entropy H(π) over valid actions (used for exploration bonus)
    """
    if not valid_edges:
        return None, None, None

    scores = policy(state, valid_edges)
    probs = torch.softmax(scores, dim=0)
    m = torch.distributions.Categorical(probs)
    
    if training:
        idx = m.sample()
    else:
        idx = torch.argmax(probs)
        
    logp = m.log_prob(idx)
    entropy = m.entropy()
    action = valid_edges[int(idx.item())]
    return action, logp, entropy


# ---------------------------------------------------------------------------
# Episode rollout
# ---------------------------------------------------------------------------

def run_episode(
    env: GraphEnv,
    policy: EdgePolicy,
    gamma: float = 0.99,
    training: bool = True,
) -> Tuple[Episode, nx.Graph, Dict]:
    """
    Collect one full episode.

    Invariant: len(episode.logps) == len(episode.rewards) == len(episode.entropies)

    When no valid actions remain (graph fully constrained), the environment is
    evaluated once to get a terminal reward.  That reward goes into
    ``Episode.terminal_r`` — NOT into ``Episode.rewards`` — so no gradient is
    computed for a step that was never taken.  The terminal reward still
    influences the discounted returns of earlier steps inside train_policy().
    """
    state = env.reset()
    ep_logps: List[torch.Tensor] = []
    ep_rewards: List[float] = []
    ep_entropies: List[torch.Tensor] = []
    terminal_r = 0.0
    info_final: Dict = {}

    while True:
        valid_edges = env.valid_actions()

        if not valid_edges:
            r, _, info = env._evaluate()
            terminal_r = r
            info_final = info
            break

        action, logp, entropy = select_action(policy, state, valid_edges, training=training)
        if action is None:
            break

        state, reward, done, info = env.step(action)
        ep_logps.append(logp)
        ep_rewards.append(reward)
        ep_entropies.append(entropy)

        if done:
            info_final = info
            break

    episode = Episode(
        logps=ep_logps,
        rewards=ep_rewards,
        entropies=ep_entropies,
        terminal_r=terminal_r,
    )
    return episode, env.G.copy(), info_final


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_policy(
    devices: List[Tuple[str, int]],
    constraints: Constraints,
    episodes: int = 200,
    lr: float = 1e-3,
    gamma: float = 0.99,
    entropy_coef: float = 0.01,
    baseline_alpha: float = 0.1,
    lambda_cost: float = 0.0,
    lambda_resilience: float = 0.0,
    variable_n: bool = False,
) -> Tuple["EdgePolicy", nx.Graph, float, Dict]:
    """
    REINFORCE with EMA baseline and entropy bonus.

    Parameters
    ----------
    episodes      : number of training episodes
    lr            : Adam learning rate
    gamma         : discount factor for returns
    entropy_coef  : weight of the entropy bonus (encourages exploration)
    baseline_alpha: EMA coefficient for the running baseline
                    (higher -> baseline adapts faster)
    lambda_cost   : weight penalty of the edge deployment cost
    lambda_resilience: weight bonus of the redundancy resilience proxy
    variable_n    : whether to randomly sample network size per episode

    Returns
    -------
    policy       : trained EdgePolicy
    best_graph   : graph with highest cumulative reward seen during training
    best_reward  : cumulative reward of best_graph
    history      : {'episode_rewards': [...], 'success': [...]} for plotting
    """
    env = GraphEnv(
        devices,
        constraints,
        lambda_cost=lambda_cost,
        lambda_resilience=lambda_resilience,
    )
    policy = EdgePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    if constraints.servers_connect_only_to_switches:
        env._ensure_server_switch_connections()

    best_graph: Optional[nx.Graph] = None
    best_reward = float("-inf")
    history: Dict[str, List] = {"episode_rewards": [], "success": []}

    # Running EMA baseline: tracks expected episode return across episodes
    baseline = 0.0

    for ep in range(episodes):
        if variable_n:
            # Sample random network configuration
            n_nodes = random.randint(8, 14)
            n_servers = random.randint(1, 2)
            n_switches = random.randint(2, 3)
            n_routers = random.randint(1, 2)
            n_end = max(2, n_nodes - (n_servers + n_switches + n_routers))
            ep_devices = [("Server", n_servers), ("Switch", n_switches), ("Router", n_routers), ("EndDevice", n_end)]
            env = GraphEnv(
                ep_devices,
                constraints,
                lambda_cost=lambda_cost,
                lambda_resilience=lambda_resilience,
            )
            if constraints.servers_connect_only_to_switches:
                env._ensure_server_switch_connections()

        episode, graph, info = run_episode(env, policy, gamma)

        # ----------------------------------------------------------------
        # Compute discounted returns.
        # Start from terminal_r so it propagates backwards via discounting.
        # ----------------------------------------------------------------
        returns: List[float] = []
        G = episode.terminal_r
        for r in reversed(episode.rewards):
            G = r + gamma * G
            returns.insert(0, G)

        total_reward = sum(episode.rewards) + episode.terminal_r

        # ----------------------------------------------------------------
        # Update EMA baseline (cross-episode variance reduction).
        # ----------------------------------------------------------------
        baseline = baseline_alpha * total_reward + (1.0 - baseline_alpha) * baseline

        # ----------------------------------------------------------------
        # Policy gradient update.
        # ----------------------------------------------------------------
        if episode.logps:
            returns_t = torch.tensor(returns, dtype=torch.float32)

            # Subtract baseline: reduces variance without introducing bias
            returns_t = returns_t - baseline

            # Within-episode scale normalisation for gradient stability
            if len(returns_t) > 1:
                returns_t = returns_t / (returns_t.std() + 1e-8)

            policy_loss = [
                -logp * ret
                for logp, ret in zip(episode.logps, returns_t)
            ]

            # Entropy bonus: maximising H(π) keeps the policy exploratory
            entropy_bonus = -entropy_coef * torch.stack(episode.entropies).mean()

            loss = torch.stack(policy_loss).sum() + entropy_bonus
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ----------------------------------------------------------------
        # Bookkeeping
        # ----------------------------------------------------------------
        history["episode_rewards"].append(total_reward)
        history["success"].append(info.get("status", "").startswith("success"))

        if total_reward > best_reward:
            best_reward = total_reward
            best_graph = graph

        if ep % 20 == 0:
            recent_sr = sum(history["success"][-20:]) / min(20, ep + 1)
            print(
                f"  ep={ep:4d}  reward={total_reward:7.3f}  "
                f"baseline={baseline:7.3f}  "
                f"sr(last20)={recent_sr:.0%}  "
                f"status={info.get('status', '?')}"
            )

    return policy, best_graph, best_reward, history  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Random baseline
# ---------------------------------------------------------------------------

def run_random_baseline(
    devices: List[Tuple[str, int]],
    constraints: Constraints,
    episodes: int = 100,
) -> Dict:
    """
    Run a uniformly-random policy (random choice from valid_actions).
    Returns the same history-dict format as train_policy() for easy comparison.
    """
    env = GraphEnv(devices, constraints)
    rewards: List[float] = []
    successes: List[bool] = []

    for _ in range(episodes):
        env.reset()
        ep_reward = 0.0
        info_final: Dict = {}

        while True:
            valid_edges = env.valid_actions()
            if not valid_edges:
                _, r, _, info = env._evaluate()
                ep_reward += r
                info_final = info
                break
            action = random.choice(valid_edges)
            _, reward, done, info = env.step(action)
            ep_reward += reward
            if done:
                info_final = info
                break

        rewards.append(ep_reward)
        successes.append(info_final.get("status", "").startswith("success"))

    return {"episode_rewards": rewards, "success": successes}


# ---------------------------------------------------------------------------
# Training-curve plotting
# ---------------------------------------------------------------------------

def plot_training_curves(
    rl_history: Dict,
    random_history: Optional[Dict] = None,
    filename: str = "training_curve.png",
) -> Optional[str]:
    """
    Save a two-panel training figure:
      left  – episode reward (smoothed) vs. random baseline mean
      right – rolling success rate vs. random baseline rate

    Requires matplotlib; prints a warning and returns None if not installed.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed — skipping training curve. "
              "Run: pip install matplotlib")
        return None

    rl_rewards = rl_history["episode_rewards"]
    rl_success = rl_history["success"]
    n_ep = len(rl_rewards)
    window = max(5, n_ep // 10)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # ── Reward panel ──────────────────────────────────────────────────────
    ax1.plot(rl_rewards, alpha=0.25, color="steelblue", linewidth=0.8,
             label="_nolegend_")
    if n_ep >= window:
        smoothed = np.convolve(rl_rewards, np.ones(window) / window, mode="valid")
        ax1.plot(range(window - 1, n_ep), smoothed,
                 color="steelblue", linewidth=2, label=f"RL (w={window} avg)")
    else:
        ax1.plot(rl_rewards, color="steelblue", linewidth=2, label="RL")

    if random_history:
        r_mean = float(np.mean(random_history["episode_rewards"]))
        ax1.axhline(r_mean, color="firebrick", linestyle="--", linewidth=1.5,
                    label=f"Random baseline  μ={r_mean:.3f}")

    ax1.set_title("Episode Reward", fontsize=13)
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Cumulative Reward")
    ax1.legend()
    ax1.grid(alpha=0.3)

    # ── Success-rate panel ────────────────────────────────────────────────
    if n_ep >= window:
        rl_sr = np.convolve(
            [int(s) for s in rl_success],
            np.ones(window) / window, mode="valid"
        )
        ax2.plot(range(window - 1, n_ep), rl_sr,
                 color="steelblue", linewidth=2, label="RL success rate")
    else:
        ax2.plot([int(s) for s in rl_success],
                 color="steelblue", linewidth=2, label="RL success rate")

    if random_history:
        r_sr = float(np.mean(random_history["success"]))
        ax2.axhline(r_sr, color="firebrick", linestyle="--", linewidth=1.5,
                    label=f"Random baseline  {r_sr:.1%}")

    ax2.set_title("Success Rate (Rolling Window)", fontsize=13)
    ax2.set_ylim(-0.05, 1.1)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Success Rate")
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Training curve saved -> {filename}")
    return filename


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_policy(policy: EdgePolicy, filepath: str) -> None:
    """Persist policy weights to disk."""
    dirpath = os.path.dirname(filepath)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)
    torch.save(policy.state_dict(), filepath)
    print(f"Policy saved -> {filepath}")


def load_policy(
    filepath: str,
    node_feat_dim: int = 5,
    hidden: int = 64,
) -> EdgePolicy:
    """Load policy weights from disk. Returns an eval-mode EdgePolicy."""
    policy = EdgePolicy(node_feat_dim=node_feat_dim, hidden=hidden)
    policy.load_state_dict(torch.load(filepath, weights_only=True))
    policy.eval()
    return policy


# ---------------------------------------------------------------------------
# Rule-based topology generators
# ---------------------------------------------------------------------------

def generate_topology(
    devices: List[Tuple[str, int]],
    topology_type: str,
) -> Tuple[nx.Graph, Dict[int, str]]:
    """Deterministic graph constructors for six classical topology types."""
    nodes = make_nodes(devices)
    node_types = {i: t for i, t in nodes}
    G = nx.Graph()
    G.add_nodes_from([i for i, _ in nodes])

    all_nodes = [i for i, _ in nodes]
    n = len(all_nodes)

    if topology_type == "star":
        center = all_nodes[0]
        for node in all_nodes[1:]:
            G.add_edge(center, node)

    elif topology_type == "ring":
        for i in range(n):
            G.add_edge(all_nodes[i], all_nodes[(i + 1) % n])

    elif topology_type == "mesh":
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(all_nodes[i], all_nodes[j])

    elif topology_type == "tree":
        servers = [i for i, t in node_types.items() if t == "Server"]
        switches = [i for i, t in node_types.items() if t == "Switch"]
        routers = [i for i, t in node_types.items() if t == "Router"]
        end_devices_list = [i for i, t in node_types.items() if t == "EndDevice"]

        for switch in switches:
            for server in servers:
                G.add_edge(server, switch)

        lower_devices = routers + end_devices_list
        for i, switch in enumerate(switches):
            if i < len(lower_devices):
                G.add_edge(switch, lower_devices[i])

        remaining: List[int] = []
        if len(servers) > len(switches):
            remaining.extend(servers[len(switches):])
        if len(switches) > len(lower_devices):
            remaining.extend(switches[len(lower_devices):])
        if len(lower_devices) > len(switches):
            remaining.extend(lower_devices[len(switches):])

        for device in remaining:
            if all_nodes:
                G.add_edge(device, all_nodes[0])

    elif topology_type == "bus":
        for i in range(n - 1):
            G.add_edge(all_nodes[i], all_nodes[i + 1])

    elif topology_type == "hybrid":
        servers = [i for i, t in node_types.items() if t == "Server"]
        switches = [i for i, t in node_types.items() if t == "Switch"]
        routers = [i for i, t in node_types.items() if t == "Router"]
        end_devices_list = [i for i, t in node_types.items() if t == "EndDevice"]

        if n >= 4:
            hub_size = min(3, n // 2)
            center = all_nodes[0]
            for i in range(1, hub_size + 1):
                if i < n:
                    G.add_edge(center, all_nodes[i])
            remaining_nodes = all_nodes[hub_size + 1:]
            if len(remaining_nodes) >= 2:
                for i in range(len(remaining_nodes)):
                    G.add_edge(remaining_nodes[i], remaining_nodes[(i + 1) % len(remaining_nodes)])
                if remaining_nodes:
                    G.add_edge(all_nodes[1], remaining_nodes[0])
        else:
            center = all_nodes[0]
            for node in all_nodes[1:]:
                G.add_edge(center, node)

        lower_devices = routers + end_devices_list
        for i, switch in enumerate(switches):
            if lower_devices and i < len(lower_devices):
                G.add_edge(switch, lower_devices[i])

    return G, node_types


def calculate_topology_success_rates(
    servers: int,
    switches: int,
    routers: int,
    end_devices: int,
) -> Dict[str, int]:
    """
    Compute a suitability score (0–100) for each topology type based on
    **measured** structural properties of the actually-generated graph for
    this device configuration.  Scores are NOT heuristic — every point
    is derived from a real networkx metric.

    Scoring breakdown:
      50 pts  –  graph is connected (primary correctness criterion)
      30 pts  –  edge density in a "useful" range (neither too sparse nor mesh)
      20 pts  –  low average shortest-path length (good reachability)
    """
    total = servers + switches + routers + end_devices
    if total < 2:
        return {t: 0 for t in ["star", "ring", "mesh", "tree", "bus", "hybrid"]}

    devices: List[Tuple[str, int]] = []
    if servers > 0:
        devices.append(("Server", servers))
    if switches > 0:
        devices.append(("Switch", switches))
    if routers > 0:
        devices.append(("Router", routers))
    if end_devices > 0:
        devices.append(("EndDevice", end_devices))

    scores: Dict[str, int] = {}
    for topo in ["star", "ring", "mesh", "tree", "bus", "hybrid"]:
        try:
            G, _ = generate_topology(devices, topo)
            score = 0

            if nx.is_connected(G):
                score += 50

                # Density: penalise both extremes (too sparse ↔ full mesh)
                density = nx.density(G)
                ideal_lo = 0.15 if total > 6 else 0.30
                ideal_hi = 0.50 if total > 6 else 0.80
                if ideal_lo <= density <= ideal_hi:
                    score += 30
                else:
                    deviation = min(
                        abs(density - ideal_lo),
                        abs(density - ideal_hi),
                    )
                    score += max(0, 30 - int(deviation * 60))

                # Reachability: lower average path length → higher score
                avg_path = nx.average_shortest_path_length(G)
                max_path = float(total - 1)  # linear chain (bus) has the worst avg path
                score += max(0, 20 - int((avg_path / max_path) * 20))

            scores[topo] = min(100, max(0, score))
        except Exception:
            scores[topo] = 0

    return scores


# ---------------------------------------------------------------------------
# Visualisation (standalone mode only — web app uses vis.js via app.py)
# ---------------------------------------------------------------------------

def visualize_graph(
    graph: nx.Graph,
    node_types: Dict[int, str],
    filename: str = "network.html",
) -> str:
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")

    images = {
        "Server": "assets/server.png",
        "Router": "assets/router.png",
        "Switch": "assets/switch.png",
        "EndDevice": "assets/desktop.png",
    }
    hierarchy_levels = {"Server": -400, "Switch": -200, "Router": 0, "EndDevice": 200}

    nodes_by_type: Dict[str, List] = {
        "Server": [], "Switch": [], "Router": [], "EndDevice": []
    }
    for node in graph.nodes():
        ntype = node_types.get(node, "Unknown")
        if ntype in nodes_by_type:
            nodes_by_type[ntype].append(node)

    for node in graph.nodes():
        ntype = node_types.get(node, "Unknown")
        img = images.get(ntype)
        y_pos = hierarchy_levels.get(ntype, 0)
        type_nodes = nodes_by_type.get(ntype, [node])
        x_spacing = 150
        x_offset = (len(type_nodes) - 1) * x_spacing / 2
        x_pos = (type_nodes.index(node) * x_spacing) - x_offset

        if img:
            net.add_node(node, label=f"{ntype}\n{node}", image=img,
                         shape="image", size=40, x=x_pos, y=y_pos, physics=False)
        else:
            net.add_node(node, label=f"{ntype}\n{node}", color="#808080",
                         size=30, x=x_pos, y=y_pos, physics=False)

    for u, v in graph.edges():
        net.add_edge(u, v, width=2)

    net.set_options("""
    var options = {
      "physics": { "enabled": false },
      "layout": { "hierarchical": { "enabled": false } }
    }
    """)
    net.save_graph(filename)
    print(f"Network visualization saved -> {filename}")
    return filename


def simulate_failures(graph: nx.Graph, n_trials: int = 100) -> Tuple[float, float]:
    """
    Simulate failures on the graph and return (mean_survival_rate, mean_path_degradation).
    
    For each trial, randomly decide to remove one edge (link failure) or one node (device failure).
    If the graph remains connected, survival = 1.0, and path degradation is calculated as:
      (new_avg_path - old_avg_path) / old_avg_path
    If the graph becomes disconnected, survival = 0.0, and path degradation is 1.0 (max penalty).
    """
    if graph.number_of_nodes() <= 1:
        return 1.0, 0.0

    # Calculate initial average shortest path length (if connected)
    if nx.is_connected(graph):
        initial_path_len = nx.average_shortest_path_length(graph)
    else:
        initial_path_len = float(graph.number_of_nodes() - 1)

    survival_results = []
    degradation_results = []

    for _ in range(n_trials):
        # 50% link failure, 50% node failure
        trial_type = random.choice(["link", "node"])
        
        # Make a copy
        G_temp = graph.copy()
        
        if trial_type == "link" and G_temp.number_of_edges() > 0:
            edge = random.choice(list(G_temp.edges()))
            G_temp.remove_edge(*edge)
        elif trial_type == "node" and G_temp.number_of_nodes() > 1:
            node = random.choice(list(G_temp.nodes()))
            G_temp.remove_node(node)
            
        # Evaluate connectivity
        if nx.is_connected(G_temp):
            survival = 1.0
            new_path_len = nx.average_shortest_path_length(G_temp)
            if initial_path_len > 0:
                degradation = max(0.0, (new_path_len - initial_path_len) / initial_path_len)
            else:
                degradation = 0.0
        else:
            survival = 0.0
            degradation = 1.0 # 100% path length increase penalty
            
        survival_results.append(survival)
        degradation_results.append(degradation)
        
    mean_survival = sum(survival_results) / len(survival_results) if survival_results else 1.0
    mean_degradation = sum(degradation_results) / len(degradation_results) if degradation_results else 0.0
    return mean_survival, mean_degradation


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    devices = [("Server", 3), ("Switch", 2), ("Router", 2), ("EndDevice", 2)]
    constraints = Constraints.from_strings(["connected"])

    # ── Random baseline ───────────────────────────────────────────────────
    print("=" * 60)
    print("Running random baseline (100 episodes)...")
    random_history = run_random_baseline(devices, constraints, episodes=100)
    r_mean = sum(random_history["episode_rewards"]) / len(random_history["episode_rewards"])
    r_sr = sum(random_history["success"]) / len(random_history["success"])
    print(f"  mean_reward={r_mean:.3f}  success_rate={r_sr:.1%}")

    # ── RL training ───────────────────────────────────────────────────────
    print("=" * 60)
    print("Training RL policy (200 episodes)...")
    policy, best_graph, best_reward, history = train_policy(
        devices, constraints, episodes=200
    )

    rl_sr = sum(history["success"]) / len(history["success"])
    print("=" * 60)
    print(f"RL Agent — best_reward={best_reward:.3f}  success_rate={rl_sr:.1%}")
    print(
        f"Best graph: {best_graph.number_of_nodes()} nodes, "
        f"{best_graph.number_of_edges()} edges, "
        f"connected={nx.is_connected(best_graph)}"
    )

    # ── Save checkpoint ───────────────────────────────────────────────────
    save_policy(policy, os.path.join("checkpoints", "policy_standalone.pt"))

    # ── Training curves ───────────────────────────────────────────────────
    plot_training_curves(history, random_history, "training_curve.png")

    # ── Visualise ─────────────────────────────────────────────────────────
    nodes = make_nodes(devices)
    node_types = {i: t for i, t in nodes}
    html_file = visualize_graph(best_graph, node_types)

    # ── Export JSON ───────────────────────────────────────────────────────
    network_data = {
        "nodes": [{"id": i, "type": t} for i, t in nodes],
        "edges": list(best_graph.edges()),
    }
    with open("best_network.json", "w") as f:
        json.dump(network_data, f, indent=2)
    print("Network data saved -> best_network.json")

    html_path = os.path.abspath(html_file)
    print(f"Opening visualization in browser...")
    webbrowser.open(f"file:///{html_path}")