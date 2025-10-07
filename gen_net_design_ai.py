"""
Generative AI for Automated Network Design — Minimal RL Prototype

Overview
- Environment: builds a network by adding one edge at a time under constraints
- Policy: small PyTorch policy (REINFORCE) over valid edge choices
- Constraints supported (extendable):
    * is_tree
    * is_connected
    * server→switch only (servers cannot connect directly to routers/servers)
- Devices: list of (type, count), e.g., [("Server",5),("Router",2),("Switch",3)]
- Output: best graphs as NetworkX and JSON (node types preserved)

Notes
- This is a compact teaching prototype: good for proving the loop end‑to‑end.
- Reward shaping is intentionally simple; tune as needed.
- You can drop this file into a venv with: pip install torch networkx

Author: you + ChatGPT
"""
from __future__ import annotations
import json
import math
import random
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import networkx as nx
import torch
import torch.nn as nn
import torch.optim as optim
from pyvis.network import Network

# -----------------------------
# Utility: Device inventory → typed nodes
# -----------------------------

def make_nodes(devices: List[Tuple[str, int]]) -> List[Tuple[int, str]]:
    """Return list of (node_id, type)."""
    nodes = []
    nid = 0
    for t, c in devices:
        for _ in range(c):
            nodes.append((nid, t))
            nid += 1
    return nodes

# -----------------------------
# Constraint helpers
# -----------------------------

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
        # if tree is required, connected is implied
        if c.must_be_tree:
            c.must_be_connected = True
        return c

# -----------------------------
# Environment: add-edge MDP
# -----------------------------

class GraphEnv:
    def __init__(self, devices: List[Tuple[str, int]], constraints: Constraints, max_steps: Optional[int] = None):
        self.nodes = make_nodes(devices)  # [(id, type)]
        self.type_of = {i: t for i, t in self.nodes}
        self.n = len(self.nodes)
        self.G = nx.Graph()
        self.G.add_nodes_from([i for i, _ in self.nodes])
        self.constraints = constraints
        self.steps = 0
        self.max_steps = max_steps or (self.n * 2)  # heuristic
        self.done = False

    # ---- Constraint checks per edge ----
    def _edge_respects_type_rules(self, u: int, v: int) -> bool:
        if not self.constraints.servers_connect_only_to_switches:
            return True
        tu, tv = self.type_of[u], self.type_of[v]
        # servers can connect only to switches. So any edge with a server must have the other endpoint a switch.
        if tu == "Server" and tv != "Switch":
            return False
        if tv == "Server" and tu != "Switch":
            return False
        return True

    def _edge_keeps_tree_property(self, u: int, v: int) -> bool:
        if not self.constraints.must_be_tree:
            return True
        # For tree, adding an edge must not create a cycle
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

    def _state(self):
        # Simple state: degree + one-hot type for each node (aggregated pairwise later by policy)
        degs = {i: self.G.degree(i) for i in self.G.nodes}
        type_map = {"Server": 0, "Router": 1, "Switch": 2}
        features = []
        for i in range(self.n):
            tvec = [0, 0, 0]
            tvec[type_map.get(self.type_of[i], 1)] = 1
            features.append([degs[i]] + tvec)
        return torch.tensor(features, dtype=torch.float32)

    def _evaluate(self) -> Tuple[float, bool, Dict]:
        # Base shaping
        reward = 0.0
        # Encourage connecting different components early
        reward += 0.05 * sum(1 for _ in nx.connected_components(self.G)) ** -1

        # Terminal conditions
        terminal = False
        info = {}

        # If tree required, check cycle and final size
        if self.constraints.must_be_tree:
            if nx.is_forest(self.G):
                # Tree must have n-1 edges and be connected
                if self.G.number_of_edges() == self.n - 1 and nx.is_connected(self.G):
                    reward += 1.0
                    terminal = True
                    info["status"] = "success_tree"
            else:
                reward -= 0.5  # made a cycle under tree constraint (should be prevented by valid_actions)

        # If only connected required (no tree), finish when connected and at least n-1 edges
        elif self.constraints.must_be_connected:
            if nx.is_connected(self.G) and self.G.number_of_edges() >= self.n - 1:
                reward += 0.7
                terminal = True
                info["status"] = "success_connected"

        # Step/episode limits
        if self.steps >= self.max_steps:
            terminal = True
            info["status"] = info.get("status", "max_steps")
            # mild penalty if unfinished
            if not (nx.is_connected(self.G) if self.constraints.must_be_connected else True):
                reward -= 0.2

        return reward, terminal, info

    def reset(self):
        self.G = nx.Graph()
        self.G.add_nodes_from([i for i, _ in self.nodes])
        self.steps = 0
        self.done = False
        return self._state()

# -----------------------------
# Policy: score candidate edges
# -----------------------------

class EdgePolicy(nn.Module):
    def __init__(self, node_feat_dim=4, hidden=64):
        super().__init__()
        # We embed nodes, then score a pair (u,v)
        self.node_mlp = nn.Sequential(
            nn.Linear(node_feat_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU()
        )
        self.edge_mlp = nn.Sequential(
            nn.Linear(hidden * 3, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )

    def forward(self, node_feats: torch.Tensor, pairs: List[Tuple[int, int]]):
        # node_feats: [N, F]
        H = self.node_mlp(node_feats)  # [N, H]
        # Compose pair feature as [h_u, h_v, |h_u - h_v|]
        e_feats = []
        for u, v in pairs:
            hu, hv = H[u], H[v]
            e_feats.append(torch.cat([hu, hv, torch.abs(hu - hv)], dim=-1))
        E = torch.stack(e_feats, dim=0) if e_feats else torch.zeros((0, H.shape[1] * 3))
        scores = self.edge_mlp(E).squeeze(-1)  # [E]
        return scores

# -----------------------------
# REINFORCE training loop
# -----------------------------

@dataclass
class Episode:
    logps: List[torch.Tensor]
    rewards: List[float]


def select_action(policy: EdgePolicy, state: torch.Tensor, valid_edges: List[Tuple[int, int]], training=True):
    if not valid_edges:
        return None, None
    
    if training:
        scores = policy(state, valid_edges)
        probs = torch.softmax(scores, dim=0)
        m = torch.distributions.Categorical(probs)
        idx = m.sample()
        logp = m.log_prob(idx)
        action = valid_edges[int(idx.item())]
        return action, logp
    else:
        with torch.no_grad():
            scores = policy(state, valid_edges)
            probs = torch.softmax(scores, dim=0)
            m = torch.distributions.Categorical(probs)
            idx = m.sample()
            logp = m.log_prob(idx)
            action = valid_edges[int(idx.item())]
            return action, logp


def run_episode(env: GraphEnv, policy: EdgePolicy, gamma=0.99) -> Tuple[Episode, nx.Graph, Dict]:
    state = env.reset()
    ep_logps, ep_rewards = [], []
    info_final = {}
    while True:
        valid_edges = env.valid_actions()
        if not valid_edges:
            # no action possible; end episode
            _, r, term, info = env._evaluate()
            info_final = info
            ep_rewards.append(r)
            break
        action, logp = select_action(policy, state, valid_edges)
        if action is None:
            _, r, term, info = env._evaluate()
            info_final = info
            ep_rewards.append(r)
            break
        state, reward, done, info = env.step(action)
        ep_logps.append(logp)
        ep_rewards.append(reward)
        info_final = info
        if done:
            break
    
    episode = Episode(logps=ep_logps, rewards=ep_rewards)
    return episode, env.G.copy(), info_final


def compute_returns(rewards: List[float], gamma=0.99) -> List[float]:
    """Compute discounted returns."""
    returns = []
    G = 0
    for r in reversed(rewards):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))


def train_policy(policy: EdgePolicy, episodes: List[Episode], optimizer, gamma=0.99):
    """REINFORCE update."""
    policy_loss = 0
    has_gradients = False
    
    for ep in episodes:
        if not ep.logps:  # Skip episodes with no actions
            continue
            
        returns = compute_returns(ep.rewards, gamma)
        returns = torch.tensor(returns, dtype=torch.float32)
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        for logp, G in zip(ep.logps, returns[:len(ep.logps)]):
            policy_loss -= logp * G
            has_gradients = True
    
    if has_gradients:
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()
        return policy_loss.item()
    else:
        return 0.0


def main():
    """Main training loop and demonstration."""
    # Example configuration
    devices = [("Server", 3), ("Router", 2), ("Switch", 2)]
    constraints = Constraints.from_strings(["tree", "connected", "servers_connect_only_to_switches"])
    
    print(f"Devices: {devices}")
    print(f"Constraints: tree={constraints.must_be_tree}, connected={constraints.must_be_connected}, server-switch={constraints.servers_connect_only_to_switches}")
    
    # Initialize environment and policy
    env = GraphEnv(devices, constraints)
    policy = EdgePolicy(node_feat_dim=4, hidden=32)
    optimizer = optim.Adam(policy.parameters(), lr=0.01)
    
    best_graph = None
    best_reward = float('-inf')
    
    # Training loop
    print("\nStarting training...")
    for epoch in range(50):
        episodes = []
        total_reward = 0
        
        # Collect episodes
        for _ in range(5):
            episode, graph, info = run_episode(env, policy)
            episodes.append(episode)
            ep_reward = sum(episode.rewards)
            total_reward += ep_reward
            
            # Track best graph
            if ep_reward > best_reward:
                best_reward = ep_reward
                best_graph = graph.copy()
                # Add node types to the graph
                for node_id, node_type in env.nodes:
                    best_graph.nodes[node_id]['type'] = node_type
        
        # Train policy
        loss = train_policy(policy, episodes, optimizer)
        avg_reward = total_reward / len(episodes)
        
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Avg Reward = {avg_reward:.3f}, Loss = {loss:.3f}")
    
    print(f"\nTraining completed. Best reward: {best_reward:.3f}")
    
    # Display best network
    if best_graph:
        print("\nBest Network Design:")
        print(f"Nodes: {best_graph.number_of_nodes()}")
        print(f"Edges: {best_graph.number_of_edges()}")
        print(f"Connected: {nx.is_connected(best_graph)}")
        print(f"Is Tree: {nx.is_tree(best_graph)}")
            
        print("\nNode Details:")
        for node in best_graph.nodes(data=True):
            node_id, data = node
            node_type = data.get('type', 'Unknown')
            degree = best_graph.degree(node_id)
            print(f"  Node {node_id}: {node_type} (degree: {degree})")
        
        print("\nEdges:")
        for u, v in best_graph.edges():
            u_type = best_graph.nodes[u].get('type', 'Unknown')
            v_type = best_graph.nodes[v].get('type', 'Unknown')
            print(f"  {u}({u_type}) -- {v}({v_type})")
        
        # Export to JSON
        graph_data = {
            'nodes': [{'id': n, 'type': best_graph.nodes[n].get('type', 'Unknown')} for n in best_graph.nodes()],
            'edges': [{'source': u, 'target': v} for u, v in best_graph.edges()]
        }
        
        with open('best_network.json', 'w') as f:
            json.dump(graph_data, f, indent=2)
        print("\nNetwork exported to 'best_network.json'")
        
        # Create interactive visualization
        visualize_network(best_graph)


def visualize_network(graph):
    """Create interactive network visualization using pyvis."""
    net = Network(height="600px", width="100%", bgcolor="#222222", font_color="white")
    
    # Image mapping for device types (local image paths)
    images = {
        "Server": "assets/server.png",
        "Router": "assets/router.png", 
        "Switch": "assets/switch.png"
    }
    
    # Add nodes with images and labels
    for node_id, data in graph.nodes(data=True):
        node_type = data.get('type', 'Unknown')
        image_url = images.get(node_type)
        
        if image_url:
            net.add_node(node_id, 
                        label=f"{node_type}\n{node_id}", 
                        image=image_url,
                        shape="image",
                        size=30)
        else:
            net.add_node(node_id, 
                        label=f"{node_type}\n{node_id}", 
                        color="#gray", 
                        size=25)
    
    # Add edges
    for u, v in graph.edges():
        net.add_edge(u, v, width=2)
    
    # Configure physics
    net.set_options("""
    var options = {
      "physics": {
        "enabled": true,
        "stabilization": {"iterations": 100}
      }
    }
    """)
    
    # Save and show
    net.save_graph("network_visualization.html")
    print("\nInteractive visualization saved as 'network_visualization.html'")
    print("Open the HTML file in your browser to view the network!")


if __name__ == "__main__":
    main()
