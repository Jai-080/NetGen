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
import webbrowser
import os
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
    
    def _ensure_server_switch_connections(self):
        """Ensure all servers are connected to all switches"""
        servers = [i for i, t in self.nodes if t == "Server"]
        switches = [i for i, t in self.nodes if t == "Switch"]
        
        for server in servers:
            for switch in switches:
                if not self.G.has_edge(server, switch):
                    self.G.add_edge(server, switch)

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
        type_map = {"Server": 0, "Router": 1, "Switch": 2, "EndDevice": 3}
        features = []
        for i in range(self.n):
            tvec = [0, 0, 0, 0]  # Updated to handle 4 device types
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
        
        # Ensure all servers connect to all switches if constraint is enabled
        if self.constraints.servers_connect_only_to_switches:
            self._ensure_server_switch_connections()
        
        self.steps = 0
        self.done = False
        return self._state()

# -----------------------------
# Policy: score candidate edges
# -----------------------------

class EdgePolicy(nn.Module):
    def __init__(self, node_feat_dim=5, hidden=64):  # Updated to 5 features (degree + 4 device types)
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
            ep_rewards.append(r)
            info_final = info
            break
        
        action, logp = select_action(policy, state, valid_edges)
        if action is None:
            break
        
        state, reward, done, info = env.step(action)
        ep_logps.append(logp)
        ep_rewards.append(reward)
        
        if done:
            info_final = info
            break
    
    return Episode(ep_logps, ep_rewards), env.G.copy(), info_final


def generate_topology(devices, topology_type):
    """Generate network based on specified topology"""
    nodes = make_nodes(devices)
    node_types = {i: t for i, t in nodes}
    G = nx.Graph()
    G.add_nodes_from([i for i, _ in nodes])
    
    all_nodes = [i for i, _ in nodes]
    n = len(all_nodes)
    
    if topology_type == "star":
        # Star topology: one central node connected to all others
        center = all_nodes[0]
        for node in all_nodes[1:]:
            G.add_edge(center, node)
    
    elif topology_type == "ring":
        # Ring topology: nodes connected in a circle
        for i in range(n):
            G.add_edge(all_nodes[i], all_nodes[(i + 1) % n])
    
    elif topology_type == "mesh":
        # Mesh topology: every node connected to every other node
        for i in range(n):
            for j in range(i + 1, n):
                G.add_edge(all_nodes[i], all_nodes[j])
    
    elif topology_type == "tree":
        # Tree topology: hierarchical structure with switch-to-lower-device connections
        servers = [i for i, t in node_types.items() if t == "Server"]
        switches = [i for i, t in node_types.items() if t == "Switch"]
        routers = [i for i, t in node_types.items() if t == "Router"]
        end_devices = [i for i, t in node_types.items() if t == "EndDevice"]
        
        # Connect all switches to servers
        for switch in switches:
            for server in servers:
                G.add_edge(server, switch)
        
        # Ensure every switch connects to at least one lower level device
        lower_devices = routers + end_devices
        for i, switch in enumerate(switches):
            if i < len(lower_devices):
                G.add_edge(switch, lower_devices[i])
        
        # Connect remaining devices in tree structure
        remaining_devices = []
        if len(servers) > len(switches):
            remaining_devices.extend(servers[len(switches):])
        if len(switches) > len(lower_devices):
            remaining_devices.extend(switches[len(lower_devices):])
        if len(lower_devices) > len(switches):
            remaining_devices.extend(lower_devices[len(switches):])
        
        # Connect remaining devices to existing tree
        for device in remaining_devices:
            if all_nodes:
                # Connect to first available node
                G.add_edge(device, all_nodes[0])
    
    elif topology_type == "bus":
        # Bus topology: linear connection
        for i in range(n - 1):
            G.add_edge(all_nodes[i], all_nodes[i + 1])
    
    elif topology_type == "hybrid":
        # Hybrid topology: combination of star and ring with switch-to-lower-device connections
        servers = [i for i, t in node_types.items() if t == "Server"]
        switches = [i for i, t in node_types.items() if t == "Switch"]
        routers = [i for i, t in node_types.items() if t == "Router"]
        end_devices = [i for i, t in node_types.items() if t == "EndDevice"]
        
        if n >= 4:
            # Create a central hub with some nodes
            hub_size = min(3, n // 2)
            center = all_nodes[0]
            
            # Star part
            for i in range(1, hub_size + 1):
                if i < n:
                    G.add_edge(center, all_nodes[i])
            
            # Ring part with remaining nodes
            remaining = all_nodes[hub_size + 1:]
            if len(remaining) >= 2:
                for i in range(len(remaining)):
                    G.add_edge(remaining[i], remaining[(i + 1) % len(remaining)])
                # Connect ring to star
                if remaining:
                    G.add_edge(all_nodes[1], remaining[0])
        else:
            # Fallback to star for small networks
            center = all_nodes[0]
            for node in all_nodes[1:]:
                G.add_edge(center, node)
        
        # Ensure every switch connects to at least one lower level device
        lower_devices = routers + end_devices
        for i, switch in enumerate(switches):
            if lower_devices and i < len(lower_devices):
                G.add_edge(switch, lower_devices[i])
    
    return G, node_types

def train_policy(devices, constraints, episodes=100, lr=1e-3):
    env = GraphEnv(devices, constraints)
    policy = EdgePolicy()
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Pre-connect all servers to all switches if constraint is enabled
    if constraints.servers_connect_only_to_switches:
        env._ensure_server_switch_connections()
    
    best_graph = None
    best_reward = float('-inf')
    
    for ep in range(episodes):
        episode, graph, info = run_episode(env, policy)
        
        # Calculate returns
        returns = []
        G = 0
        for r in reversed(episode.rewards):
            G = r + 0.99 * G
            returns.insert(0, G)
        
        # Policy gradient update
        if episode.logps:
            returns = torch.tensor(returns, dtype=torch.float32)
            if len(returns) > 1:
                returns = (returns - returns.mean()) / (returns.std() + 1e-8)
            
            policy_loss = []
            for logp, ret in zip(episode.logps, returns):
                policy_loss.append(-logp * ret)
            
            optimizer.zero_grad()
            loss = torch.stack(policy_loss).sum()
            loss.backward()
            optimizer.step()
        
        total_reward = sum(episode.rewards)
        if total_reward > best_reward:
            best_reward = total_reward
            best_graph = graph
        
        if ep % 20 == 0:
            print(f"Episode {ep}: Reward={total_reward:.3f}, Status={info.get('status', 'unknown')}")
    
    return policy, best_graph, best_reward


def visualize_graph(graph, node_types, filename="network.html"):
    net = Network(height="800px", width="100%", bgcolor="#222222", font_color="white")
    
    # Image mapping for device types using assets folder
    images = {
        "Server": "assets/server.png",
        "Router": "assets/router.png", 
        "Switch": "assets/switch.png",
        "EndDevice": "assets/desktop.png"
    }
    
    # Hierarchical positioning: Server(top) -> Switch -> Router -> EndDevice(bottom)
    hierarchy_levels = {
        "Server": -400,    # Top level (negative Y = top in vis.js)
        "Switch": -200,    # Second level  
        "Router": 0,       # Third level
        "EndDevice": 200   # Bottom level (positive Y = bottom in vis.js)
    }
    
    # Group nodes by type for horizontal spacing
    nodes_by_type = {"Server": [], "Switch": [], "Router": [], "EndDevice": []}
    for node in graph.nodes():
        node_type = node_types.get(node, "Unknown")
        if node_type in nodes_by_type:
            nodes_by_type[node_type].append(node)
    
    for node in graph.nodes():
        node_type = node_types.get(node, "Unknown")
        image_path = images.get(node_type)
        
        # Calculate position for hierarchical layout
        y_pos = hierarchy_levels.get(node_type, 0)
        type_nodes = nodes_by_type.get(node_type, [node])
        x_spacing = 150
        x_offset = (len(type_nodes) - 1) * x_spacing / 2
        x_pos = (type_nodes.index(node) * x_spacing) - x_offset
        
        if image_path:
            net.add_node(node, 
                        label=f"{node_type}\n{node}", 
                        image=image_path,
                        shape="image",
                        size=40,
                        x=x_pos,
                        y=y_pos,
                        physics=False)
        else:
            net.add_node(node, 
                        label=f"{node_type}\n{node}", 
                        color="#gray", 
                        size=30,
                        x=x_pos,
                        y=y_pos,
                        physics=False)
    
    for edge in graph.edges():
        net.add_edge(edge[0], edge[1], width=2)
    
    # Configure layout options
    net.set_options("""
    var options = {
      "physics": {
        "enabled": false
      },
      "layout": {
        "hierarchical": {
          "enabled": false
        }
      }
    }
    """)
    
    net.save_graph(filename)
    print(f"Network visualization saved to {filename}")
    return filename


if __name__ == "__main__":
    # Example configuration with end devices
    devices = [("Server", 3), ("Switch", 2), ("Router", 2), ("EndDevice", 2)]
    constraints = Constraints.from_strings(["connected", "server switch"])
    
    print("Training network design AI...")
    policy, best_graph, best_reward = train_policy(devices, constraints, episodes=100)
    
    print(f"\nBest network found with reward: {best_reward:.3f}")
    print(f"Nodes: {best_graph.number_of_nodes()}, Edges: {best_graph.number_of_edges()}")
    print(f"Connected: {nx.is_connected(best_graph)}")
    
    # Create node type mapping for visualization
    nodes = make_nodes(devices)
    node_types = {i: t for i, t in nodes}
    
    # Visualize the best network
    html_file = visualize_graph(best_graph, node_types)
    
    # Save network as JSON
    network_data = {
        "nodes": [{"id": i, "type": t} for i, t in nodes],
        "edges": list(best_graph.edges())
    }
    
    with open("best_network.json", "w") as f:
        json.dump(network_data, f, indent=2)
    
    print("Network data saved to best_network.json")
    
    # Automatically open the visualization in browser
    html_path = os.path.abspath(html_file)
    print(f"Opening visualization in browser...")
    webbrowser.open(f"file:///{html_path}")