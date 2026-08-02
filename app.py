"""
app.py — NetGen Flask backend

Routes
------
GET  /          → index.html
POST /generate  → generate or RL-train a network topology, return vis.js JSON
"""

import hashlib
import math
import os
import random
import time
import numpy as np

import networkx as nx
import torch
from flask import Flask, jsonify, render_template, request

from gen_net_design_ai import (
    Constraints,
    GraphEnv,
    calculate_topology_success_rates,
    generate_topology,
    load_policy,
    make_nodes,
    run_episode,
    save_policy,
    train_policy,
    simulate_failures,
    get_edge_cost,
)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Policy cache (in-memory + on-disk checkpoints)
# ---------------------------------------------------------------------------

CHECKPOINT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints")
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

_policy_cache: dict = {}  # config_hash → EdgePolicy


def _config_hash(devices: list, constraints: Constraints) -> str:
    """Stable hash for a (devices, constraints) configuration."""
    key = str(sorted(devices)) + str(sorted(vars(constraints).items()))
    return hashlib.md5(key.encode()).hexdigest()[:12]


def _get_policy(devices: list, constraints: Constraints, episodes: int = 80, lambda_cost: float = 0.0, lambda_resilience: float = 0.0):
    """
    Return a trained EdgePolicy for this config.
    Priority: in-memory cache -> on-disk checkpoint -> train from scratch.
    """
    chk_hash = _config_hash(devices, constraints) + f"_c{lambda_cost}_r{lambda_resilience}"

    if chk_hash in _policy_cache:
        return _policy_cache[chk_hash]

    chk_path = os.path.join(CHECKPOINT_DIR, f"policy_{chk_hash}.pt")
    if os.path.exists(chk_path):
        try:
            policy = load_policy(chk_path)
            _policy_cache[chk_hash] = policy
            app.logger.info(f"Loaded policy from checkpoint {chk_path}")
            return policy
        except Exception as exc:
            app.logger.warning(f"Corrupt checkpoint {chk_path}, retraining. ({exc})")

    # Only fallback to global policy if parameters are default (no cost or resilience pressure)
    if lambda_cost == 0.0 and lambda_resilience == 0.0:
        global_path = os.path.join(CHECKPOINT_DIR, "global_policy.pt")
        if os.path.exists(global_path):
            try:
                policy = load_policy(global_path)
                _policy_cache[chk_hash] = policy
                app.logger.info(f"Loaded pre-trained generalized global policy from {global_path}")
                return policy
            except Exception as exc:
                app.logger.warning(f"Failed to load global policy: {exc}")

    # Fallback: Train from scratch and persist
    policy, _, _, _ = train_policy(
        devices,
        constraints,
        episodes=episodes,
        lambda_cost=lambda_cost,
        lambda_resilience=lambda_resilience
    )
    save_policy(policy, chk_path)
    _policy_cache[chk_hash] = policy
    return policy


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/generate", methods=["POST"])
def generate_network():
    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "error": "Request body must be JSON"}), 400

        # ── Parse and validate device counts ─────────────────────────────
        try:
            servers = int(data.get("servers", 0))
            switches = int(data.get("switches", 0))
            routers = int(data.get("routers", 0))
            end_devices = int(data.get("end_devices", 0))
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Device counts must be integers"}), 400

        for field_name, val in [
            ("servers", servers), ("switches", switches),
            ("routers", routers), ("end_devices", end_devices),
        ]:
            if val < 0 or val > 15:
                return jsonify({
                    "success": False,
                    "error": f"'{field_name}' must be between 0 and 15",
                }), 400

        total_devices = servers + switches + routers + end_devices
        if total_devices == 0:
            return jsonify({"success": False, "error": "Please specify at least one device"})
        if total_devices == 1:
            return jsonify({"success": False, "error": "Please specify at least 2 devices"})

        # ── Topology selection ────────────────────────────────────────────
        topology = data.get("topology", "hybrid")
        valid_topologies = {"hybrid", "star", "ring", "mesh", "tree", "bus", "random", "rl"}
        if topology not in valid_topologies:
            return jsonify({"success": False, "error": f"Invalid topology: '{topology}'"}), 400

        if topology == "random":
            topology = random.choice(["hybrid", "star", "ring", "mesh", "tree", "bus"])

        # ── Build device list ─────────────────────────────────────────────
        devices = []
        if servers > 0:
            devices.append(("Server", servers))
        if switches > 0:
            devices.append(("Switch", switches))
        if routers > 0:
            devices.append(("Router", routers))
        if end_devices > 0:
            devices.append(("EndDevice", end_devices))

        # ── Generate graph ────────────────────────────────────────────────
        if topology == "rl":
            # Only enforce server→switch constraint when both types are present
            use_srv_sw = servers > 0 and switches > 0
            constraints = Constraints(
                must_be_connected=True,
                servers_connect_only_to_switches=use_srv_sw,
            )
            policy = _get_policy(devices, constraints, episodes=80)
            env = GraphEnv(devices, constraints)
            policy.eval()
            with torch.no_grad():
                _, best_graph, _ = run_episode(env, policy)
            node_types = {i: t for i, t in make_nodes(devices)}
            topology_label = "rl"
        else:
            best_graph, node_types = generate_topology(devices, topology)
            topology_label = topology

        # ── Real suitability scores (computed from actual graph metrics) ──
        success_rates = calculate_topology_success_rates(
            servers, switches, routers, end_devices
        )

        # ── Build vis.js-ready payload ────────────────────────────────────
        network_data = generate_network_data(best_graph, node_types, topology_label)

        return jsonify({
            "success": True,
            "network": network_data,
            "topology_used": topology_label,
            "stats": {
                "nodes": best_graph.number_of_nodes(),
                "edges": best_graph.number_of_edges(),
                "connected": nx.is_connected(best_graph),
            },
            "success_rates": success_rates,
        })

    except Exception:
        app.logger.exception("Unhandled error in /generate")
        return jsonify({"success": False, "error": "Internal server error — check server logs"})


# ---------------------------------------------------------------------------
# Live Comparison Route (Task 4)
# ---------------------------------------------------------------------------

_compare_cache: dict = {}

def calculate_graph_cost(graph: nx.Graph, node_types: dict) -> float:
    total_cost = 0.0
    for u, v in graph.edges():
        total_cost += get_edge_cost(node_types[u], node_types[v])
    return total_cost

@app.route("/compare", methods=["POST"])
def compare_topologies():
    try:
        data = request.json
        if data is None:
            return jsonify({"success": False, "error": "Request body must be JSON"}), 400

        try:
            servers = int(data.get("servers", 0))
            switches = int(data.get("switches", 0))
            routers = int(data.get("routers", 0))
            end_devices = int(data.get("end_devices", 0))
        except (TypeError, ValueError):
            return jsonify({"success": False, "error": "Device counts must be integers"}), 400

        total_devices = servers + switches + routers + end_devices
        if total_devices < 2:
            return jsonify({"success": False, "error": "Please specify at least 2 devices"}), 400

        # Build stable config
        devices = []
        if servers > 0:
            devices.append(("Server", servers))
        if switches > 0:
            devices.append(("Switch", switches))
        if routers > 0:
            devices.append(("Router", routers))
        if end_devices > 0:
            devices.append(("EndDevice", end_devices))

        use_srv_sw = servers > 0 and switches > 0
        constraints = Constraints(
            must_be_connected=True,
            servers_connect_only_to_switches=use_srv_sw,
        )

        config_key = f"{servers}_{switches}_{routers}_{end_devices}"
        if config_key in _compare_cache:
            return jsonify({"success": True, "results": _compare_cache[config_key]})

        results = {}

        # 1. Random baseline
        start_time = time.time()
        env_rand = GraphEnv(devices, constraints)
        rand_successes = []
        rand_costs = []
        rand_resilience = []
        node_types = {i: t for i, t in make_nodes(devices)}
        for _ in range(5):
            env_rand.reset()
            info = {}
            while True:
                valid_edges = env_rand.valid_actions()
                if not valid_edges:
                    _, _, inf = env_rand._evaluate()
                    info = inf
                    break
                action = random.choice(valid_edges)
                _, _, done, inf = env_rand.step(action)
                if done:
                    info = inf
                    break
            rand_successes.append(info.get("status", "").startswith("success"))
            rand_costs.append(calculate_graph_cost(env_rand.G, node_types))
            survival, _ = simulate_failures(env_rand.G, n_trials=30)
            rand_resilience.append(survival)
        results["random"] = {
            "success_rate": float(np.mean(rand_successes)),
            "cost": float(np.mean(rand_costs)),
            "resilience": float(np.mean(rand_resilience)),
            "time": (time.time() - start_time) / 5
        }

        # 2. Heuristic Hybrid
        start_time = time.time()
        try:
            G_heur, node_types_heur = generate_topology(devices, "hybrid")
            env_heur = GraphEnv(devices, constraints)
            env_heur.G = G_heur.copy()
            _, _, info = env_heur._evaluate()
            heur_success = info.get("status", "").startswith("success")
            heur_cost = calculate_graph_cost(G_heur, node_types_heur)
            heur_survival, _ = simulate_failures(G_heur, n_trials=30)
            results["hybrid"] = {
                "success_rate": 1.0 if heur_success else 0.0,
                "cost": heur_cost,
                "resilience": heur_survival,
                "time": time.time() - start_time
            }
        except Exception:
            results["hybrid"] = {"success_rate": 0.0, "cost": 999.0, "resilience": 0.0, "time": 0.0}

        # 3. MST Baseline
        start_time = time.time()
        try:
            n = len(node_types)
            K = nx.Graph()
            K.add_nodes_from(range(n))
            for u in range(n):
                for v in range(u + 1, n):
                    cost = get_edge_cost(node_types[u], node_types[v])
                    if use_srv_sw:
                        if (node_types[u] == "Server" and node_types[v] != "Switch") or \
                           (node_types[v] == "Server" and node_types[u] != "Switch"):
                            cost += 1000.0
                    K.add_edge(u, v, weight=cost)
            MST_G = nx.minimum_spanning_tree(K, weight='weight')
            env_mst = GraphEnv(devices, constraints)
            env_mst.G = MST_G.copy()
            _, _, info = env_mst._evaluate()
            mst_success = info.get("status", "").startswith("success")
            mst_cost = calculate_graph_cost(MST_G, node_types)
            mst_survival, _ = simulate_failures(MST_G, n_trials=30)
            results["mst"] = {
                "success_rate": 1.0 if mst_success else 0.0,
                "cost": mst_cost,
                "resilience": mst_survival,
                "time": time.time() - start_time
            }
        except Exception:
            results["mst"] = {"success_rate": 0.0, "cost": 999.0, "resilience": 0.0, "time": 0.0}

        # 4. RL Agent (lambda_cost=0.5, lambda_resilience=0.2)
        start_time = time.time()
        # Fetch or train policy
        policy = _get_policy(devices, constraints, episodes=100, lambda_cost=0.5, lambda_resilience=0.2)
        policy.eval()
        env_rl = GraphEnv(devices, constraints, lambda_cost=0.5, lambda_resilience=0.2)
        rl_successes = []
        rl_costs = []
        rl_resilience = []
        with torch.no_grad():
            for _ in range(5):
                _, graph, info = run_episode(env_rl, policy, training=False)
                rl_successes.append(info.get("status", "").startswith("success"))
                rl_costs.append(calculate_graph_cost(graph, node_types))
                survival, _ = simulate_failures(graph, n_trials=30)
                rl_resilience.append(survival)
        results["rl"] = {
            "success_rate": float(np.mean(rl_successes)),
            "cost": float(np.mean(rl_costs)),
            "resilience": float(np.mean(rl_resilience)),
            "time": (time.time() - start_time) / 5
        }

        # Save to cache
        _compare_cache[config_key] = results

        return jsonify({"success": True, "results": results})

    except Exception:
        app.logger.exception("Unhandled error in /compare")
        return jsonify({"success": False, "error": "Internal server error — check server logs"})


# ---------------------------------------------------------------------------
# Network layout helper (vis.js JSON)
# ---------------------------------------------------------------------------

def generate_network_data(graph: nx.Graph, node_types: dict, topology: str = "hybrid") -> dict:
    """Convert a networkx graph to a vis.js-compatible nodes/edges payload."""
    images = {
        "Server": "static/assets/server.png",
        "Router": "static/assets/router.png",
        "Switch": "static/assets/switch.png",
        "EndDevice": "static/assets/desktop.png",
    }
    hierarchy_levels = {"Server": -400, "Switch": -200, "Router": 0, "EndDevice": 200}

    nodes_by_type: dict = {"Server": [], "Switch": [], "Router": [], "EndDevice": []}
    for node in graph.nodes():
        ntype = node_types.get(node, "Unknown")
        if ntype in nodes_by_type:
            nodes_by_type[ntype].append(node)

    nodes_data = []

    if topology == "star":
        all_nodes = list(graph.nodes())
        n = len(all_nodes)
        if n > 1:
            center_node = all_nodes[0]
            center_type = node_types.get(center_node, "Unknown")
            nodes_data.append({
                "id": center_node,
                "label": f"{center_type}\n{center_node}",
                "image": images.get(center_type, ""),
                "shape": "image", "size": 50,
                "x": 0, "y": 0, "physics": False,
            })
            radius = 300
            for i, node in enumerate(all_nodes[1:]):
                angle = 2 * math.pi * i / (n - 1)
                ntype = node_types.get(node, "Unknown")
                nodes_data.append({
                    "id": node,
                    "label": f"{ntype}\n{node}",
                    "image": images.get(ntype, ""),
                    "shape": "image", "size": 40,
                    "x": radius * math.cos(angle),
                    "y": radius * math.sin(angle),
                    "physics": False,
                })
        else:
            node = all_nodes[0]
            ntype = node_types.get(node, "Unknown")
            nodes_data.append({
                "id": node, "label": f"{ntype}\n{node}",
                "image": images.get(ntype, ""),
                "shape": "image", "size": 40,
                "x": 0, "y": 0, "physics": False,
            })

    elif topology == "ring":
        all_nodes = list(graph.nodes())
        n = len(all_nodes)
        radius = 250
        for i, node in enumerate(all_nodes):
            angle = 2 * math.pi * i / n
            ntype = node_types.get(node, "Unknown")
            nodes_data.append({
                "id": node, "label": f"{ntype}\n{node}",
                "image": images.get(ntype, ""),
                "shape": "image", "size": 40,
                "x": radius * math.cos(angle),
                "y": radius * math.sin(angle),
                "physics": False,
            })

    else:
        # Hierarchical layout for mesh, tree, bus, hybrid, rl, and any other type
        for node in graph.nodes():
            ntype = node_types.get(node, "Unknown")
            y_pos = hierarchy_levels.get(ntype, 0)
            type_nodes = nodes_by_type.get(ntype, [node])
            x_spacing = 250 if topology == "mesh" else 150
            x_offset = (len(type_nodes) - 1) * x_spacing / 2
            x_pos = (type_nodes.index(node) * x_spacing) - x_offset
            nodes_data.append({
                "id": node, "label": f"{ntype}\n{node}",
                "image": images.get(ntype, ""),
                "shape": "image", "size": 40,
                "x": x_pos, "y": y_pos, "physics": False,
            })

    edges_data = [{"from": u, "to": v, "width": 2} for u, v in graph.edges()]
    return {"nodes": nodes_data, "edges": edges_data}


if __name__ == "__main__":
    app.run(debug=True, port=5000)