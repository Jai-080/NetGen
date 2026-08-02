"""
benchmark.py
------------
Proper multi-objective benchmark sweep across network design configurations.
Compares classical baselines (Random, Heuristics, MST) and evaluates the trained
RL policy under a variety of cost penalty weights (lambda_cost) to map the
Cost-vs-Resilience Pareto frontier.
"""

import os
import time
import csv
import random
import numpy as np
import networkx as nx
import torch

from gen_net_design_ai import (
    Constraints,
    GraphEnv,
    EdgePolicy,
    train_policy,
    run_episode,
    simulate_failures,
    generate_topology,
    make_nodes,
    get_edge_cost
)

# ---------------------------------------------------------------------------
# Hardware Access/Trunk/WAN Cost Model Helper
# ---------------------------------------------------------------------------

def calculate_graph_cost(graph: nx.Graph, node_types: dict) -> float:
    """Sum the total hardware cost of all edges in the graph."""
    total_cost = 0.0
    for u, v in graph.edges():
        total_cost += get_edge_cost(node_types[u], node_types[v])
    return total_cost


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def run_random_eval(devices, constraints, n_runs=30):
    """Evaluate a random edge selection policy."""
    env = GraphEnv(devices, constraints)
    successes = []
    costs = []
    path_lengths = []
    resilience_scores = []
    
    node_types = {i: t for i, t in make_nodes(devices)}
    
    for _ in range(n_runs):
        env.reset()
        info = {}
        
        while True:
            valid_edges = env.valid_actions()
            if not valid_edges:
                _, _, inf = env._evaluate()
                info = inf
                break
            action = random.choice(valid_edges)
            _, _, done, inf = env.step(action)
            if done:
                info = inf
                break
                
        is_success = info.get("status", "").startswith("success")
        successes.append(is_success)
        costs.append(calculate_graph_cost(env.G, node_types))
        
        if nx.is_connected(env.G):
            path_lengths.append(nx.average_shortest_path_length(env.G))
        else:
            path_lengths.append(float(env.n - 1))
            
        survival, _ = simulate_failures(env.G, n_trials=50)
        resilience_scores.append(survival)
        
    return {
        "success_rate": np.mean(successes),
        "mean_cost": np.mean(costs),
        "mean_path_len": np.mean(path_lengths),
        "resilience": np.mean(resilience_scores),
    }

def run_heuristic_eval(devices, constraints, topo_type="hybrid"):
    """Evaluate a deterministic rule-based topology layout."""
    try:
        G, node_types = generate_topology(devices, topo_type)
        env = GraphEnv(devices, constraints)
        env.G = G.copy()
        _, _, info = env._evaluate()
        is_success = info.get("status", "").startswith("success")
        
        cost = calculate_graph_cost(G, node_types)
        if nx.is_connected(G):
            path_len = nx.average_shortest_path_length(G)
        else:
            path_len = float(len(G) - 1)
            
        survival, _ = simulate_failures(G, n_trials=50)
        
        return {
            "success_rate": float(is_success),
            "mean_cost": cost,
            "mean_path_len": path_len,
            "resilience": survival,
        }
    except Exception:
        return {
            "success_rate": 0.0,
            "mean_cost": 999.0,
            "mean_path_len": 999.0,
            "resilience": 0.0,
        }

def run_mst_eval(devices, constraints):
    """Evaluate Minimum Spanning Tree (MST) on synthetic cost graph."""
    nodes = make_nodes(devices)
    node_types = {i: t for i, t in nodes}
    n = len(nodes)
    
    K = nx.Graph()
    K.add_nodes_from([i for i, _ in nodes])
    
    for u in range(n):
        for v in range(u + 1, n):
            cost = get_edge_cost(node_types[u], node_types[v])
            if constraints.servers_connect_only_to_switches:
                if (node_types[u] == "Server" and node_types[v] != "Switch") or \
                   (node_types[v] == "Server" and node_types[u] != "Switch"):
                    cost += 1000.0
            K.add_edge(u, v, weight=cost)
            
    MST_G = nx.minimum_spanning_tree(K, weight='weight')
    
    env = GraphEnv(devices, constraints)
    env.G = MST_G.copy()
    _, _, info = env._evaluate()
    is_success = info.get("status", "").startswith("success")
    
    cost = calculate_graph_cost(MST_G, node_types)
    if nx.is_connected(MST_G):
        path_len = nx.average_shortest_path_length(MST_G)
    else:
        path_len = float(n - 1)
        
    survival, _ = simulate_failures(MST_G, n_trials=50)
    
    return {
        "success_rate": float(is_success),
        "mean_cost": cost,
        "mean_path_len": path_len,
        "resilience": survival,
    }

def run_rl_eval(devices, constraints, lambda_cost, lambda_resilience, episodes=100, n_evals=20):
    """Train and evaluate policy under specific multi-objective parameters."""
    start_train = time.time()
    policy, _, _, _ = train_policy(
        devices,
        constraints,
        episodes=episodes,
        lambda_cost=lambda_cost,
        lambda_resilience=lambda_resilience
    )
    train_time = time.time() - start_train
    avg_ep_time = train_time / episodes
    
    # Evaluate
    env = GraphEnv(devices, constraints)
    successes = []
    costs = []
    path_lengths = []
    resilience_scores = []
    node_types = {i: t for i, t in make_nodes(devices)}
    
    policy.eval()
    with torch.no_grad():
        for _ in range(n_evals):
            _, graph, info = run_episode(env, policy, training=False)
            is_success = info.get("status", "").startswith("success")
            successes.append(is_success)
            costs.append(calculate_graph_cost(graph, node_types))
            
            if nx.is_connected(graph):
                path_lengths.append(nx.average_shortest_path_length(graph))
            else:
                path_lengths.append(float(env.n - 1))
                
            survival, _ = simulate_failures(graph, n_trials=50)
            resilience_scores.append(survival)
            
    return {
        "success_rate": np.mean(successes),
        "mean_cost": np.mean(costs),
        "mean_path_len": np.mean(path_lengths),
        "resilience": np.mean(resilience_scores),
        "train_time": train_time,
        "avg_ep_time": avg_ep_time
    }


# ---------------------------------------------------------------------------
# Sweep Sweeper & Pareto Frontier Construction
# ---------------------------------------------------------------------------

CONFIGURATIONS = [
    (
        "Config_Medium_Hierarchy",
        [("Server", 3), ("Switch", 3), ("Router", 2), ("EndDevice", 4)],
        Constraints(must_be_connected=True, servers_connect_only_to_switches=True),
        "N=12, Hierarchy constraints (Primary Evaluation)"
    ),
    (
        "Config_Replication_Hierarchy",
        [("Server", 2), ("Switch", 3), ("Router", 2), ("EndDevice", 3)],
        Constraints(must_be_connected=True, servers_connect_only_to_switches=True),
        "N=10, Hierarchy constraints (Replication Verification)"
    )
]

LAMBDA_SWEEP = [0.0, 0.1, 0.2, 0.3, 0.5, 0.8, 1.0]

def plot_pareto(rl_frontier_data, mst_data, heur_data, filename="pareto_frontier.png"):
    """Plot Pareto frontier of Cost vs Resilience and save as PNG."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping Pareto plot generation.")
        return
        
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 1. Plot classical baselines as single points/lines
    for name, c, r, marker, color in [
        ("MST (Cheap & Fragile)", mst_data["mean_cost"], mst_data["resilience"], "D", "firebrick"),
        ("Heuristic Hybrid (Expensive)", heur_data["mean_cost"], heur_data["resilience"], "s", "orange")
    ]:
        ax.scatter(c, r, marker=marker, color=color, s=150, zorder=5, label=name)
        
    # 2. Plot RL lambda sweep frontier
    rl_costs = [d["mean_cost"] for d in rl_frontier_data]
    rl_res = [d["resilience"] for d in rl_frontier_data]
    lambdas = [d["lambda"] for d in rl_frontier_data]
    
    # Sort by cost for a clean line plot
    sort_indices = np.argsort(rl_costs)
    sorted_costs = np.array(rl_costs)[sort_indices]
    sorted_res = np.array(rl_res)[sort_indices]
    
    ax.plot(sorted_costs, sorted_res, color="steelblue", linestyle="-", linewidth=2.5, zorder=3)
    
    # Annotate lambdas
    for c, r, l in zip(rl_costs, rl_res, lambdas):
        ax.scatter(c, r, color="steelblue", s=80, zorder=4)
        ax.annotate(f"λ={l}", (c, r), textcoords="offset points", xytext=(5, 5), fontsize=9)
        
    ax.set_title("Pareto Frontier: Network Deployment Cost vs. Failure Resilience", fontsize=13, fontweight="bold")
    ax.set_xlabel("Average Hardware Deployment Cost ($)", fontsize=11)
    ax.set_ylabel("Failure Resilience Rate (Survival %)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename, dpi=120)
    plt.close(fig)
    print(f"Pareto frontier plot successfully saved to {filename}")


def run_sweeps():
    print("=" * 80)
    print("Starting Multi-Objective Tradeoff & Pareto Frontier Sweeps...")
    print("=" * 80)
    
    results = []
    
    # Keep track of primary config (Config_Medium_Hierarchy) Pareto points for plotting
    primary_rl_points = []
    primary_mst = None
    primary_heur = None
    
    # Checkstop flags
    primary_checkstop_passed = False
    replication_checkstop_passed = False
    
    for name, devices, constraints, desc in CONFIGURATIONS:
        n_nodes = len(make_nodes(devices))
        print(f"\n>>> EVALUATING CONFIGURATION: {name} ({desc})")
        
        # 1. Run baselines
        print("  Evaluating Random Baseline...")
        rand_res = run_random_eval(devices, constraints)
        print("  Evaluating Heuristic Hybrid...")
        heur_res = run_heuristic_eval(devices, constraints, "hybrid")
        print("  Evaluating MST Baseline...")
        mst_res = run_mst_eval(devices, constraints)
        
        if name == "Config_Medium_Hierarchy":
            primary_mst = mst_res
            primary_heur = heur_res
            
        print("-" * 80)
        print(f"{'Method/Parameter':<20} | {'Success':<8} | {'Cost':<6} | {'Path Len':<8} | {'Resilience':<10}")
        print("-" * 80)
        print(f"{'Random':<20} | {rand_res['success_rate']:.1%}  | {rand_res['mean_cost']:<6.1f} | {rand_res['mean_path_len']:<8.2f} | {rand_res['resilience']:.1%}")
        print(f"{'Heuristic Hybrid':<20} | {heur_res['success_rate']:.1%}  | {heur_res['mean_cost']:<6.1f} | {heur_res['mean_path_len']:<8.2f} | {heur_res['resilience']:.1%}")
        print(f"{'MST Baseline':<20} | {mst_res['success_rate']:.1%}  | {mst_res['mean_cost']:<6.1f} | {mst_res['mean_path_len']:<8.2f} | {mst_res['resilience']:.1%}")
        
        # 2. Sweep lambda_cost values for RL (with lambda_resilience = 0.2)
        print("\n  Sweeping lambda_cost weights for RL...")
        config_rl_points = []
        
        for l_cost in LAMBDA_SWEEP:
            rl_res = run_rl_eval(devices, constraints, lambda_cost=l_cost, lambda_resilience=0.2, episodes=100)
            print(f"  RL (lambda_cost={l_cost:.1f})  | {rl_res['success_rate']:.1%}  | {rl_res['mean_cost']:<6.1f} | {rl_res['mean_path_len']:<8.2f} | {rl_res['resilience']:.1%}")
            
            # Save results row
            row = {
                "Config": name,
                "N": n_nodes,
                "Method": f"RL_lambda_{l_cost}",
                "Success": rl_res["success_rate"],
                "Cost": rl_res["mean_cost"],
                "PathLen": rl_res["mean_path_len"],
                "Resilience": rl_res["resilience"]
            }
            results.append(row)
            
            pt = {"lambda": l_cost, "mean_cost": rl_res["mean_cost"], "resilience": rl_res["resilience"], "success_rate": rl_res["success_rate"]}
            config_rl_points.append(pt)
            if name == "Config_Medium_Hierarchy":
                primary_rl_points.append(pt)
                
        # Save baseline rows
        for b_name, b_res in [("Random", rand_res), ("Heuristic_Hybrid", heur_res), ("MST", mst_res)]:
            results.append({
                "Config": name,
                "N": n_nodes,
                "Method": b_name,
                "Success": b_res["success_rate"],
                "Cost": b_res["mean_cost"],
                "PathLen": b_res["mean_path_len"],
                "Resilience": b_res["resilience"]
            })
            
        # Verify if Pareto-optimal middle ground checkstop passes for this configuration
        # Middle ground definition: RL cost is at least 15% cheaper than Heuristic Hybrid,
        # AND RL resilience is at least 15% (15 percentage points) higher than MST.
        print("\n  Checking Multi-Objective Tradeoff for config...")
        best_tradeoff_pt = None
        for pt in config_rl_points:
            # We look at cases where RL succeeds (Success rate >= 90%)
            if pt["success_rate"] >= 0.90:
                cost_reduction = (heur_res["mean_cost"] - pt["mean_cost"]) / heur_res["mean_cost"]
                resilience_advantage = pt["resilience"] - mst_res["resilience"]
                if cost_reduction >= 0.15 and resilience_advantage >= 0.15:
                    best_tradeoff_pt = pt
                    break
                    
        if best_tradeoff_pt:
            print(f"  [RESULT] Tradeoff check passed at lambda={best_tradeoff_pt['lambda']}!")
            print(f"    RL Cost: {best_tradeoff_pt['mean_cost']:.1f} vs Heuristic: {heur_res['mean_cost']:.1f} (-{ (heur_res['mean_cost'] - best_tradeoff_pt['mean_cost'])/heur_res['mean_cost']:.1%} reduction)")
            print(f"    RL Resilience: {best_tradeoff_pt['resilience']:.1%} vs MST: {mst_res['resilience']:.1%} (+{best_tradeoff_pt['resilience'] - mst_res['resilience']:.1%} points higher)")
            if name == "Config_Medium_Hierarchy":
                primary_checkstop_passed = True
            elif name == "Config_Replication_Hierarchy":
                replication_checkstop_passed = True
        else:
            print("  [RESULT] Tradeoff check FAILED (no lambda weight found that simultaneously beats Heuristic Hybrid cost and MST resilience).")
            
        print("=" * 80)
        
    # Write to CSV
    csv_file = "benchmark_pareto_results.csv"
    with open(csv_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["Config", "N", "Method", "Success", "Cost", "PathLen", "Resilience"])
        writer.writeheader()
        writer.writerows(results)
    print(f"Detailed multi-objective results saved to {csv_file}")
    
    # 3. Plot the Pareto frontier for the primary configuration
    if primary_mst and primary_heur:
        plot_pareto(primary_rl_points, primary_mst, primary_heur, "pareto_frontier.png")
        
    # ── Strict Checkstop Status ──────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL CHECKSTOP STATUS")
    print("=" * 60)
    print(f"Primary Configuration Tradeoff Reached:    {primary_checkstop_passed}")
    print(f"Replication Configuration Tradeoff Reached:  {replication_checkstop_passed}")
    
    if primary_checkstop_passed and replication_checkstop_passed:
        print("\n[CHECKSTOP STATUS] ALL PASSED!")
        print("  -> RL successfully demonstrated a Pareto-optimal tradeoff (15% lower cost than Heuristic, 15% better resilience than MST) on BOTH configurations.")
        print("  -> Safe to proceed with Tasks 3, 4, and 5.")
    else:
        print("\n[CHECKSTOP STATUS] FAILED!")
        print("  -> Multi-objective tradeoff did not replicate cleanly on both configurations.")
        print("  -> STOPPING execution of subsequent tasks.")

if __name__ == "__main__":
    run_sweeps()
