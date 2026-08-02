# NetGen — AI-Powered Network Design Generator

Automated network topology design using reinforcement learning (REINFORCE).

## Architecture

NetGen has two operational modes:

| Mode | Entry point | How it works |
|------|-------------|--------------|
| **Web app** | `app.py` | Six deterministic topology generators **plus** an RL mode that runs REINFORCE to construct a connected graph edge-by-edge. Policies are cached on disk and reloaded automatically. |
| **Standalone** | `gen_net_design_ai.py` | Trains from scratch, compares against a random-policy baseline, saves training curves, and opens a PyVis visualization. |

## Quick Start

```bash
# Install all dependencies
pip install flask networkx torch pyvis matplotlib

# Run the web app
python app.py
# Open: http://localhost:5000

# Or run the standalone RL training demo
python gen_net_design_ai.py
```

### Using the virtual environment (recommended)

```bash
# One-time setup
setup_venv.bat

# Start a shell with the venv active, then run
activate_venv.bat
python app.py
```

## Features

- **RL mode**: REINFORCE policy gradient with entropy bonus and EMA baseline trains a graph-construction policy and caches it to `checkpoints/`
- **Six topology types**: star, ring, mesh, tree, bus, hybrid (rule-based, deterministic)
- **Real suitability scores**: computed from actual graph metrics — connectivity, edge density, average path length — not heuristics
- **Training curves**: `training_curve.png` plots RL reward and success rate vs. random baseline

## RL Implementation Details

| Component | Detail |
|-----------|--------|
| Algorithm | REINFORCE (Monte-Carlo policy gradient) |
| Baseline | Exponential moving average of episode returns (cross-episode variance reduction) |
| Exploration | Entropy bonus `H(π)` in the loss; `entropy_coef=0.01` |
| Policy | Node-embedding MLP → edge-scoring MLP over `[h_u, h_v, \|h_u−h_v\|]` |
| State | Per-node features: degree + 4-dim device-type one-hot |

## Hyperparameters

All configurable as arguments to `train_policy()`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `episodes` | 200 | Training episodes |
| `lr` | 1e-3 | Adam learning rate |
| `gamma` | 0.99 | Discount factor |
| `entropy_coef` | 0.01 | Entropy bonus weight |
| `baseline_alpha` | 0.1 | EMA learning rate for the baseline |

## Testing

```bash
pip install pytest
pytest tests/test_netgen.py -v
```

## Output Files

| File | Description |
|------|-------------|
| `training_curve.png` | RL vs. random baseline reward and success rate |
| `best_network.json` | Best graph found during standalone training |
| `network.html` | Interactive PyVis visualization (standalone mode) |
| `checkpoints/` | Persisted policy weights (`.pt` files), loaded on next request |

## License

MIT License — Feel free to use and modify!