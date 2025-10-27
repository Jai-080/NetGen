# NetGen - AI-Powered Network Design Generator

Generative AI for automated network topology design using reinforcement learning.

## Features

- **Constraint-based Design**: Supports tree topology, connectivity, and device-specific rules
- **Device Types**: Servers, Routers, Switches with customizable connection rules
- **AI Learning**: REINFORCE algorithm learns optimal network topologies
- **Interactive Visualization**: PyVis-powered network diagrams with custom device images
- **Export Options**: JSON and HTML outputs for further analysis

## Quick Start

1. **Install Dependencies**:
```bash
pip install networkx torch pyvis
```

2. **Run the Generator**:
```bash
python gen_net_design_ai.py
```

3. **View Results**:
   - Open `network_visualization.html` in your browser
   - Check `best_network.json` for network data

## Configuration

Modify the `main()` function to customize:
- Device counts: `[("Server", 3), ("Router", 2), ("Switch", 2)]`
- Constraints: `["tree", "connected", "servers_connect_only_to_switches"]`
- Training epochs and learning parameters

## Output

The AI generates:
- Optimal network topology respecting all constraints
- Interactive visualization with custom device images
- JSON export for integration with other tools

## Example Network

```
Server0 ──┐
Server1 ──┤── Switch6 ──┐
          │             │
          └── Router4 ──┤── Switch5 ── Server2
                        │
                        └── Router3
```

## License

MIT License - Feel free to use and modify!