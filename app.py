from flask import Flask, render_template, request, jsonify
import json
from gen_net_design_ai import *

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generate', methods=['POST'])
def generate_network():
    try:
        # Get parameters from request
        data = request.json
        servers = int(data.get('servers', 0))
        switches = int(data.get('switches', 0))
        routers = int(data.get('routers', 0))
        end_devices = int(data.get('end_devices', 0))
        episodes = 50  # Fixed episodes for consistent performance
        
        # Validate that at least one device is specified
        total_devices = servers + switches + routers + end_devices
        if total_devices == 0:
            return jsonify({'success': False, 'error': 'Please specify at least one device'})
        
        # Ensure minimum viable network (at least 2 devices for connections)
        if total_devices == 1:
            return jsonify({'success': False, 'error': 'Please specify at least 2 devices to create connections'})
        
        # Create device configuration (only include devices with count > 0)
        devices = []
        if servers > 0:
            devices.append(("Server", servers))
        if switches > 0:
            devices.append(("Switch", switches))
        if routers > 0:
            devices.append(("Router", routers))
        if end_devices > 0:
            devices.append(("EndDevice", end_devices))
        constraints = Constraints.from_strings(["connected", "server switch"])
        
        # Train and generate network
        policy, best_graph, best_reward = train_policy(devices, constraints, episodes=episodes)
        
        # Create node type mapping
        nodes = make_nodes(devices)
        node_types = {i: t for i, t in nodes}
        
        # Generate network data for visualization
        network_data = generate_network_data(best_graph, node_types)
        
        return jsonify({
            'success': True,
            'network': network_data,
            'stats': {
                'reward': round(best_reward, 3),
                'nodes': best_graph.number_of_nodes(),
                'edges': best_graph.number_of_edges(),
                'connected': nx.is_connected(best_graph)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def generate_network_data(graph, node_types):
    # Image mapping for device types
    images = {
        "Server": "static/assets/server.png",
        "Router": "static/assets/router.png", 
        "Switch": "static/assets/switch.png",
        "EndDevice": "static/assets/desktop.png"
    }
    
    # Hierarchical positioning
    hierarchy_levels = {
        "Server": -400,
        "Switch": -200,
        "Router": 0,
        "EndDevice": 200
    }
    
    # Group nodes by type
    nodes_by_type = {"Server": [], "Switch": [], "Router": [], "EndDevice": []}
    for node in graph.nodes():
        node_type = node_types.get(node, "Unknown")
        if node_type in nodes_by_type:
            nodes_by_type[node_type].append(node)
    
    nodes_data = []
    for node in graph.nodes():
        node_type = node_types.get(node, "Unknown")
        
        # Calculate position
        y_pos = hierarchy_levels.get(node_type, 0)
        type_nodes = nodes_by_type.get(node_type, [node])
        x_spacing = 150
        x_offset = (len(type_nodes) - 1) * x_spacing / 2
        x_pos = (type_nodes.index(node) * x_spacing) - x_offset
        
        nodes_data.append({
            'id': node,
            'label': f"{node_type}\n{node}",
            'image': images.get(node_type, ""),
            'shape': 'image',
            'size': 40,
            'x': x_pos,
            'y': y_pos,
            'physics': False
        })
    
    edges_data = []
    for edge in graph.edges():
        edges_data.append({
            'from': edge[0],
            'to': edge[1],
            'width': 2
        })
    
    return {'nodes': nodes_data, 'edges': edges_data}

if __name__ == '__main__':
    app.run(debug=True, port=5000)