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
        topology = data.get('topology', 'hybrid')
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
        
        # Generate network based on topology
        if topology == 'random':
            import random
            topology_options = ['hybrid', 'star', 'ring', 'mesh', 'tree', 'bus']
            topology = random.choice(topology_options)
        
        best_graph, node_types = generate_topology(devices, topology)
        best_reward = 1.0  # Fixed reward for predefined topologies
        
        # Create node type mapping
        nodes = make_nodes(devices)
        node_types = {i: t for i, t in nodes}
        
        # Generate network data for visualization
        network_data = generate_network_data(best_graph, node_types, topology)
        
        return jsonify({
            'success': True,
            'network': network_data,
            'stats': {
                'nodes': best_graph.number_of_nodes(),
                'edges': best_graph.number_of_edges(),
                'connected': nx.is_connected(best_graph)
            }
        })
        
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

def get_topology_recommendation(servers, switches, routers, end_devices):
    """Recommend best topology based on device parameters"""
    total_devices = servers + switches + routers + end_devices
    
    # Check for hierarchical structure (servers + switches + lower devices)
    has_hierarchy = servers > 0 and switches > 0 and (routers > 0 or end_devices > 0)
    
    # Check for peer-to-peer (no servers or switches)
    is_peer_to_peer = servers == 0 and switches == 0
    
    # Check for single device type dominance
    device_types = sum([1 for count in [servers, switches, routers, end_devices] if count > 0])
    
    if total_devices <= 2:
        return {
            "message": "Bus topology is recommended for very small networks (≤2 devices) with simple linear connection.",
            "topology": "bus"
        }
    elif total_devices == 3:
        return {
            "message": "Star topology is recommended for small networks (3 devices) as it's simple and centralized.",
            "topology": "star"
        }
    elif has_hierarchy and device_types >= 3:
        return {
            "message": "Tree topology is recommended for hierarchical networks with servers, switches, and lower-level devices.",
            "topology": "tree"
        }
    elif total_devices >= 8 and device_types >= 2:
        return {
            "message": "Mesh topology is recommended for large networks (≥8 devices) to ensure high redundancy and fault tolerance.",
            "topology": "mesh"
        }
    elif is_peer_to_peer and total_devices >= 4:
        return {
            "message": "Ring topology is recommended for peer-to-peer networks without hierarchical structure.",
            "topology": "ring"
        }
    elif total_devices >= 4 and total_devices <= 7 and device_types >= 2:
        return {
            "message": "Hybrid topology is recommended for medium networks that need both centralized and distributed connectivity.",
            "topology": "hybrid"
        }
    elif servers > 0 and total_devices <= 6:
        return {
            "message": "Star topology is recommended when you have servers that need to centrally connect to other devices.",
            "topology": "star"
        }
    else:
        return {
            "message": "Bus topology is recommended for simple linear connections between devices.",
            "topology": "bus"
        }

def generate_network_data(graph, node_types, topology='hybrid'):
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
    
    if topology == 'star':
        # Circular arrangement for star topology
        import math
        all_nodes = list(graph.nodes())
        n = len(all_nodes)
        
        if n > 1:
            # Center node (first node)
            center_node = all_nodes[0]
            center_type = node_types.get(center_node, "Unknown")
            nodes_data.append({
                'id': center_node,
                'label': f"{center_type}\n{center_node}",
                'image': images.get(center_type, ""),
                'shape': 'image',
                'size': 50,  # Larger center node
                'x': 0,
                'y': 0,
                'physics': False
            })
            
            # Arrange other nodes in circle
            radius = 300
            for i, node in enumerate(all_nodes[1:]):
                angle = 2 * math.pi * i / (n - 1)
                x_pos = radius * math.cos(angle)
                y_pos = radius * math.sin(angle)
                
                node_type = node_types.get(node, "Unknown")
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
        else:
            # Single node case
            node = all_nodes[0]
            node_type = node_types.get(node, "Unknown")
            nodes_data.append({
                'id': node,
                'label': f"{node_type}\n{node}",
                'image': images.get(node_type, ""),
                'shape': 'image',
                'size': 40,
                'x': 0,
                'y': 0,
                'physics': False
            })
    
    elif topology == 'ring':
        # Ring arrangement - all nodes in a circle
        import math
        all_nodes = list(graph.nodes())
        n = len(all_nodes)
        
        radius = 250
        for i, node in enumerate(all_nodes):
            angle = 2 * math.pi * i / n
            x_pos = radius * math.cos(angle)
            y_pos = radius * math.sin(angle)
            
            node_type = node_types.get(node, "Unknown")
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
    

    else:
        # Default hierarchical layout for other topologies
        for node in graph.nodes():
            node_type = node_types.get(node, "Unknown")
            
            # Calculate position with wider spacing for mesh topology
            y_pos = hierarchy_levels.get(node_type, 0)
            type_nodes = nodes_by_type.get(node_type, [node])
            
            # Use wider spacing for mesh topology
            if topology == 'mesh':
                x_spacing = 250  # Wider spacing for mesh
            else:
                x_spacing = 150  # Normal spacing
                
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