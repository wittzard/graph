import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np

# Create a larger sample graph with 10 nodes
nodes = {f'Node{i}': (np.cos(2 * np.pi * i / 10), np.sin(2 * np.pi * i / 10)) for i in range(10)}
edges = [(f'Node{i}', f'Node{(i+1) % 10}') for i in range(10)]
edges += [(f'Node{i}', f'Node{(i+2) % 10}') for i in range(10)]

G = nx.Graph()
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Define routes for two vehicles
routes = {
    'vehicle_1': ['Node0', 'Node1', 'Node2', 'Node3', 'Node4', 'Node5', 'Node6', 'Node7', 'Node8', 'Node9', 'Node0'],
    'vehicle_2': ['Node5', 'Node6', 'Node7', 'Node8', 'Node9', 'Node0', 'Node1', 'Node2', 'Node3', 'Node4', 'Node5']
}

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

# Draw the graph
pos = nodes
nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
nx.draw_networkx_nodes(G, pos, ax=ax, node_color='blue', node_size=500)
nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

# Set plot limits
ax.set_xlim(-1.5, 1.5)
ax.set_ylim(-1.5, 1.5)

# Define colors for vehicles
colors = {'vehicle_1': 'green', 'vehicle_2': 'orange'}

# Function to update the animation
def update(frame):
    ax.clear()  # Clear previous frame
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='blue', node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

    # Plot the path for each vehicle
    for vehicle, route in routes.items():
        path = route[:frame + 1]
        if len(path) > 1:
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)], ax=ax,
                                   edge_color=colors[vehicle], width=2, alpha=0.7)
        
        # Plot the vehicle's position
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color=colors[vehicle], node_size=500, label=vehicle)

    # Add a legend for vehicle colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Vehicle 1'),
               plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Vehicle 2')]
    ax.legend(handles=handles)

# Create the animation
ani = FuncAnimation(fig, update, frames=max(len(route) for route in routes.values()), repeat=False, interval=500)

# Show the animation
plt.show()
