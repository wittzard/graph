import matplotlib.pyplot as plt
import networkx as nx
from matplotlib.animation import FuncAnimation
import numpy as np

# Create a sample graph with 12 nodes
nodes = {f'Node{i}': (np.cos(2 * np.pi * i / 12), np.sin(2 * np.pi * i / 12)) for i in range(12)}
edges = [(f'Node{i}', f'Node{(i+1) % 12}') for i in range(12)]
edges += [(f'Node{i}', f'Node{(i+2) % 12}') for i in range(12)]

G = nx.Graph()
G.add_nodes_from(nodes.keys())
G.add_edges_from(edges)

# Define routes for each iteration with 12 nodes
iterations = {
    0: [[0, 7, 0], [0, 5, 4, 1, 6, 8, 0]],
    1: [[0, 11, 10, 0], [0, 2, 1, 4, 0], [0, 6, 0]],
    2: [[0, 5, 7, 0], [0, 8, 10, 0], [0, 3, 4, 6, 2, 0]],
    3: [[0, 7, 0], [0, 8, 10, 0], [0, 6, 1, 4, 3, 0]],
    4: [[0, 3, 1, 2, 0], [0, 11, 10, 0], [0, 5, 4, 6, 0]],
    5: [[0, 9, 8, 11, 0], [0, 5, 7, 0], [0, 2, 1, 4, 6, 0]],
    6: [[0, 2, 1, 6, 3, 0], [0, 5, 7, 0], [0, 8, 9, 0]],
    7: [[0, 2, 1, 6, 3, 0], [0, 5, 7, 0], [0, 8, 9, 10, 0]],
    8: [[0, 5, 7, 0], [0, 2, 1, 4, 6, 0], [0, 8, 9, 10, 11, 0]],
    9: [[0, 5, 7, 0], [0, 3, 6, 1, 2, 0], [0, 11, 10, 0]],
    10: [[0, 8, 11, 10, 0], [0, 5, 2, 1, 4, 0]],
    11: [[0, 7, 0], [0, 8, 11, 10, 0], [0, 5, 4, 1, 0]],
    12: [[0, 7, 0], [0, 4, 1, 6, 0], [0, 10, 11, 0]],
    13: [[0, 7, 0], [0, 8, 9, 10, 11, 0], [0, 6, 3, 0]],
    14: [[0, 8, 9, 0], [0, 4, 6, 0], [0, 10, 11, 0], [0, 5, 3, 1, 2, 0]],
    15: [[0, 7, 0], [0, 11, 10, 9, 0], [0, 3, 4, 6, 1, 2, 0]],
    16: [[0, 8, 9, 0], [0, 6, 4, 0], [0, 10, 0], [0, 5, 3, 1, 2, 0]],
    17: [[0, 2, 1, 3, 0], [0, 11, 6, 4, 0]],
    18: [[0, 5, 3, 1, 2, 0], [0, 8, 11, 10, 0], [0, 6, 0]],
    19: [[0, 7, 0], [0, 2, 1, 4, 3, 0], [0, 8, 9, 10, 0]]
}

# Initialize plot
fig, ax = plt.subplots(figsize=(10, 10))

# Draw the graph
pos = nodes

# Define colors for vehicles
colors = ['green', 'orange', 'blue', 'red']

# Function to update the animation
def update(frame):
    iteration = frame // len(routes)
    step = frame % len(routes)

    ax.clear()  # Clear previous frame
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.5)
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color='blue', node_size=500)
    nx.draw_networkx_labels(G, pos, ax=ax, font_size=12, font_weight='bold')

    # Plot the path for each vehicle in the current iteration
    routes = [[f'Node{node}' for node in route] for route in iterations[iteration]]
    for idx, route in enumerate(routes):
        path = route[:step + 1]
        if len(path) > 1:
            nx.draw_networkx_edges(G, pos, edgelist=[(path[i], path[i + 1]) for i in range(len(path) - 1)], ax=ax,
                                   edge_color=colors[idx % len(colors)], width=2, alpha=0.7)
        
        # Plot the vehicle's position
        nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color=colors[idx % len(colors)], node_size=500, label=f'Vehicle {idx+1}')

    # Add iteration label
    ax.text(0.95, 0.95, f'Iteration: {iteration}', transform=ax.transAxes, fontsize=14,
            verticalalignment='top', horizontalalignment='right', bbox=dict(facecolor='white', alpha=0.8, edgecolor='black'))

    # Add a legend for vehicle colors
    handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'Vehicle {idx+1}') for idx, color in enumerate(colors[:len(routes)])]
    ax.legend(handles=handles)

# Create the animation
total_frames = sum(len(route) for routes in iterations.values() for route in routes)
ani = FuncAnimation(fig, update, frames=total_frames, repeat=False, interval=1000)

# Show the animation
plt.show()
