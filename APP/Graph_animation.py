import streamlit as st
from streamlit_force_graph_simulator import ForceGraphSimulation, st_graph
import networkx as nx

G = nx.erdos_renyi_graph(5,0.8,directed=True)
F = ForceGraphSimulation(G)

F.add_node(5)
F.add_edge(4,5)
F.add_edge(3,5)
F.save_event()

F.add_node(6)
F.add_edge(5,6)
F.save_event()

F.remove_node(5)
F.save_event()

props = {
    'height':300,
    'cooldownTicks':1000 ,
    'linkDirectionalArrowLength':3.5,
    'linkDirectionalArrowRelPos':1
}

st_graph(
    F.initial_graph_json,
    F.events,
    time_interval = 1000,
    graphprops=props,
    continuous_play = True,
    directed = True,
    key='my_graph'
)