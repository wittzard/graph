import simpy
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import matplotlib.animation as animation

# ข้อมูลพื้นฐาน
time_matrix = np.array([
    [0, 3, 2, 3, 5, 6, 2, 4, 4, 5],
    [3, 0, 2, 4, 3, 4, 3, 2, 3, 2],
    [2, 2, 0, 3, 4, 4, 1, 3, 2, 4],
    [3, 4, 3, 0, 6, 6, 2, 5, 3, 6],
    [5, 3, 4, 6, 0, 2, 5, 1, 3, 0],
    [6, 4, 4, 6, 2, 0, 5, 2, 3, 2],
    [2, 3, 1, 2, 5, 5, 0, 3, 2, 4],
    [4, 2, 3, 5, 1, 2, 3, 0, 2, 1],
    [4, 3, 2, 3, 3, 3, 2, 2, 0, 3],
    [5, 2, 4, 6, 0, 2, 4, 1, 3, 0]
])

# ข้อมูลลูกค้า
customers = [i for i in range(len(time_matrix))]

# ข้อมูลรถยนต์
num_vehicles = 4
depot_index = 0
max_time_per_vehicle = 30  # เวลาสูงสุดที่รถยนต์สามารถใช้

class Vehicle:
    def __init__(self, env, id, depot_index, max_time):
        self.env = env
        self.id = id
        self.current_location = depot_index
        self.route = []
        self.time_spent = 0
        self.max_time = max_time
        self.finished = False
        self.served_customers = []
        self.positions = [depot_index]  # บันทึกตำแหน่งการเดินทางของรถยนต์

    def travel_to(self, customer_index):
        travel_time = time_matrix[self.current_location][customer_index]
        yield self.env.timeout(travel_time)
        self.time_spent += travel_time
        self.current_location = customer_index
        self.positions.append(customer_index)

    def serve_customer(self, customer):
        service_time = 1  # เวลาการให้บริการคงที่
        yield self.env.timeout(service_time)
        self.time_spent += service_time
        self.served_customers.append(customer)

    def run(self):
        while not self.finished:
            if not self.route:
                yield self.env.timeout(1)  # รอการมอบหมายใหม่
                continue

            next_customer_index = self.route.pop(0)
            
            # เดินทางไปยังลูกค้า
            yield self.env.process(self.travel_to(next_customer_index))
            
            # ให้บริการลูกค้า
            yield self.env.process(self.serve_customer(next_customer_index))
            
            # ตรวจสอบว่ารถยนต์ต้องกลับไปที่ depot หรือไม่
            if not self.route or self.time_spent + time_matrix[self.current_location][depot_index] > self.max_time:
                yield self.env.process(self.travel_to(depot_index))
                self.finished = True

class NearestNeighborStrategy:
    def __init__(self, time_matrix, depot_index):
        self.time_matrix = time_matrix
        self.depot_index = depot_index

    def choose_route(self, vehicle, unvisited_customers):
        current_location = vehicle.current_location
        route = []
        while unvisited_customers:
            nearest_customer = min(
                unvisited_customers,
                key=lambda cust: self.time_matrix[current_location][cust]
            )
            route.append(nearest_customer)
            unvisited_customers.remove(nearest_customer)
            current_location = nearest_customer
        return route

def vrptw_simulation(env):
    vehicles = [Vehicle(env, i, depot_index, max_time_per_vehicle) for i in range(num_vehicles)]
    strategy = NearestNeighborStrategy(time_matrix, depot_index)
    
    unvisited_customers = [c for c in customers if c != depot_index]
    
    for vehicle in vehicles:
        vehicle.route = strategy.choose_route(vehicle, unvisited_customers.copy())
        env.process(vehicle.run())

    while any(not v.finished for v in vehicles):
        yield env.timeout(1)

    all_results = {
        'served_customers': {v.id: v.served_customers for v in vehicles},
        'positions': {v.id: v.positions for v in vehicles}
    }
    return all_results

# สร้างสภาพแวดล้อมการจำลอง
env = simpy.Environment()
results_process = env.process(vrptw_simulation(env))
env.run()

# เก็บผลลัพธ์
results = results_process.value

# สร้างกราฟ
G = nx.Graph()
for i in range(len(time_matrix)):
    G.add_node(i)

for i in range(len(time_matrix)):
    for j in range(len(time_matrix)):
        if i != j:
            G.add_edge(i, j, weight=time_matrix[i][j])

# สร้างแอนิเมชัน
fig, ax = plt.subplots()

def update(num):
    ax.clear()
    pos = nx.spring_layout(G)
    nx.draw(G, pos, ax=ax, with_labels=True, node_color='lightblue', edge_color='gray', node_size=500, font_size=10)
    
    for vehicle_id, positions in results['positions'].items():
        route = positions
        route_edges = [(route[i], route[i+1]) for i in range(len(route)-1)]
        nx.draw_networkx_edges(G, pos, edgelist=route_edges, edge_color='r', width=2)
    
    ax.set_title(f"Time: {num}")

ani = animation.FuncAnimation(fig, update, frames=range(0, 40), repeat=False)

plt.show()
