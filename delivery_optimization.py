import osmnx as ox
import networkx as nx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from haversine import haversine
import time
import itertools
from math import radians, sin, cos, sqrt, atan2
from typing import Dict, List, Tuple, Set, Optional
import contextily as ctx
import heapq

ox.settings.log_console = True
ox.settings.use_cache = True

class StreetRouter:
    """Clase para manejar el routing a nivel de calles entre nodos OSM"""
    
    def __init__(self, graph, projected: bool = True):
        self.graph = graph
        self.projected = projected
        self.nodes_data = {node: data for node, data in graph.nodes(data=True)}
        
        # Si el grafo está proyectado, necesitamos las coordenadas originales para Haversine
        if projected and 'lat' not in list(self.nodes_data.values())[0]:
            print("Advertencia: Grafo proyectado sin coordenadas originales")
    
    def get_node_coords(self, node: int) -> Tuple[float, float]:
        """Obtiene coordenadas del nodo (lat, lon)"""
        data = self.nodes_data[node]
        if self.projected and 'y' in data and 'x' in data:
            # Para grafos proyectados, y=lat, x=lon
            return data['y'], data['x']
        elif 'lat' in data and 'lon' in data:
            return data['lat'], data['lon']
        else:
            return data['y'], data['x']
    
    def haversine_distance(self, node1: int, node2: int) -> float:
        """Calcula distancia Haversine entre dos nodos (para heurística A*)"""
        try:
            coord1 = self.get_node_coords(node1)
            coord2 = self.get_node_coords(node2)
            return haversine(coord1, coord2, unit='m')
        except Exception as e:
            print(f"Error en Haversine: {e}")
            # Fallback: distancia Euclidiana aproximada
            x1, y1 = self.get_node_coords(node1)
            x2, y2 = self.get_node_coords(node2)
            return sqrt((x2 - x1)**2 + (y2 - y1)**2) * 111000  # Aproximación
    
    def euclidean_distance(self, node1: int, node2: int) -> float:
        """Distancia Euclidiana para grafos proyectados"""
        coord1 = self.get_node_coords(node1)
        coord2 = self.get_node_coords(node2)
        return sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    
    def bfs(self, start: int, goal: int) -> Tuple[List[int], float, int]:
        """BFS - baseline no ponderado"""
        visited = set()
        queue = [(start, [start])]
        expanded_nodes = 0
        
        while queue:
            current, path = queue.pop(0)
            expanded_nodes += 1
            
            if current == goal:
                distance = self._calculate_path_distance(path)
                return path, distance, expanded_nodes
            
            if current not in visited:
                visited.add(current)
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        return [], float('inf'), expanded_nodes
    
    def dfs(self, start: int, goal: int) -> Tuple[List[int], float, int]:
        """DFS - baseline no ponderado"""
        visited = set()
        stack = [(start, [start])]
        expanded_nodes = 0
        
        while stack:
            current, path = stack.pop()
            expanded_nodes += 1
            
            if current == goal:
                distance = self._calculate_path_distance(path)
                return path, distance, expanded_nodes
            
            if current not in visited:
                visited.add(current)
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        stack.append((neighbor, path + [neighbor]))
        
        return [], float('inf'), expanded_nodes
    
    def dijkstra(self, start: int, goal: int) -> Tuple[List[int], float, int]:
        """Algoritmo de Dijkstra - control ponderado"""
        try:
            path = nx.shortest_path(self.graph, start, goal, weight='length', method='dijkstra')
            distance = self._calculate_path_distance(path)
            
            # Para Dijkstra, estimamos nodos expandidos como el número de nodos en el grafo conectado
            expanded_nodes = len(nx.descendants(self.graph, start)) + len(nx.ancestors(self.graph, goal))
            
            return path, distance, expanded_nodes
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            print(f"Error en Dijkstra: {e}")
            return [], float('inf'), 0
    
    def astar(self, start: int, goal: int) -> Tuple[List[int], float, int]:
        """Algoritmo A* con heurística Euclidiana/Haversine"""
        start_time = time.time()
        expanded_nodes = 0
        
        # Usar cola de prioridad
        open_set = []
        heapq.heappush(open_set, (0, start))
        
        came_from = {}
        g_score = {node: float('inf') for node in self.graph.nodes()}
        g_score[start] = 0
        
        f_score = {node: float('inf') for node in self.graph.nodes()}
        f_score[start] = self.euclidean_distance(start, goal)
        
        open_set_hash = {start}
        
        while open_set:
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            expanded_nodes += 1
            
            if current == goal:
                # Reconstruir camino
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                distance = self._calculate_path_distance(path)
                return path, distance, expanded_nodes
            
            for neighbor in self.graph.neighbors(current):
                # Calcular distancia hasta el vecino
                edge_data = self.graph.get_edge_data(current, neighbor)
                if not edge_data:
                    continue
                    
                # Tomar la primera arista
                first_key = list(edge_data.keys())[0]
                edge_length = edge_data[first_key].get('length', float('inf'))
                
                if edge_length == float('inf'):
                    continue
                
                tentative_g_score = g_score[current] + edge_length
                
                if tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f_score[neighbor] = tentative_g_score + self.euclidean_distance(neighbor, goal)
                    
                    if neighbor not in open_set_hash:
                        heapq.heappush(open_set, (f_score[neighbor], neighbor))
                        open_set_hash.add(neighbor)
        
        return [], float('inf'), expanded_nodes
    
    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calcula la distancia total de un camino"""
        if len(path) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                # Tomar la primera arista (puede haber múltiples en grafos multidigraph)
                first_key = list(edge_data.keys())[0]
                total_distance += edge_data[first_key].get('length', 0)
        
        return total_distance

class DeliveryPlanner:
    """Clase para planificar el tour completo de entregas (TSP)"""
    
    def __init__(self, street_router: StreetRouter, delivery_points: Dict[str, int], 
                 depot_node: int, motorcycle_speed: float = 30.0):
        self.router = street_router
        self.delivery_points = delivery_points  # {point_id: node_id}
        self.depot_node = depot_node
        self.motorcycle_speed = motorcycle_speed  # km/h
        self.distance_cache = {}  # Cache para distancias entre puntos
    
    def get_direct_distance(self, point1: str, point2: str) -> float:
        """Distancia en línea recta entre dos puntos de entrega"""
        key = tuple(sorted([point1, point2]))
        if key in self.distance_cache:
            return self.distance_cache[key]
        
        if point1 == 'depot':
            node1 = self.depot_node
        else:
            node1 = self.delivery_points[point1]
        
        if point2 == 'depot':
            node2 = self.depot_node
        else:
            node2 = self.delivery_points[point2]
        
        distance = self.router.euclidean_distance(node1, node2)
        self.distance_cache[key] = distance
        return distance
    
    def mst_heuristic(self, current_point: str, unvisited: Set[str]) -> float:
        """Heurística MST para A* del TSP"""
        if not unvisited:
            return self.get_direct_distance(current_point, 'depot')
        
        # Crear grafo completo de puntos no visitados + depot
        points = list(unvisited) + ['depot']
        n = len(points)
        
        if n == 1:
            return self.get_direct_distance(current_point, points[0])
        
        # Calcular todas las aristas
        edges = []
        for i in range(n):
            for j in range(i + 1, n):
                dist = self.get_direct_distance(points[i], points[j])
                edges.append((dist, points[i], points[j]))
        
        # Ordenar aristas por distancia
        edges.sort()
        
        # Kruskal's algorithm para MST
        parent = {point: point for point in points}
        
        def find(point):
            if parent[point] != point:
                parent[point] = find(parent[point])
            return parent[point]
        
        def union(point1, point2):
            root1, root2 = find(point1), find(point2)
            if root1 != root2:
                parent[root2] = root1
                return True
            return False
        
        mst_weight = 0
        count = 0
        
        for dist, u, v in edges:
            if union(u, v):
                mst_weight += dist
                count += 1
                if count == n - 1:
                    break
        
        # Añadir distancia desde punto actual al MST
        if unvisited:
            min_dist_to_mst = min(self.get_direct_distance(current_point, p) for p in unvisited)
            mst_weight += min_dist_to_mst
        
        return mst_weight
    
    def nearest_neighbor(self) -> Tuple[List[str], float, float]:
        """Algoritmo greedy: nearest neighbor"""
        unvisited = set(self.delivery_points.keys())
        current = 'depot'
        path = [current]
        total_distance = 0
        
        while unvisited:
            # Encontrar el más cercano usando distancia real
            nearest = None
            min_dist = float('inf')
            
            for point in unvisited:
                dist = self._get_route_distance(current, point)
                if dist < min_dist:
                    min_dist = dist
                    nearest = point
            
            if nearest is None:
                break
                
            total_distance += min_dist
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
        
        # Regresar al depot
        return_dist = self._get_route_distance(current, 'depot')
        total_distance += return_dist
        path.append('depot')
        total_time = self._calculate_time(total_distance)
        
        return path, total_distance, total_time
    
    def fixed_order(self, order: List[str]) -> Tuple[List[str], float, float]:
        """Tour en orden fijo (baseline)"""
        path = ['depot'] + order + ['depot']
        total_distance = 0
        
        for i in range(len(path) - 1):
            total_distance += self._get_route_distance(path[i], path[i + 1])
        
        total_time = self._calculate_time(total_distance)
        return path, total_distance, total_time
    
    def _get_route_distance(self, point1: str, point2: str) -> float:
        """Obtiene distancia de ruta entre dos puntos (con cache)"""
        cache_key = (point1, point2)
        if cache_key in self.distance_cache:
            return self.distance_cache[cache_key]
        
        if point1 == 'depot':
            node1 = self.depot_node
        else:
            node1 = self.delivery_points[point1]
        
        if point2 == 'depot':
            node2 = self.depot_node
        else:
            node2 = self.delivery_points[point2]
        
        # Usar A* para calcular distancia real
        _, distance, _ = self.router.astar(node1, node2)
        self.distance_cache[cache_key] = distance
        return distance
    
    def _calculate_time(self, distance_meters: float) -> float:
        """Calcula tiempo en minutos dada la distancia"""
        distance_km = distance_meters / 1000
        return distance_km / (self.motorcycle_speed / 60)

class ExperimentRunner:
    """Clase para ejecutar experimentos y generar métricas"""
    
    def __init__(self, city_graph, delivery_points, depot_node):
        self.city_graph = city_graph
        self.delivery_points = delivery_points
        self.depot_node = depot_node
        self.street_router = StreetRouter(city_graph, projected=True)
    
    def run_street_algorithms_comparison(self, start: int, goal: int) -> Dict:
        """Compara algoritmos de routing a nivel de calles"""
        results = {}
        paths = {}
        
        print(f"Comparando algoritmos de {start} a {goal}")
        
        # BFS
        print("Ejecutando BFS...")
        bfs_path, bfs_distance, bfs_nodes = self.street_router.bfs(start, goal)
        results['BFS'] = {
            'distance': bfs_distance,
            'expanded_nodes': bfs_nodes,
            'path_length': len(bfs_path)
        }
        paths['BFS'] = bfs_path
        
        # DFS
        print("Ejecutando DFS...")
        dfs_path, dfs_distance, dfs_nodes = self.street_router.dfs(start, goal)
        results['DFS'] = {
            'distance': dfs_distance,
            'expanded_nodes': dfs_nodes,
            'path_length': len(dfs_path)
        }
        paths['DFS'] = dfs_path
        
        # Dijkstra
        print("Ejecutando Dijkstra...")
        dijkstra_path, dijkstra_distance, dijkstra_nodes = self.street_router.dijkstra(start, goal)
        results['Dijkstra'] = {
            'distance': dijkstra_distance,
            'expanded_nodes': dijkstra_nodes,
            'path_length': len(dijkstra_path)
        }
        paths['Dijkstra'] = dijkstra_path
        
        # A*
        print("Ejecutando A*...")
        astar_path, astar_distance, astar_nodes = self.street_router.astar(start, goal)
        results['A*'] = {
            'distance': astar_distance,
            'expanded_nodes': astar_nodes,
            'path_length': len(astar_path)
        }
        paths['A*'] = astar_path
        
        return results, paths
    
    def run_delivery_experiments(self, speeds: List[float] = [24, 30, 36]) -> Dict:
        """Ejecuta experimentos de delivery completos"""
        results = {}
        
        for speed in speeds:
            print(f"\nEjecutando experimento con velocidad: {speed} km/h")
            planner = DeliveryPlanner(self.street_router, self.delivery_points, 
                                    self.depot_node, speed)
            
            # Nearest Neighbor
            print("Calculando Nearest Neighbor...")
            nn_path, nn_distance, nn_time = planner.nearest_neighbor()
            
            # Orden fijo (mismo orden que NN pero sin optimización)
            fixed_order = nn_path[1:-1]  # Excluir depot inicial y final
            print("Calculando Orden Fijo...")
            fixed_path, fixed_distance, fixed_time = planner.fixed_order(fixed_order)
            
            results[speed] = {
                'Nearest Neighbor': {
                    'path': nn_path,
                    'distance': nn_distance,
                    'time': nn_time
                },
                'Fixed Order': {
                    'path': fixed_path,
                    'distance': fixed_distance,
                    'time': fixed_time
                }
            }
        
        return results

class Visualizer:
    """Clase para visualizar resultados en mapas"""
    
    def __init__(self, graph, delivery_points, depot_node):
        self.graph = graph
        self.delivery_points = delivery_points
        self.depot_node = depot_node
        self.node_positions = {node: (data['x'], data['y']) 
                             for node, data in graph.nodes(data=True)}
    
    def plot_complete_tour(self, tour_path: List[str], delivery_planner: DeliveryPlanner, 
                          filename: str = 'complete_tour.png'):
        """Visualiza el tour completo de entregas"""
        fig, ax = plt.subplots(figsize=(15, 12))
        
        # Plot grafo base
        ox.plot_graph(self.graph, ax=ax, show=False, close=False, 
                     node_size=0, edge_color='gray', edge_alpha=0.3)
        
        # Plot rutas entre puntos
        colors = plt.cm.Set3(np.linspace(0, 1, len(tour_path) - 1))
        
        for i in range(len(tour_path) - 1):
            point1, point2 = tour_path[i], tour_path[i + 1]
            
            if point1 == 'depot':
                node1 = self.depot_node
            else:
                node1 = self.delivery_points[point1]
            
            if point2 == 'depot':
                node2 = self.depot_node
            else:
                node2 = self.delivery_points[point2]
            
            # Obtener ruta
            path, distance, _ = delivery_planner.router.astar(node1, node2)
            
            # Plot ruta
            if len(path) > 1:
                path_coords = [(self.node_positions[node][0], self.node_positions[node][1]) 
                             for node in path]
                xs, ys = zip(*path_coords)
                ax.plot(xs, ys, color=colors[i], linewidth=3, 
                       label=f'Tramo {i+1}: {point1} → {point2}')
        
        # Plot puntos de entrega
        delivery_coords = []
        for point_id, node_id in self.delivery_points.items():
            coords = self.node_positions[node_id]
            delivery_coords.append((point_id, coords[0], coords[1]))
            ax.scatter([coords[0]], [coords[1]], color='blue', s=150, zorder=5)
            ax.annotate(f'{point_id}', (coords[0], coords[1]), xytext=(5, 5), 
                       textcoords='offset points', fontsize=12, fontweight='bold')
        
        # Depot
        depot_coords = self.node_positions[self.depot_node]
        ax.scatter([depot_coords[0]], [depot_coords[1]], color='red', 
                  s=200, marker='s', label='Almacén', zorder=5)
        
        ax.set_title('Tour Completo de Entregas', fontsize=16, fontweight='bold')
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Tour completo guardado como '{filename}'")
    
    def plot_street_comparison(self, paths_dict: Dict, filename: str = 'street_comparison.png'):
        """Compara diferentes algoritmos de routing en calles"""
        # Filtrar algoritmos que encontraron ruta
        valid_algorithms = {alg: path for alg, path in paths_dict.items() 
                          if path and len(path) > 1}
        
        if not valid_algorithms:
            print("No hay rutas válidas para comparar")
            return
        
        n_algorithms = len(valid_algorithms)
        fig, axes = plt.subplots(2, 2, figsize=(20, 16))
        axes = axes.flatten()
        
        for idx, (algorithm, path) in enumerate(valid_algorithms.items()):
            if idx >= len(axes):
                break
                
            ax = axes[idx]
            ox.plot_graph(self.graph, ax=ax, show=False, close=False, 
                         node_size=0, edge_color='gray', edge_alpha=0.3)
            
            path_coords = [(self.node_positions[node][0], self.node_positions[node][1]) 
                         for node in path]
            xs, ys = zip(*path_coords)
            ax.plot(xs, ys, color='red', linewidth=4, label='Ruta')
            
            # Marcar inicio y fin
            start_coords = self.node_positions[path[0]]
            end_coords = self.node_positions[path[-1]]
            
            ax.scatter([start_coords[0]], [start_coords[1]], color='green', 
                      s=150, marker='o', label='Inicio', zorder=5)
            ax.scatter([end_coords[0]], [end_coords[1]], color='blue', 
                      s=150, marker='s', label='Fin', zorder=5)
            
            ax.set_title(f'{algorithm}', fontsize=14, fontweight='bold')
            ax.legend()
        
        # Ocultar ejes vacíos
        for idx in range(n_algorithms, len(axes)):
            axes[idx].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Comparación de calles guardada como '{filename}'")

def load_osm_data(place_name: str = "Guadalajara, Mexico") -> nx.MultiDiGraph:
    """Carga datos OSM para una ciudad"""
    print(f"Cargando datos OSM para: {place_name}")
    
    try:
        # Descargar grafo de calles (sin proyectar inicialmente)
        graph = ox.graph_from_place(place_name, network_type='drive')
        
        print(f"Grafo original: {len(graph.nodes)} nodos, {len(graph.edges)} aristas")
        
        # Proyectar a metros para cálculos de distancia precisos
        graph_proj = ox.project_graph(graph)
        
        print(f"Grafo proyectado: {len(graph_proj.nodes)} nodos")
        return graph_proj
    
    except Exception as e:
        print(f"Error cargando OSM: {e}")
        print("Creando grafo de ejemplo...")
        return create_sample_graph()

def create_sample_graph() -> nx.MultiDiGraph:
    """Crea un grafo de ejemplo para testing"""
    G = nx.MultiDiGraph()
    
    # Crear un grid simple
    nodes = []
    for i in range(8):
        for j in range(8):
            node_id = i * 8 + j
            # Coordenadas en un área pequeña de Guadalajara
            lat = 20.67 + i * 0.005
            lon = -103.35 + j * 0.005
            G.add_node(node_id, y=lat, x=lon)
            nodes.append(node_id)
    
    # Añadir aristas con distancias realistas
    for i in range(8):
        for j in range(8):
            node_id = i * 8 + j
            
            # Conectar con nodo derecho
            if j < 7:
                right_id = i * 8 + (j + 1)
                distance = haversine((20.67 + i*0.005, -103.35 + j*0.005), 
                                   (20.67 + i*0.005, -103.35 + (j+1)*0.005), unit='m')
                G.add_edge(node_id, right_id, length=distance)
                G.add_edge(right_id, node_id, length=distance)
            
            # Conectar con nodo inferior
            if i < 7:
                down_id = (i + 1) * 8 + j
                distance = haversine((20.67 + i*0.005, -103.35 + j*0.005), 
                                   (20.67 + (i+1)*0.005, -103.35 + j*0.005), unit='m')
                G.add_edge(node_id, down_id, length=distance)
                G.add_edge(down_id, node_id, length=distance)
    
    print(f"Grafo de ejemplo creado: {len(G.nodes)} nodos, {len(G.edges)} aristas")
    return G

def create_sample_delivery_points(graph: nx.MultiDiGraph, num_points: int = 6) -> Tuple[Dict, int]:
    """Crea puntos de entrega de ejemplo"""
    nodes = list(graph.nodes())
    
    if len(nodes) < num_points + 1:
        print(f"Grafo muy pequeño, usando todos los nodos disponibles")
        num_points = min(num_points, len(nodes) - 1)
    
    # Seleccionar nodos aleatorios pero asegurando que estén conectados
    selected_nodes = []
    remaining_nodes = nodes.copy()
    
    # Empezar con un nodo aleatorio como depot
    depot_node = np.random.choice(remaining_nodes)
    selected_nodes.append(depot_node)
    remaining_nodes.remove(depot_node)
    
    # Seleccionar puntos de entrega que estén conectados al depot
    for i in range(num_points):
        if not remaining_nodes:
            break
            
        # Preferir nodos que estén conectados al grafo principal
        candidate = np.random.choice(remaining_nodes)
        selected_nodes.append(candidate)
        remaining_nodes.remove(candidate)
    
    delivery_points = {}
    for i in range(1, len(selected_nodes)):
        delivery_points[f'P{i}'] = selected_nodes[i]
    
    print(f"Depot: nodo {depot_node}")
    print(f"Puntos de entrega: {list(delivery_points.keys())}")
    
    return delivery_points, depot_node

def main():
    """Función principal del proyecto"""
    print("=== OPTIMIZACIÓN DE RUTA MULTI-ENTREGA EN CIUDAD (OSM) ===\n")
    
    # 1. Cargar datos OSM
    city_graph = load_osm_data("Guadalajara, Mexico")
    
    # 2. Crear puntos de entrega de ejemplo
    delivery_points, depot_node = create_sample_delivery_points(city_graph, 6)
    
    # 3. Inicializar componentes
    experiment_runner = ExperimentRunner(city_graph, delivery_points, depot_node)
    visualizer = Visualizer(city_graph, delivery_points, depot_node)
    
    # 4. Experimento E1: Comparación de algoritmos de calles
    print("\n--- EXPERIMENTO E1: Comparación Algoritmos Calles ---")
    
    # Seleccionar dos puntos que estén conectados
    sample_points = list(delivery_points.values())[:2]
    if len(sample_points) >= 2:
        street_results, street_paths = experiment_runner.run_street_algorithms_comparison(
            sample_points[0], sample_points[1]
        )
        
        # Mostrar resultados
        street_df = pd.DataFrame(street_results).T
        print("\nResultados Algoritmos Calles:")
        print(street_df)
        
        # Guardar visualización
        visualizer.plot_street_comparison(street_paths, 'street_comparison.png')
    else:
        print("No hay suficientes puntos para comparación de algoritmos de calles")
        street_results, street_paths = {}, {}
    
    # 5. Experimento E2: Planificación de entregas completas
    print("\n--- EXPERIMENTO E2: Planificación Entregas Completas ---")
    
    delivery_results = experiment_runner.run_delivery_experiments([24, 30, 36])
    
    # Procesar resultados
    summary_data = []
    for speed, methods in delivery_results.items():
        for method, data in methods.items():
            summary_data.append({
                'Velocidad (km/h)': speed,
                'Método': method,
                'Distancia (m)': data['distance'],
                'Tiempo (min)': data['time']
            })
    
    summary_df = pd.DataFrame(summary_data)
    print("\nResumen Entregas Completas:")
    print(summary_df)
    
    # 6. Visualizaciones
    print("\n--- GENERANDO VISUALIZACIONES ---")
    
    # Tour completo (usar mejor ruta encontrada)
    best_tour = None
    best_distance = float('inf')
    best_speed = None
    
    for speed, methods in delivery_results.items():
        for method, data in methods.items():
            if data['distance'] < best_distance and len(data['path']) > 2:
                best_distance = data['distance']
                best_tour = data['path']
                best_speed = speed
    
    if best_tour:
        planner = DeliveryPlanner(experiment_runner.street_router, delivery_points, 
                                depot_node, best_speed)
        visualizer.plot_complete_tour(best_tour, planner, 'complete_tour.png')
    else:
        print("No se encontró un tour válido para visualizar")
    
    # 7. Exportar resultados
    print("\n--- EXPORTANDO RESULTADOS ---")
    
    if street_results:
        street_df.to_csv('street_algorithms_results.csv')
        print("✓ Resultados de algoritmos calles exportados a 'street_algorithms_results.csv'")
    
    summary_df.to_csv('delivery_planning_results.csv')
    print("✓ Resultados de planificación exportados a 'delivery_planning_results.csv'")
    
    # 8. Análisis final
    print("\n=== ANÁLISIS DE RESULTADOS ===")
    
    if 30 in delivery_results:
        base_results = delivery_results[30]
        greedy_distance = base_results['Nearest Neighbor']['distance']
        fixed_distance = base_results['Fixed Order']['distance']
        
        if greedy_distance < float('inf'):
            gap = ((fixed_distance - greedy_distance) / fixed_distance) * 100
            print(f"Mejora Nearest Neighbor vs Orden Fijo: {gap:.2f}%")
        
        print(f"\nDistancia total más corta: {best_distance:.2f} metros")
        print(f"Tiempo estimado: {best_distance/1000/(30/60):.2f} minutos (a 30 km/h)")
    
    print("\n¡Proyecto completado exitosamente!")

if __name__ == "__main__":
    main()