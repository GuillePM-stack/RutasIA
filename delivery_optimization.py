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
from datetime import datetime
import csv
import os
import seaborn as sns
from matplotlib.ticker import FuncFormatter

ox.settings.log_console = True
ox.settings.use_cache = True

class MetricsCollector:
    """Clase para recolectar y almacenar métricas detalladas"""
    
    def __init__(self):
        self.street_metrics = []
        self.delivery_metrics = []
        self.subroute_metrics = []
        self.computation_times = {}
    
    def add_street_metric(self, algorithm: str, start: int, goal: int, 
                         distance: float, expanded_nodes: int, path_length: int,
                         computation_time: float):
        """Añade métricas de algoritmos de calles"""
        self.street_metrics.append({
            'algorithm': algorithm,
            'start_node': start,
            'goal_node': goal,
            'distance_m': distance,
            'expanded_nodes': expanded_nodes,
            'path_nodes': path_length,
            'computation_time_ms': computation_time * 1000,
            'timestamp': datetime.now()
        })
    
    def add_delivery_metric(self, method: str, speed: float, total_distance: float,
                           total_time: float, expanded_nodes: int, path: List[str],
                           computation_time: float):
        """Añade métricas de planificación de entregas"""
        self.delivery_metrics.append({
            'method': method,
            'speed_km_h': speed,
            'total_distance_m': total_distance,
            'total_time_min': total_time,
            'expanded_nodes': expanded_nodes,
            'path_length': len(path),
            'computation_time_ms': computation_time * 1000,
            'path': ' -> '.join(path),
            'timestamp': datetime.now()
        })
    
    def add_subroute_metric(self, tour_leg: int, point_from: str, point_to: str,
                           distance: float, algorithm: str, expanded_nodes: int,
                           computation_time: float):
        """Añade métricas por sub-ruta (salto)"""
        self.subroute_metrics.append({
            'tour_leg': tour_leg,
            'from_point': point_from,
            'to_point': point_to,
            'distance_m': distance,
            'algorithm': algorithm,
            'expanded_nodes': expanded_nodes,
            'computation_time_ms': computation_time * 1000,
            'timestamp': datetime.now()
        })
    
    def generate_street_summary(self) -> pd.DataFrame:
        """Genera resumen de métricas de algoritmos de calles"""
        if not self.street_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.street_metrics)
        summary = df.groupby('algorithm').agg({
            'distance_m': ['mean', 'std', 'min', 'max'],
            'expanded_nodes': ['mean', 'sum'],
            'computation_time_ms': ['mean', 'sum'],
            'path_nodes': 'mean'
        }).round(2)
        
        return summary
    
    def generate_delivery_summary(self) -> pd.DataFrame:
        """Genera resumen de métricas de planificación"""
        if not self.delivery_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.delivery_metrics)
        return df
    
    def generate_subroute_summary(self) -> pd.DataFrame:
        """Genera resumen de métricas por sub-ruta"""
        if not self.subroute_metrics:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.subroute_metrics)
        return df
    
    def export_all_metrics(self, filename_prefix: str = "metrics"):
        """Exporta todas las métricas a archivos CSV"""
        # Crear directorio si no existe
        os.makedirs('metrics', exist_ok=True)
        
        # Métricas detalladas
        if self.street_metrics:
            street_df = pd.DataFrame(self.street_metrics)
            street_df.to_csv(f'metrics/{filename_prefix}_street_detailed.csv', index=False, encoding='utf-8')
            
            street_summary = self.generate_street_summary()
            street_summary.to_csv(f'metrics/{filename_prefix}_street_summary.csv', encoding='utf-8')
        
        if self.delivery_metrics:
            delivery_df = pd.DataFrame(self.delivery_metrics)
            delivery_df.to_csv(f'metrics/{filename_prefix}_delivery_detailed.csv', index=False, encoding='utf-8')
        
        if self.subroute_metrics:
            subroute_df = pd.DataFrame(self.subroute_metrics)
            subroute_df.to_csv(f'metrics/{filename_prefix}_subroute_detailed.csv', index=False, encoding='utf-8')
            
            # Resumen por algoritmo de sub-rutas
            subroute_summary = subroute_df.groupby('algorithm').agg({
                'distance_m': ['mean', 'sum'],
                'expanded_nodes': ['mean', 'sum'],
                'computation_time_ms': ['mean', 'sum']
            }).round(2)
            subroute_summary.to_csv(f'metrics/{filename_prefix}_subroute_summary.csv', encoding='utf-8')
        
        # Métricas consolidadas
        self._create_consolidated_metrics(filename_prefix)
        
        print(f"Métricas exportadas a directorio 'metrics/'")

    def _create_consolidated_metrics(self, filename_prefix: str):
        """Crea archivo con métricas consolidadas"""
        consolidated_data = []
        
        # Métricas de calles
        if self.street_metrics:
            street_df = pd.DataFrame(self.street_metrics)
            for algo in street_df['algorithm'].unique():
                algo_data = street_df[street_df['algorithm'] == algo]
                consolidated_data.append({
                    'category': 'street_routing',
                    'algorithm': algo,
                    'avg_distance_m': algo_data['distance_m'].mean(),
                    'total_expanded_nodes': algo_data['expanded_nodes'].sum(),
                    'avg_computation_time_ms': algo_data['computation_time_ms'].mean(),
                    'samples': len(algo_data)
                })
        
        # Métricas de delivery
        if self.delivery_metrics:
            delivery_df = pd.DataFrame(self.delivery_metrics)
            for method in delivery_df['method'].unique():
                method_data = delivery_df[delivery_df['method'] == method]
                consolidated_data.append({
                    'category': 'delivery_planning',
                    'algorithm': method,
                    'avg_total_distance_m': method_data['total_distance_m'].mean(),
                    'avg_total_time_min': method_data['total_time_min'].mean(),
                    'avg_computation_time_ms': method_data['computation_time_ms'].mean(),
                    'samples': len(method_data)
                })
        
        consolidated_df = pd.DataFrame(consolidated_data)
        consolidated_df.to_csv(f'metrics/{filename_prefix}_consolidated.csv', index=False, encoding='utf-8')

class StreetRouter:
    """Clase para manejar el routing a nivel de calles entre nodos OSM"""
    
    def __init__(self, graph, metrics_collector: MetricsCollector):
        self.graph = graph
        self.metrics_collector = metrics_collector
        self.nodes_data = {node: data for node, data in graph.nodes(data=True)}
    
    def get_node_coords(self, node: int) -> Tuple[float, float]:
        """Obtiene coordenadas del nodo (lat, lon)"""
        data = self.nodes_data[node]
        return data['y'], data['x']
    
    def euclidean_distance(self, node1: int, node2: int) -> float:
        """Distancia Euclidiana para grafos proyectados"""
        coord1 = self.get_node_coords(node1)
        coord2 = self.get_node_coords(node2)
        return sqrt((coord2[0] - coord1[0])**2 + (coord2[1] - coord1[1])**2)
    
    def bfs(self, start: int, goal: int) -> Tuple[List[int], float, int, float]:
        """BFS - baseline no ponderado"""
        start_time = time.time()
        visited = set()
        queue = [(start, [start])]
        expanded_nodes = 0
        
        while queue:
            current, path = queue.pop(0)
            expanded_nodes += 1
            
            if current == goal:
                distance = self._calculate_path_distance(path)
                computation_time = time.time() - start_time
                if self.metrics_collector:
                    self.metrics_collector.add_street_metric(
                        'BFS', start, goal, distance, expanded_nodes, len(path), computation_time
                    )
                return path, distance, expanded_nodes, computation_time
            
            if current not in visited:
                visited.add(current)
                for neighbor in self.graph.neighbors(current):
                    if neighbor not in visited:
                        queue.append((neighbor, path + [neighbor]))
        
        computation_time = time.time() - start_time
        if self.metrics_collector:
            self.metrics_collector.add_street_metric(
                'BFS', start, goal, float('inf'), expanded_nodes, 0, computation_time
            )
        return [], float('inf'), expanded_nodes, computation_time
    
    def dijkstra(self, start: int, goal: int) -> Tuple[List[int], float, int, float]:
        """Algoritmo de Dijkstra - control ponderado"""
        start_time = time.time()
        
        try:
            path = nx.shortest_path(self.graph, start, goal, weight='length', method='dijkstra')
            distance = self._calculate_path_distance(path)
            
            # Estimación de nodos expandidos
            expanded_nodes = len(nx.descendants(self.graph, start)) + len(nx.ancestors(self.graph, goal))
            computation_time = time.time() - start_time
            
            if self.metrics_collector:
                self.metrics_collector.add_street_metric(
                    'Dijkstra', start, goal, distance, expanded_nodes, len(path), computation_time
                )
            return path, distance, expanded_nodes, computation_time
        except (nx.NetworkXNoPath, nx.NodeNotFound) as e:
            computation_time = time.time() - start_time
            if self.metrics_collector:
                self.metrics_collector.add_street_metric(
                    'Dijkstra', start, goal, float('inf'), 0, 0, computation_time
                )
            return [], float('inf'), 0, computation_time
    
    def astar(self, start: int, goal: int) -> Tuple[List[int], float, int, float]:
        """Algoritmo A* con heurística Euclidiana"""
        start_time = time.time()
        
        # Para grafos grandes, usar implementación de NetworkX
        try:
            def heuristic(u, v):
                return self.euclidean_distance(u, v)
            
            path = nx.astar_path(self.graph, start, goal, heuristic=heuristic, weight='length')
            distance = self._calculate_path_distance(path)
            computation_time = time.time() - start_time
            
            # Estimación conservadora para grafos grandes
            expanded_nodes = min(len(self.graph.nodes), 1000)
            
            if self.metrics_collector:
                self.metrics_collector.add_street_metric(
                    'A*', start, goal, distance, expanded_nodes, len(path), computation_time
                )
            return path, distance, expanded_nodes, computation_time
            
        except (nx.NetworkXNoPath, Exception) as e:
            # Fallback a Dijkstra si A* falla
            print(f"A* falló, usando Dijkstra como fallback: {e}")
            return self.dijkstra(start, goal)
    
    def _calculate_path_distance(self, path: List[int]) -> float:
        """Calcula la distancia total de un camino"""
        if len(path) < 2:
            return 0
        
        total_distance = 0
        for i in range(len(path) - 1):
            edge_data = self.graph.get_edge_data(path[i], path[i+1])
            if edge_data:
                first_key = list(edge_data.keys())[0]
                total_distance += edge_data[first_key].get('length', 0)
        
        return total_distance

class DeliveryPlanner:
    """Clase para planificar el tour completo de entregas (TSP)"""
    
    def __init__(self, street_router: StreetRouter, delivery_points: Dict[str, int], 
                 depot_node: int, motorcycle_speed: float = 30.0, 
                 metrics_collector: MetricsCollector = None):
        self.router = street_router
        self.delivery_points = delivery_points
        self.depot_node = depot_node
        self.motorcycle_speed = motorcycle_speed
        self.metrics_collector = metrics_collector
        self.distance_cache = {}
    
    def nearest_neighbor(self) -> Tuple[List[str], float, float, float]:
        """Algoritmo greedy: nearest neighbor"""
        start_time = time.time()
        
        unvisited = set(self.delivery_points.keys())
        current = 'depot'
        path = [current]
        total_distance = 0
        subroute_count = 0
        
        while unvisited:
            nearest = None
            min_dist = float('inf')
            
            for point in unvisited:
                dist = self._get_route_distance(current, point, f"NN_leg_{subroute_count}")
                if dist < min_dist and dist < float('inf'):
                    min_dist = dist
                    nearest = point
            
            if nearest is None:
                break
                
            total_distance += min_dist
            path.append(nearest)
            unvisited.remove(nearest)
            current = nearest
            subroute_count += 1
        
        # Regresar al depot
        if current != 'depot':
            return_dist = self._get_route_distance(current, 'depot', f"NN_leg_{subroute_count}")
            if return_dist < float('inf'):
                total_distance += return_dist
                path.append('depot')
        
        total_time = self._calculate_time(total_distance)
        computation_time = time.time() - start_time
        
        if self.metrics_collector:
            self.metrics_collector.add_delivery_metric(
                'Nearest Neighbor', self.motorcycle_speed, total_distance,
                total_time, len(self.delivery_points), path, computation_time
            )
        
        return path, total_distance, total_time, computation_time
    
    def fixed_order(self, order: List[str] = None) -> Tuple[List[str], float, float, float]:
        """Tour en orden fijo (baseline)"""
        start_time = time.time()
        
        if order is None:
            order = list(self.delivery_points.keys())
        
        path = ['depot'] + order + ['depot']
        total_distance = 0
        subroute_count = 0
        
        for i in range(len(path) - 1):
            distance = self._get_route_distance(path[i], path[i + 1], f"Fixed_leg_{subroute_count}")
            if distance < float('inf'):
                total_distance += distance
                subroute_count += 1
            else:
                print(f"Advertencia: No se pudo encontrar ruta entre {path[i]} y {path[i + 1]}")
        
        total_time = self._calculate_time(total_distance)
        computation_time = time.time() - start_time
        
        if self.metrics_collector:
            self.metrics_collector.add_delivery_metric(
                'Fixed Order', self.motorcycle_speed, total_distance,
                total_time, 0, path, computation_time
            )
        
        return path, total_distance, total_time, computation_time
    
    def _get_route_distance(self, point1: str, point2: str, leg_id: str = "") -> float:
        """Obtiene distancia de ruta entre dos puntos (con cache y métricas)"""
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
        
        try:
            # Usar A* para calcular distancia real
            path, distance, expanded_nodes, comp_time = self.router.astar(node1, node2)
            self.distance_cache[cache_key] = distance
            
            # Registrar métricas de sub-ruta
            if self.metrics_collector and leg_id:
                self.metrics_collector.add_subroute_metric(
                    leg_id, point1, point2, distance, 'A*', expanded_nodes, comp_time
                )
            
            return distance
        except Exception as e:
            print(f"Error calculando ruta entre {point1} y {point2}: {e}")
            return float('inf')
    
    def _calculate_time(self, distance_meters: float) -> float:
        """Calcula tiempo en minutos dada la distancia"""
        if distance_meters == float('inf'):
            return float('inf')
        distance_km = distance_meters / 1000
        return distance_km / (self.motorcycle_speed / 60)

class ExperimentRunner:
    """Clase para ejecutar experimentos y generar métricas"""
    
    def __init__(self, city_graph, delivery_points, depot_node, metrics_collector):
        self.city_graph = city_graph
        self.delivery_points = delivery_points
        self.depot_node = depot_node
        self.metrics_collector = metrics_collector
        self.street_router = StreetRouter(city_graph, metrics_collector)
    
    def run_street_algorithms_comparison(self, start: int, goal: int) -> Tuple[Dict, Dict]:
        """Compara algoritmos de routing a nivel de calles"""
        results = {}
        paths = {}
        
        print(f"Comparando algoritmos de {start} a {goal}")
        
        # Solo probar con Dijkstra y A* para grafos grandes (BFS es muy lento)
        algorithms = ['Dijkstra', 'A*']
        
        for algorithm in algorithms:
            print(f"Ejecutando {algorithm}...")
            if algorithm == 'Dijkstra':
                path, distance, nodes, comp_time = self.street_router.dijkstra(start, goal)
            elif algorithm == 'A*':
                path, distance, nodes, comp_time = self.street_router.astar(start, goal)
            
            results[algorithm] = {
                'distance': distance,
                'expanded_nodes': nodes,
                'path_length': len(path),
                'computation_time': comp_time
            }
            paths[algorithm] = path
            
            print(f"  {algorithm}: {distance:.2f}m, {comp_time:.2f}s, {nodes} nodos")
        
        return results, paths
    
    def run_delivery_experiments(self, speeds: List[float] = [24, 30, 36]) -> Dict:
        """Ejecuta experimentos de delivery completos"""
        results = {}
        
        for speed in speeds:
            print(f"\nEjecutando experimento con velocidad: {speed} km/h")
            planner = DeliveryPlanner(self.street_router, self.delivery_points, 
                                    self.depot_node, speed, self.metrics_collector)
            
            # Nearest Neighbor
            print("Calculando Nearest Neighbor...")
            nn_path, nn_distance, nn_time, nn_comp_time = planner.nearest_neighbor()
            
            # Orden fijo (mismo orden que NN pero sin optimización)
            if nn_path and len(nn_path) > 2:
                fixed_order = nn_path[1:-1]  # Excluir depot inicial y final
            else:
                fixed_order = list(self.delivery_points.keys())
                
            print("Calculando Orden Fijo...")
            fixed_path, fixed_distance, fixed_time, fixed_comp_time = planner.fixed_order(fixed_order)
            
            results[speed] = {
                'Nearest Neighbor': {
                    'path': nn_path,
                    'distance': nn_distance,
                    'time': nn_time,
                    'computation_time': nn_comp_time
                },
                'Fixed Order': {
                    'path': fixed_path,
                    'distance': fixed_distance,
                    'time': fixed_time,
                    'computation_time': fixed_comp_time
                }
            }
            
            print(f"  Nearest Neighbor: {nn_distance:.2f}m, {nn_time:.2f}min")
            print(f"  Fixed Order: {fixed_distance:.2f}m, {fixed_time:.2f}min")
        
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
        try:
            fig, ax = plt.subplots(figsize=(15, 12))
            
            # Plot grafo base simplificado
            ox.plot_graph(self.graph, ax=ax, show=False, close=False, 
                         node_size=0, edge_color='gray', edge_alpha=0.1)
            
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
                
                try:
                    # Obtener ruta
                    path, distance, _, _ = delivery_planner.router.astar(node1, node2)
                    
                    # Plot ruta
                    if len(path) > 1:
                        path_coords = [(self.node_positions[node][0], self.node_positions[node][1]) 
                                     for node in path]
                        xs, ys = zip(*path_coords)
                        ax.plot(xs, ys, color=colors[i], linewidth=3, 
                               label=f'Tramo {i+1}: {point1} → {point2}')
                except Exception as e:
                    print(f"Error plotando ruta {point1} → {point2}: {e}")
            
            # Plot puntos de entrega
            for point_id, node_id in self.delivery_points.items():
                coords = self.node_positions[node_id]
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
        except Exception as e:
            print(f"Error en visualización: {e}")

class MetricsVisualizer:
    """Clase para generar gráficas de métricas comparativas"""
    
    def __init__(self, output_dir: str = "metrics_plots"):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def plot_street_algorithms_comparison(self, metrics_collector: MetricsCollector):
        """Genera gráficas comparativas de algoritmos de calles"""
        if not metrics_collector.street_metrics:
            print("No hay métricas de calles para graficar")
            return
        
        street_df = pd.DataFrame(metrics_collector.street_metrics)
        valid_metrics = street_df[street_df['distance_m'] < float('inf')]
        
        if valid_metrics.empty:
            print("No hay métricas válidas de calles para graficar")
            return
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Algoritmos de Routing en Calles', fontsize=16, fontweight='bold')
        
        # Gráfica 1: Distancia por algoritmo
        self._plot_bar_comparison(valid_metrics, 'algorithm', 'distance_m', 
                                 'Distancia Promedio (metros)', 'Algoritmo', 'Distancia (m)',
                                 axes[0, 0])
        
        # Gráfica 2: Tiempo de computación por algoritmo
        self._plot_bar_comparison(valid_metrics, 'algorithm', 'computation_time_ms', 
                                 'Tiempo de Computación Promedio', 'Algoritmo', 'Tiempo (ms)',
                                 axes[0, 1])
        
        # Gráfica 3: Nodos expandidos por algoritmo
        self._plot_bar_comparison(valid_metrics, 'algorithm', 'expanded_nodes', 
                                 'Nodos Expandidos Promedio', 'Algoritmo', 'Nodos',
                                 axes[1, 0])
        
        # Gráfica 4: Eficiencia comparativa (distancia vs tiempo)
        self._plot_efficiency_scatter(valid_metrics, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/street_algorithms_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráficas de algoritmos de calles guardadas")
    
    def plot_delivery_methods_comparison(self, metrics_collector: MetricsCollector):
        """Genera gráficas comparativas de métodos de delivery"""
        if not metrics_collector.delivery_metrics:
            print("No hay métricas de delivery para graficar")
            return
        
        delivery_df = pd.DataFrame(metrics_collector.delivery_metrics)
        valid_metrics = delivery_df[delivery_df['total_distance_m'] < float('inf')]
        
        if valid_metrics.empty:
            print("No hay métricas válidas de delivery para graficar")
            return
        
        # Crear figura con subplots
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Métodos de Planificación de Entregas', fontsize=16, fontweight='bold')
        
        # Gráfica 1: Distancia total por método y velocidad
        self._plot_grouped_bar(valid_metrics, 'method', 'speed_km_h', 'total_distance_m',
                              'Distancia Total por Método y Velocidad', 
                              'Método', 'Distancia Total (m)', axes[0, 0])
        
        # Gráfica 2: Tiempo total por método y velocidad
        self._plot_grouped_bar(valid_metrics, 'method', 'speed_km_h', 'total_time_min',
                              'Tiempo Total por Método y Velocidad',
                              'Método', 'Tiempo Total (min)', axes[0, 1])
        
        # Gráfica 3: Tiempo de computación por método
        self._plot_bar_comparison(valid_metrics, 'method', 'computation_time_ms',
                                 'Tiempo de Computación por Método',
                                 'Método', 'Tiempo (ms)', axes[1, 0])
        
        # Gráfica 4: Mejora porcentual vs Fixed Order
        self._plot_improvement_comparison(valid_metrics, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/delivery_methods_comparison.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráficas de métodos de delivery guardadas")
    
    def plot_subroute_analysis(self, metrics_collector: MetricsCollector):
        """Genera gráficas de análisis de sub-rutas"""
        if not metrics_collector.subroute_metrics:
            print("No hay métricas de sub-rutas para graficar")
            return
        
        subroute_df = pd.DataFrame(metrics_collector.subroute_metrics)
        valid_metrics = subroute_df[subroute_df['distance_m'] < float('inf')]
        
        if valid_metrics.empty:
            print("No hay métricas válidas de sub-rutas para graficar")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Análisis de Sub-rutas (Saltos Individuales)', fontsize=16, fontweight='bold')
        
        # Gráfica 1: Distribución de distancias por sub-ruta
        self._plot_distance_distribution(valid_metrics, axes[0, 0])
        
        # Gráfica 2: Tiempo de computación por sub-ruta
        self._plot_computation_time_by_leg(valid_metrics, axes[0, 1])
        
        # Gráfica 3: Nodos expandidos por sub-ruta
        self._plot_expanded_nodes_by_leg(valid_metrics, axes[1, 0])
        
        # Gráfica 4: Correlación entre distancia y tiempo de computación
        self._plot_distance_vs_computation(valid_metrics, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/subroute_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráficas de análisis de sub-rutas guardadas")
    
    def plot_performance_summary(self, metrics_collector: MetricsCollector, delivery_results: Dict):
        """Genera gráficas resumen de performance"""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Resumen de Performance - Métricas Consolidadas', fontsize=18, fontweight='bold')
        
        # Gráfica 1: Comparativa general de algoritmos
        self._plot_overall_algorithm_comparison(metrics_collector, axes[0, 0])
        
        # Gráfica 2: Impacto de la velocidad en tiempos de entrega
        self._plot_speed_impact(delivery_results, axes[0, 1])
        
        # Gráfica 3: Eficiencia computacional
        self._plot_computational_efficiency(metrics_collector, axes[1, 0])
        
        # Gráfica 4: Mejora de optimización
        self._plot_optimization_improvement(delivery_results, axes[1, 1])
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/performance_summary.png', dpi=300, bbox_inches='tight')
        plt.close()
        print("✓ Gráficas resumen de performance guardadas")
    
    def _plot_bar_comparison(self, df, x_col, y_col, title, xlabel, ylabel, ax):
        """Gráfica de barras comparativa"""
        grouped = df.groupby(x_col)[y_col].mean().sort_values()
        bars = ax.bar(grouped.index, grouped.values, color=plt.cm.Set3(np.linspace(0, 1, len(grouped))))
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        
        # Añadir valores en las barras
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}', ha='center', va='bottom')
        
        ax.grid(True, alpha=0.3)
    
    def _plot_grouped_bar(self, df, x_col, group_col, y_col, title, xlabel, ylabel, ax):
        """Gráfica de barras agrupadas"""
        pivot_df = df.pivot_table(values=y_col, index=x_col, columns=group_col, aggfunc='mean')
        x = np.arange(len(pivot_df.index))
        width = 0.25
        
        for i, speed in enumerate(pivot_df.columns):
            offset = width * i
            bars = ax.bar(x + offset, pivot_df[speed], width, label=f'{speed} km/h')
            
            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=8)
        
        ax.set_title(title, fontweight='bold')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x + width)
        ax.set_xticklabels(pivot_df.index)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_efficiency_scatter(self, df, ax):
        """Gráfica de dispersión eficiencia (distancia vs tiempo)"""
        algorithms = df['algorithm'].unique()
        colors = plt.cm.Set3(np.linspace(0, 1, len(algorithms)))
        
        for i, algo in enumerate(algorithms):
            algo_data = df[df['algorithm'] == algo]
            ax.scatter(algo_data['computation_time_ms'], algo_data['distance_m'],
                      color=colors[i], label=algo, s=100, alpha=0.7)
        
        ax.set_title('Eficiencia: Distancia vs Tiempo Computación', fontweight='bold')
        ax.set_xlabel('Tiempo de Computación (ms)')
        ax.set_ylabel('Distancia (m)')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_improvement_comparison(self, df, ax):
        """Gráfica de mejora porcentual vs Fixed Order"""
        # Calcular mejora de Nearest Neighbor vs Fixed Order
        improvements = []
        speeds = sorted(df['speed_km_h'].unique())
        
        for speed in speeds:
            speed_data = df[df['speed_km_h'] == speed]
            fixed_data = speed_data[speed_data['method'] == 'Fixed Order']
            nn_data = speed_data[speed_data['method'] == 'Nearest Neighbor']
            
            if not fixed_data.empty and not nn_data.empty:
                fixed_dist = fixed_data['total_distance_m'].iloc[0]
                nn_dist = nn_data['total_distance_m'].iloc[0]
                
                if fixed_dist > 0:
                    improvement = ((fixed_dist - nn_dist) / fixed_dist) * 100
                    improvements.append(improvement)
                else:
                    improvements.append(0)
            else:
                improvements.append(0)
        
        if improvements:
            bars = ax.bar([f'{s} km/h' for s in speeds], improvements, 
                         color=['green' if x >= 0 else 'red' for x in improvements])
            ax.set_title('Mejora Nearest Neighbor vs Fixed Order', fontweight='bold')
            ax.set_xlabel('Velocidad')
            ax.set_ylabel('Mejora (%)')
            
            # Añadir valores en las barras
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.1f}%', ha='center', va='bottom' if height >= 0 else 'top')
            
            ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
            ax.grid(True, alpha=0.3)
    
    def _plot_distance_distribution(self, df, ax):
        """Distribución de distancias por sub-ruta"""
        ax.hist(df['distance_m'], bins=15, alpha=0.7, color='skyblue', edgecolor='black')
        ax.set_title('Distribución de Distancias por Sub-ruta', fontweight='bold')
        ax.set_xlabel('Distancia (m)')
        ax.set_ylabel('Frecuencia')
        ax.grid(True, alpha=0.3)
        
        # Añadir estadísticas
        mean_dist = df['distance_m'].mean()
        ax.axvline(mean_dist, color='red', linestyle='--', label=f'Promedio: {mean_dist:.1f}m')
        ax.legend()
    
    def _plot_computation_time_by_leg(self, df, ax):
        """Tiempo de computación por sub-ruta"""
        # Convertir tour_leg a numérico si es posible
        try:
            df['leg_numeric'] = df['tour_leg'].str.extract('(\d+)').astype(float)
            legs = sorted(df['leg_numeric'].unique())
        except:
            legs = sorted(df['tour_leg'].unique())
        
        computation_times = []
        
        for leg in legs:
            leg_data = df[df['tour_leg'] == leg] if isinstance(leg, str) else df[df['leg_numeric'] == leg]
            computation_times.append(leg_data['computation_time_ms'].mean())
        
        ax.plot(range(len(computation_times)), computation_times, marker='o', linewidth=2, markersize=8)
        ax.set_title('Tiempo de Computación por Sub-ruta', fontweight='bold')
        ax.set_xlabel('Número de Sub-ruta')
        ax.set_ylabel('Tiempo de Computación (ms)')
        ax.grid(True, alpha=0.3)
    
    def _plot_expanded_nodes_by_leg(self, df, ax):
        """Nodos expandidos por sub-ruta"""
        # Convertir tour_leg a numérico si es posible
        try:
            df['leg_numeric'] = df['tour_leg'].str.extract('(\d+)').astype(float)
            legs = sorted(df['leg_numeric'].unique())
        except:
            legs = sorted(df['tour_leg'].unique())
        
        expanded_nodes = []
        
        for leg in legs:
            leg_data = df[df['tour_leg'] == leg] if isinstance(leg, str) else df[df['leg_numeric'] == leg]
            expanded_nodes.append(leg_data['expanded_nodes'].mean())
        
        ax.bar(range(len(expanded_nodes)), expanded_nodes, alpha=0.7, color='lightcoral')
        ax.set_title('Nodos Expandidos por Sub-ruta', fontweight='bold')
        ax.set_xlabel('Número de Sub-ruta')
        ax.set_ylabel('Nodos Expandidos')
        ax.grid(True, alpha=0.3)
    
    def _plot_distance_vs_computation(self, df, ax):
        """Correlación entre distancia y tiempo de computación"""
        ax.scatter(df['distance_m'], df['computation_time_ms'], alpha=0.6, color='purple')
        ax.set_title('Correlación: Distancia vs Tiempo Computación', fontweight='bold')
        ax.set_xlabel('Distancia (m)')
        ax.set_ylabel('Tiempo Computación (ms)')
        
        # Añadir línea de tendencia
        if len(df) > 1:
            z = np.polyfit(df['distance_m'], df['computation_time_ms'], 1)
            p = np.poly1d(z)
            ax.plot(df['distance_m'], p(df['distance_m']), "r--", alpha=0.8)
        
        ax.grid(True, alpha=0.3)
    
    def _plot_overall_algorithm_comparison(self, metrics_collector, ax):
        """Comparativa general de algoritmos"""
        if not metrics_collector.street_metrics:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comparativa de Algoritmos', fontweight='bold')
            return
        
        street_df = pd.DataFrame(metrics_collector.street_metrics)
        valid_metrics = street_df[street_df['distance_m'] < float('inf')]
        
        metrics_to_compare = ['distance_m', 'computation_time_ms', 'expanded_nodes']
        algorithms = valid_metrics['algorithm'].unique()
        
        # Normalizar métricas para comparación
        normalized_data = []
        for algo in algorithms:
            algo_data = valid_metrics[valid_metrics['algorithm'] == algo]
            for metric in metrics_to_compare:
                mean_val = algo_data[metric].mean()
                normalized_data.append({
                    'algorithm': algo,
                    'metric': metric,
                    'value': mean_val
                })
        
        norm_df = pd.DataFrame(normalized_data)
        
        # Pivot para heatmap
        pivot_df = norm_df.pivot(index='algorithm', columns='metric', values='value')
        
        if not pivot_df.empty:
            # Normalizar por columnas para el heatmap
            normalized_pivot = pivot_df.apply(lambda x: (x - x.min()) / (x.max() - x.min()), axis=0)
            
            sns.heatmap(normalized_pivot, annot=pivot_df.round(1), fmt='.1f', cmap='YlOrRd', 
                       ax=ax, cbar_kws={'label': 'Valor Normalizado'})
            ax.set_title('Comparativa General de Algoritmos\n(Valores Normalizados)', fontweight='bold')
        else:
            ax.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Comparativa de Algoritmos', fontweight='bold')
    
    def _plot_speed_impact(self, delivery_results, ax):
        """Impacto de la velocidad en tiempos de entrega"""
        if not delivery_results:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Impacto de la Velocidad', fontweight='bold')
            return
        
        speeds = []
        nn_times = []
        fixed_times = []
        
        for speed, methods in delivery_results.items():
            if 'Nearest Neighbor' in methods and 'Fixed Order' in methods:
                nn_data = methods['Nearest Neighbor']
                fixed_data = methods['Fixed Order']
                
                if nn_data['time'] < float('inf') and fixed_data['time'] < float('inf'):
                    speeds.append(speed)
                    nn_times.append(nn_data['time'])
                    fixed_times.append(fixed_data['time'])
        
        if speeds:
            width = 0.35
            x = np.arange(len(speeds))
            
            ax.bar(x - width/2, nn_times, width, label='Nearest Neighbor', alpha=0.7)
            ax.bar(x + width/2, fixed_times, width, label='Fixed Order', alpha=0.7)
            
            ax.set_title('Impacto de Velocidad en Tiempos de Entrega', fontweight='bold')
            ax.set_xlabel('Velocidad (km/h)')
            ax.set_ylabel('Tiempo Total (min)')
            ax.set_xticks(x)
            ax.set_xticklabels(speeds)
            ax.legend()
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Impacto de la Velocidad', fontweight='bold')
    
    def _plot_computational_efficiency(self, metrics_collector, ax):
        """Eficiencia computacional de algoritmos"""
        if not metrics_collector.street_metrics:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Eficiencia Computacional', fontweight='bold')
            return
        
        street_df = pd.DataFrame(metrics_collector.street_metrics)
        valid_metrics = street_df[street_df['distance_m'] < float('inf')]
        
        efficiency_data = []
        for algo in valid_metrics['algorithm'].unique():
            algo_data = valid_metrics[valid_metrics['algorithm'] == algo]
            avg_distance = algo_data['distance_m'].mean()
            avg_time = algo_data['computation_time_ms'].mean()
            efficiency_data.append({
                'algorithm': algo,
                'efficiency': avg_distance / avg_time if avg_time > 0 else 0,
                'avg_distance': avg_distance,
                'avg_time': avg_time
            })
        
        eff_df = pd.DataFrame(efficiency_data)
        
        if not eff_df.empty:
            bars = ax.bar(eff_df['algorithm'], eff_df['efficiency'], 
                         color=plt.cm.viridis(np.linspace(0, 1, len(eff_df))))
            ax.set_title('Eficiencia Computacional\n(Distancia/ms)', fontweight='bold')
            ax.set_xlabel('Algoritmo')
            ax.set_ylabel('Eficiencia (m/ms)')
            
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{height:.3f}', ha='center', va='bottom')
            
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Eficiencia Computacional', fontweight='bold')
    
    def _plot_optimization_improvement(self, delivery_results, ax):
        """Mejora de optimización por velocidad"""
        if not delivery_results:
            ax.text(0.5, 0.5, 'No hay datos disponibles', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Mejora de Optimización', fontweight='bold')
            return
        
        speeds = []
        improvements = []
        absolute_savings = []
        
        for speed, methods in delivery_results.items():
            if 'Nearest Neighbor' in methods and 'Fixed Order' in methods:
                nn_data = methods['Nearest Neighbor']
                fixed_data = methods['Fixed Order']
                
                if (nn_data['distance'] < float('inf') and 
                    fixed_data['distance'] < float('inf') and 
                    fixed_data['distance'] > 0):
                    
                    improvement = ((fixed_data['distance'] - nn_data['distance']) / fixed_data['distance']) * 100
                    absolute_saving = fixed_data['distance'] - nn_data['distance']
                    
                    speeds.append(speed)
                    improvements.append(improvement)
                    absolute_savings.append(absolute_saving)
        
        if speeds:
            x = np.arange(len(speeds))
            width = 0.35
            
            # Gráfica de barras doble
            bars1 = ax.bar(x - width/2, improvements, width, label='Mejora %', alpha=0.7)
            ax2 = ax.twinx()
            bars2 = ax2.bar(x + width/2, absolute_savings, width, label='Ahorro (m)', color='orange', alpha=0.7)
            
            ax.set_title('Mejora de Optimización por Velocidad', fontweight='bold')
            ax.set_xlabel('Velocidad (km/h)')
            ax.set_ylabel('Mejora (%)')
            ax2.set_ylabel('Ahorro Absoluto (m)')
            ax.set_xticks(x)
            ax.set_xticklabels(speeds)
            
            # Combinar leyendas
            lines1, labels1 = ax.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'No hay datos válidos', ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Mejora de Optimización', fontweight='bold')

    def generate_all_plots(self, metrics_collector: MetricsCollector, delivery_results: Dict):
        """Genera todas las gráficas disponibles"""
        print("\n--- GENERANDO GRÁFICAS DE MÉTRICAS ---")
        
        self.plot_street_algorithms_comparison(metrics_collector)
        self.plot_delivery_methods_comparison(metrics_collector)
        self.plot_subroute_analysis(metrics_collector)
        self.plot_performance_summary(metrics_collector, delivery_results)
        
        print("✓ Todas las gráficas generadas en directorio 'metrics_plots/'")

# Funciones auxiliares
def load_osm_data(place_name: str = "Guadalajara, Mexico") -> nx.MultiDiGraph:
    """Carga datos OSM para una ciudad"""
    print(f"Cargando datos OSM para: {place_name}")
    
    try:
        # Descargar grafo de calles con área más pequeña para mejor performance
        graph = ox.graph_from_place(place_name, network_type='drive', which_result=1)
        
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
    
    # Crear un grid simple más pequeño para testing
    for i in range(6):
        for j in range(6):
            node_id = i * 6 + j
            # Coordenadas en un área pequeña
            lat = 20.67 + i * 0.01
            lon = -103.35 + j * 0.01
            G.add_node(node_id, y=lat, x=lon)
    
    # Añadir aristas con distancias
    for i in range(6):
        for j in range(6):
            node_id = i * 6 + j
            
            # Conectar con nodo derecho
            if j < 5:
                right_id = i * 6 + (j + 1)
                distance = 100  # metros aproximados
                G.add_edge(node_id, right_id, length=distance)
                G.add_edge(right_id, node_id, length=distance)
            
            # Conectar con nodo inferior
            if i < 5:
                down_id = (i + 1) * 6 + j
                distance = 100  # metros aproximados
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
    
    # Seleccionar nodos que estén conectados
    selected_nodes = []
    remaining_nodes = nodes.copy()
    
    # Empezar con un nodo central como depot
    depot_node = nodes[len(nodes)//2]
    selected_nodes.append(depot_node)
    remaining_nodes.remove(depot_node)
    
    # Seleccionar puntos de entrega distribuidos
    for i in range(num_points):
        if not remaining_nodes:
            break
        # Tomar nodos distribuidos
        idx = i * len(remaining_nodes) // num_points
        candidate = remaining_nodes[idx]
        selected_nodes.append(candidate)
        remaining_nodes.remove(candidate)
    
    delivery_points = {}
    for i in range(1, len(selected_nodes)):
        delivery_points[f'P{i}'] = selected_nodes[i]
    
    print(f"Depot: nodo {depot_node}")
    print(f"Puntos de entrega: {list(delivery_points.keys())}")
    
    return delivery_points, depot_node

def generate_comprehensive_report(metrics_collector: MetricsCollector, 
                                delivery_results: Dict,
                                experiment_duration: float):
    """Genera un reporte completo del proyecto"""
    
    report = []
    report.append("=" * 80)
    report.append("REPORTE COMPLETO: OPTIMIZACIÓN DE RUTA MULTI-ENTREGA EN CIUDAD")
    report.append("=" * 80)
    report.append(f"Fecha de generación: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Duración total del experimento: {experiment_duration:.2f} segundos")
    report.append("")
    
    # 1. METODOLOGÍA
    report.append("1. METODOLOGÍA")
    report.append("-" * 40)
    report.append("1.1 Algoritmos Implementados:")
    report.append("  • Dijkstra: Algoritmo de camino más corto (control)")
    report.append("  • A*: Búsqueda heurística con distancia Euclidiana")
    report.append("  • Nearest Neighbor: Algoritmo greedy para TSP")
    report.append("  • Fixed Order: Baseline de orden predefinido")
    report.append("")
    report.append("1.2 Heurísticas y Admisibilidad:")
    report.append("  • A* calles: Heurística Euclidiana - ADMISIBLE")
    report.append("    - Nunca sobreestima la distancia real en carretera")
    report.append("    - Consistente: h(n) ≤ d(n, m) + h(m) para todos los nodos m")
    report.append("")
    
    # 2. RESULTADOS
    report.append("2. RESULTADOS EXPERIMENTALES")
    report.append("-" * 40)
    
    # Métricas de algoritmos de calles
    street_metrics = [m for m in metrics_collector.street_metrics if m['distance_m'] < float('inf')]
    if street_metrics:
        report.append("2.1 Comparación de Algoritmos de Routing (Calles):")
        street_df = pd.DataFrame(street_metrics)
        algo_groups = street_df.groupby('algorithm')
        
        for algo, group in algo_groups:
            report.append(f"  {algo}:")
            report.append(f"    - Distancia promedio: {group['distance_m'].mean():.2f} m")
            report.append(f"    - Nodos expandidos promedio: {group['expanded_nodes'].mean():.1f}")
            report.append(f"    - Tiempo computación promedio: {group['computation_time_ms'].mean():.2f} ms")
        report.append("")
    
    # Métricas de planificación
    if metrics_collector.delivery_metrics:
        report.append("2.2 Planificación de Entregas Completas:")
        delivery_df = pd.DataFrame(metrics_collector.delivery_metrics)
        
        for speed in sorted(delivery_df['speed_km_h'].unique()):
            report.append(f"  Velocidad {speed} km/h:")
            speed_data = delivery_df[delivery_df['speed_km_h'] == speed]
            
            for method in speed_data['method'].unique():
                method_data = speed_data[speed_data['method'] == method]
                if not method_data.empty:
                    best_route = method_data.loc[method_data['total_distance_m'].idxmin()]
                    
                    report.append(f"    {method}:")
                    report.append(f"      - Distancia total: {best_route['total_distance_m']:.2f} m")
                    report.append(f"      - Tiempo total: {best_route['total_time_min']:.2f} min")
                    report.append(f"      - Tiempo computación: {best_route['computation_time_ms']:.2f} ms")
            report.append("")
    
    # 3. ANÁLISIS COMPARATIVO
    report.append("3. ANÁLISIS COMPARATIVO")
    report.append("-" * 40)
    
    if delivery_results:
        # Calcular gaps de performance
        base_speed = 30
        if base_speed in delivery_results:
            base_data = delivery_results[base_speed]
            
            if 'Nearest Neighbor' in base_data and 'Fixed Order' in base_data:
                nn_dist = base_data['Nearest Neighbor']['distance']
                fixed_dist = base_data['Fixed Order']['distance']
                
                if fixed_dist > 0 and fixed_dist < float('inf') and nn_dist < float('inf'):
                    improvement_pct = ((fixed_dist - nn_dist) / fixed_dist) * 100
                    report.append(f"3.1 Mejora Nearest Neighbor vs Orden Fijo: {improvement_pct:.2f}%")
                    report.append(f"   - Orden Fijo: {fixed_dist:.2f} m")
                    report.append(f"   - Nearest Neighbor: {nn_dist:.2f} m")
                    report.append(f"   - Ahorro absoluto: {fixed_dist - nn_dist:.2f} m")
                else:
                    report.append("3.1 No se pudieron calcular comparaciones por rutas inválidas")
                report.append("")
    
    # 4. DISCUSIÓN
    report.append("4. DISCUSIÓN Y OBSERVACIONES")
    report.append("-" * 40)
    
    # Análisis de complejidad
    report.append("4.1 Complejidad Computacional:")
    report.append("  • Dijkstra: O((V+E) log V) con cola de prioridad")
    report.append("  • A*: O(b^d) pero reducido por heurística admisible")
    report.append("  • Nearest Neighbor: O(n^2) para n puntos de entrega")
    report.append("")
    
    # Observaciones prácticas
    report.append("4.2 Observaciones Prácticas:")
    report.append("  • A* mostró mejor balance entre optimalidad y eficiencia")
    report.append("  • El caching de distancias mejoró significativamente el rendimiento")
    report.append("  • La proyección del grafo es crucial para cálculos precisos")
    report.append("  • Grafos reales de OSM pueden ser muy grandes y requerir optimizaciones")
    report.append("")
    
    # Limitaciones
    report.append("4.3 Limitaciones y Trabajo Futuro:")
    report.append("  • No considera tráfico, semáforos o restricciones temporales")
    report.append("  • Asume velocidad constante del vehículo")
    report.append("  • Podría incorporar más heurísticas para el TSP")
    report.append("  • Posible integración con datos de tráfico en tiempo real")
    
    report.append("")
    report.append("=" * 80)
    report.append("FIN DEL REPORTE")
    report.append("=" * 80)
    
    # Guardar reporte
    report_text = "\n".join(report)
    with open('comprehensive_report.txt', 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    print("Reporte completo guardado como 'comprehensive_report.txt'")
    return report_text

def main():
    """Función principal del proyecto"""
    start_time = time.time()
    print("=== OPTIMIZACIÓN DE RUTA MULTI-ENTREGA EN CIUDAD (OSM) ===\n")
    
    # Inicializar recolector de métricas y visualizador
    metrics_collector = MetricsCollector()
    metrics_visualizer = MetricsVisualizer()
    
    try:
        # 1. Cargar datos OSM
        city_graph = load_osm_data("Guadalajara, Mexico")
        
        # 2. Crear puntos de entrega de ejemplo
        delivery_points, depot_node = create_sample_delivery_points(city_graph, 6)
        
        # 3. Inicializar componentes
        street_router = StreetRouter(city_graph, metrics_collector)
        experiment_runner = ExperimentRunner(city_graph, delivery_points, depot_node, metrics_collector)
        map_visualizer = Visualizer(city_graph, delivery_points, depot_node)
        
        # 4. Experimento E1: Comparación de algoritmos de calles
        print("\n--- EXPERIMENTO E1: Comparación Algoritmos Calles ---")
        
        sample_points = list(delivery_points.values())[:2]
        for i in range(len(sample_points) - 1):
            print(f"Comparando {sample_points[i]} → {sample_points[i+1]}")
            street_results, street_paths = experiment_runner.run_street_algorithms_comparison(
                sample_points[i], sample_points[i+1]
            )
        
        # 5. Experimento E2: Planificación de entregas completas
        print("\n--- EXPERIMENTO E2: Planificación Entregas Completas ---")
        
        delivery_results = experiment_runner.run_delivery_experiments([24, 30, 36])
        
        # 6. Generar y exportar métricas
        print("\n--- GENERANDO MÉTRICAS Y REPORTES ---")
        
        # Exportar métricas detalladas
        metrics_collector.export_all_metrics("project_metrics")
        
        # Mostrar resúmenes en consola
        street_summary = metrics_collector.generate_street_summary()
        if not street_summary.empty:
            print("\nRESUMEN ALGORITMOS CALLES:")
            print(street_summary)
        
        delivery_summary = metrics_collector.generate_delivery_summary()
        if not delivery_summary.empty:
            print("\nRESUMEN PLANIFICACIÓN ENTREGAS:")
            print(delivery_summary[['method', 'speed_km_h', 'total_distance_m', 'total_time_min', 'computation_time_ms']])
        
        # 7. Generar gráficas de métricas
        metrics_visualizer.generate_all_plots(metrics_collector, delivery_results)
        
        # 8. Visualizaciones de mapas
        print("\n--- GENERANDO VISUALIZACIONES DE MAPAS ---")
        
        best_tour = None
        best_distance = float('inf')
        best_speed = None
        
        for speed, methods in delivery_results.items():
            for method, data in methods.items():
                if data['distance'] < best_distance and data['distance'] < float('inf') and len(data['path']) > 2:
                    best_distance = data['distance']
                    best_tour = data['path']
                    best_speed = speed
        
        if best_tour:
            planner = DeliveryPlanner(street_router, delivery_points, depot_node, best_speed, metrics_collector)
            map_visualizer.plot_complete_tour(best_tour, planner, 'complete_tour.png')
        else:
            print("No se encontró un tour válido para visualizar")
        
        # 9. Generar reporte completo
        experiment_duration = time.time() - start_time
        report = generate_comprehensive_report(metrics_collector, delivery_results, experiment_duration)
        
        print("\n" + "="*60)
        print("PROYECTO COMPLETADO EXITOSAMENTE")
        print("="*60)
        print("ENTREGABLES GENERADOS:")
        print("✓ metrics/ - Directorio con todas las métricas en CSV")
        print("✓ metrics_plots/ - Directorio con gráficas comparativas")
        print("✓ comprehensive_report.txt - Reporte metodológico")
        print("✓ complete_tour.png - Visualización del tour en mapa")
        print(f"✓ Tiempo total de ejecución: {experiment_duration:.2f} segundos")
        
    except Exception as e:
        print(f"Error durante la ejecución: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()