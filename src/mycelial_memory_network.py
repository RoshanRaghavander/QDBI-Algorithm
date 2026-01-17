"""
Mycelial Memory Networks (MMN) module for QBDI framework.
This module implements the bio-inspired memory structure for UAV navigation and coordination.
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.spatial import Voronoi, voronoi_plot_2d
import os

class MycelialMemoryNetwork:
    """
    Implementation of Mycelial Memory Networks (MMN) as described in the QBDI paper.
    
    This class implements a bio-inspired memory structure that mimics mycelial networks
    for distributed information storage and retrieval in UAV swarms.
    """
    
    def __init__(self, num_uavs, memory_capacity=100):
        """
        Initialize the mycelial memory network.
        
        Args:
            num_uavs (int): Number of UAVs in the swarm
            memory_capacity (int): Maximum number of memory nodes in the network
        """
        self.num_uavs = num_uavs
        self.memory_capacity = memory_capacity
        
        # Memory nodes: position, value, timestamp, importance
        self.memory_nodes = []
        
        # Connection graph between memory nodes
        self.connections = nx.Graph()
        
        # UAV-specific memory access patterns
        self.uav_access_patterns = [[] for _ in range(num_uavs)]
        
    def add_memory_node(self, position, value, importance=1.0):
        """
        Add a new memory node to the network.
        
        Args:
            position (numpy.ndarray): 2D or 3D position vector
            value (float or numpy.ndarray): Value stored in the memory node
            importance (float): Importance of the memory node (0.0 to 1.0)
            
        Returns:
            int: Index of the added memory node
        """
        timestamp = len(self.memory_nodes)  # Use current size as timestamp
        
        # Create new memory node
        node = {
            'position': np.array(position),
            'value': value,
            'timestamp': timestamp,
            'importance': importance,
            'access_count': 0
        }
        
        # Add to memory nodes
        self.memory_nodes.append(node)
        node_idx = len(self.memory_nodes) - 1
        
        # Add to connection graph
        self.connections.add_node(node_idx)
        
        # Connect to nearby nodes (based on spatial proximity)
        self._update_connections(node_idx)
        
        # If capacity exceeded, remove least important node
        if len(self.memory_nodes) > self.memory_capacity:
            self._prune_memory()
            
        return node_idx
    
    def _update_connections(self, node_idx, connection_radius=5.0):
        """
        Update connections for a memory node.
        
        Args:
            node_idx (int): Index of the memory node
            connection_radius (float): Maximum distance for connection
        """
        new_node = self.memory_nodes[node_idx]
        
        for i, node in enumerate(self.memory_nodes):
            if i != node_idx:
                distance = np.linalg.norm(new_node['position'] - node['position'])
                
                # Connect if within radius
                if distance < connection_radius:
                    # Weight inversely proportional to distance
                    weight = 1.0 / (1.0 + distance)
                    self.connections.add_edge(node_idx, i, weight=weight)
    
    def _prune_memory(self):
        """
        Remove least important memory nodes when capacity is exceeded.
        """
        # Calculate node scores based on importance, recency, and connectivity
        scores = []
        
        for i, node in enumerate(self.memory_nodes):
            # Score based on importance, recency, and connectivity
            importance_factor = node['importance']
            recency_factor = 1.0 / (1.0 + (len(self.memory_nodes) - node['timestamp']))
            connectivity_factor = len(list(self.connections.neighbors(i))) / len(self.memory_nodes)
            access_factor = np.log1p(node['access_count'])
            
            score = importance_factor * 0.4 + recency_factor * 0.2 + connectivity_factor * 0.2 + access_factor * 0.2
            scores.append((i, score))
        
        # Sort by score (ascending)
        scores.sort(key=lambda x: x[1])
        
        # Remove node with lowest score
        node_to_remove = scores[0][0]
        
        # Remove from connection graph
        self.connections.remove_node(node_to_remove)
        
        # Remove from memory nodes
        self.memory_nodes.pop(node_to_remove)
        
        # Update node indices in graph
        mapping = {i: i if i < node_to_remove else i-1 for i in range(len(self.memory_nodes)+1)}
        self.connections = nx.relabel_nodes(self.connections, mapping)
    
    def query_memory(self, position, uav_idx=None, k=3, update_usage=True):
        """
        Query the memory network for information at a given position.
        
        Args:
            position (numpy.ndarray): Query position
            uav_idx (int, optional): Index of the querying UAV
            k (int): Number of nearest neighbors to consider
            
        Returns:
            tuple: (values, weights) - weighted values from nearby memory nodes
        """
        if not self.memory_nodes:
            return None, None
        
        # Find k nearest memory nodes
        distances = []
        for i, node in enumerate(self.memory_nodes):
            dist = np.linalg.norm(position - node['position'])
            distances.append((i, dist))
        
        # Sort by distance
        distances.sort(key=lambda x: x[1])
        nearest_indices = [idx for idx, _ in distances[:k]]
        
        # Calculate weights based on distance (inverse distance weighting)
        nearest_distances = np.array([dist for _, dist in distances[:k]])
        if np.all(nearest_distances < 1e-10):  # If query point is exactly on a memory node
            weights = np.zeros_like(nearest_distances)
            weights[0] = 1.0
        else:
            weights = 1.0 / (nearest_distances + 1e-10)
            weights = weights / np.sum(weights)  # Normalize
        
        values = [self.memory_nodes[idx]['value'] for idx in nearest_indices]
        
        if update_usage:
            for idx in nearest_indices:
                self.memory_nodes[idx]['access_count'] += 1
        
            if uav_idx is not None:
                self.uav_access_patterns[uav_idx].append(nearest_indices)
        
        return values, weights
    
    def update_memory_value(self, node_idx, new_value, importance_delta=0.0):
        """
        Update the value and importance of a memory node.
        
        Args:
            node_idx (int): Index of the memory node
            new_value: New value to store
            importance_delta (float): Change in importance
        """
        if 0 <= node_idx < len(self.memory_nodes):
            self.memory_nodes[node_idx]['value'] = new_value
            self.memory_nodes[node_idx]['importance'] += importance_delta
            
            # Ensure importance stays in [0, 1]
            self.memory_nodes[node_idx]['importance'] = max(0.0, min(1.0, self.memory_nodes[node_idx]['importance']))
    
    def visualize_memory_network(self, title="Mycelial Memory Network"):
        """
        Visualize the memory network.
        
        Args:
            title (str): Title for the visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Create position dictionary for networkx
        pos = {i: node['position'][:2] for i, node in enumerate(self.memory_nodes)}
        
        # Get node importances for node sizes
        node_sizes = [300 * node['importance'] for node in self.memory_nodes]
        
        # Get edge weights for edge widths
        edge_weights = [self.connections[u][v]['weight'] * 2 for u, v in self.connections.edges()]
        
        # Draw the graph
        nx.draw_networkx(
            self.connections,
            pos=pos,
            node_size=node_sizes,
            node_color='skyblue',
            edge_color='gray',
            width=edge_weights,
            with_labels=True,
            font_size=8,
            alpha=0.8
        )
        
        plt.title(title)
        plt.axis('off')
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "memory_network.png")
        plt.savefig(save_path)
        plt.close()
    
    def visualize_memory_heatmap(self, grid_size=20, title="Memory Value Heatmap"):
        """
        Visualize the memory values as a heatmap.
        
        Args:
            grid_size (int): Size of the grid for heatmap
            title (str): Title for the visualization
        """
        if not self.memory_nodes:
            return
            
        # Determine bounds of the environment
        positions = np.array([node['position'][:2] for node in self.memory_nodes])
        min_x, min_y = np.min(positions, axis=0) - 1
        max_x, max_y = np.max(positions, axis=0) + 1
        
        # Create grid
        x = np.linspace(min_x, max_x, grid_size)
        y = np.linspace(min_y, max_y, grid_size)
        X, Y = np.meshgrid(x, y)
        
        # Query memory at each grid point
        Z = np.zeros((grid_size, grid_size))
        for i in range(grid_size):
            for j in range(grid_size):
                position = np.array([X[i, j], Y[i, j]])
                values, weights = self.query_memory(position, update_usage=False)
                
                if values is not None and all(isinstance(v, (int, float)) for v in values):
                    # Weighted average of values
                    Z[i, j] = np.sum(np.array(values) * weights)
        
        plt.figure(figsize=(10, 8))
        plt.contourf(X, Y, Z, cmap='viridis', levels=20)
        plt.colorbar(label='Memory Value')
        
        # Plot memory node positions
        plt.scatter(positions[:, 0], positions[:, 1], c='red', s=50, marker='o')
        
        plt.title(title)
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "memory_heatmap.png")
        plt.savefig(save_path)
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create a mycelial memory network for 5 UAVs
    mmn = MycelialMemoryNetwork(5, memory_capacity=20)
    
    # Add random memory nodes
    for _ in range(15):
        position = np.random.uniform(-10, 10, size=2)
        value = np.random.uniform(0, 1)
        importance = np.random.uniform(0.3, 1.0)
        mmn.add_memory_node(position, value, importance)
    
    # Query memory at random positions
    for i in range(5):
        query_pos = np.random.uniform(-10, 10, size=2)
        values, weights = mmn.query_memory(query_pos, uav_idx=i)
        
        if values is not None:
            weighted_value = np.sum(np.array(values) * weights)
            print(f"UAV {i} query at {query_pos}: weighted value = {weighted_value:.4f}")
    
    # Visualize the memory network
    mmn.visualize_memory_network()
    mmn.visualize_memory_heatmap()
