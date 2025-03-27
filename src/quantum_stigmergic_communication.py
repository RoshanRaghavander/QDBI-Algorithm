"""
Quantum Stigmergic Communication (SQEC) module for QBDI framework.
This module implements the entanglement-inspired implicit coordination model.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform

class QuantumStigmergicCommunication:
    """
    Implementation of Quantum Stigmergic Communication (SQEC) as described in the QBDI paper.
    
    This class implements the implicit coordination model:
    Sij(t) = ∫E Γ(pi, pj) · χ(ei, ej)de
    
    Where:
    - Γ(pi, pj) = exp(-‖pi - pj‖²/2σ²) is the spatial correlation function
    - χ(ei, ej) measures entanglement strength between UAV states
    """
    
    def __init__(self, num_uavs, sigma=1.0):
        """
        Initialize the quantum stigmergic communication model.
        
        Args:
            num_uavs (int): Number of UAVs in the swarm
            sigma (float): Parameter for spatial correlation function
        """
        self.num_uavs = num_uavs
        self.sigma = sigma
        self.positions = np.zeros((num_uavs, 3))  # 3D positions of UAVs
        self.states = np.zeros((num_uavs, 2))     # Quantum-inspired states (2D for simplicity)
        self.communication_matrix = np.zeros((num_uavs, num_uavs))
        
    def set_uav_position(self, uav_index, position):
        """
        Set the position of a UAV.
        
        Args:
            uav_index (int): Index of the UAV
            position (numpy.ndarray): 3D position vector [x, y, z]
        """
        self.positions[uav_index] = position
        
    def set_uav_state(self, uav_index, state):
        """
        Set the quantum-inspired state of a UAV.
        
        Args:
            uav_index (int): Index of the UAV
            state (numpy.ndarray): 2D state vector
        """
        self.states[uav_index] = state / np.linalg.norm(state)  # Normalize state
        
    def calculate_spatial_correlation(self):
        """
        Calculate the spatial correlation function Γ(pi, pj) for all UAV pairs.
        
        Returns:
            numpy.ndarray: Matrix of spatial correlations
        """
        # Calculate pairwise distances between UAVs
        distances = squareform(pdist(self.positions))
        
        # Calculate spatial correlation: Γ(pi, pj) = exp(-‖pi - pj‖²/2σ²)
        spatial_correlation = np.exp(-distances**2 / (2 * self.sigma**2))
        
        # Set diagonal elements to 1 (self-correlation)
        np.fill_diagonal(spatial_correlation, 1.0)
        
        return spatial_correlation
    
    def calculate_entanglement_strength(self):
        """
        Calculate the entanglement strength χ(ei, ej) between UAV states.
        
        Returns:
            numpy.ndarray: Matrix of entanglement strengths
        """
        # Calculate dot products between normalized state vectors as a measure of entanglement
        entanglement = np.zeros((self.num_uavs, self.num_uavs))
        
        for i in range(self.num_uavs):
            for j in range(self.num_uavs):
                # Simplified entanglement measure based on state similarity
                entanglement[i, j] = np.abs(np.dot(self.states[i], self.states[j]))
        
        return entanglement
    
    def update_communication_matrix(self):
        """
        Update the communication matrix based on spatial correlation and entanglement.
        
        This implements: Sij(t) = Γ(pi, pj) · χ(ei, ej)
        (simplified from the integral form)
        
        Returns:
            numpy.ndarray: Updated communication matrix
        """
        spatial_correlation = self.calculate_spatial_correlation()
        entanglement_strength = self.calculate_entanglement_strength()
        
        # Element-wise multiplication of spatial correlation and entanglement strength
        self.communication_matrix = spatial_correlation * entanglement_strength
        
        return self.communication_matrix
    
    def get_communication_strength(self, uav_i, uav_j):
        """
        Get the communication strength between two UAVs.
        
        Args:
            uav_i (int): Index of the first UAV
            uav_j (int): Index of the second UAV
            
        Returns:
            float: Communication strength
        """
        return self.communication_matrix[uav_i, uav_j]
    
    def visualize_communication_matrix(self, title="Quantum Stigmergic Communication Matrix"):
        """
        Visualize the communication matrix.
        
        Args:
            title (str): Title for the visualization
        """
        plt.figure(figsize=(10, 8))
        plt.imshow(self.communication_matrix, cmap='viridis', interpolation='nearest')
        plt.colorbar(label='Communication Strength')
        plt.xlabel('UAV Index')
        plt.ylabel('UAV Index')
        plt.title(title)
        plt.xticks(range(self.num_uavs))
        plt.yticks(range(self.num_uavs))
        plt.savefig('/home/ubuntu/qbdi_implementation/communication_matrix.png')
        plt.close()
        
    def visualize_uav_positions(self, title="UAV Positions with Communication Links"):
        """
        Visualize UAV positions and communication links.
        
        Args:
            title (str): Title for the visualization
        """
        plt.figure(figsize=(10, 8))
        
        # Plot UAV positions
        plt.scatter(self.positions[:, 0], self.positions[:, 1], s=100, c='blue', marker='o')
        
        # Plot communication links
        for i in range(self.num_uavs):
            for j in range(i+1, self.num_uavs):
                strength = self.communication_matrix[i, j]
                if strength > 0.2:  # Only show significant links
                    plt.plot([self.positions[i, 0], self.positions[j, 0]],
                             [self.positions[i, 1], self.positions[j, 1]],
                             'k-', alpha=strength, linewidth=strength*3)
        
        # Add UAV indices
        for i in range(self.num_uavs):
            plt.text(self.positions[i, 0], self.positions[i, 1], str(i),
                     fontsize=12, ha='center', va='center', color='white',
                     bbox=dict(facecolor='blue', alpha=0.7))
        
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        plt.savefig('/home/ubuntu/qbdi_implementation/uav_positions.png')
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create a quantum stigmergic communication model for 5 UAVs
    sqec = QuantumStigmergicCommunication(5, sigma=2.0)
    
    # Set random positions for UAVs
    for i in range(5):
        position = np.random.uniform(-10, 10, size=3)
        sqec.set_uav_position(i, position)
    
    # Set random quantum-inspired states for UAVs
    for i in range(5):
        state = np.random.normal(0, 1, size=2)
        sqec.set_uav_state(i, state)
    
    # Update and print communication matrix
    comm_matrix = sqec.update_communication_matrix()
    print("Communication Matrix:")
    print(comm_matrix)
    
    # Visualize results
    sqec.visualize_communication_matrix()
    sqec.visualize_uav_positions()
