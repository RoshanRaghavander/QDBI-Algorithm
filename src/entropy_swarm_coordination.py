"""
Entropy-Based Swarm Coordination module for QBDI framework.
This module implements the Shannon entropy function for measuring swarm coherence.
"""

import numpy as np
import matplotlib.pyplot as plt
import os

class EntropySwarmCoordination:
    """
    Implementation of entropy-based swarm coordination as described in the QBDI paper.
    
    This class implements the Shannon entropy function:
    Hswarm(t) = -∑i Pi log Pi
    
    Where Pi represents the probability of UAV i following an optimal trajectory.
    """
    
    def __init__(self, num_uavs):
        """
        Initialize the entropy-based swarm coordination.
        
        Args:
            num_uavs (int): Number of UAVs in the swarm
        """
        self.num_uavs = num_uavs
        self.trajectory_probabilities = np.ones(num_uavs) / num_uavs  # Initialize with uniform probabilities
        self.entropy_history = []
        
    def update_trajectory_probability(self, uav_index, probability):
        """
        Update the probability of a UAV following an optimal trajectory.
        
        Args:
            uav_index (int): Index of the UAV
            probability (float): Probability value between 0 and 1
        """
        if probability < 0 or probability > 1:
            raise ValueError("Probability must be between 0 and 1")
            
        self.trajectory_probabilities[uav_index] = probability
        
        # Normalize probabilities to ensure they sum to 1
        self.trajectory_probabilities = self.trajectory_probabilities / np.sum(self.trajectory_probabilities)
        
    def calculate_swarm_entropy(self):
        """
        Calculate the Shannon entropy of the swarm.
        
        Returns:
            float: Entropy value
        """
        # Filter out zero probabilities to avoid log(0)
        valid_probs = self.trajectory_probabilities[self.trajectory_probabilities > 0]
        
        # Calculate entropy: -∑i Pi log Pi
        swarm_entropy = -np.sum(valid_probs * np.log(valid_probs))
        
        # Store in history
        self.entropy_history.append(swarm_entropy)
        
        return swarm_entropy
    
    def get_entropy_history(self):
        """
        Get the history of entropy values.
        
        Returns:
            list: History of entropy values
        """
        return self.entropy_history
    
    def visualize_entropy_history(self, title="Swarm Entropy Over Time"):
        """
        Visualize the history of entropy values.
        
        Args:
            title (str): Title for the visualization
        """
        plt.figure(figsize=(10, 6))
        plt.plot(self.entropy_history, marker='o', linestyle='-')
        plt.xlabel('Time Step')
        plt.ylabel('Swarm Entropy')
        plt.title(title)
        plt.grid(True, alpha=0.3)
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "entropy_history.png")
        plt.savefig(save_path)
        plt.close()
        
    def visualize_trajectory_probabilities(self, title="UAV Trajectory Probabilities"):
        """
        Visualize the current trajectory probabilities of the UAVs.
        
        Args:
            title (str): Title for the visualization
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.num_uavs), self.trajectory_probabilities)
        plt.xlabel('UAV Index')
        plt.ylabel('Probability of Optimal Trajectory')
        plt.title(title)
        plt.xticks(range(self.num_uavs))
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "trajectory_probabilities.png")
        plt.savefig(save_path)
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create an entropy-based swarm coordination for 5 UAVs
    swarm_coord = EntropySwarmCoordination(5)
    
    # Simulate 10 time steps
    for t in range(10):
        # Update trajectory probabilities (example values)
        for i in range(5):
            # In a real scenario, these would be based on the UAV's performance
            # Here we're just using random values for demonstration
            prob = np.random.uniform(0.1, 1.0)
            swarm_coord.update_trajectory_probability(i, prob)
        
        # Calculate and print entropy
        entropy = swarm_coord.calculate_swarm_entropy()
        print(f"Time step {t}, Swarm entropy: {entropy:.4f}")
    
    # Visualize results
    swarm_coord.visualize_entropy_history()
    swarm_coord.visualize_trajectory_probabilities()
