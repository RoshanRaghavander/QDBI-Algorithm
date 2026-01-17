"""
Quantum Approximate Optimization Algorithm (QAOA) based decision model for UAV swarm coordination.
This module implements the QAOA-based quantum decision model as described in the QBDI research paper.
"""

import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import os

class QAOADecisionModel:
    """
    Implementation of the QAOA-based quantum decision model for UAV swarm coordination.
    
    This class implements the Hamiltonian function:
    HQBDI = ∑(i,j) Jij zi zj + ∑i hi zi
    
    Where:
    - zi ∈ {-1, 1} represents the decision state of UAV i
    - Jij represents interaction strength between UAVs
    - hi represents an external field (environmental influence)
    """
    
    def __init__(self, num_uavs):
        """
        Initialize the QAOA decision model.
        
        Args:
            num_uavs (int): Number of UAVs in the swarm
        """
        self.num_uavs = num_uavs
        self.J = np.zeros((num_uavs, num_uavs))  # Interaction strength matrix
        self.h = np.zeros(num_uavs)  # External field vector
        self.z = np.ones(num_uavs)  # Decision state vector, initialized to all 1s
        
    def set_interaction_strength(self, i, j, strength):
        """
        Set the interaction strength between two UAVs.
        
        Args:
            i (int): Index of first UAV
            j (int): Index of second UAV
            strength (float): Interaction strength value
        """
        self.J[i, j] = strength
        self.J[j, i] = strength  # Ensure symmetry
        
    def set_external_field(self, i, field_value):
        """
        Set the external field (environmental influence) for a UAV.
        
        Args:
            i (int): Index of UAV
            field_value (float): External field value
        """
        self.h[i] = field_value
        
    def compute_hamiltonian(self, z=None):
        """
        Compute the Hamiltonian function value.
        
        Args:
            z (numpy.ndarray, optional): Decision state vector. If None, use the current state.
            
        Returns:
            float: Value of the Hamiltonian function
        """
        if z is None:
            z = self.z
            
        interaction_term = 0
        for i in range(self.num_uavs):
            for j in range(i + 1, self.num_uavs):
                interaction_term += self.J[i, j] * z[i] * z[j]
        
        # Second term: ∑i hi zi
        field_term = np.sum(self.h * z)
        
        return interaction_term + field_term
    
    def optimize_decision_state(self, num_iterations=100, num_restarts=10):
        """
        Optimize the decision state to minimize the Hamiltonian.
        
        This implements a simplified version of QAOA by using classical optimization
        to find the minimum energy state of the Hamiltonian.
        
        Args:
            num_iterations (int): Number of optimization iterations
            
        Returns:
            numpy.ndarray: Optimized decision state vector
        """
        best_state = None
        best_energy = None
        
        for _ in range(num_restarts):
            state = 2 * np.random.randint(0, 2, size=self.num_uavs) - 1
            energy = self.compute_hamiltonian(state)
            
            for _ in range(num_iterations):
                flip_index = np.random.randint(0, self.num_uavs)
                candidate = state.copy()
                candidate[flip_index] *= -1
                candidate_energy = self.compute_hamiltonian(candidate)
                
                if candidate_energy < energy:
                    state = candidate
                    energy = candidate_energy
            
            if best_energy is None or energy < best_energy:
                best_state = state
                best_energy = energy
        
        self.z = best_state
        return best_state
    
    def calculate_energy_difference(self, optimal_state):
        """
        Calculate the energy difference between current state and optimal state.
        
        This implements: E[HQBDI] - Hopt
        
        Args:
            optimal_state (numpy.ndarray): The optimal decision state
            
        Returns:
            float: Energy difference
        """
        current_energy = self.compute_hamiltonian(self.z)
        optimal_energy = self.compute_hamiltonian(optimal_state)
        
        return current_energy - optimal_energy
    
    def visualize_decision_states(self, title="UAV Decision States"):
        """
        Visualize the current decision states of the UAVs.
        
        Args:
            title (str): Title for the visualization
        """
        plt.figure(figsize=(10, 6))
        plt.bar(range(self.num_uavs), self.z, color=['red' if z == -1 else 'blue' for z in self.z])
        plt.xlabel('UAV Index')
        plt.ylabel('Decision State')
        plt.title(title)
        plt.xticks(range(self.num_uavs))
        plt.yticks([-1, 1])
        plt.grid(True, alpha=0.3)
        results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
        os.makedirs(results_dir, exist_ok=True)
        save_path = os.path.join(results_dir, "decision_states.png")
        plt.savefig(save_path)
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create a QAOA decision model for 5 UAVs
    model = QAOADecisionModel(5)
    
    # Set interaction strengths (example values)
    for i in range(5):
        for j in range(i+1, 5):
            strength = np.random.uniform(-1, 1)
            model.set_interaction_strength(i, j, strength)
    
    # Set external field values (example values)
    for i in range(5):
        field = np.random.uniform(-0.5, 0.5)
        model.set_external_field(i, field)
    
    # Optimize decision states
    optimal_state = model.optimize_decision_state()
    
    # Print results
    print("Optimal decision state:", optimal_state)
    print("Hamiltonian value:", model.compute_hamiltonian())
    
    # Visualize decision states
    model.visualize_decision_states()
