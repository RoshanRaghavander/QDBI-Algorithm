"""
UAV Swarm Simulation Environment for QBDI framework.
This module integrates all QBDI components into a cohesive simulation environment.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
from tqdm import tqdm
import time
import os

# Import QBDI components
from qaoa_decision_model import QAOADecisionModel
from entropy_swarm_coordination import EntropySwarmCoordination
from quantum_stigmergic_communication import QuantumStigmergicCommunication
from mycelial_memory_network import MycelialMemoryNetwork

class UAV:
    """
    UAV class representing a single drone in the swarm.
    """
    
    def __init__(self, uav_id, initial_position, max_speed=1.0):
        """
        Initialize a UAV.
        
        Args:
            uav_id (int): Unique identifier for the UAV
            initial_position (numpy.ndarray): Initial 3D position [x, y, z]
            max_speed (float): Maximum speed of the UAV
        """
        self.id = uav_id
        self.position = np.array(initial_position)
        self.velocity = np.zeros(3)
        self.max_speed = max_speed
        self.target = None
        self.decision_state = 1  # Binary decision state (1 or -1)
        self.trajectory_probability = 0.5  # Probability of following optimal trajectory
        self.quantum_state = np.random.normal(0, 1, size=2)  # Quantum-inspired state
        self.quantum_state = self.quantum_state / np.linalg.norm(self.quantum_state)
        self.history = [np.copy(initial_position)]
        
    def set_target(self, target_position):
        """
        Set target position for the UAV.
        
        Args:
            target_position (numpy.ndarray): Target 3D position
        """
        self.target = np.array(target_position)
        
    def update_position(self, dt=0.1):
        """
        Update UAV position based on current velocity.
        
        Args:
            dt (float): Time step
        """
        if self.target is not None:
            # Direction to target
            direction = self.target - self.position
            distance = np.linalg.norm(direction)
            
            if distance > 0.1:  # If not already at target
                # Normalize direction and scale by max speed
                direction = direction / distance * self.max_speed
                self.velocity = direction
            else:
                self.velocity = np.zeros(3)
        
        # Update position
        self.position += self.velocity * dt
        
        # Record history
        self.history.append(np.copy(self.position))
        
    def get_position(self):
        """
        Get current position of the UAV.
        
        Returns:
            numpy.ndarray: Current 3D position
        """
        return self.position
    
    def get_history(self):
        """
        Get position history of the UAV.
        
        Returns:
            list: List of 3D positions
        """
        return self.history


class Obstacle:
    """
    Obstacle class representing a static obstacle in the environment.
    """
    
    def __init__(self, position, radius):
        """
        Initialize an obstacle.
        
        Args:
            position (numpy.ndarray): 2D position [x, y]
            radius (float): Radius of the obstacle
        """
        self.position = np.array(position)
        self.radius = radius
        
    def check_collision(self, uav_position, buffer=0.5):
        """
        Check if a UAV collides with this obstacle.
        
        Args:
            uav_position (numpy.ndarray): UAV position
            buffer (float): Additional buffer distance
            
        Returns:
            bool: True if collision, False otherwise
        """
        # Check 2D distance (ignore z-coordinate)
        distance = np.linalg.norm(self.position - uav_position[:2])
        return distance < (self.radius + buffer)


class QBDISwarmSimulation:
    """
    Main simulation class for QBDI UAV swarm.
    """
    
    def __init__(self, num_uavs=10, world_size=100.0, obstacle_count=5):
        """
        Initialize the QBDI swarm simulation.
        
        Args:
            num_uavs (int): Number of UAVs in the swarm
            world_size (float): Size of the simulation world
            obstacle_count (int): Number of obstacles
        """
        self.num_uavs = num_uavs
        self.world_size = world_size
        self.uavs = []
        self.obstacles = []
        self.target_position = np.array([world_size * 0.8, world_size * 0.8, 10.0])
        self.time_step = 0
        
        # Initialize QBDI components
        self.qaoa_model = QAOADecisionModel(num_uavs)
        self.entropy_coord = EntropySwarmCoordination(num_uavs)
        self.sqec = QuantumStigmergicCommunication(num_uavs, sigma=world_size/10)
        self.mmn = MycelialMemoryNetwork(num_uavs, memory_capacity=100)
        
        # Performance metrics
        self.success_count = 0
        self.collision_count = 0
        self.energy_usage = np.zeros(num_uavs)
        self.decision_latency = []
        
        # Create UAVs with random initial positions
        for i in range(num_uavs):
            initial_position = np.array([
                np.random.uniform(0, world_size * 0.2),
                np.random.uniform(0, world_size * 0.2),
                np.random.uniform(5, 15)
            ])
            uav = UAV(i, initial_position, max_speed=2.0)
            uav.set_target(self.target_position)
            self.uavs.append(uav)
        
        # Create random obstacles
        for _ in range(obstacle_count):
            position = np.array([
                np.random.uniform(world_size * 0.3, world_size * 0.7),
                np.random.uniform(world_size * 0.3, world_size * 0.7)
            ])
            radius = np.random.uniform(3, 8)
            self.obstacles.append(Obstacle(position, radius))
        
        # Initialize interaction strengths in QAOA model
        for i in range(num_uavs):
            for j in range(i+1, num_uavs):
                strength = np.random.uniform(-0.5, 0.5)
                self.qaoa_model.set_interaction_strength(i, j, strength)
    
    def update_external_fields(self):
        """
        Update external fields in QAOA model based on obstacle proximity.
        """
        for i, uav in enumerate(self.uavs):
            position = uav.get_position()
            
            # Calculate field based on obstacles
            field_value = 0
            for obstacle in self.obstacles:
                # 2D distance to obstacle
                distance = np.linalg.norm(obstacle.position - position[:2])
                
                # Field strength inversely proportional to distance
                if distance < obstacle.radius * 3:
                    field_value -= 1.0 / (distance - obstacle.radius + 0.1)
            
            # Add target attraction
            target_distance = np.linalg.norm(self.target_position - position)
            field_value += 0.5 / (target_distance + 1.0)
            
            # Set external field
            self.qaoa_model.set_external_field(i, field_value)
    
    def update_quantum_states(self):
        """
        Update quantum-inspired states based on UAV positions and obstacles.
        """
        for i, uav in enumerate(self.uavs):
            position = uav.get_position()
            
            # Create a 2D quantum state based on position and obstacles
            state = np.zeros(2)
            
            # Component based on position relative to target
            direction = self.target_position - position
            state[0] = np.arctan2(direction[1], direction[0]) / np.pi
            
            # Component based on obstacle avoidance
            obstacle_influence = 0
            for obstacle in self.obstacles:
                distance = np.linalg.norm(obstacle.position - position[:2])
                if distance < obstacle.radius * 3:
                    obstacle_influence += 1.0 / (distance - obstacle.radius + 0.1)
            state[1] = np.tanh(obstacle_influence)
            
            # Normalize state
            state = state / np.linalg.norm(state)
            
            # Update UAV's quantum state
            uav.quantum_state = state
            
            # Update in SQEC model
            self.sqec.set_uav_state(i, state)
    
    def update_memory_network(self):
        """
        Update mycelial memory network with new information.
        """
        # Add memory nodes at obstacle locations with high importance
        for obstacle in self.obstacles:
            # Check if obstacle is already in memory
            add_new = True
            for node in self.mmn.memory_nodes:
                if np.linalg.norm(node['position'][:2] - obstacle.position) < 1.0:
                    add_new = False
                    break
            
            if add_new:
                position = np.append(obstacle.position, [0])  # Add z=0
                value = -1.0  # Negative value for obstacles
                self.mmn.add_memory_node(position, value, importance=0.9)
        
        # Add memory nodes at UAV positions with varying importance
        for i, uav in enumerate(self.uavs):
            position = uav.get_position()
            
            # Value based on decision state and trajectory probability
            value = uav.decision_state * uav.trajectory_probability
            
            # Add memory node with moderate importance
            self.mmn.add_memory_node(position, value, importance=0.5)
    
    def optimize_swarm_decisions(self):
        """
        Optimize UAV decision states using QAOA model.
        
        Returns:
            float: Decision latency (computation time)
        """
        start_time = time.time()
        
        # Update external fields based on environment
        self.update_external_fields()
        
        # Optimize decision states
        optimal_states = self.qaoa_model.optimize_decision_state()
        
        # Update UAV decision states
        for i, uav in enumerate(self.uavs):
            uav.decision_state = optimal_states[i]
        
        # Calculate decision latency
        latency = time.time() - start_time
        self.decision_latency.append(latency)
        
        return latency
    
    def update_trajectory_probabilities(self):
        """
        Update trajectory probabilities based on UAV performance.
        """
        for i, uav in enumerate(self.uavs):
            position = uav.get_position()
            
            # Calculate probability based on:
            # 1. Distance to target
            target_distance = np.linalg.norm(self.target_position - position)
            target_factor = 1.0 - min(1.0, target_distance / self.world_size)
            
            # 2. Obstacle avoidance
            obstacle_factor = 1.0
            for obstacle in self.obstacles:
                distance = np.linalg.norm(obstacle.position - position[:2])
                if distance < obstacle.radius * 2:
                    obstacle_factor *= distance / (obstacle.radius * 2)
            
            # 3. Energy efficiency
            energy_factor = 1.0 - min(1.0, self.energy_usage[i] / 100.0)
            
            # Combine factors
            probability = 0.4 * target_factor + 0.4 * obstacle_factor + 0.2 * energy_factor
            
            # Update UAV's trajectory probability
            uav.trajectory_probability = probability
            
            # Update in entropy coordination model
            self.entropy_coord.update_trajectory_probability(i, probability)
    
    def update_uav_positions(self):
        """
        Update UAV positions based on QBDI algorithms.
        """
        # Update UAV positions in SQEC model
        for i, uav in enumerate(self.uavs):
            self.sqec.set_uav_position(i, uav.get_position())
        
        # Update communication matrix
        self.sqec.update_communication_matrix()
        
        # Calculate swarm entropy
        swarm_entropy = self.entropy_coord.calculate_swarm_entropy()
        
        # Update UAV positions
        for i, uav in enumerate(self.uavs):
            position = uav.get_position()
            
            # Get memory information from nearby locations
            memory_values, memory_weights = self.mmn.query_memory(position, uav_idx=i)
            
            # Get communication strengths with other UAVs
            comm_strengths = [self.sqec.get_communication_strength(i, j) for j in range(self.num_uavs) if j != i]
            
            # Adjust velocity based on decision state and communication
            if uav.decision_state == 1:  # Positive decision state
                # Direct path to target
                uav.update_position()
            else:  # Negative decision state
                # Obstacle avoidance mode
                # Find nearest obstacle
                nearest_obstacle = None
                min_distance = float('inf')
                
                for obstacle in self.obstacles:
                    distance = np.linalg.norm(obstacle.position - position[:2])
                    if distance < min_distance:
                        min_distance = distance
                        nearest_obstacle = obstacle
                
                if nearest_obstacle and min_distance < nearest_obstacle.radius * 3:
                    # Calculate avoidance direction
                    avoidance_direction = position[:2] - nearest_obstacle.position
                    avoidance_direction = avoidance_direction / np.linalg.norm(avoidance_direction)
                    
                    # Create temporary target for avoidance
                    temp_target = np.array([
                        position[0] + avoidance_direction[0] * 10,
                        position[1] + avoidance_direction[1] * 10,
                        position[2]
                    ])
                    
                    uav.set_target(temp_target)
                    uav.update_position()
                    
                    # Reset target to original
                    uav.set_target(self.target_position)
                else:
                    # No nearby obstacle, continue to target
                    uav.update_position()
            
            # Update energy usage (proportional to velocity magnitude)
            self.energy_usage[i] += np.linalg.norm(uav.velocity) * 0.1
    
    def check_collisions(self):
        """
        Check for collisions between UAVs and obstacles.
        
        Returns:
            int: Number of collisions detected
        """
        collision_count = 0
        
        for uav in self.uavs:
            position = uav.get_position()
            
            # Check collision with obstacles
            for obstacle in self.obstacles:
                if obstacle.check_collision(position):
                    collision_count += 1
                    break
        
        self.collision_count += collision_count
        return collision_count
    
    def check_success(self):
        """
        Check if UAVs have reached the target.
        
        Returns:
            int: Number of UAVs that reached the target
        """
        success_count = 0
        
        for uav in self.uavs:
            position = uav.get_position()
            distance = np.linalg.norm(self.target_position - position)
            
            if distance < 5.0:  # Target reached threshold
                success_count += 1
        
        self.success_count = success_count
        return success_count
    
    def run_simulation(self, num_steps=100):
        """
        Run the simulation for a specified number of steps.
        
        Args:
            num_steps (int): Number of simulation steps
            
        Returns:
            dict: Simulation results and metrics
        """
        # Initialize results storage
        results = {
            'uav_trajectories': [[] for _ in range(self.num_uavs)],
            'entropy_history': [],
            'success_rate': 0.0,
            'collision_rate': 0.0,
            'energy_usage': 0.0,
            'decision_latency': 0.0
        }
        
        # Run simulation steps
        for step in tqdm(range(num_steps), desc="Running simulation"):
            # Optimize swarm decisions
            if step % 5 == 0:  # Optimize every 5 steps to reduce computation
                self.optimize_swarm_decisions()
            
            # Update quantum states
            self.update_quantum_states()
            
            # Update trajectory probabilities
            self.update_trajectory_probabilities()
            
            # Update UAV positions
            self.update_uav_positions()
            
            # Update memory network
            if step % 10 == 0:  # Update memory less frequently
                self.update_memory_network()
            
            # Check collisions
            self.check_collisions()
            
            # Check success
            success_count = self.check_success()
            
            # Calculate swarm entropy
            entropy = self.entropy_coord.calculate_swarm_entropy()
            results['entropy_history'].append(entropy)
            
            # Store UAV positions
            for i, uav in enumerate(self.uavs):
                results['uav_trajectories'][i].append(np.copy(uav.get_position()))
        
        # Calculate final metrics
        results['success_rate'] = self.success_count / self.num_uavs * 100
        results['collision_rate'] = self.collision_count / (self.num_uavs * num_steps) * 100
        results['energy_usage'] = np.mean(self.energy_usage)
        results['decision_latency'] = np.mean(self.decision_latency) * 1000  # Convert to ms
        
        return results
    
    def visualize_simulation(self, results, save_path='/home/ubuntu/qbdi_implementation/simulation_result.png'):
        """
        Visualize the simulation results.
        
        Args:
            results (dict): Simulation results
            save_path (str): Path to save the visualization
        """
        plt.figure(figsize=(12, 10))
        
        # Plot world boundaries
        plt.plot([0, self.world_size, self.world_size, 0, 0],
                 [0, 0, self.world_size, self.world_size, 0],
                 'k--', alpha=0.3)
        
        # Plot obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle.position[0], obstacle.position[1]),
                               obstacle.radius, color='red', alpha=0.5)
            plt.gca().add_patch(circle)
        
        # Plot target
        plt.scatter(self.target_position[0], self.target_position[1],
                   s=200, c='green', marker='*', label='Target')
        
        # Plot UAV trajectories
        for i, trajectory in enumerate(results['uav_trajectories']):
            trajectory = np.array(trajectory)
            plt.plot(trajectory[:, 0], trajectory[:, 1], '-', linewidth=1, alpha=0.7,
                    label=f'UAV {i}' if i == 0 else None)
            
            # Plot final position
            plt.scatter(trajectory[-1, 0], trajectory[-1, 1], s=50, marker='o')
        
        # Add metrics as text
        metrics_text = (
            f"Success Rate: {results['success_rate']:.1f}%\n"
            f"Collision Rate: {results['collision_rate']:.1f}%\n"
            f"Avg Energy Usage: {results['energy_usage']:.1f} J/UAV\n"
            f"Decision Latency: {results['decision_latency']:.1f} ms"
        )
        plt.text(self.world_size * 0.05, self.world_size * 0.95, metrics_text,
                bbox=dict(facecolor='white', alpha=0.7))
        
        plt.title('QBDI UAV Swarm Simulation')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.legend(loc='upper right')
        plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()
        
        # Save figure
        plt.savefig(save_path)
        plt.close()
        
        # Also create a plot of entropy history
        plt.figure(figsize=(10, 6))
        plt.plot(results['entropy_history'], marker='o', markersize=3, linestyle='-')
        plt.xlabel('Time Step')
        plt.ylabel('Swarm Entropy')
        plt.title('Swarm Entropy Over Time')
        plt.grid(True, alpha=0.3)
        plt.savefig('/home/ubuntu/qbdi_implementation/entropy_over_time.png')
        plt.close()
    
    def create_animation(self, results, save_path='/home/ubuntu/qbdi_implementation/simulation_animation.gif'):
        """
        Create an animation of the simulation.
        
        Args:
            results (dict): Simulation results
            save_path (str): Path to save the animation
        """
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Maximum number of frames
        num_frames = len(results['uav_trajectories'][0])
        
        # Initialize UAV scatter plot
        uav_scatter = ax.scatter([], [], s=50, c='blue', marker='o')
        
        # Plot world boundaries
        ax.plot([0, self.world_size, self.world_size, 0, 0],
               [0, 0, self.world_size, self.world_size, 0],
               'k--', alpha=0.3)
        
        # Plot obstacles
        for obstacle in self.obstacles:
            circle = plt.Circle((obstacle.position[0], obstacle.position[1]),
                               obstacle.radius, color='red', alpha=0.5)
            ax.add_patch(circle)
        
        # Plot target
        ax.scatter(self.target_position[0], self.target_position[1],
                 s=200, c='green', marker='*', label='Target')
        
        # Add metrics text
        metrics_text = ax.text(self.world_size * 0.05, self.world_size * 0.95, "",
                             bbox=dict(facecolor='white', alpha=0.7))
        
        # Set axis properties
        ax.set_xlim(0, self.world_size)
        ax.set_ylim(0, self.world_size)
        ax.set_title('QBDI UAV Swarm Simulation')
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.grid(True, alpha=0.3)
        
        # Animation update function
        def update(frame):
            # Get UAV positions at this frame
            positions = np.array([trajectory[frame] for trajectory in results['uav_trajectories']])
            
            # Update UAV scatter plot
            uav_scatter.set_offsets(positions[:, :2])
            
            # Update metrics text
            current_success = sum(1 for pos in positions if np.linalg.norm(pos - self.target_position) < 5.0)
            success_rate = current_success / self.num_uavs * 100
            
            metrics_str = (
                f"Time Step: {frame}\n"
                f"Current Success: {success_rate:.1f}%\n"
                f"Swarm Entropy: {results['entropy_history'][frame]:.3f}"
            )
            metrics_text.set_text(metrics_str)
            
            return uav_scatter, metrics_text
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=num_frames, interval=100, blit=True)
        
        # Save animation
        anim.save(save_path, writer='pillow', fps=10)
        plt.close()


# Example usage
if __name__ == "__main__":
    # Create and run simulation
    simulation = QBDISwarmSimulation(num_uavs=10, world_size=100.0, obstacle_count=5)
    results = simulation.run_simulation(num_steps=100)
    
    # Visualize results
    simulation.visualize_simulation(results)
    simulation.create_animation(results)
    
    # Print metrics
    print("\nSimulation Results:")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Collision Rate: {results['collision_rate']:.1f}%")
    print(f"Average Energy Usage: {results['energy_usage']:.1f} J/UAV")
    print(f"Decision Latency: {results['decision_latency']:.1f} ms")
