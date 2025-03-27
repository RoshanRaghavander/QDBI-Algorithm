"""
Test script for QBDI implementation.
This script tests the core QBDI algorithms and the UAV swarm simulation.
"""

import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add the src directory to the Python path
sys.path.append('/home/ubuntu/qbdi_implementation/src')

# Import QBDI components
from qaoa_decision_model import QAOADecisionModel
from entropy_swarm_coordination import EntropySwarmCoordination
from quantum_stigmergic_communication import QuantumStigmergicCommunication
from mycelial_memory_network import MycelialMemoryNetwork
from uav_swarm_simulation import QBDISwarmSimulation

def test_qaoa_decision_model():
    """Test the QAOA-based quantum decision model."""
    print("\n=== Testing QAOA Decision Model ===")
    
    # Create a QAOA decision model for 5 UAVs
    model = QAOADecisionModel(5)
    
    # Set interaction strengths
    for i in range(5):
        for j in range(i+1, 5):
            strength = np.random.uniform(-1, 1)
            model.set_interaction_strength(i, j, strength)
    
    # Set external field values
    for i in range(5):
        field = np.random.uniform(-0.5, 0.5)
        model.set_external_field(i, field)
    
    # Optimize decision states
    print("Optimizing decision states...")
    optimal_state = model.optimize_decision_state()
    
    # Print results
    print(f"Optimal decision state: {optimal_state}")
    print(f"Hamiltonian value: {model.compute_hamiltonian()}")
    
    # Visualize decision states
    model.visualize_decision_states()
    print("Decision states visualization saved to decision_states.png")
    
    return True

def test_entropy_swarm_coordination():
    """Test the entropy-based swarm coordination."""
    print("\n=== Testing Entropy Swarm Coordination ===")
    
    # Create an entropy-based swarm coordination for 5 UAVs
    swarm_coord = EntropySwarmCoordination(5)
    
    # Simulate 10 time steps
    print("Simulating 10 time steps...")
    for t in range(10):
        # Update trajectory probabilities
        for i in range(5):
            prob = np.random.uniform(0.1, 1.0)
            swarm_coord.update_trajectory_probability(i, prob)
        
        # Calculate entropy
        entropy = swarm_coord.calculate_swarm_entropy()
        print(f"Time step {t}, Swarm entropy: {entropy:.4f}")
    
    # Visualize results
    swarm_coord.visualize_entropy_history()
    swarm_coord.visualize_trajectory_probabilities()
    print("Entropy history visualization saved to entropy_history.png")
    print("Trajectory probabilities visualization saved to trajectory_probabilities.png")
    
    return True

def test_quantum_stigmergic_communication():
    """Test the quantum stigmergic communication."""
    print("\n=== Testing Quantum Stigmergic Communication ===")
    
    # Create a quantum stigmergic communication model for 5 UAVs
    sqec = QuantumStigmergicCommunication(5, sigma=2.0)
    
    # Set random positions for UAVs
    print("Setting random UAV positions and states...")
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
    print("Communication matrix visualization saved to communication_matrix.png")
    print("UAV positions visualization saved to uav_positions.png")
    
    return True

def test_mycelial_memory_network():
    """Test the mycelial memory network."""
    print("\n=== Testing Mycelial Memory Network ===")
    
    # Create a mycelial memory network for 5 UAVs
    mmn = MycelialMemoryNetwork(5, memory_capacity=20)
    
    # Add random memory nodes
    print("Adding random memory nodes...")
    for _ in range(15):
        position = np.random.uniform(-10, 10, size=2)
        value = np.random.uniform(0, 1)
        importance = np.random.uniform(0.3, 1.0)
        mmn.add_memory_node(position, value, importance)
    
    # Query memory at random positions
    print("Querying memory at random positions...")
    for i in range(5):
        query_pos = np.random.uniform(-10, 10, size=2)
        values, weights = mmn.query_memory(query_pos, uav_idx=i)
        
        if values is not None:
            weighted_value = np.sum(np.array(values) * weights)
            print(f"UAV {i} query at {query_pos}: weighted value = {weighted_value:.4f}")
    
    # Visualize the memory network
    mmn.visualize_memory_network()
    mmn.visualize_memory_heatmap()
    print("Memory network visualization saved to memory_network.png")
    print("Memory heatmap visualization saved to memory_heatmap.png")
    
    return True

def test_uav_swarm_simulation(num_uavs=5, num_steps=50):
    """Test the UAV swarm simulation."""
    print(f"\n=== Testing UAV Swarm Simulation with {num_uavs} UAVs for {num_steps} steps ===")
    
    # Create and run simulation
    simulation = QBDISwarmSimulation(num_uavs=num_uavs, world_size=100.0, obstacle_count=3)
    print("Running simulation...")
    results = simulation.run_simulation(num_steps=num_steps)
    
    # Visualize results
    simulation.visualize_simulation(results)
    print("Simulation visualization saved to simulation_result.png")
    
    # Create animation
    print("Creating animation...")
    simulation.create_animation(results)
    print("Simulation animation saved to simulation_animation.gif")
    
    # Print metrics
    print("\nSimulation Results:")
    print(f"Success Rate: {results['success_rate']:.1f}%")
    print(f"Collision Rate: {results['collision_rate']:.1f}%")
    print(f"Average Energy Usage: {results['energy_usage']:.1f} J/UAV")
    print(f"Decision Latency: {results['decision_latency']:.1f} ms")
    
    return True

def run_all_tests():
    """Run all tests."""
    print("Starting QBDI implementation tests...")
    
    tests = [
        test_qaoa_decision_model,
        test_entropy_swarm_coordination,
        test_quantum_stigmergic_communication,
        test_mycelial_memory_network,
        test_uav_swarm_simulation
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"Error in {test.__name__}: {str(e)}")
            results.append(False)
    
    # Print summary
    print("\n=== Test Summary ===")
    for i, test in enumerate(tests):
        status = "PASSED" if results[i] else "FAILED"
        print(f"{test.__name__}: {status}")
    
    return all(results)

if __name__ == "__main__":
    success = run_all_tests()
    if success:
        print("\nAll tests passed successfully!")
    else:
        print("\nSome tests failed. Check the output for details.")
