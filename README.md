# Quantum Biomimetic Distributed Intelligence (QBDI) Implementation

This repository contains a Python implementation of the Quantum Biomimetic Distributed Intelligence (QBDI) framework for UAV swarm coordination as described in the research paper.

## Overview

QBDI is a novel framework that integrates quantum computing principles with biologically inspired intelligence systems for UAV swarm coordination. The implementation includes:

1. **QAOA-based Quantum Decision Model** - Optimizes UAV decision states using quantum-inspired algorithms
2. **Entropy-Based Swarm Coordination** - Measures and optimizes swarm coherence
3. **Quantum Stigmergic Communication (SQEC)** - Enables implicit coordination through an entanglement-inspired model
4. **Mycelial Memory Networks (MMN)** - Bio-inspired memory structure for distributed information storage

## Repository Structure

- `src/` - Source code for all QBDI components
  - `qaoa_decision_model.py` - Implementation of the QAOA-based quantum decision model
  - `entropy_swarm_coordination.py` - Implementation of entropy-based swarm coordination
  - `quantum_stigmergic_communication.py` - Implementation of quantum stigmergic communication
  - `mycelial_memory_network.py` - Implementation of mycelial memory networks
  - `uav_swarm_simulation.py` - Integration of all components into a UAV swarm simulation
- `results/` - Visualization outputs and simulation results
- `test_qbdi.py` - Test script for all components

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/qbdi-implementation.git
cd qbdi-implementation

# Install required packages
pip install numpy matplotlib scipy networkx scikit-learn pandas seaborn tqdm
```

## Usage

### Running the Simulation

```python
from src.uav_swarm_simulation import QBDISwarmSimulation

# Create and run simulation
simulation = QBDISwarmSimulation(num_uavs=10, world_size=100.0, obstacle_count=5)
results = simulation.run_simulation(num_steps=100)

# Visualize results
simulation.visualize_simulation(results)
simulation.create_animation(results)

# Print metrics
print(f"Success Rate: {results['success_rate']:.1f}%")
print(f"Collision Rate: {results['collision_rate']:.1f}%")
print(f"Average Energy Usage: {results['energy_usage']:.1f} J/UAV")
print(f"Decision Latency: {results['decision_latency']:.1f} ms")
```

### Testing Individual Components

```python
# Test QAOA Decision Model
from src.qaoa_decision_model import QAOADecisionModel
model = QAOADecisionModel(5)
# ... (see test_qbdi.py for examples)

# Test Entropy-Based Swarm Coordination
from src.entropy_swarm_coordination import EntropySwarmCoordination
# ... (see test_qbdi.py for examples)

# Test Quantum Stigmergic Communication
from src.quantum_stigmergic_communication import QuantumStigmergicCommunication
# ... (see test_qbdi.py for examples)

# Test Mycelial Memory Networks
from src.mycelial_memory_network import MycelialMemoryNetwork
# ... (see test_qbdi.py for examples)
```

## Mathematical Formulation

### QAOA-Based Quantum Decision Model

The Hamiltonian function governing UAV decisions:

```
HQBDI = ∑(i,j) Jij zi zj + ∑i hi zi
```

Where:
- zi ∈ {-1, 1} represents the decision state of UAV i
- Jij represents interaction strength between UAVs
- hi represents an external field (environmental influence)

### Entropy-Based Swarm Coordination

Shannon entropy function for measuring swarm coherence:

```
Hswarm(t) = -∑i Pi log Pi
```

Where Pi represents the probability of UAV i following an optimal trajectory.

### Quantum Stigmergic Communication

Implicit coordination model:

```
Sij(t) = ∫E Γ(pi, pj) · χ(ei, ej)de
```

Where:
- Γ(pi, pj) = exp(-‖pi - pj‖²/2σ²) is the spatial correlation function
- χ(ei, ej) measures entanglement strength between UAV states

## Results

See the `results/` directory for visualizations and the `results_summary.md` file for a detailed analysis of the simulation results.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on the research paper "Quantum Biomimetic Distributed Intelligence (QBDI): A Quantum-Inspired Framework for UAV Swarm Coordination" by Roshan Raghavander N.
