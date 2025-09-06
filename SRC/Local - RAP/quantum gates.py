from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit.visualization import array_to_latex
import math

# Step 1: Initialize a 2-qubit quantum circuit
qc = QuantumCircuit(2)

# Step 2: Get and print the initial state of the qubits (|00⟩ state)
initial_state = Statevector.from_instruction(qc)
print("Initial Statevector:", initial_state)

# Step 3: Apply Hadamard gates to both qubits
qc.h(0)  # Apply Hadamard gate to qubit 0
qc.h(1)  # Apply Hadamard gate to qubit 1
# qc.cz(0, 1)
qc.crz(2*math.pi, 0, 1)


# Step 4: Get and print the state after applying the Hadamard gates
final_state = Statevector.from_instruction(qc)
print("Final Statevector:", final_state)
