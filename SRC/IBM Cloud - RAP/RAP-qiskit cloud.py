from qiskit import QuantumCircuit
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.quantum_info import SparsePauliOp, Statevector
import numpy as np
import matplotlib.pyplot as plt
from qiskit.transpiler import InstructionDurations

# Constants
π = np.pi
Δ = 2  # Detuning - default 1
Ω = 2  # Coupling strength - default 1 
Tf0 = 50 * π / Δ  # Total duration of RAP

# Define the pulses
def Ω_pulse(t):
    return Ω * np.sin(np.pi * t / Tf0)

def Δ_pulse(t):
    return -Δ * np.cos(np.pi * t / Tf0)

# Time grid
num_steps_values = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# num_steps = 500
# time_intervals = np.linspace(0, Tf0, num_steps)
# time_step_size = time_intervals[1] - time_intervals[0]

populations = []

def apply_rap_dynamics_split(qc):
    populations_0 = []
    populations_1 = []
    state = Statevector.from_label('1')
    for num_step in num_steps_values:
        time_intervals = np.linspace(0, Tf0, num_step)
        step_size = time_intervals[1] - time_intervals[0]
        for t in time_intervals:
            omega_t = Ω_pulse(t)
            delta_t = Δ_pulse(t)

            # Calculate the rotation angles for this step
            theta_x = omega_t * step_size
            theta_z = delta_t * step_size

            # Apply the sequence of gates for RAP
            qc.rz(theta_z, 0)  # Z-rotation for detuning
            qc.rx(2 * theta_x, 0)  # X-rotation for coupling

            # Update the statevector
            state = Statevector.from_instruction(qc)
            populations_qiskit = np.abs(state.data) ** 2

            # Measure the populations
            populations.append(populations_qiskit)
            populations_0 = [p[0] for p in populations]  # Extract the first item (|0⟩ population)
            populations_1 = [p[1] for p in populations]  # Extract the second item (|1⟩ population)
    return num_steps_values, populations_0, populations_1

# Create Quantum Circuit
qc = QuantumCircuit(1)
qc.x(0)  # Start in |1⟩ state

# IBM Quantum Cloud Setup
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)

# Transpile the circuit
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
transpiled_circuit = pm.run(qc)
print("We are connected")
steps, pops_0, pops_1 =apply_rap_dynamics_split(qc)

# Check qubit count
num_qubits = transpiled_circuit.num_qubits

# Define one-qubit observable matching the transpiled circuit
observable = SparsePauliOp("Z" * num_qubits)

# Construct the EstimatorV2
estimator = EstimatorV2(backend=backend)

# Submit the job to measure the observable
job = estimator.run(pubs=[(transpiled_circuit, observable)])

# Retrieve results
print(f">>> Job ID: {job.job_id()}")
# job_result = job.result()
# expectation_value = job_result[0].data.evs  # Extract expectation value

# # Calculate populations
# population_0 = (1 + expectation_value) / 2
# population_1 = (1 - expectation_value) / 2

# Final state
final_state_qiskit = Statevector.from_instruction(qc)
final_populations_qiskit = np.abs(final_state_qiskit.data) ** 2
durations = InstructionDurations.from_backend(backend)
print(durations)

# Display populations
print("Populations (|0⟩, |1⟩):", final_populations_qiskit)

# # Extract populations for |0⟩ and |1⟩ states
# populations_0 = [p[0] for p in populations]  # Extract the first item (|0⟩ population)
# populations_1 = [p[1] for p in populations]  # Extract the second item (|1⟩ population)

# Plotting the results
plt.figure(figsize=(10, 5))
plt.plot(steps, pops_0, 'bo-', label='Final Population of |0⟩')
plt.plot(steps, pops_1, 'ro-', label='Final Population of |1⟩')
plt.xlabel('Number of Steps')
plt.ylabel('Final Population')
plt.title('Final Qubit State Populations vs. Number of Steps')
plt.legend()
plt.grid()
plt.show()