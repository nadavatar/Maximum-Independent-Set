from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
import numpy as np
import matplotlib.pyplot as plt

# Constants
π = np.pi
Δ = 1  # Detuning
Ω = 1  # Coupling strength
Tf0 = 50 * π / Δ  # Total duration of RAP


populations = []

# Define the pulses
def Ω_pulse(t):
    return Ω * np.sin(np.pi * t / Tf0)

def Δ_pulse(t):
    return -Δ * np.cos(np.pi * t / Tf0)

# Time grid
# num_steps = 500
num_steps_values = [100, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500, 5000]
# time_intervals = np.linspace(0, Tf0, num_steps)
# time_step_size = time_intervals[1] - time_intervals[0]

def apply_rap_dynamics_split(qc):
    populations_0 = []
    populations_1 = []
    state = Statevector.from_label('1')
    for num_step in num_steps_values:
        time_intervals = np.linspace(0, Tf0, num_step)
        print(time_intervals)
        step_size = time_intervals[1] - time_intervals[0]
        print(str(len(time_intervals)))

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
        print(populations)
        populations_0 = [p[0] for p in populations]  # Extract the first item (|0⟩ population)
        populations_1 = [p[1] for p in populations]  # Extract the second item (|1⟩ population)   

    return num_steps_values, populations_0, populations_1

# Create Quantum Circuit
qc = QuantumCircuit(1)
qc.x(0)  # Start in |1⟩ state
initial_state_qiskit = Statevector.from_instruction(qc)
start_populations_qiskit = np.abs(initial_state_qiskit.data) ** 2
print("Start Populations (|0⟩, |1⟩):", start_populations_qiskit)

# Apply RAP dynamics using the refined method
steps, pops_0, pops_1 =apply_rap_dynamics_split(qc)

# Final state
final_state_qiskit = Statevector.from_instruction(qc)
final_populations_qiskit = np.abs(final_state_qiskit.data) ** 2

# Display results
print("Final Statevector:", final_state_qiskit)
print("Final Populations (|0⟩, |1⟩):", final_populations_qiskit)

# Extract populations for |0⟩ and |1⟩ states
# populations_0 = [p[0] for p in populations]  # Extract the first item (|0⟩ population)
# populations_1 = [p[1] for p in populations]  # Extract the second item (|1⟩ population)
# print(pops_1)
# print(pops_0)

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
