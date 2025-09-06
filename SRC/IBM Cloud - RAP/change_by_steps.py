import argparse
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.quantum_info import SparsePauliOp

# Constants
π = np.pi
Δ = 2.0
Ω = 2.0
Tf0 = 50 * π / Δ

def Ω_pulse(t, Tf):
    """Coupling strength pulse."""
    return Ω * np.sin(np.pi * t / Tf)

def Δ_pulse(t, Tf):
    """Detuning pulse."""
    return -Δ * np.cos(np.pi * t / Tf)

def apply_rap_dynamics_split(qc, time_intervals):
    """
    Applies the RAP dynamics to the given quantum circuit `qc`
    for each time step in time_intervals.
    Returns a list of population arrays at each step.
    """
    populations = []
    state = Statevector.from_label('1')  # Start in |1⟩
    step_size = time_intervals[1] - time_intervals[0]

    for t in time_intervals:
        omega_t = Ω_pulse(t, Tf0)
        delta_t = Δ_pulse(t, Tf0)

        # Calculate rotation angles for this step
        theta_x = omega_t * step_size
        theta_z = delta_t * step_size

        # Apply the sequence of gates for RAP
        qc.rz(theta_z, 0)
        qc.rx(2 * theta_x, 0)

        # Update statevector
        state = Statevector.from_instruction(qc)
        populations.append(np.abs(state.data)**2)

    return populations

def run_rap(num_steps=500, plot=True):
    """
    Core logic for running the RAP experiment.
    
    :param num_steps: Number of time steps for the RAP evolution
    :param plot: Whether to plot the results at the end
    :return: (populations, final_populations) 
             where populations is a list over time steps 
             and final_populations is (|0>, |1>) at the end
    """
    # Build time grid
    time_intervals = np.linspace(0, Tf0, num_steps)

    # Create Quantum Circuit
    qc = QuantumCircuit(1)
    qc.x(0)  # Start in |1⟩

    # Apply RAP
    populations = apply_rap_dynamics_split(qc, time_intervals)

    # (Optional) Connect to IBM Quantum Cloud, transpile, etc.
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)

    from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_circuit = pm.run(qc)

    # Define the observable
    num_qubits = transpiled_circuit.num_qubits
    observable = SparsePauliOp("Z" * num_qubits)

    # Construct Estimator and submit
    estimator = EstimatorV2(backend=backend)
    job = estimator.run(pubs=[(transpiled_circuit, observable)])
    print(f">>> Job ID: {job.job_id()}")

    # Calculate final populations locally
    final_state = Statevector.from_instruction(qc)
    final_populations = np.abs(final_state.data) ** 2
    print("Final populations (|0>, |1>):", final_populations)

    # Optionally plot
    if plot:
        pops_0 = [p[0] for p in populations]
        pops_1 = [p[1] for p in populations]

        plt.figure(figsize=(8, 5))
        plt.plot(time_intervals, pops_0,
         label=f"Population |0⟩ (final={pops_0[-1]:.6f})")
        plt.plot(time_intervals, pops_1,
         label=f"Population |1⟩ (final={pops_1[-1]:.6f})")
        plt.title("Qubit State Populations During RAP Dynamics, Number Of Steps = " + str(num_steps))
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.legend()
        plt.grid(True)
        plt.show(block=True)
        # plt.savefig("populations.png")

        # plt.plot([0, 1, 2], [0, 1, 4], label="Simple test")
        # plt.legend()
        # plt.savefig("test_plot.png")
        # plt.show()

    return populations, final_populations

def main():
    parser = argparse.ArgumentParser(description="Run RAP experiment on Qiskit with specified time steps.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=500,
        help="Number of time steps for the RAP evolution (default 500)."
    )
    args = parser.parse_args()

    run_rap(num_steps=args.num_steps, plot=True)

if __name__ == "__main__":
    main()
