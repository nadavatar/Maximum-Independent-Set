import argparse
from typing import List, Tuple, Optional
import numpy as np
import matplotlib.pyplot as plt

from qiskit import QuantumCircuit
from qiskit.quantum_info import Statevector, SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

# Constants
π = np.pi
# Δ = 2.0
# Tf0 = 50 * π / Δ

def Ω_pulse(Ω: float, t: float, Tf: float) -> float:
    """
    Calculate the coupling strength pulse at time t.
    
    Args:
        Ω: Coupling strength amplitude
        t: Current time
        Tf: Total duration of the pulse
        
    Returns:
        float: Coupling strength at time t
    """
    return Ω * np.sin(np.pi * t / Tf)

def Δ_pulse(Δ: float, t: float, Tf: float) -> float:
    """
    Calculate the detuning pulse at time t.
    
    Args:
        Δ: Detuning amplitude
        t: Current time
        Tf: Total duration of the pulse
        
    Returns:
        float: Detuning at time t
    """
    return -Δ * np.cos(np.pi * t / Tf)

def apply_rap_dynamics_split(Ω: float, Δ: float, qc: QuantumCircuit, 
                           time_intervals: np.ndarray, Tf0: float) -> List[np.ndarray]:
    """
    Apply the RAP dynamics to the given quantum circuit for each time step.
    
    Args:
        Ω: Coupling strength amplitude
        Δ: Detuning amplitude
        qc: Quantum circuit to apply dynamics to
        time_intervals: Array of time points
        Tf0: Total duration of the RAP sequence
        
    Returns:
        List[np.ndarray]: List of population arrays at each time step
    """
    populations = []
    state = Statevector.from_label('1')  # Start in |1⟩
    step_size = time_intervals[1] - time_intervals[0]

    for t in time_intervals:
        omega_t = Ω_pulse(Ω, t, Tf0)
        delta_t = Δ_pulse(Δ, t, Tf0)

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

def setup_quantum_backend() -> Tuple[QuantumCircuit, SparsePauliOp]:
    """
    Set up the quantum backend and prepare the circuit and observable.
    
    Returns:
        Tuple[QuantumCircuit, SparsePauliOp]: Transpiled circuit and observable
    """
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    
    qc = QuantumCircuit(1)
    qc.x(0)  # Start in |1⟩
    
    pm = generate_preset_pass_manager(backend=backend, optimization_level=1)
    transpiled_circuit = pm.run(qc)
    
    num_qubits = transpiled_circuit.num_qubits
    observable = SparsePauliOp("Z" * num_qubits)
    
    return transpiled_circuit, observable

def plot_populations(time_intervals: np.ndarray, populations: List[np.ndarray], 
                    Ω: float, Δ: float) -> None:
    """
    Plot the population evolution over time.
    
    Args:
        time_intervals: Array of time points
        populations: List of population arrays
        Ω: Coupling strength amplitude
        Δ: Detuning amplitude
    """
    pops_0 = [p[0] for p in populations]
    pops_1 = [p[1] for p in populations]

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(time_intervals, pops_0, label=f"Population |0⟩ (final={pops_0[-1]:.9f})")
    ax.plot(time_intervals, pops_1, label=f"Population |1⟩ (final={pops_1[-1]:.9f})")

    ax.xaxis.set_major_locator(plt.MaxNLocator(10))
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.05))

    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax.set_title(f"Qubit Populations During RAP Dynamics (Ω={Ω}, Δ={Δ})")
    ax.set_xlabel("Time")
    ax.set_ylabel("Population")
    ax.legend()

    plt.tight_layout()
    plt.show(block=True)

def run_rap(num_steps: int = 500, Ω: float = 1.0, Δ: float = 1.0, 
           plot: bool = True) -> Tuple[List[np.ndarray], np.ndarray]:
    """
    Run the RAP experiment with specified parameters.
    
    Args:
        num_steps: Number of time steps for the RAP evolution
        Ω: Coupling strength amplitude
        Δ: Detuning amplitude
        plot: Whether to plot the results
        
    Returns:
        Tuple[List[np.ndarray], np.ndarray]: Populations over time and final populations
    """
    # Build time grid
    Tf0 = 50 * π / Δ
    time_intervals = np.linspace(0, Tf0, num_steps)

    # Create and prepare quantum circuit
    qc = QuantumCircuit(1)
    qc.x(0)  # Start in |1⟩

    # Apply RAP dynamics
    populations = apply_rap_dynamics_split(Ω, Δ, qc, time_intervals, Tf0)

    # Set up quantum backend and run experiment
    transpiled_circuit, observable = setup_quantum_backend()
    
    # Run the experiment
    service = QiskitRuntimeService()
    backend = service.least_busy(simulator=False, operational=True)
    estimator = EstimatorV2(backend=backend)
    job = estimator.run(pubs=[(transpiled_circuit, observable)])
    print(f">>> Job ID: {job.job_id()}")

    # Calculate final populations
    final_state = Statevector.from_instruction(qc)
    final_populations = np.abs(final_state.data) ** 2
    print("Final populations (|0>, |1>):", final_populations)

    # Plot if requested
    if plot:
        plot_populations(time_intervals, populations, Ω, Δ)

    return populations, final_populations

def main() -> None:
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(description="Run RAP experiment on Qiskit with specified time steps.")
    parser.add_argument(
        "--num_steps",
        type=int,
        default=500,
        help="Number of time steps for the RAP evolution (default 500)."
    )
    parser.add_argument(
        "--omega",
        type=float,
        default=1.0,
        help="Coupling strength amplitude (default 1.0)."
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1.0,
        help="Detuning amplitude (default 1.0)."
    )
    args = parser.parse_args()

    run_rap(num_steps=args.num_steps, Ω=args.omega, Δ=args.delta, plot=True)

if __name__ == "__main__":
    main()
