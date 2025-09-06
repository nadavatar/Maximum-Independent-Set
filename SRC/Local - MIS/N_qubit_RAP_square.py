# -*- coding: utf-8 -*-
import os
from pathlib import Path

import numpy as np
import pandas as pd
import networkx as nx
from itertools import combinations
from functools import reduce
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import eigsh
from qiskit_dynamics import Solver
from qiskit_dynamics.signals import Signal

# ─────────────────────────────────────────────
# Output folder (Windows path with Hebrew + spaces)
SAVE_DIR = Path(r"C:\\Users\\nadav\\OneDrive\\Documents\\פרויקט גמר\\MIS")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_XLSX = SAVE_DIR / "mis_results.xlsx"

# Constants
π = np.pi
Δ = 1.0
Ω = 1.0
V = 10
num_qubits = 7
Tf0 = 50 * π / Δ
t = np.linspace(0, Tf0, 600)
iterations = 100  # Number of random graphs to simulate

# Pauli matrices
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# Pulses
def Ω_pulse(t_val): return Ω * np.sin(π * t_val / Tf0)
def Δ_pulse(t_val): return Δ * np.cos(π * t_val / Tf0)

# Apply operator on i-th qubit
def operator_on_qubit(pauli, pos, N):
    op = [I2] * N
    op[pos] = pauli
    return reduce(np.kron, op)

# Check if state is valid MIS
def is_valid_mis(state, edges):
    for i, j in edges:
        if state[i] == '1' and state[j] == '1':
            return False
    return True

# Prepare all possible edges
all_possible_edges = list(combinations(range(num_qubits), 2))

# Store results
results_list = []

for i in range(iterations):
    print("Starting iteration", i + 1)
    edges_idx = np.random.choice(len(all_possible_edges), 9, replace=False)
    graph_edges = [all_possible_edges[i] for i in edges_idx]

    # Build Hamiltonians
    H_X = sum(operator_on_qubit(sx, i, num_qubits) for i in range(num_qubits))
    H_n = sum(0.5 * (np.eye(2**num_qubits) - operator_on_qubit(sz, i, num_qubits)) for i in range(num_qubits))
    H_int = sum(
        V * (0.5 * (np.eye(2**num_qubits) - operator_on_qubit(sz, i, num_qubits))) @
            (0.5 * (np.eye(2**num_qubits) - operator_on_qubit(sz, j, num_qubits)))
        for i, j in graph_edges
    )

    # Initial state
    initial_state = np.zeros((2**num_qubits, 1), dtype=complex)
    initial_state[0, 0] = 1.0

    solver = Solver(hamiltonian_operators=[H_X, H_n, H_int])
    signals = [
        Signal(lambda τ: Ω_pulse(τ), carrier_freq=0),
        Signal(lambda τ: -Δ_pulse(τ), carrier_freq=0),
        Signal(lambda τ: 1.0, carrier_freq=0)
    ]

    results = solver.solve(
        t_span=(0, Tf0),
        y0=initial_state,
        t_eval=t,
        rtol=1e-10,
        atol=1e-12,
        signals=signals
    )

    psi_t = np.squeeze(results.y)
    populations = np.abs(psi_t)**2
    final_pops = populations[-1]
    states = [format(i, f'0{num_qubits}b') for i in range(2**num_qubits)]

    mis_states = [s for s in states if is_valid_mis(s, graph_edges)]
    mis_probs = {s: final_pops[int(s, 2)] for s in mis_states}
    max_mis_size = max(s.count('1') for s in mis_states)
    d_mis = sum(1 for s in mis_states if s.count('1') == max_mis_size)
    d_mis_minus_1 = sum(1 for s in mis_states if s.count('1') == max_mis_size - 1)
    hardness = d_mis_minus_1 / d_mis if d_mis else 0

    best_mis_states = [s for s in mis_states if s.count('1') == max_mis_size]
    fidelities = [mis_probs[s] for s in best_mis_states]
    best_fidelity = max(fidelities) if fidelities else 0.0
    fidelity_loss = 1.0 - best_fidelity

    # Final Hamiltonian and min gap
    H_final = csr_matrix(H_X - H_n + H_int)
    # (Optional: use which="SA" to target ground/first excited precisely)
    eigvals = eigsh(H_final, k=2, which="SA", return_eigenvectors=False)
    min_gap = float(np.abs(eigvals[1] - eigvals[0]))

    # Format the actual graph as a set of edge pairs: {(i,j), (k,l), ...}
    edge_set_str = "{" + ", ".join(f"({i},{j})" for i, j in sorted(graph_edges)) + "}"

    results_list.append({
        # Put the actual edge set in this column (per your request)
        "G(N,E)": edge_set_str,
        # (Optional: keep explicit sizes too)
        "N": num_qubits,
        "|E|": len(graph_edges),
        "|MIS|": max_mis_size,
        "d_|MIS|": d_mis,
        "d_|MIS|-1": d_mis_minus_1,
        "Hardness": hardness,
        "Fidelity": best_fidelity,
        "1 - Fidelity": fidelity_loss,
        "Min Energy Gap": min_gap
    })
    print("end of iteration", i + 1)

# Convert to DataFrame and export to Excel in the requested folder
df_results = pd.DataFrame(results_list)
df_results.to_excel(OUTPUT_XLSX, index=False)
print(f"Saved to {OUTPUT_XLSX}")
