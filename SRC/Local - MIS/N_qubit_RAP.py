from qiskit_dynamics import Solver
from qiskit_dynamics.signals import Signal
import numpy as np
import matplotlib.pyplot as plt
from functools import reduce

# ─────────────────────────────────────────────────────────────────────
# PARAMETERS
π = np.pi
Δ = 1.0
Ω = 1.0
V = 10
N = 5  # <<< Set number of qubits here
Tf0 = 50 * π / Δ
t = np.linspace(0, Tf0, 600)

# Control pulses
def Ω_pulse(t_val): return Ω * np.sin(π * t_val / Tf0)
def Δ_pulse(t_val): return Δ * np.cos(π * t_val / Tf0)

# ─────────────────────────────────────────────────────────────────────
# PAULI MATRICES
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)

# ─────────────────────────────────────────────────────────────────────
# N-QUBIT OPERATOR GENERATORS
def operator_on_qubit(pauli, pos, N):
    ops = [I2] * N
    ops[pos] = pauli
    return reduce(np.kron, ops)

# HAMILTONIAN
H_X = sum(operator_on_qubit(sx, i, N) for i in range(N))
H_n = sum(0.5 * (np.eye(2**N) - operator_on_qubit(sz, i, N)) for i in range(N))

def zz_interaction(i, j, N):
    zi = 0.5 * (np.eye(2**N) - operator_on_qubit(sz, i, N))
    zj = 0.5 * (np.eye(2**N) - operator_on_qubit(sz, j, N))
    return zi @ zj

H_int = sum(V * zz_interaction(i, i + 1, N) for i in range(N - 1))

# ─────────────────────────────────────────────────────────────────────
# INITIAL STATE
initial_state = np.zeros((2**N, 1), dtype=complex)
initial_state[0, 0] = 1.0  # |00...0⟩

# QISKIT DYNAMICS SOLVER
solver = Solver(hamiltonian_operators=[H_X, H_n, H_int])
signals = [
    Signal(envelope=lambda τ: Ω_pulse(τ), carrier_freq=0),
    Signal(envelope=lambda τ: -Δ_pulse(τ), carrier_freq=0),
    Signal(envelope=lambda τ: 1.0, carrier_freq=0)
]

results = solver.solve(
    t_span=(0, Tf0),
    y0=initial_state,
    t_eval=t,
    rtol=1e-10,
    atol=1e-12,
    signals=signals
)

# ─────────────────────────────────────────────────────────────────────
# POST-PROCESSING
psi_t = np.squeeze(results.y)  # shape: (len(t), dim)
populations = np.abs(psi_t)**2
labels = [f"|{format(i, f'0{N}b')}⟩" for i in range(2**N)]

# Final population
final_pop = populations[-1]
label_pop_pairs = list(zip(labels, final_pop))

# Include zero state
zero_state = f"|{'0'*N}⟩"
if zero_state not in dict(label_pop_pairs):
    label_pop_pairs.append((zero_state, 0.0))

# Top 10 + zero
top_pop = sorted(label_pop_pairs, key=lambda x: x[1], reverse=True)
top_set = {x[0] for x in top_pop[:10]}
if zero_state not in top_set:
    top_pop = top_pop[:9] + [(zero_state, dict(label_pop_pairs)[zero_state])]

top_pop = sorted(top_pop, key=lambda x: x[1], reverse=True)

# ─────────────────────────────────────────────────────────────────────
# PRINT RESULTS
print(f"\nTop 10 + Zero-State Populations for N = {N} Qubits:")
for state, prob in top_pop:
    print(f"{state}: {prob:.7f}")

# ─────────────────────────────────────────────────────────────────────
# PLOT: STATE POPULATIONS (top 10 only)
top_indices = [labels.index(state) for state, _ in top_pop]
legend_labels = [f"{labels[i]}: {populations[-1, i]:.7f}" for i in top_indices]

plt.figure(figsize=(12, 6))
for i, idx in enumerate(top_indices):
    plt.plot(t, populations[:, idx], label=legend_labels[i], linewidth=1.3)

plt.xlabel("Time")
plt.ylabel("Population")
plt.title(f"{N}-Qubit RAP: Top 10 + Zero State Populations")
plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────
# PLOT: CONTROL PULSES
Ω_vals = [Ω_pulse(tt) for tt in t]
Δ_vals = [Δ_pulse(tt) for tt in t]
plt.figure(figsize=(10, 3.5))
plt.plot(t, Ω_vals, label=r"$\Omega(t)$", color="navy", linewidth=1.6)
plt.plot(t, Δ_vals, label=r"$\Delta(t)$", color="crimson", linewidth=1.6)
plt.xlabel("Time")
plt.ylabel("Amplitude")
plt.title("RAP Control Pulses: Ω(t), Δ(t)")
plt.legend(loc="upper right", fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()
