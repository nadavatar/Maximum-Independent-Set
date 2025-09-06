from qiskit_dynamics import Solver
from qiskit_dynamics.signals import Signal
import numpy as np
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────
# RAP parameters
π = np.pi
Δ = 1.0
Ω = 1.0
Tf0 = 50 * π / Δ
V = 10
t = np.linspace(0, Tf0, 600)

def Ω_pulse(t_val): return Ω * np.sin(π * t_val / Tf0)
def Δ_pulse(t_val): return Δ * np.cos(π * t_val / Tf0)

# ─────────────────────────────────────────────────────────────────────
# Pauli matrices and Kronecker helpers
sx = np.array([[0, 1], [1, 0]], dtype=complex)
sz = np.array([[1, 0], [0, -1]], dtype=complex)
I2 = np.eye(2, dtype=complex)
def kron3(a, b, c): return np.kron(a, np.kron(b, c))

# Qubit operators
X1, X2, X3 = kron3(sx, I2, I2), kron3(I2, sx, I2), kron3(I2, I2, sx)
Z1, Z2, Z3 = kron3(sz, I2, I2), kron3(I2, sz, I2), kron3(I2, I2, sz)
I8 = np.eye(8, dtype=complex)

# RAP Hamiltonian
H_X = X1 + X2 + X3
n1 = 0.5 * (I8 - Z1)
n2 = 0.5 * (I8 - Z2)
n3 = 0.5 * (I8 - Z3)
H_n = n1 + n2 + n3
H_int = V * (n1 @ n2 + n2 @ n3)  # Nearest-neighbor blockade

# ─────────────────────────────────────────────────────────────────────
# Initial state |000⟩
initial_state = np.zeros((8, 1), dtype=complex)
initial_state[0, 0] = 1.0

# Solver
solver = Solver(hamiltonian_operators=[H_X, H_n, H_int])
signals = [
    Signal(lambda τ: Ω_pulse(τ), carrier_freq=0),
    Signal(lambda τ: -Δ_pulse(τ), carrier_freq=0),  # minus sign is crucial
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

# ─────────────────────────────────────────────────────────────────────
# Post-process results
psi_t = np.squeeze(results.y)
populations = np.abs(psi_t)**2

# ─────────────────────────────────────────────────────────────────────
# Plot: state populations
labels = [
    r"$|000\rangle$", r"$|001\rangle$", r"$|010\rangle$", r"$|011\rangle$",
    r"$|100\rangle$", r"$|101\rangle$", r"$|110\rangle$", r"$|111\rangle$"
]
cmap = plt.get_cmap("tab10")
plt.figure(figsize=(10, 5))
for k in range(8):
    plt.plot(t, populations[:, k], label=labels[k], color=cmap(k), linewidth=1.2)
plt.xlabel("Time")
plt.ylabel("Population")
plt.title("3-Qubit RAP: Discretisized Rydberg MIS Hamiltonian")
plt.legend(loc="upper right", ncol=2, fontsize="small")
plt.grid(True)
plt.tight_layout()
plt.show()

# ─────────────────────────────────────────────────────────────────────
# Plot: control pulses
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

# ─── 3. Print final state probabilities ─────────────────────────
print("Final state probabilities:")
for i, p in enumerate(populations[-1]):
    print(f"{labels[i]}: {p:.4f}")
