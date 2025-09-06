# run_ibm_digitized_3qubit_v3.py
from qiskit import QuantumCircuit
from qiskit.circuit.library import PauliEvolutionGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime import QiskitRuntimeService, EstimatorV2 as Estimator, EstimatorOptions

# Build 3-qubit MIS Hamiltonian
paulis = ['ZII', 'IZI', 'IIZ', 'ZZI', 'IZZ']
coeffs = [1.0, 1.0, 1.0, 0.5, 0.5]
H = SparsePauliOp(paulis, coeffs=coeffs)

# Trotter evolution circuit
reps = 2
time = 1.0
step_gate = PauliEvolutionGate(H, time=time / reps, synthesis='trotter')
qc = QuantumCircuit(3)
for _ in range(reps):
    qc.append(step_gate, qc.qubits)
qc.measure_all()
print(qc.draw())

# Connect to IBM Quantum and select least busy real backend
service = QiskitRuntimeService()
backend = service.least_busy(simulator=False, operational=True)
print("Running on:", backend.name)

# Use EstimatorV2 with proper options
options = EstimatorOptions()
options.resilience_level = 1
options.default_shots = 4096

estimator = Estimator(mode=backend, options=options)

# Submit circuit–observable PUB; run returns a JobV2
job = estimator.run([(qc, H)])
print("Job ID:", job.job_id())

# Retrieve results
pub_results = job.result()
ev = pub_results[0].data.evs
std = pub_results[0].data.stds
print("Expectation value:", ev)
print("Standard error:", std)
# Save results to a text file
with open("3qubit_RAP_MIS_Trotter100_ibmq_cloud_results.txt", "w") as f:
    f.write(f"Expectation value: {ev}\n")
    f.write(f"Standard error: {std}\n")
print("Results saved to 3qubit_RAP_MIS_Trotter100_ibmq_cloud_results.txt")