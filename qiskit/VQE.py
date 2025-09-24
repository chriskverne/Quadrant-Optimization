import numpy as np
from dataclasses import dataclass

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.quantum_info import SparsePauliOp

# Use Aer Estimator primitive (fast, supports sampling)
try:
    from qiskit_aer.primitives import Estimator as AerEstimator
    Estimator = AerEstimator
except Exception:
    # Fallback to reference Estimator (slower but OK)
    from qiskit.primitives import Estimator  # type: ignore


# -----------------------------
# Hamiltonian builders (benchmarks)
# -----------------------------

def tfim_hamiltonian(n_qubits: int, h: float = 1.0, open_boundary: bool = True) -> SparsePauliOp:
    """Transverse-Field Ising Model (TFIM)
    H = -sum_{i} Z_i Z_{i+1} - h * sum_i X_i
    Args:
        n_qubits: number of qubits (>=2)
        h: transverse field strength
        open_boundary: if False, uses periodic boundary conditions
    Returns:
        SparsePauliOp encoding the Hamiltonian
    """
    paulis = []
    coeffs = []

    # Z Z interactions
    last = n_qubits if open_boundary else n_qubits + 1
    for i in range(n_qubits - 1 + (0 if open_boundary else 1)):
        a = i
        b = (i + 1) % n_qubits
        zstring = ["I"] * n_qubits
        zstring[a] = "Z"
        zstring[b] = "Z"
        paulis.append("".join(reversed(zstring)))  # Qiskit uses little-endian order in strings
        coeffs.append(-1.0)

    # Transverse field X
    for i in range(n_qubits):
        xstring = ["I"] * n_qubits
        xstring[i] = "X"
        paulis.append("".join(reversed(xstring)))
        coeffs.append(-h)

    return SparsePauliOp.from_list([(p, c) for p, c in zip(paulis, coeffs)])


def heisenberg_xxz_hamiltonian(n_qubits: int, delta: float = 1.0, open_boundary: bool = True) -> SparsePauliOp:
    """XXZ Heisenberg chain: H = sum_i (X_i X_{i+1} + Y_i Y_{i+1} + delta * Z_i Z_{i+1})
    Widely used as a second benchmark.
    """
    paulis = []
    coeffs = []
    links = range(n_qubits - 1) if open_boundary else range(n_qubits)
    for i in links:
        a = i
        b = (i + 1) % n_qubits
        for term, coef in (("X", 1.0), ("Y", 1.0), ("Z", delta)):
            s = ["I"] * n_qubits
            s[a] = term
            s[b] = term
            paulis.append("".join(reversed(s)))
            coeffs.append(coef)
    return SparsePauliOp.from_list([(p, c) for p, c in zip(paulis, coeffs)])


# -----------------------------
# Ansatz (hardware-efficient, scalable 2-4 qubits)
# -----------------------------

def build_ansatz(n_qubits: int, n_layers: int):
    """Hardware-efficient ansatz: input-agnostic, Rx-Rz layers + CZ ring.
    Returns circuit and ParameterVector θ of length n_layers * (2*n_qubits).
    """
    qc = QuantumCircuit(n_qubits)
    thetas = ParameterVector("θ", n_layers * 2 * n_qubits)
    idx = 0
    for _ in range(n_layers):
        # single-qubit rotations
        for q in range(n_qubits):
            qc.rx(thetas[idx], q); idx += 1
            qc.rz(thetas[idx], q); idx += 1
        # entangling CZ ring (linear chain for open boundary)
        for q in range(n_qubits - 1):
            qc.cz(q, q + 1)
    return qc, thetas


# -----------------------------
# VQE core (parameter-shift SGD)
# -----------------------------

@dataclass
class VQEConfig:
    n_qubits: int = 4
    n_layers: int = 2
    lr: float = 0.1
    epochs: int = 150
    seed: int = 7
    h: float = 1.0  # TFIM field strength
    shots: int | None = None  # None -> exact (default for AerEstimator), or set an int for sampling
    model: str = "tfim"  # or "xxz"
    open_boundary: bool = True


class VQE:
    def __init__(self, cfg: VQEConfig):
        assert 2 <= cfg.n_qubits <= 4, "This script targets 2-4 qubits as requested."
        self.cfg = cfg
        self.estimator = Estimator(run_options={"shots": cfg.shots} if cfg.shots else None)
        self.ansatz, self.params = build_ansatz(cfg.n_qubits, cfg.n_layers)
        rng = np.random.default_rng(cfg.seed)
        self.theta = rng.uniform(low=-0.1, high=0.1, size=len(self.params))

        if cfg.model == "tfim":
            self.H = tfim_hamiltonian(cfg.n_qubits, h=cfg.h, open_boundary=cfg.open_boundary)
        elif cfg.model == "xxz":
            self.H = heisenberg_xxz_hamiltonian(cfg.n_qubits, delta=1.0, open_boundary=cfg.open_boundary)
        else:
            raise ValueError("Unknown model: choose 'tfim' or 'xxz'")

    def energy(self, theta: np.ndarray) -> float:
        bound = self.ansatz.assign_parameters({p: float(v) for p, v in zip(self.params, theta)})
        res = self.estimator.run([bound], [self.H]).result()
        return float(res.values[0])

    def grad_parameter_shift(self, theta: np.ndarray) -> np.ndarray:
        grads = np.zeros_like(theta)
        shift = np.pi / 2
        for i in range(theta.size):
            t_plus = theta.copy();  t_plus[i] += shift
            t_minus = theta.copy(); t_minus[i] -= shift
            e_plus = self.energy(t_plus)
            e_minus = self.energy(t_minus)
            grads[i] = 0.5 * (e_plus - e_minus)
        return grads

    def train(self):
        history = []
        for epoch in range(1, self.cfg.epochs + 1):
            E = self.energy(self.theta)
            g = self.grad_parameter_shift(self.theta)
            self.theta = self.theta - self.cfg.lr * g
            history.append(E)
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch:3d} | Energy: {E:.6f} | ||grad||: {np.linalg.norm(g):.4e}")
        final_E = self.energy(self.theta)
        print(f"\nConverged Energy: {final_E:.8f}")
        return final_E, self.theta, np.array(history)


if __name__ == "__main__":
    # Example: 2-4 qubits; TFIM is a common research benchmark observable (ground-state energy)
    cfg = VQEConfig(n_qubits=4, n_layers=2, lr=0.15, epochs=120, seed=42, h=1.0, shots=None, model="tfim")
    vqe = VQE(cfg)
    vqe.train()
