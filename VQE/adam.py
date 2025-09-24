import pennylane as qml
import pennylane.numpy as pnp
from tqdm import tqdm

# ------------------------ Problem: TFIM Hamiltonian ------------------------
# H = -J Σ_{i=0}^{N-2} Z_i Z_{i+1} - h Σ_{i=0}^{N-1} X_i  (open chain)
def make_tfim_hamiltonian(n_qubits, J=1.0, h=1.0):
    zz_ops = [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n_qubits - 1)]
    x_ops = [qml.PauliX(i) for i in range(n_qubits)]
    coeffs = [-J] * (n_qubits - 1) + [-h] * n_qubits
    ops = zz_ops + x_ops
    return qml.Hamiltonian(coeffs, ops)

# ------------------------ Ansatz & Energy QNode -----------------------------
# Hardware-efficient: per-layer Ry, Rz on each wire + CZ chain entanglement.
def build_energy_fn(n_layers, n_qubits):
    dev = qml.device("default.qubit", wires=n_qubits)
    H = make_tfim_hamiltonian(n_qubits)

    def ansatz(params):
        for l in range(n_layers):
            for w in range(n_qubits):
                qml.RY(params[l, w, 0], wires=w)
                qml.RZ(params[l, w, 1], wires=w)
            for w in range(n_qubits - 1):
                qml.CZ(wires=[w, w + 1])

    @qml.qnode(dev, interface="autograd")
    def energy(params):
        ansatz(params)
        return qml.expval(H)

    return energy

# ------------------------ Params Init --------------------------------------
def init_params(n_layers, n_qubits, seed=1234):
    rng = pnp.random.default_rng(seed)
    return rng.normal(0.0, 2*pnp.pi, size=(n_layers, n_qubits, 2))

# ------------------------ Training (VQE) -----------------------------------
def train_vqe(n_qubits, n_layers, num_epochs, lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8):
    energy = build_energy_fn(n_layers, n_qubits)
    params = init_params(n_layers, n_qubits)

    # Adam state
    m = pnp.zeros_like(params)
    v = pnp.zeros_like(params)
    t = 0

    loss_history = []
    grad_fn = qml.grad(lambda th: energy(th))

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        # Forward/loss
        E = energy(params)
        loss_history.append(E)

        # Gradients (param-shift via autograd)
        g = grad_fn(params)

        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - lr * m_hat / (pnp.sqrt(v_hat) + eps)

        if epoch % max(1, num_epochs // 10) == 0 or epoch == 1:
            print(f"Epoch {epoch:4d} | Energy: {E:.8f}")

    return params, loss_history

# ------------------------ Run ------------------------------------------------
if __name__ == "__main__":
    n_qubits = 6
    n_layers = 3
    num_epochs = 200
    params, history = train_vqe(n_qubits, n_layers, num_epochs)
