import pennylane as qml
import pennylane.numpy as pnp
from tqdm import tqdm
import numpy as np

# ------------------------ Problem: TFIM Hamiltonian ------------------------
def make_tfim_hamiltonian(n_qubits, J=1.0, h=1.0):
    zz_ops = [qml.PauliZ(i) @ qml.PauliZ(i+1) for i in range(n_qubits - 1)]
    x_ops = [qml.PauliX(i) for i in range(n_qubits)]
    coeffs = [-J] * (n_qubits - 1) + [-h] * n_qubits
    ops = zz_ops + x_ops
    return qml.Hamiltonian(coeffs, ops)

# ------------------------ Noise helpers ------------------------------------
def tphi_from_t1_t2(T1, T2, eps=1e-12):
    # 1/T2 = 1/(2T1) + 1/Tphi  ->  Tphi = 1 / (1/T2 - 1/(2T1))
    inv = np.maximum(1.0/np.array(T2) - 0.5/np.array(T1), eps)
    return 1.0 / inv

def prob_from_time(T, dt):
    # probability that an error occurs over interval dt for exponential process
    # p = 1 - exp(-dt/T)
    T = np.array(T)
    dt = np.array(dt)
    return 1.0 - np.exp(-dt / np.maximum(T, 1e-12))

def safe_clip(p):
    return float(np.clip(p, 0.0, 1.0))

def apply_decoherence(dt, T1, T2, wire):
    """Apply amplitude + phase damping corresponding to free-evolution over dt."""
    p_amp = safe_clip(prob_from_time(T1, dt))
    Tphi = tphi_from_t1_t2(T1, T2)
    p_phase = safe_clip(prob_from_time(Tphi, dt))
    if p_amp > 0:
        qml.AmplitudeDamping(p_amp, wires=wire)
    if p_phase > 0:
        qml.PhaseDamping(p_phase, wires=wire)

def apply_gate_noise_1q(err_1q_dep, dt_1q, T1, T2, wire):
    # Coherent gate time causes decoherence + optional depolarizing for control errors.
    apply_decoherence(dt_1q, T1, T2, wire)
    if err_1q_dep and err_1q_dep > 0:
        qml.DepolarizingChannel(safe_clip(err_1q_dep), wires=wire)

def apply_gate_noise_2q(err_2q_dep, dt_2q, T1s, T2s, wires):
    for w, T1, T2 in zip(wires, T1s, T2s):
        apply_decoherence(dt_2q, T1, T2, w)
    # Optional 2q depolarizing to model control errors/crosstalk on entangler
    if err_2q_dep and err_2q_dep > 0:
        try:
            qml.TwoQubitDepolarizingChannel(safe_clip(err_2q_dep), wires=wires)
        except Exception:
            # Fallback: apply local depolarizing to each qubit if TwoQubitDepolarizingChannel unavailable
            for w in wires:
                qml.DepolarizingChannel(safe_clip(err_2q_dep), wires=w)

def apply_readout_error(p_ro, wire):
    # Simple symmetric assignment error model
    if p_ro and p_ro > 0:
        qml.BitFlip(safe_clip(p_ro), wires=wire)

# ------------------------ Ansatz & Energy QNode (noisy) --------------------
def build_energy_fn(n_layers, n_qubits, noise_cfg=None, shots=2000):
    """
    noise_cfg:
      {
        "T1": [s]*n,          # seconds
        "T2": [s]*n,          # seconds
        "t_1q": 35e-9,        # seconds
        "t_2q": 200e-9,       # seconds
        "t_idle": 50e-9,      # seconds between layers (optional)
        "p_1q_dep": 0.001,    # depolarizing per 1q gate (optional)
        "p_2q_dep": 0.02,     # depolarizing per 2q gate (optional)
        "p_readout": [..]*n   # readout assignment error per qubit (optional)
      }
    """
    if noise_cfg is None:
        # default "clean" (no additional gate depol) but still mixed-state device
        noise_cfg = {
            "T1": [30e-6]*n_qubits,
            "T2": [20e-6]*n_qubits,
            "t_1q": 35e-9,
            "t_2q": 200e-9,
            "t_idle": 0.0,
            "p_1q_dep": 0.0,
            "p_2q_dep": 0.0,
            "p_readout": [0.0]*n_qubits
        }

    dev = qml.device("default.mixed", wires=n_qubits, shots=shots)
    H = make_tfim_hamiltonian(n_qubits)

    T1 = noise_cfg["T1"]
    T2 = noise_cfg["T2"]
    t_1q = noise_cfg.get("t_1q", 0.0)
    t_2q = noise_cfg.get("t_2q", 0.0)
    t_idle = noise_cfg.get("t_idle", 0.0)
    p_1q_dep = noise_cfg.get("p_1q_dep", 0.0)
    p_2q_dep = noise_cfg.get("p_2q_dep", 0.0)
    p_ro = noise_cfg.get("p_readout", [0.0]*n_qubits)

    def ansatz(params):
        for l in range(n_layers):
            # per-qubit rotations + their gate noise
            for w in range(n_qubits):
                qml.RY(params[l, w, 0], wires=w)
                apply_gate_noise_1q(p_1q_dep, t_1q, T1[w], T2[w], w)

                qml.RZ(params[l, w, 1], wires=w)
                apply_gate_noise_1q(p_1q_dep, t_1q, T1[w], T2[w], w)

            # entangling CZ chain + 2q noise
            for w in range(n_qubits - 1):
                qml.CZ(wires=[w, w + 1])
                apply_gate_noise_2q(p_2q_dep, t_2q, [T1[w], T1[w+1]], [T2[w], T2[w+1]], [w, w+1])

            # idle decoherence between layers
            if t_idle and t_idle > 0:
                for w in range(n_qubits):
                    apply_decoherence(t_idle, T1[w], T2[w], w)

        # measurement (readout) error just before expectation
        for w in range(n_qubits):
            apply_readout_error(p_ro[w], w)

    @qml.qnode(dev, interface="autograd", diff_method="parameter-shift")
    def energy(params):
        ansatz(params)
        return qml.expval(H)

    return energy

# ------------------------ Params Init --------------------------------------
def init_params(n_layers, n_qubits, seed=1234):
    rng = pnp.random.default_rng(seed)
    return rng.normal(0.0, 2*pnp.pi, size=(n_layers, n_qubits, 2))

# ------------------------ Training (VQE) -----------------------------------
def train_vqe(n_qubits, n_layers, num_epochs, noise_cfg=None, shots=2000,
              lr=0.01, beta1=0.9, beta2=0.999, eps=1e-8, seed=1234):
    energy = build_energy_fn(n_layers, n_qubits, noise_cfg=noise_cfg, shots=shots)
    params = init_params(n_layers, n_qubits, seed=seed)
    print(f'Initial energy: {energy(params)}')

    # Adam state
    fp = 0
    m = pnp.zeros_like(params)
    v = pnp.zeros_like(params)
    t = 0

    loss_history = []
    grad_fn = qml.grad(lambda th: energy(th))

    for epoch in tqdm(range(1, num_epochs + 1), desc="Epochs"):
        # Forward/loss (finite-shots => stochastic)
        E = energy(params)
        loss_history.append((fp, E.item()))

        # Gradients (parameter-shift works with channels on default.mixed)
        g = grad_fn(params)
        fp += 2*params.size

        # Adam update
        t += 1
        m = beta1 * m + (1 - beta1) * g
        v = beta2 * v + (1 - beta2) * (g ** 2)
        m_hat = m / (1 - beta1 ** t)
        v_hat = v / (1 - beta2 ** t)
        params = params - lr * m_hat / (pnp.sqrt(v_hat) + eps)

        if epoch % 5 == 0:
            print(f"Epoch {epoch:4d} | Energy (shot-est.): {float(E):.8f} FP {fp}")

    return params, loss_history

# ------------------------ Example real-ish calibration ----------------------
# Replace these with your device’s stats (seconds & probabilities).
def build_noise_from_calibration(n_qubits, calib):
    # calib = {"T1": [...], "T2": [...], "p1q": ..., "p2q": ..., "pro": [...]}
    return dict(
        T1=calib.get("T1", [30e-6]*n_qubits),
        T2=calib.get("T2", [20e-6]*n_qubits),
        t_1q=calib.get("t_1q", 35e-9),
        t_2q=calib.get("t_2q", 200e-9),
        t_idle=calib.get("t_idle", 50e-9),
        p_1q_dep=calib.get("p1q_dep", 1e-3),
        p_2q_dep=calib.get("p2q_dep", 2e-2),
        p_readout=calib.get("p_ro", [0.02]*n_qubits),
    )

# ------------------------ Run ------------------------------------------------
if __name__ == "__main__":
    n_qubits = 4
    n_layers = 1
    num_epochs = 500

    # Example: populate with your real calibration data
    calib = {
        # T1/T2 in seconds (from the first two columns, µs → s)
        "T1": [56.89e-6, 372.73e-6, 311.97e-6, 301.17e-6, 292.19e-6, 307.90e-6, 167.87e-6, 414.46e-6, 253.11e-6, 225.45e-6],
        "T2": [82.74e-6, 343.29e-6, 320.36e-6, 294.83e-6, 296.07e-6, 276.10e-6, 218.31e-6, 355.27e-6, 331.20e-6, 352.94e-6],

        # Gate durations (from the table: 1q=32 ns, 2q=88 ns), idle between layers ~2.584 µs
        "t_1q": 32e-9,
        "t_2q": 88e-9,
        "t_idle": 2.584e-6,

        # One-qubit depolarizing per 1q gate.
        # Chosen as the *median* of the per-qubit 1q error column
        # (values like 4.353e-4, 2.54e-4, ..., 1.04e-3) → ~3.4465e-4
        "p1q_dep": 3.45e-4,

        # Two-qubit depolarizing per entangler.
        # The edge-specific entries vary a lot (e.g., 0–1≈1.202e-2, 4–5≈7.253e-3, 8–9≈1.393e-2, many others ≈1–2e-3).
        # For a single scalar, a sensible “balanced” choice is the mean over listed edges ≈4.54e-3;
        # if you want to be conservative, use ~1.20e-2. Pick ONE of these:
        "p2q_dep": 4.54e-3,   # balanced average
        # "p2q_dep": 1.20e-2, # conservative (matches your initial choice)

        # Readout assignment error per qubit (from the third number in each qubit row)
        "p_ro": [5.981e-3, 6.348e-3, 6.226e-3, 4.639e-3, 3.174e-3, 3.784e-3, 5.444e-2, 3.418e-3, 1.245e-2, 6.335e-2],
    }
    
    noise_cfg = build_noise_from_calibration(n_qubits, calib)
    
    print("Target ground state energy:", np.linalg.eigvalsh(qml.matrix(make_tfim_hamiltonian(n_qubits)))[0])

    # Include finite shots to add sampling noise
    params, history = train_vqe(
        n_qubits, n_layers, num_epochs,
        noise_cfg=noise_cfg, shots=1000, lr=0.012
    )

    print(history)
