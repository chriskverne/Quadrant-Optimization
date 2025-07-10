import pennylane as qml
import pennylane.numpy as pnp
from functools import reduce # Optional, for a more compact way


def create_qnn_XOR(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, params):
        # Simple encoding
        for i in range(n_qubits):
            if inputs[i] == 1: # Try different encoding first
                qml.X(wires=i)
                # qml.RZ(pnp.pi, wires=i)

            # else:
            #     qml.H(wires=i)

        for layer in range(n_layers):
            # Rotational Gates
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RZ(params[layer][qubit][1], wires=qubit)

            # Cyclic entanglement
            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CNOT(wires=[qubit, next_qubit])

        return qml.probs(wires=[0]) # Try different output?
        # z_operators = [qml.PauliZ(i) for i in range(n_qubits)]
        # parity_observable = reduce(lambda a, b: a @ b, z_operators)
        # return qml.expval(parity_observable)

    # Return the decorated QNode function
    return circuit

