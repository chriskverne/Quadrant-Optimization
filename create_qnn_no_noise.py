import pennylane as qml
import numpy as np


def create_qnn(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, params, num_meas):
        for i in range(n_qubits):
            qml.RX(inputs[i], wires=i)

        for layer in range(n_layers):
            # Rotational Gates
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RY(params[layer][qubit][1], wires=qubit)
                qml.RZ(params[layer][qubit][2], wires=qubit)

            # Apply CNOT gates for entanglement
            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CNOT(wires=[qubit, next_qubit])

        return qml.probs(wires=range(num_meas))

    # Return the decorated QNode function
    return circuit

