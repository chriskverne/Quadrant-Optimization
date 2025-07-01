import pennylane as qml

def create_qnn_XOR(n_layers, n_qubits):
    dev = qml.device('default.qubit', wires=n_qubits)

    @qml.qnode(dev)
    def circuit(inputs, params):
        for i in range(n_qubits):
            if inputs[i] == 1:
                qml.X(wires=i)

        for layer in range(n_layers):
            # Rotational Gates
            for qubit in range(n_qubits):
                qml.RX(params[layer][qubit][0], wires=qubit)
                qml.RZ(params[layer][qubit][1], wires=qubit)

            # Apply CNOT gates for entanglement
            for qubit in range(n_qubits):
                next_qubit = (qubit + 1) % n_qubits
                qml.CNOT(wires=[qubit, next_qubit])

        return qml.probs(wires=[0])

    # Return the decorated QNode function
    return circuit

