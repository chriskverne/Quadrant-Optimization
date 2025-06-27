import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

# Qiskit imports
from qiskit import QuantumCircuit, ClassicalRegister
from qiskit.circuit import ParameterVector
from qiskit_algorithms.gradients import ParamShiftSamplerGradient
from qiskit_aer import AerSimulator

# Import your helper functions (we'll need to adapt these too)
from helper.fetch_mnist import fetch_mnist, preprocess_image
from helper.cross_entropy import cross_entropy_loss, cross_entropy_grad
from data.params import *

def cross_entropy_grad_softmax(softmax_probabilities, true_label_index):
    num_classes = len(softmax_probabilities)
    y_one_hot = np.zeros(num_classes)
    y_one_hot[true_label_index] = 1.0
    return softmax_probabilities - y_one_hot

def create_qnn_circuit(n_layers, n_qubits):
    """Create a parameterized quantum circuit for QNN"""
    qc = QuantumCircuit(n_qubits)
    
    # Input encoding parameters
    input_params = ParameterVector('x', n_qubits)
    
    # Trainable parameters
    num_params_per_layer = n_qubits * 2  # Ry->Rz, CNOT, ...
    weight_params = ParameterVector('θ', n_layers * num_params_per_layer)
    
    # Encode input data
    for i in range(n_qubits):
        qc.rx(input_params[i], i)
    
    # Variational layers
    param_idx = 0
    for layer in range(n_layers):
        # Rotation layer
        for qubit in range(n_qubits):
            qc.rx(weight_params[param_idx], qubit)
            param_idx += 1
            qc.rz(weight_params[param_idx], qubit)
            param_idx += 1
        
        # Entanglement layer (circular coupling)
        for i in range(n_qubits):
            qc.cz(i, (i + 1) % n_qubits)
    
    return qc, input_params, weight_params

def bind_circuit(qc, input_params, weight_params, input_values, param_values):
    """
    Args:
        - qc: QuantumCircuit with parameterized gates
        - input_params: ParameterVector for input encoding (Rx gates)
        - weight_params: ParameterVector for trainable weights (Ry, Rz gates)
        - input_values: single input sample [n_qubits]
        - param_values: flattened weight params [n_layers * n_qubits * 2]
    
    Returns:
        - bound_circuit: Circuit with all parameters bound to concrete values
    """
    
    # Create parameter binding dictionary
    param_dict = {}
    
    # Bind input parameters to input values
    for i in range(len(input_params)):
        param_dict[input_params[i]] = input_values[i]
    
    # Bind weight parameters to weight values
    for i in range(len(weight_params)):
        param_dict[weight_params[i]] = param_values[i]
    
    # Bind all parameters to the circuit
    bound_circuit = qc.assign_parameters(param_dict)
    
    return bound_circuit

def run_qc(bound_circ, shots=1000, num_meas_q=None):
    """
    Executes the circuit and returns the output distribution
    Only measures num_meas_q qubits returning {2^num_meas_q: c} values
    """
    backend = AerSimulator()

    # Add measurement to first num_meas_q qubits
    bound_circ.add_register(ClassicalRegister(num_meas_q))
    for i in range(num_meas_q):
        bound_circ.measure(i, i)

    # Execute the circuit
    job = backend.run(bound_circ, shots=shots)
    result = job.result()
    counts = result.get_counts(bound_circ)
    
    # convert dict to array
    result = np.zeros(2**num_meas_q, dtype=int)
    for binary_string, count in counts.items():
        index = int(binary_string, 2)
        result[index] = count

    # normalize to probabilities for softmax
    return result/shots 

def train_qnn(n_layers, n_qubits, x, y, lr=0.01, num_meas_q=None):
    qc, input_params, weight_params = create_qnn_circuit(n_layers, n_qubits)
    params = np.random.rand(n_layers * n_qubits * 2)
    fp = 0

    for time_step in range(100):
        timestep_loss = 0
        correct_predictions = 0
        s = 50
        x_t = x[time_step*s:(time_step+1)*s]
        y_t = y[time_step*s:(time_step+1)*s]

        for image, label in zip(x_t, y_t):
            bound_circ = bind_circuit(qc, input_params, weight_params, image, params)
            out = run_qc(bound_circ, num_meas_q=num_meas_q)
            fp+=1
            timestep_loss += cross_entropy_loss(out, label)
            pred = np.argmax(out)
            if pred == label:
                correct_predictions += 1

            loss_grad = cross_entropy_grad_softmax(out, label)

            # Compute N gradients with PSR
            grads = np.zeros(n_layers*n_qubits*2)
            for i in range(len(params)):
                params_plus = params.copy()
                params_plus[i] += np.pi/2
                bound_circ = bind_circuit(qc, input_params, weight_params, image, params_plus)
                out_plus = run_qc(bound_circ, num_meas_q=num_meas_q)
                fp+=1

                params_minus = params.copy()
                params_minus[i] -= np.pi/2
                bound_circ = bind_circuit(qc, input_params, weight_params, image, params_minus)
                out_minus = run_qc(bound_circ, num_meas_q=num_meas_q)
                fp+=1

                grad = (out_plus - out_minus) / 2
                grads[i] = np.dot(grad, loss_grad)
            
            params -= lr*grads # Update params with SGD
        
        print(f"\nNo FP: {fp}, Epoch {time_step+1}, Avg Loss: {timestep_loss/s:.4f}, Accuracy: {correct_predictions/s:.2%}")

n_layers = 2
n_qubits = 4
df = pd.read_csv('./data/two_digit.csv')
x = df.drop('label', axis=1).values
x = preprocess_image(x, n_qubits)
y = df['label'].values
num_meas_q = 1

train_qnn(n_layers, n_qubits, x, y, lr=0.01, num_meas_q=num_meas_q)