import numpy as np
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pandas as pd

# Qiskit imports
from qiskit import QuantumCircuit
from qiskit.circuit import ParameterVector
from qiskit.primitives import Sampler
from qiskit_algorithms.gradients import ParamShiftSamplerGradient
from qiskit_aer import AerSimulator

# Import your helper functions (we'll need to adapt these too)
from helper.fetch_mnist import fetch_mnist, preprocess_image
from data.params import *

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
            qc.ry(weight_params[param_idx], qubit)
            param_idx += 1
            qc.rz(weight_params[param_idx], qubit)
            param_idx += 1
        
        # Entanglement layer (circular coupling)
        for i in range(n_qubits):
            qc.cx(i, (i + 1) % n_qubits)
    
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

def cross_entropy_loss(output_distribution, true):
    epsilon = 1e-10
    pred = output_distribution[true]
    return -pnp.log(pred + epsilon)

def train_qnn(n_layers, n_qubits, x, y, lr=0.01):
    qc, input_params, weight_params = create_qnn_circuit(n_layers, n_qubits)
    params = np.random.rand(n_layers * n_qubits * 2)
    fp = 0

    while fp < 1000000:
        for image, label in zip(x, y):
            bound_circ = bind_circuit(qc, input_params, weight_params, image, params)
            # execute this circuit to get outputdist

            for i in range(params):
                params_plus = params.copy()
                params_plus[i] += np.pi/2
                bound_circ = bind_circuit(qc, input_params, weight_params, image, params_plus)
                out_plus = ? # execute this circuit to get outputdist

                params_minus = params.copy()
                params_minus -= np.pi/2
                bound_circ = bind_circuit(qc, input_params, weight_params, image, params_minus)
                out_minus = ? # execute this circuit to get outputdist

inp1 = np.array([1,2,3,4])
params = np.array([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
qc, input_params, weight_params = create_qnn_circuit(4, 2)

