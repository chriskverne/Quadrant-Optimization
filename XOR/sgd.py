import pennylane as qml
import pennylane.numpy as pnp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from helper.get_xor_data import get_xor_data
from helper.create_qnn_xor import create_qnn_XOR
from helper.cross_entropy import cross_entropy_loss
from data.params import *

def train_qnn_param_shift(x, y, n_qubits, n_layers, num_epochs):
    forward_pass = create_qnn_XOR(n_layers, n_qubits)
    fp = 0
    params = five_ten
    loss_history = []
    fp_history = []

    def cost_fn(params, image, label):
        out = forward_pass(image, params)
        return cross_entropy_loss(out, label)
    grad_fn = qml.grad(cost_fn, argnum=0)
    
    """Training Loop"""
    for time_step in tqdm(range(num_epochs), desc="Time step"):
        s = 100
        x_t = x[time_step*s:(time_step+1)*s]
        y_t = y[time_step*s:(time_step+1)*s]
        epoch_loss = 0
        correct_predictions = 0
        
        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {time_step+1}/{num_epochs}", leave=False):
            # Compute loss with current parameters
            out = forward_pass(image, params)
            fp+=1
            loss = cross_entropy_loss(out, label)
            epoch_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1

            # compute gradients and apply only to active params
            gradients = grad_fn(params, image, label)

            # increase fp by 2*n_active_params
            fp += 2*params.size 

            # Update active params only
            params -= 0.01* gradients

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {time_step+1}/{num_epochs}, Avg Loss: {avg_loss[0]:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history

    
# --------------------------------- Model Setup ---------------------------
n_qubits = 10
n_layers = 5
n_epochs = 400
x,y = get_xor_data(n_qubits, 10000)


train_qnn_param_shift(x, y, n_qubits, n_layers, n_epochs)