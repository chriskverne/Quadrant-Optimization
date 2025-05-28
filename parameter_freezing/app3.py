"""
Change to Probabilistc Freezing
"""

import pennylane as qml
import pennylane.numpy as pnp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import math
from tqdm import tqdm
from helper.fetch_mnist import fetch_mnist, preprocess_image
from helper.create_qnn_no_noise import create_qnn
from helper.cross_entropy import cross_entropy_loss
from data.params import *
import pandas as pd

def train_qnn_param_shift(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    fp = 0
    params = five_ten
    loss_history = []
    fp_history = []

    def cost_fn(params, image, label):
        out = forward_pass(image, params, num_measurment_gates)
        return cross_entropy_loss(out, label)
    grad_fn = qml.grad(cost_fn, argnum=0)

    # Tracks which parameters are marked as active (1) or frozen (0)
    active_p = pnp.ones_like(params)  # Initialize all as active

    # Tracks gradients to decide what to freeze
    sum_grads = pnp.zeros_like(params)

    freeze_t = 0.70
    
    """Training Loop"""
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 50
        x_t = x[epoch*s:(epoch+1)*s]
        y_t = y[epoch*s:(epoch+1)*s]
        epoch_loss = 0
        correct_predictions = 0
        
        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Compute loss with current parameters
            out = forward_pass(image, params, num_measurment_gates)
            fp+=1
            loss = cross_entropy_loss(out, label)
            epoch_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1

            # compute gradients and apply only to active params
            gradients = grad_fn(params, image, label)
            gradients *= active_p  # Only active params (1) keep their gradients

            # Add gradients to sum
            sum_grads += gradients

            # increase fp by 2*n_active_params
            fp += 2*pnp.sum(active_p)  # Count active parameters (where active_p=1)

            # Update active params only
            params -= 0.01* gradients
        
        # Decide what to freeze (mark as 0 for frozen, 1 for active)
        flat_grads = pnp.abs(sum_grads.flatten())
        probs = flat_grads / pnp.sum(flat_grads) # convert to probabilities
        n_active = int(len(flat_grads) * (1 - freeze_t))
        active_indices = pnp.random.choice(
            len(flat_grads), # how many to choose between
            size=n_active, # how many to choose
            replace=False, 
            p=probs # prob of choosing each
        )

        new_active_flat = pnp.zeros(len(flat_grads))
        new_active_flat[active_indices] = 1 # add the randomly sampled indicies to active params
        active_p = new_active_flat.reshape(params.shape)

        # Reset sum_grads for active params, keep for frozen params
        sum_grads = pnp.where(active_p == 0, sum_grads, 0)  # Frozen params (0) keep sum_grads, active params (1) reset to 0

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history

    
# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

digits = [0,1,2,3]
num_qubits = num_components = 10
num_layers = 5
num_measurment_gates = math.ceil(pnp.log2(len(digits)))
num_epochs = 500
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)