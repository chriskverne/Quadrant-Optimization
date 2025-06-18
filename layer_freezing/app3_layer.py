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
    active_l = pnp.ones(n_layers)  # Initialize all as active

    # Tracks gradients to decide what to freeze
    sum_grads = pnp.zeros_like(params)

    freeze_t = 0.80
    
    """Training Loop"""
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 50
        random_indices = pnp.random.choice(len(x), size=s, replace=False)
        x_t = x[random_indices]
        y_t = y[random_indices]
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

            # Make gradients 0 for frozen layers, else add it to sum_grads
            for l in range(n_layers):
                if active_l[l] == 0:
                    gradients[l] = 0

            sum_grads += gradients

            # increase fp by 2*n_active_params
            fp += 2*pnp.sum(active_l)*n_qubits*2

            # Update active params only
            params -= 0.01* gradients
        
        # Decide what to freeze (mark as 0 for frozen, 1 for active)
        flat_grads = pnp.array([pnp.sum(pnp.abs(sum_grads[l])) for l in range(n_layers)])
        flat_grads += 1e-8
        probs = flat_grads / pnp.sum(flat_grads)
        n_active = max(1, int(n_layers * (1 - freeze_t)))
        active_indices = pnp.random.choice(
            n_layers, # how many to choose between
            size=n_active, # how many to choose
            replace=False, 
            p=probs # prob of choosing each
        )

        active_l = pnp.zeros(n_layers)
        active_l[active_indices] = 1
        print(f'Active Layer: {active_l}')

        # Reset sum_grads for active layers, keep for frozen layers
        for l in range(n_layers):
            if active_l[l] == 1:
                sum_grads[l] = 0

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

num_qubits = num_components = 10
num_layers = 5
num_measurment_gates = 2
num_epochs = 1500
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)