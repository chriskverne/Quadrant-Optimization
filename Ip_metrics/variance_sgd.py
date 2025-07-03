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
    active_p = pnp.ones_like(params, requires_grad=False)

    # --- Setup for Online Variance Freezing ---
    freeze_t = 0.70
    tau_window_size = 100
    n_in_window = 0
    running_mean = pnp.zeros_like(params, requires_grad=False)
    running_S = pnp.zeros_like(params, requires_grad=False)

    """Epoch 0 eval"""
    x_k = x[0:100]
    y_k = y[0:100]
    temp_loss = 0
    for i, l in zip(x_k, y_k):
        o = forward_pass(i, params, num_measurment_gates)
        temp_loss += cross_entropy_loss(o, l)

    print(f'Epoch 0 loss: {temp_loss/100}')

    """Training Loop"""
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 100
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
            gradients *= active_p

            # Update online variance trackers
            n_in_window += 1
            delta = gradients - running_mean
            running_mean += delta / n_in_window
            delta2 = gradients - running_mean
            running_S += delta * delta2

            # increase fp by 2*n_active_params
            fp += 2*pnp.sum(active_p)

            # Update active params only
            params -= 0.01* gradients
            
            # --- Freezing logic moved and corrected ---
            if n_in_window >= tau_window_size:
                # Calculate the final sample variance for the window
                variance = running_S / (n_in_window - 1) if n_in_window > 1 else pnp.zeros_like(params)

                # Importance is proportional to variance (keep high-variance params active)
                importance_scores = variance.flatten() + 1e-9 # Add epsilon for stability
                probs = importance_scores / pnp.sum(importance_scores)
                
                n_active = int(len(importance_scores) * (1 - freeze_t))
                active_indices = pnp.random.choice(
                    len(importance_scores),
                    size=n_active,
                    replace=False, 
                    p=probs
                )

                new_active_flat = pnp.zeros(len(importance_scores), requires_grad=False)
                new_active_flat[active_indices] = 1
                active_p = new_active_flat.reshape(params.shape)

                # Reset trackers for the next window
                n_in_window = 0
                running_mean = pnp.where(active_p == 0, running_mean, 0)
                running_S = pnp.where(active_p == 0, running_S, 0)
        
        # Calculate average loss and accuracy for the epoch
        avg_loss = epoch_loss / len(x_t) if len(x_t) > 0 else 0
        accuracy = correct_predictions / len(x_t) if len(x_t) > 0 else 0
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
num_epochs = 500
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)