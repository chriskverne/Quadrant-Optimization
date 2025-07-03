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

def train_qnn_diagonal_fisher(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    fp = 0
    params = five_ten
    loss_history = []
    fp_history = []

    def cost_fn(params, image, label):
        out = forward_pass(image, params, num_measurment_gates)
        return cross_entropy_loss(out, label)

    # --- CHANGE 1: Define a new function for the log-likelihood gradient ---
    # This function computes the log of the probability of the true label.
    def log_likelihood_fn(params, image, label):
        out = forward_pass(image, params, num_measurment_gates)
        # Apply softmax to get probabilities and add a small epsilon for numerical stability
        probs = pnp.exp(out) / pnp.sum(pnp.exp(out)) + 1e-9
        return pnp.log(probs[label])

    # Gradient of the log-likelihood for the Fisher information
    log_like_grad_fn = qml.grad(log_likelihood_fn, argnum=0)
    
    # Gradient of the loss function for the parameter updates
    loss_grad_fn = qml.grad(cost_fn, argnum=0)

    # Tracks which parameters are active (1) or frozen (0)
    active_p = pnp.ones_like(params)

    # --- CHANGE 2: Setup for Fisher Information ---
    # Tracks the diagonal Fisher information
    fisher_info = pnp.zeros_like(params)

    freeze_t = 0.70

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
        
        # Reset Fisher info at the start of each epoch to get a fresh estimate
        fisher_info = pnp.zeros_like(params)

        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Compute loss with current parameters
            out = forward_pass(image, params, num_measurment_gates)
            fp += 1
            loss = cross_entropy_loss(out, label)
            epoch_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1

            # --- CHANGE 3: Calculate diagonal Fisher and accumulate ---
            # This requires an extra gradient calculation per sample
            log_like_grads = log_like_grad_fn(params, image, label)
            # The diagonal Fisher for a single sample is the square of the log-likelihood gradient
            fisher_info += pnp.square(log_like_grads)
            
            # Here, an extra 2*n_params forward passes are simulated for the gradient of the log-likelihood
            fp += 2 * len(params.flatten())

            # Compute gradients for the parameter update
            loss_gradients = loss_grad_fn(params, image, label)
            loss_gradients *= active_p
            fp += 2 * pnp.sum(active_p)

            # Update active params only
            params -= 0.01 * loss_gradients
        
        # --- CHANGE 4: Freezing logic based on Fisher Information ---
        # We use the accumulated Fisher info over the epoch as the importance score
        # Note: We average the Fisher info over the batch size
        avg_fisher_info = fisher_info / len(x_t)
        importance_scores = avg_fisher_info.flatten() + 1e-9 # Add epsilon for stability
        probs = importance_scores / pnp.sum(importance_scores)
        
        n_active = int(len(importance_scores) * (1 - freeze_t))
        active_indices = pnp.random.choice(
            len(importance_scores),
            size=n_active,
            replace=False,
            p=probs
        )

        new_active_flat = pnp.zeros(len(importance_scores))
        new_active_flat[active_indices] = 1
        active_p = new_active_flat.reshape(params.shape)

        # No need to reset fisher_info here as it's reset at the start of the epoch

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
num_epochs = 500
x = preprocess_image(x, num_components)

train_qnn_diagonal_fisher(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)