import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pennylane.numpy as pnp
import math
import pandas as pd
from tqdm import tqdm
from helper.fetch_mnist import preprocess_image
from helper.create_qnn_no_noise import create_qnn
from helper.cross_entropy import cross_entropy_loss
from data.params import *

"""
Stochastic Gradient Post Descent

Todo:
Drop Ry in ansatz
Restrict Rx to 0-pi range
How much gradient varies (STD)

1) Try EMA Gradients so the average is weighted more based on recent gradients
2) Maybe reset grad_history for unfrozen parameters?? I.e. full reset for parameters frozen for a long time
"""

def train_qnn_param_shift(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    freeze_t = 0.80
    unfreeze_p = 0.10
    fp=0    
    params = three_six_two

    # Tracks which parameters are marked as frozen
    frozen_p = pnp.zeros_like(params)

    # Tracks the duration each parameter has been frozen for
    frozen_dur = pnp.zeros_like(params)

    # Tracks sum of gradients for each parameter
    grad_history = pnp.zeros_like(params)

    # Tracks number of grads for each param (to get average gradient over an epoch)
    no_grads = pnp.zeros_like(params)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 50
        #indices = pnp.random.choice(len(x), size=s, replace=False)
        x_t = x[epoch*s:(epoch+1)*s] #x[indices]
        y_t = y[epoch*s:(epoch+1)*s]#y[indices]
        total_loss = 0
        correct_predictions = 0
        
        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Compute forward pass with current parameters
            out = forward_pass(image, params, num_measurment_gates)
            fp+=1
            loss = cross_entropy_loss(out, label)
            total_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1

            # Calculate gradients
            dL_dp = pnp.zeros_like(out)
            grads = pnp.zeros_like(params)
            dL_dp[label] = -1.0 / (out[label] + 1e-10)

            for l in range(n_layers):
                for q in range(n_qubits):
                    for g in range(2):
                        if frozen_p[l,q,g] == 1:
                            frozen_dur[l,q,g] += 1
                            continue
                        
                        no_grads[l,q,g]+=1

                        params_plus = params.copy()
                        params_plus[l,q,g] += pnp.pi/2
                        params_minus = params.copy()
                        params_minus[l,q,g] -= pnp.pi/2

                        grad = (forward_pass(image, params_plus, num_measurment_gates) - forward_pass(image, params_minus, num_measurment_gates))/2
                        fp+=2
                        grads[l,q,g] = pnp.dot(dL_dp, grad)

            # Add gradients to param history
            grad_history += grads # Not negative. I.e. negative and positive gradients will cancel each other out which can prevent osciliation (might be good or bad not sure)

            # Update params which havent been frozen
            params -= 0.01*grads

        # Decide what to freeze
        avg_grad = grad_history / no_grads
        sorted_abs_history = pnp.sort(pnp.abs(avg_grad.flatten()))
        idx = int(len(sorted_abs_history) * freeze_t)
        threshold = sorted_abs_history[idx]
        frozen_p = pnp.where(pnp.abs(avg_grad) <= threshold, 1, 0)

        # Decide what to unfreeze
        frozen_indices_flat = pnp.where(frozen_p.flatten() == 1)[0]
        frozen_durations = frozen_dur.flatten()[frozen_indices_flat]
        num_to_unfreeze = int(pnp.ceil(unfreeze_p * frozen_indices_flat.size))
        longest_frozen_relative_indices = pnp.argsort(frozen_durations)[-num_to_unfreeze:] # Get indices of top longest durations
        indices_to_unfreeze_flat = frozen_indices_flat[longest_frozen_relative_indices]
        multi_dim_indices_to_unfreeze = pnp.unravel_index(indices_to_unfreeze_flat, params.shape)

        # Reset count / grads for unfrozen params
        frozen_p[multi_dim_indices_to_unfreeze] = 0
        frozen_dur[multi_dim_indices_to_unfreeze] = 0
        grad_history[multi_dim_indices_to_unfreeze] = 0
        no_grads[multi_dim_indices_to_unfreeze] = 0


        avg_loss = total_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        print(f"\nNo FP: {fp}, Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params

    
# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/two_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

digits = [0,1]
num_qubits = num_components = 6
num_layers = 3
num_measurment_gates = math.ceil(pnp.log2(len(digits)))
num_epochs = 40
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)