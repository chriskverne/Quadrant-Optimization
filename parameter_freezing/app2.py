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
    # Tracks the duration each parameter has been frozen for
    frozen_dur = pnp.zeros_like(params)
    # Tracks gradients to decide what to freeze
    sum_grads = pnp.zeros_like(params)

    freeze_t = 0.80
    temperature = 1000
    
    """Training Loop"""
    for time_step in tqdm(range(num_epochs), desc="Time step"):
        s = 50
        x_t = x[time_step*s:(time_step+1)*s]
        y_t = y[time_step*s:(time_step+1)*s]
        epoch_loss = 0
        correct_predictions = 0
        
        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {time_step+1}/{num_epochs}", leave=False):
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
            frozen_dur += (1-active_p) # add 1 to all params set to 0

            # Add gradients to sum
            sum_grads += gradients

            # increase fp by 2*n_active_params
            fp += 2*pnp.sum(active_p)  # Count active parameters (where active_p=1)

            # Update active params only
            params -= 0.01* gradients
        
        # Decide what to freeze (mark as 0 for frozen, 1 for active)
        sorted_abs_history = pnp.sort(pnp.abs(sum_grads.flatten()))
        idx = int(len(sorted_abs_history) * freeze_t)
        threshold = sorted_abs_history[idx]
        active_p = pnp.where(pnp.abs(sum_grads) <= threshold, 0, 1)  # Small gradients become frozen (0)

        # Probabilistic unfreezing
        unfreeze_probs = pnp.minimum(frozen_dur / temperature, 1) # max unfreezing 100%
        random_vals = pnp.random.random(params.shape)
        newly_unfrozen = (random_vals < unfreeze_probs)
        #print(pnp.sum(newly_unfrozen))
        active_p = pnp.where(newly_unfrozen, 1, active_p)

        # Reset sum_grads & frozen_dur for active params, keep for frozen params
        sum_grads = pnp.where(active_p == 0, sum_grads, 0) 
        frozen_dur = pnp.where(active_p == 0, frozen_dur, 0)


        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {time_step+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
    return params, loss_history

    
# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

digits = [0,1,2,3]
num_qubits = num_components = 10
num_layers = 5
num_measurment_gates = math.ceil(pnp.log2(len(digits)))
num_epochs = 300
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)