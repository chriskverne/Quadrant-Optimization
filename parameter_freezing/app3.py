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
Also look at the QFI when deciding what to freeze
"""

def train_qnn_param_shift(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    freeze_t = 0.80
    print(f"Freezing T set to {freeze_t*100}%")
    unfreeze_p = 0.10
    fp=0    
    params = five_ten

    # Tracks which parameters are marked as frozen
    frozen_p = pnp.zeros_like(params)

    # Tracks the duration each parameter has been frozen for
    frozen_dur = pnp.zeros_like(params)

    # Tracks gradients to decide what to freeze
    tt_param_grads = pnp.zeros_like(params)

    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 50 # time interval representing when to freeze/unfreeze parameters
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
                        
                        params_plus = params.copy()
                        params_plus[l,q,g] += pnp.pi/2
                        params_minus = params.copy()
                        params_minus[l,q,g] -= pnp.pi/2

                        out_plus = forward_pass(image, params_plus, num_measurment_gates)
                        out_minus = forward_pass(image, params_minus, num_measurment_gates)
                        grad = (out_plus - out_minus)/2
                        fp+=2
                        grads[l,q,g] = pnp.dot(dL_dp, grad)

            # Add gradients to param history
            tt_param_grads += grads # sums the total gradient over interval t

            # Update params which havent been frozen
            params -= 0.01*grads

        # Decide what to freeze (i.e freeze_t% of parameters with smallest tt_param_grads)
        sorted_abs_history = pnp.sort(pnp.abs(tt_param_grads.flatten()))
        idx = int(len(sorted_abs_history) * freeze_t)
        threshold = sorted_abs_history[idx]
        frozen_p = pnp.where(pnp.abs(tt_param_grads) <= threshold, 1, 0)

        # Decide what to unfreeze and set their freeze_dur to 0 (maybe not needed?)

        # Reset grads for unfrozen params. I.e. a parameter isn't frozen set tt_param_grads[l,q,g] for that parameter equal to 0
        tt_param_grads = pnp.where(frozen_p == 1, tt_param_grads, 0)

        avg_loss = total_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        print(f"\nNo FP: {fp}, Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params

    
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