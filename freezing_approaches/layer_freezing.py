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


def train_qnn_param_shift(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    beta = 0.9 # Weightage of recent grad v.s. EMA
    lr = 0.01
    fp=0    
    params = three_six

    # Tracks which layers are marked as frozen
    frozen_l = pnp.zeros_like(n_layers) # 0 = unfrozen, 1 = frozen

    # Tracks the duration each layer has been frozen for
    frozen_dur = pnp.zeros_like(n_layers)

    # Tracks combined EMA for each layer
    ema_grad = pnp.zeros_like(params)

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
                # Skip Frozen layers
                if frozen_l[l] == 1:
                    frozen_dur[l] += 1
                    continue
                for q in range(n_qubits):
                    for g in range(2):
                        params_plus = params.copy()
                        params_plus[l,q,g] += pnp.pi/2
                        params_minus = params.copy()
                        params_minus[l,q,g] -= pnp.pi/2

                        grad = (forward_pass(image, params_plus, num_measurment_gates) - forward_pass(image, params_minus, num_measurment_gates))/2
                        fp+=2
                        grads[l,q,g] = pnp.dot(dL_dp, grad)

            # Add gradients to param history
            potential_new_ema = beta * ema_grad + (1 - beta) * grads
            is_frozen_mask = (frozen_l[:, None, None] == 1)
            ema_grad = pnp.where(is_frozen_mask, ema_grad, potential_new_ema) # ema_grad if layer is frozen, otherwise potential_new_ema

            # Update params which havent been frozen
            params -= lr*grads

        # Decide what to freeze based on smallest EMA grads
        avg_ema_layer = pnp.mean(pnp.abs(ema_grad), axis=(1, 2)) # [1, l] array
        idx_to_unfreeze = pnp.argmax(avg_ema_layer)
        frozen_l = pnp.ones(n_layers) # freeze all layers
        frozen_l[idx_to_unfreeze] = 0 # Unfreeze layer with largest avg EMA

        # Decide what to unfreeze
        if epoch != 0 and epoch % 5 == 0:
            frozen_l = pnp.zeros_like(n_layers)
            frozen_dur = pnp.zeros_like(n_layers)

        # Prin epoch data
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