from scipy.optimize import minimize
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

# Alternative approach: Mini-batch optimization with manual loop
def train_qnn(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    params = two_four  # Keep original shape for reference
    original_shape = params.shape  # Store the original shape
    loss_history = []
    fp_history = []
    fp_count = 0
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 50
        random_indices = pnp.random.choice(len(x), size=s, replace=False)
        x_batch = x[random_indices]
        y_batch = y[random_indices]
        
        params_flat = params.flatten()  # Flatten for optimization
        
        def epoch_cost_function(p):
            nonlocal fp_count
            p_reshaped = p.reshape(original_shape)  # Use stored shape
            total_loss = 0
            for image, label in zip(x_batch, y_batch):
                out = forward_pass(image, p_reshaped, num_measurment_gates)
                loss = cross_entropy_loss(out, label)
                total_loss += loss
                fp_count += 1
            return total_loss / len(x_batch)
        
        # Optimize for this batch
        result = minimize(
            epoch_cost_function,
            params_flat,
            method='Nelder-Mead',
            options={'maxiter': 10}  # Limit iterations per batch
        )
        params_flat = result.x
        params = params_flat.reshape(original_shape)  # Reshape back to original shape
        
        # Evaluate performance
        test_loss = 0
        correct = 0
        test_size = 50
        test_indices = pnp.random.choice(len(x), size=test_size, replace=False)
            
        for i in test_indices:
            # params is already in the correct shape now
            out = forward_pass(x[i], params, num_measurment_gates)
            test_loss += cross_entropy_loss(out, y[i])
            if pnp.argmax(out) == y[i]:
                correct += 1
            
        avg_loss = test_loss / test_size
        accuracy = correct / test_size
        loss_history.append(round(float(avg_loss), 4))
        fp_history.append(fp_count)
            
        print(f"Epoch {epoch}, FP: {fp_count}, Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
    
    return params, fp_history, loss_history

# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/two_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

num_qubits = num_components = 4
num_layers = 2
num_measurment_gates = 1
num_epochs = 1000
x = preprocess_image(x, num_components)

optimized_params, fp_hist, loss_hist = train_qnn(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)