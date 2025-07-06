from scipy.optimize import minimize
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

# Alternative approach: Mini-batch optimization with manual loop
def train_qnn(x, y, n_qubits, n_layers, num_epochs):
    forward_pass = create_qnn_XOR(n_layers, n_qubits)
    params = two_four
    original_shape = params.shape 
    loss_history = []
    fp_history = []
    fp_count = 0
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 100
        #random_indices = pnp.random.choice(len(x), size=s, replace=False)
        #x_batch = x[random_indices]
        #y_batch = y[random_indices]
        x_t = x[epoch*s:(epoch+1)*s]
        y_t = y[epoch*s:(epoch+1)*s]
        
        params_flat = params.flatten()  # Flatten for optimization
        
        def epoch_cost_function(p):
            nonlocal fp_count
            p_reshaped = p.reshape(original_shape)  # Use stored shape
            total_loss = 0
            for image, label in zip(x_t, y_t):
                out = forward_pass(image, p_reshaped)
                loss = cross_entropy_loss(out, label)
                total_loss += loss
                fp_count += 1
            return total_loss / len(x_t)
        
        # Optimize for this batch
        result = minimize(
            epoch_cost_function,
            params_flat,
            method='Nelder-Mead',
            options={'maxiter': 50}
            #options={'maxiter': 10}  # Limit iterations per batch
        )

        params_flat = result.x
        params = params_flat.reshape(original_shape)  # Reshape back to original shape
        
        # Evaluate performance
        test_loss = 0
        correct = 0
        test_size = 100
        test_indices = pnp.random.choice(len(x), size=test_size, replace=False)
            
        for i in test_indices:
            # params is already in the correct shape now
            out = forward_pass(x[i], params)
            test_loss += cross_entropy_loss(out, y[i])
            if pnp.argmax(out) == y[i]:
                correct += 1
            
        avg_loss = test_loss / test_size
        accuracy = correct / test_size
        loss_history.append(round(float(avg_loss[0]), 4))
        fp_history.append(fp_count)
            
        print(f"Epoch {epoch}, FP: {fp_count}, Loss: {avg_loss[0]:.4f}, Accuracy: {accuracy:.2%}")
    
    return params, fp_history, loss_history

# --------------------------------- Model Setup ---------------------------
n_qubits = 4
n_layers = 2
n_epochs = 400
x,y = get_xor_data(n_qubits, 100000)


train_qnn(x, y, n_qubits, n_layers, n_epochs)