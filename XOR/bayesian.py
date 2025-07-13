from skopt import gp_minimize
from skopt.space import Real
import pennylane.numpy as pnp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from helper.create_qnn_xor import create_qnn_XOR
from helper.get_xor_data import get_xor_data
from helper.cross_entropy import cross_entropy_loss
from data.params import *

# Alternative approach: Mini-batch optimization with Bayesian optimization
def train_qnn(x, y, n_qubits, n_layers, num_epochs):
    forward_pass = create_qnn_XOR(n_layers, n_qubits)
    params = three_eight
    original_shape = params.shape 
    loss_history = []
    fp_history = []
    fp_count = 0
    
    n_params = params.size
    dimensions = [Real(0, 2*pnp.pi, name=f'param_{i}') for i in range(n_params)]
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 100
        #random_indices = pnp.random.choice(len(x), size=s, replace=False)
        #x_batch = x[random_indices]
        #y_batch = y[random_indices]
        x_t = x[epoch*s:(epoch+1)*s]
        y_t = y[epoch*s:(epoch+1)*s]
        
        params_flat = params.flatten() 
        
        def epoch_cost_function(p):
            nonlocal fp_count
            p_array = pnp.array(p)  # Convert to numpy array
            p_reshaped = p_array.reshape(original_shape)  # Use stored shape
            total_loss = 0
            for image, label in zip(x_t, y_t):
                out = forward_pass(image, p_reshaped)
                loss = cross_entropy_loss(out, label)
                total_loss += loss
                fp_count += 1
            
            # Convert to Python float to ensure it's a scalar
            avg_loss = total_loss / len(x_t)
            return float(avg_loss[0])
        
        # Bayesian optimization for this batch
        result = gp_minimize(
            epoch_cost_function,
            dimensions,
            x0=params_flat.tolist(),  # Starting point
            n_calls=25,  # Number of function evaluations (equivalent to maxiter)
            acq_func='EI',  # Expected Improvement acquisition function
            #random_state=42,
            n_initial_points=1  # Number of random points to start with
        )

        params_flat = pnp.array(result.x)
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
n_qubits = 8
n_layers = 3
n_epochs = 400
x,y = get_xor_data(n_qubits, 100000)


train_qnn(x, y, n_qubits, n_layers, n_epochs)