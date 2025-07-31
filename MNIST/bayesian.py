from skopt import gp_minimize
from skopt.space import Real
import pennylane as qml
import pennylane.numpy as pnp
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from helper.fetch_mnist import preprocess_image
from helper.create_qnn_no_noise import create_qnn
from helper.cross_entropy import cross_entropy_loss
from data.params import *
import pandas as pd

# Alternative approach: Mini-batch optimization with Bayesian optimization
def train_qnn(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    params = five_ten #pnp.random.uniform(0, 2*pnp.pi, size=(2, 4, 2)) #two_four  
    original_shape = params.shape 
    loss_history = []
    fp_history = []
    fp_count = 0
    
    # Define search space for Bayesian optimization
    # Each parameter can range from 0 to 2*pi (typical for quantum circuits)
    n_params = params.size
    dimensions = [Real(0, 2*pnp.pi, name=f'param_{i}') for i in range(n_params)]
    
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
            p_array = pnp.array(p)  # Convert to numpy array
            p_reshaped = p_array.reshape(original_shape)  # Use stored shape
            total_loss = 0
            for image, label in zip(x_t, y_t):
                out = forward_pass(image, p_reshaped, num_measurment_gates)
                loss = cross_entropy_loss(out, label)
                total_loss += loss
                fp_count += 1
            
            # Convert to Python float to ensure it's a scalar
            avg_loss = total_loss / len(x_t)
            return float(avg_loss)
        
        # Bayesian optimization for this batch
        result = gp_minimize(
            epoch_cost_function,
            dimensions,
            x0=params_flat.tolist(),  # Starting point
            n_calls=25,  # Number of function evaluations (equivalent to maxiter)
            acq_func='EI',  # Expected Improvement acquisition function
            random_state=42,
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
df = pd.read_csv('../data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

num_qubits = num_components = 10
num_layers = 5
num_measurment_gates = 2
num_epochs = 1000
x = preprocess_image(x, num_components)

optimized_params, fp_hist, loss_hist = train_qnn(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)