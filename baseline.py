import pennylane as qml
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer, GradientDescentOptimizer
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

    # Track loss and accuracy history
    loss_history = []

    # Define the cost function for a single sample
    def cost_fn(params, image, label):
        out = forward_pass(image, params, num_measurment_gates)
        return cross_entropy_loss(out, label)

    grad_fn = qml.grad(cost_fn, argnum=0)
    
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

            # compute gradients and update params
            gradients = grad_fn(params, image, label)
            # increase fp by 2*n_active_params
            params -= 0.01* gradients
            
        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        
        # Store values for plotting later
        loss_history.append(avg_loss)

        # Print metrics for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history

    
# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('./data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

digits = [0,1,2,3]
num_qubits = num_components = 10
num_layers = 5
num_measurment_gates = math.ceil(pnp.log2(len(digits)))
num_epochs = 300
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)