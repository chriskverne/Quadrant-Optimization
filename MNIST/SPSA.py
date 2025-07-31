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

    opt = qml.SPSAOptimizer(A=0.1) 

    """Training Loop"""
    for time_step in tqdm(range(num_epochs), desc="Time step"):
        s = 100
        # x_t = x[time_step*s:(time_step+1)*s]
        # y_t = y[time_step*s:(time_step+1)*s]
        random_indices = pnp.random.choice(len(x), size=s, replace=False)
        x_t = x[random_indices]
        y_t = y[random_indices]
        epoch_loss = 0
        correct_predictions = 0
        
        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {time_step+1}/{num_epochs}", leave=False):
            def cost_fn(params):
                out = forward_pass(image, params, num_measurment_gates)
                return cross_entropy_loss(out, label)
            
            # Compute loss with current parameters
            out = forward_pass(image, params, num_measurment_gates)
            #fp+=1
            loss = cross_entropy_loss(out, label)
            epoch_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1

            # increase fp by 2
            fp += 2

            # Update active params only
            params = opt.step(cost_fn, params)

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(round(float(avg_loss), 4))
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {time_step+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")
        if time_step != 0 and time_step % 100 == 0:
            print(f'x = [{fp_history}]')
            print(f'y = [{loss_history}]')

    return params, fp_history, loss_history

    
# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

num_qubits = num_components = 10
num_layers = 5
num_measurment_gates = 2
num_epochs = 2000
x = preprocess_image(x, num_components)


train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)