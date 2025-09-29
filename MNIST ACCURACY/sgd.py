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

def train_qnn_param_shift(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs, x_test, y_test):
    forward_pass = create_qnn(n_layers, n_qubits)
    fp = 0
    if n_qubits == 4:
        params = two_four
    elif n_qubits == 8:
        params = three_eight
    elif n_qubits == 10:
        params = five_ten
    loss_history = []
    fp_history = []
    eval_acc_history = []
    rng = pnp.random.default_rng(0)

    def cost_fn(params, image, label):
        out = forward_pass(image, params, num_measurment_gates)
        return cross_entropy_loss(out, label)
    grad_fn = qml.grad(cost_fn, argnum=0)
    
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

            # increase fp by 2*n_active_params
            fp += 2*params.size 

            # Update active params only
            params -= 0.01* gradients

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {time_step+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

        if True:
            x_eval, y_eval = x_test, y_test
            correct = 0
            for xi, yi in zip(x_eval, y_eval):
                out_eval = forward_pass(xi, params, num_measurment_gates)
                if pnp.argmax(out_eval) == yi:
                    correct += 1
            acc = correct / len(x_eval)
            eval_acc_history.append((fp, acc))
            print(f"[Eval @ {fp} FP] Accuracy on 1000 random samples: {acc:.2%}")
            print(eval_acc_history) 

    return params, loss_history

    
# --------------------------------- Model Setup ---------------------------
df = pd.read_csv('../data/four_digit.csv')
x = df.drop('label', axis=1).values
y = df['label'].values

num_qubits = num_components = 4
num_layers = 2
num_measurment_gates = 2
num_epochs = 30
x = preprocess_image(x, num_components)

rng = pnp.random.default_rng(0)
perm = rng.permutation(len(x))
split = int(0.75 * len(x))
x_train, y_train = x[perm[:split]], y[perm[:split]]
x_test,  y_test  = x[perm[split:]], y[perm[split:]]

train_qnn_param_shift(x_train, y_train, num_qubits, num_layers, num_measurment_gates, num_epochs, x_test, y_test)