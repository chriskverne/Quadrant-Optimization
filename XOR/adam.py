import pennylane as qml
import pennylane.numpy as pnp
import os
import sys #
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
from helper.create_qnn_xor import create_qnn_XOR
from helper.cross_entropy import cross_entropy_loss
from helper.get_xor_data import get_xor_data
from data.params import *


def train_qnn_param_shift(x, y, n_qubits, n_layers, num_epochs):
    forward_pass = create_qnn_XOR(n_layers, n_qubits)
    fp = 0
    params = two_four
    loss_history = []
    fp_history = []
    acc_curve=[]

    def cost_fn(params, image, label):
        out = forward_pass(image, params)
        return cross_entropy_loss(out, label)
    
    grad_fn = qml.grad(cost_fn, argnum=0)

    # Adam optimizer parameters
    alpha = 0.01  # learning rate (default Adam value)
    beta1 = 0.9    # exponential decay rate for first moment
    beta2 = 0.999  # exponential decay rate for second moment
    epsilon = 1e-8 # small constant to prevent division by zero
    
    # Adam optimizer state variables
    m = pnp.zeros_like(params)  # first moment (mean of gradients)
    v = pnp.zeros_like(params)  # second moment (variance of gradients)
    t=0
    
    """Training Loop"""
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 100
        x_t = x[epoch*s:(epoch+1)*s]
        y_t = y[epoch*s:(epoch+1)*s]
        epoch_loss = 0
        correct_predictions = 0
                
        for i, (str,bit) in enumerate(tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False), start=1):
            # Increment time step for Adam
            t += 1
            
            # Compute loss with current parameters
            out = forward_pass(str, params)
            fp += 1
            loss = cross_entropy_loss(out, bit)
            epoch_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == bit:
                correct_predictions += 1

            # compute gradients and apply only to active params
            gradients = grad_fn(params, str, bit)

            # increase fp by 2*n_active_params
            fp += 2*params.size

            # Adam optimizer update
            m = beta1 * m + (1 - beta1) * gradients
            #m = beta1 * m * + (1 - beta1) * gradients
            v = beta2 * v + (1 - beta2) * (gradients ** 2)
            m_hat = m / (1 - beta1 ** t)
            v_hat = v / (1 - beta2 ** t)
            
            # Update parameters using Adam formula
            adam_update = alpha * m_hat / (pnp.sqrt(v_hat) + epsilon)
            params = params - adam_update  # Use explicit assignment for clarity

            if i % 20 == 0:
                # Sample 100 random XOR points (fresh each time)
                x_eval, y_eval = get_xor_data(n_qubits, 500)

                # Forward passes only (no grads)
                correct_eval = 0
                for xe, ye in zip(x_eval, y_eval):
                    out_eval = forward_pass(xe, params)
                    pred_eval = pnp.argmax(out_eval)
                    if int(pred_eval) == int(ye):
                        correct_eval += 1
                eval_acc = correct_eval / len(x_eval)
                acc_curve.append((fp,eval_acc))
                # eval_history.append((epoch, i, float(eval_acc)))
                print(f"\nNo FP: {fp}, Avg Accuracy: {eval_acc}")
                print(acc_curve)

        # Calculate average loss and accuracy
        avg_loss = epoch_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        loss_history.append(avg_loss)
        fp_history.append(fp)
        print(f"\nNo FP: {fp}, Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss[0]:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history
    
# --------------------------------- Model Setup ---------------------------
n_qubits = 4
n_layers = 2
n_epochs = 100
x,y = get_xor_data(n_qubits, 100000)

train_qnn_param_shift(x, y, n_qubits, n_layers, n_epochs)