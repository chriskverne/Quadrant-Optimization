import pennylane as qml
import pennylane.numpy as pnp
from pennylane.optimize import AdamOptimizer
import math
from tqdm import tqdm
from fetch_mnist import fetch_mnist, preprocess_image
from create_qnn_no_noise import create_qnn
from cross_entropy import cross_entropy_loss

def train_qnn_param_shift(x, y, n_qubits, n_layers, num_measurment_gates, num_epochs):
    forward_pass = create_qnn(n_layers, n_qubits)
    
    # Initialize the Adam optimizer from PennyLane
    opt = AdamOptimizer(stepsize=0.01)
    
    params = pnp.random.uniform(0, 2*pnp.pi, size=(n_layers, n_qubits, 3))

    # Track loss and accuracy history
    loss_history = []
    accuracy_history = []
    
    for epoch in tqdm(range(num_epochs), desc="Epochs"):
        s = 100
        indices = pnp.random.choice(len(x), size=s, replace=False)
        x_t = x[indices]
        y_t = y[indices]
        total_loss = 0
        correct_predictions = 0
        
        for image, label in tqdm(zip(x_t, y_t), total=len(x_t), desc=f"Epoch {epoch+1}/{num_epochs}", leave=False):
            # Define the cost function for this sample
            def cost(params):
                out = forward_pass(image, params, num_measurment_gates)
                return cross_entropy_loss(out, label)
            
            # Compute forward pass with current parameters
            out = forward_pass(image, params, num_measurment_gates)
            loss = cross_entropy_loss(out, label)
            total_loss += loss
            
            # Check if prediction is correct
            pred = pnp.argmax(out)
            if pred == label:
                correct_predictions += 1
            
            # Update parameters using Adam optimizer
            params = opt.step(cost, params)
        
        # Calculate average loss and accuracy
        avg_loss = total_loss / len(x_t)
        accuracy = correct_predictions / len(x_t)
        
        # Store values for plotting later
        loss_history.append(avg_loss)
        accuracy_history.append(accuracy)

        # Print metrics for the epoch
        print(f"Epoch {epoch+1}/{num_epochs}, Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2%}")

    return params, loss_history, accuracy_history

    
# --------------------------------- Model Setup ---------------------------
digits = [0,1]
num_qubits = num_components = 6 # each PCA value encoded on each qubit
num_layers = 3
num_measurment_gates = math.ceil(pnp.log2(len(digits)))
num_epochs = 40
x,y = fetch_mnist(digits, 1000)
x = preprocess_image(x, num_components)
train_qnn_param_shift(x, y, num_qubits, num_layers, num_measurment_gates, num_epochs)