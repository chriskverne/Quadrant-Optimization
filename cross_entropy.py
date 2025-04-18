import pennylane.numpy as pnp
def cross_entropy_loss(output_distribution, true):
    # Convert tensor to int if needed
    if hasattr(true, 'item'):
        true = true.item()  # Convert tensor to Python scalar

    epsilon = 1e-10
    # get the models probability of right guess
    pred = output_distribution[true]

    return -pnp.log(pred + epsilon)