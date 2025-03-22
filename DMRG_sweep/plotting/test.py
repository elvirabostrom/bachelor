import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

"""
C++
# Load rank3 tensor and reshape it
def load_tensor(filename, shape, dtype=np.float64):
    data = np.fromfile(filename, dtype=dtype)  # Read binary data
    tensor = data.reshape(shape)  # Reshape into rank3 form
    return tensor

# Load test tensor
potential_test = load_tensor("output/potential_test.bin", shape = (10, 10, 10))
"""

# python:
potential_test = np.load("output/tensor_test.npy")

# PLot 2D slices of rank3 tensor along Z axis - slices in XY plane
def plot_tensor_slices(tensor):
    num_slices = tensor.shape[0]  # Number of slices along Z

    fig, axes = plt.subplots(1, num_slices, figsize = (15, 5))
    for i in range(num_slices):
        axes[i].imshow(tensor[:, :, i], cmap = 'viridis', aspect = 'auto')
        axes[i].set_title(f"$z_{i}$")
        axes[i].axis("off")

    plt.show()

# Plot 
plot_tensor_slices(potential_test)

# 3D plot - chatGPT:)
def plot_3d_tensor(tensor):
    fig = go.Figure(data = go.Volume(
        x = np.repeat(np.arange(tensor.shape[0]), tensor.shape[1] * tensor.shape[2]),
        y = np.tile(np.repeat(np.arange(tensor.shape[1]), tensor.shape[2]), tensor.shape[0]),
        z = np.tile(np.arange(tensor.shape[2]), tensor.shape[0] * tensor.shape[1]),
        value = tensor.flatten(),
        opacity = 0.3,  
        isomin = tensor.min(),
        isomax = tensor.max(),
        surface_count = 10
    ))
    fig.show()

plot_3d_tensor(potential_test)

