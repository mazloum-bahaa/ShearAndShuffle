import numpy as np
import matplotlib.pyplot as plt

# File paths
initial_file = "Flow_N900_alpha2.500_phi1.400_delta0.010_gamma2.000_w0.500_sigm_obs10.000_K10.000_fx_ext0.000800_dt0.100/data_0300"
final_file = "Flow_N900_alpha2.500_phi1.400_delta0.010_gamma2.000_w0.500_sigm_obs10.000_K10.000_fx_ext0.000800_dt0.100/data_0500"

# Load data (first row contains box dimensions)
with open(initial_file, 'r') as f:
    Lx, Ly = map(float, f.readline().split()[2:4])  # Extract Lx, Ly from first row

initial_data = np.loadtxt(initial_file, skiprows=1)
final_data = np.loadtxt(final_file, skiprows=1)

# Extract x and y coordinates
x_initial, y_initial = initial_data[:, 0], initial_data[:, 1]
x_final, y_final = final_data[:, 0], final_data[:, 1]

# Compute displacement with periodic boundary conditions
dx = x_final - x_initial
dy = y_final - y_initial

# Apply minimum image convention (account for wrap-around)
dx = dx - Lx * np.round(dx / Lx)
dy = dy - Ly * np.round(dy / Ly)

# Plot displacement vectors
plt.figure(figsize=(8, 8),dpi=400)
plt.quiver(x_initial, y_initial, dx, dy, angles='xy', scale_units='xy', scale=1, color='b')

# Formatting
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Particle Displacement Vectors (with PBC)")
plt.axis("equal")  # Equal aspect ratio
plt.grid()

# Show plot
plt.show()
