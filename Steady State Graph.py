import numpy as np
import matplotlib.pyplot as plt

# Load data from the file
file_name = "Flow_N900_alpha2.500_phi1.400_delta0.010_gamma2.000_w0.500_sigm_obs10.000_K10.000_fx_ext0.000800_dt0.100/data_total_step1000000_output_step100_run1.txt"
data = np.loadtxt(file_name)

# Extract columns
time = data[:, 0]          # First column: Time
energy = data[:, 1]        # Second column: Energy
pressure = data[:, 2]      # Third column: Pressure
shear_stress = data[:, 3]  # Fourth column: Shear Stress

# Limit data to time between 0 and 100
mask = (time >= 0) & (time <= 100)
time = time[mask]
energy = energy[mask]
pressure = pressure[mask]
shear_stress = shear_stress[mask]

# Create subplots
plt.figure(figsize=(10, 6))

# Plot Energy vs Time
plt.subplot(3, 1, 1)
plt.plot(time, energy, label="Energy", color='b')
plt.xlabel("Time")
plt.ylabel("Energy")
plt.title("Energy vs Time")
plt.grid(True)

# Plot Pressure vs Time
plt.subplot(3, 1, 2)
plt.plot(time, pressure, label="Pressure", color='r')
plt.xlabel("Time")
plt.ylabel("Pressure")
plt.title("Pressure vs Time")
plt.grid(True)

# Plot Shear Stress vs Time
plt.subplot(3, 1, 3)
plt.plot(time, shear_stress, label="Shear Stress", color='g')
plt.xlabel("Time")
plt.ylabel("Shear Stress")
plt.title("Shear Stress vs Time")
plt.grid(True)

# Function to find steady state (when Energy and Pressure stabilize)
def find_steady_state(time, energy, pressure, window=50, threshold=1e-3):
    """
    Finds the time when energy and pressure become stable.
    Stability is defined as the standard deviation in a rolling window being below a threshold.
    
    :param time: Time array
    :param energy: Energy array
    :param pressure: Pressure array
    :param window: Number of points to check for stability
    :param threshold: Stability threshold for standard deviation
    :return: Steady-state time
    """
    for i in range(len(time) - window):
        energy_std = np.std(energy[i:i+window])
        pressure_std = np.std(pressure[i:i+window])
        if energy_std < threshold and pressure_std < threshold:
            return time[i]  # Return first steady-state time
    return None  # No steady state found

# Find steady-state time
steady_state_time = find_steady_state(time, energy, pressure)

# Display steady-state time
if steady_state_time is not None:
    print(f"Steady state detected at time: {steady_state_time:.3f}")
else:
    print("No steady state detected.")

# Adjust layout and show the plot
plt.tight_layout()
plt.show()

# Return steady state time
steady_state_time

