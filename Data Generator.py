import numpy as np
from scipy.signal import find_peaks
import os

def load_data(file_path):
    data = np.loadtxt(file_path, skiprows=1)
    return data[:, [0, 1, 3]]  # x, y, diameter

def compute_rdf(data, box_size, dr=0.1, r_max=None):
    positions = data[:, :2]
    N = len(positions)
    Lx, Ly = box_size
    
    if r_max is None:
        r_max = min(Lx, Ly) / 2.0
    
    nbins = int(r_max / dr)
    r = np.linspace(dr, r_max, nbins)
    g_r = np.zeros(nbins)
    norm = np.pi * ((r + dr)**2 - r**2) * (N / (Lx * Ly))
    
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist = np.sqrt(dx**2 + dy**2)
            if dist < r_max:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    g_r[bin_idx] += 2
    g_r /= (N * norm)
    return r, g_r

def find_first_minimum(r, g_r, smoothing_window=5):
    g_r_smooth = np.convolve(g_r, np.ones(smoothing_window)/smoothing_window, mode='same')
    peaks, _ = find_peaks(g_r_smooth)
    if len(peaks) > 0:
        first_peak = peaks[0]
        for i in range(first_peak, len(g_r_smooth)-1):
            if g_r_smooth[i] < g_r_smooth[i-1] and g_r_smooth[i] < g_r_smooth[i+1]:
                return r[i]
    return r[len(r)//2]

def compute_neighbors(X, Lx, Ly, cutoff):
    N = len(X)
    neighbors = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            dx = X[i, 0] - X[j, 0]
            dy = X[i, 1] - X[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist = np.sqrt(dx**2 + dy**2)
            if dist < cutoff:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors

def compute_dmin2(X0, X1, Lx, Ly, cutoff=1.2):
    N = X0.shape[0]
    dmin2_values = np.full(N, np.nan)
    neighbors = compute_neighbors(X0, Lx, Ly, cutoff)

    for i in range(N):
        neighbors_idx = list(neighbors[i])
        if len(neighbors_idx) < 3:
            dmin2_values[i] = 0.0
            continue

        R = X0[neighbors_idx, :2] - X0[i, :2]
        R[:, 0] -= Lx * np.round(R[:, 0] / Lx)
        R[:, 1] -= Ly * np.round(R[:, 1] / Ly)

        R_prime = X1[neighbors_idx, :2] - X1[i, :2]
        R_prime[:, 0] -= Lx * np.round(R_prime[:, 0] / Lx)
        R_prime[:, 1] -= Ly * np.round(R_prime[:, 1] / Ly)

        R_T = R.T
        R_T_R = R_T @ R
        if np.linalg.det(R_T_R) < 1e-10:
            dmin2_values[i] = 0.0
            continue
        
        inv_R_T_R = np.linalg.inv(R_T_R)
        J = inv_R_T_R @ (R_T @ R_prime)
        residuals = R_prime - R @ J
        dmin2_values[i] = np.mean(np.sum(residuals**2, axis=1))

    return dmin2_values

def compute_neighbor_lists(data, box_size, r_min=None, r_max=None):
    positions = data[:, :2]
    N = len(positions)
    Lx, Ly = box_size
    
    if r_min is None or r_max is None:
        r, g_r = compute_rdf(data, box_size)
        r_min = 1.05
        r_max = 1.25
    neighbors_rmin = {i: set() for i in range(N)}
    neighbors_rmax = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist = np.sqrt(dx**2 + dy**2)
            if dist < r_max:
                neighbors_rmax[i].add(j)
                neighbors_rmax[j].add(i)
                if dist < r_min:
                    neighbors_rmin[i].add(j)
                    neighbors_rmin[j].add(i)
    return neighbors_rmin, neighbors_rmax, (r_min, r_max)

def detect_t1_events(neighbors_t_rmin, neighbors_t_rmax, 
                    neighbors_t1_rmin, neighbors_t1_rmax,
                    data_t, data_t1, box_size, cutoffs, threshold=1.0):
    r_min, r_max = cutoffs
    t1_events = []
    t1_particles = set()
    Lx, Ly = box_size
    N = len(data_t)
    dist_t = np.zeros((N, N))
    dist_t1 = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            dx = data_t[i, 0] - data_t[j, 0]
            dy = data_t[i, 1] - data_t[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist_t[i,j] = dist_t[j,i] = np.sqrt(dx**2 + dy**2)
            dx = data_t1[i, 0] - data_t1[j, 0]
            dy = data_t1[i, 1] - data_t1[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist_t1[i,j] = dist_t1[j,i] = np.sqrt(dx**2 + dy**2)
    for i in range(N):
        displacement = np.abs(data_t[i, :2] - data_t1[i, :2])
        displacement = np.minimum(displacement, box_size - displacement)
        if np.linalg.norm(displacement) > threshold:
            continue
        broken = set()
        for j in neighbors_t_rmin[i]:
            if dist_t1[i,j] > r_max:
                broken.add(j)
        formed = set()
        for j in neighbors_t1_rmin[i]:
            if j not in neighbors_t_rmax[i] or dist_t[i,j] > r_max:
                formed.add(j)
        if broken or formed:
            t1_events.append((i, broken, formed))
            t1_particles.update([i] + list(broken) + list(formed))
    return t1_events, t1_particles

# MAIN EXECUTION
if __name__ == "__main__":
    base_path = "Flow_N900_alpha2.500_phi1.400_delta0.150_gamma2.000_w0.500_sigm_obs10.000_K10.000_fx_ext0.000800_dt0.100"
    output_file = "output_timestep=180.txt"
    
    # Initialize with header only (overwrite if exists)
    with open(output_file, 'w') as f:
        f.write("900 35.3216440125 17.6608220062\n")
    
    for n in range(500):
        file_num = 200 + n*18
        file_t = os.path.join(base_path, f"data_{file_num:05d}")
        file_t1 = os.path.join(base_path, f"data_{file_num+18:05d}")
        
        print(f"Processing files {file_num} and {file_num+18}...")
        
        try:
            data_t = load_data(file_t)
            data_t1 = load_data(file_t1)
        except:
            print(f"Could not load files {file_t} or {file_t1}, skipping...")
            continue

        Ly = 17.6608220062
        box_size = np.array([2*Ly, Ly])
        Lx, Ly = box_size

        neighbors_t_rmin, neighbors_t_rmax, cutoffs = compute_neighbor_lists(data_t, box_size)
        neighbors_t1_rmin, neighbors_t1_rmax, _ = compute_neighbor_lists(data_t1, box_size, *cutoffs)

        t1_events, t1_particles = detect_t1_events(
            neighbors_t_rmin, neighbors_t_rmax,
            neighbors_t1_rmin, neighbors_t1_rmax,
            data_t, data_t1, box_size, cutoffs
        )

        dmin2_values = compute_dmin2(data_t[:, :2], data_t1[:, :2], Lx, Ly, cutoffs[1])

        # Output: x, y, diameter, T1_flag, dmin2
        T1_flag_array = np.zeros(len(data_t), dtype=int)
        for i in t1_particles:
            T1_flag_array[i] = 1

        output_data = np.column_stack((data_t[:, 0], data_t[:, 1], data_t[:, 2], T1_flag_array, dmin2_values))
        
        # Append to output file without header
        with open(output_file, 'a') as f:
            np.savetxt(f, output_data, fmt="%.6f")
        
    print(f"All data saved to {output_file}")