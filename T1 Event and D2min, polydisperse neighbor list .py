import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def load_data(file_path):
    """Load particle data (x,y,diameter) from text file"""
    data = np.loadtxt(file_path, skiprows=1)
    return data[:, [0, 1, 3]]  # x, y, diameter

def normalized_distance(pos1, pos2, diam1, diam2, box_size):
    """Calculate normalized distance between two particles with PBC"""
    dx = pos1[0] - pos2[0]
    dy = pos1[1] - pos2[1]
    
    # Apply periodic boundary conditions
    dx -= box_size[0] * np.round(dx / box_size[0])
    dy -= box_size[1] * np.round(dy / box_size[1])
    
    dist = np.sqrt(dx**2 + dy**2)
    sigma_avg = (diam1 + diam2) / 2
    return dist / sigma_avg, dx, dy, sigma_avg

def compute_rdf(data, box_size, dr=0.05, r_max=6.0):
    """Compute RDF using normalized distances up to r=6.0"""
    positions = data[:, :2]
    diameters = data[:, 2]
    N = len(positions)
    Lx, Ly = box_size
    
    nbins = int(r_max / dr)
    r = np.linspace(dr, r_max, nbins)
    g_r = np.zeros(nbins)
    norm = np.pi * ((r + dr)**2 - r**2) * (N / (Lx * Ly))
    
    for i in range(N):
        for j in range(i + 1, N):
            r_ij, _, _, _ = normalized_distance(
                positions[i], positions[j],
                diameters[i], diameters[j],
                box_size
            )
            
            if r_ij < r_max:
                bin_idx = int(r_ij / dr)
                if bin_idx < nbins:
                    g_r[bin_idx] += 2  # Count i-j and j-i
    
    g_r /= (N * norm)
    return r, g_r

def find_first_minimum(r, g_r, smoothing_window=5):
    """Find first minimum in RDF with smoothing"""
    g_r_smooth = np.convolve(g_r, np.ones(smoothing_window)/smoothing_window, mode='same')
    peaks, _ = find_peaks(g_r_smooth)
    
    if len(peaks) > 0:
        first_peak = peaks[0]
        for i in range(first_peak, len(g_r_smooth)-1):
            if g_r_smooth[i] < g_r_smooth[i-1] and g_r_smooth[i] < g_r_smooth[i+1]:
                return r[i]
    
    return r[len(r)//2]  # Fallback if no minimum found

def compute_neighbor_lists(data, box_size):
    """Compute TWO neighbor lists using r_min and r_max cutoffs"""
    positions = data[:, :2]
    diameters = data[:, 2]
    N = len(positions)
    Lx, Ly = box_size
    
    # First compute RDF to find cutoffs
    r, g_r = compute_rdf(data, box_size)
    r_min = 1.05
    r_max = 1.25
    rdf_data = (r, g_r, r_min, r_max)
    
    # Create both neighbor lists
    neighbors_rmin = {i: set() for i in range(N)}
    neighbors_rmax = {i: set() for i in range(N)}
    
    for i in range(N):
        for j in range(i + 1, N):
            r_ij, _, _, _ = normalized_distance(
                positions[i], positions[j],
                diameters[i], diameters[j],
                box_size
            )
            
            # Add to rmax list if within r_max
            if r_ij < r_max:
                neighbors_rmax[i].add(j)
                neighbors_rmax[j].add(i)
                
                # Add to rmin list if also within r_min
                if r_ij < r_min:
                    neighbors_rmin[i].add(j)
                    neighbors_rmin[j].add(i)
    
    return neighbors_rmin, neighbors_rmax, (r_min, r_max), rdf_data

def compute_dmin2(X0, X1, diameters, box_size, r_cut):
    """Compute Dmin² using normalized neighbor lists"""
    N = X0.shape[0]
    dmin2_values = np.full(N, np.nan)
    
    _, neighbors_rmax, _, _ = compute_neighbor_lists(
        np.column_stack((X0, diameters)), box_size
    )
    neighbors = neighbors_rmax  # Use rmax neighbors
    
    for i in range(N):
        neighbor_indices = list(neighbors[i])
        if len(neighbor_indices) < 3:
            continue
        
        # Collect neighbor positions (with PBC)
        R = []
        R_prime = []
        sigmas = []
        
        for j in neighbor_indices:
            # Reference configuration (t)
            r_ij, dx, dy, sigma_avg = normalized_distance(
                X0[i], X0[j], diameters[i], diameters[j], box_size
            )
            R.append([dx, dy])
            sigmas.append(sigma_avg)
            
            # Current configuration (t+1)
            _, dx_prime, dy_prime, _ = normalized_distance(
                X1[i], X1[j], diameters[i], diameters[j], box_size
            )
            R_prime.append([dx_prime, dy_prime])
        
        R = np.array(R)
        R_prime = np.array(R_prime)
        sigmas = np.array(sigmas)
        
        # Weight by sigma to account for size differences
        W = np.diag(1 / sigmas)
        R_weighted = W @ R
        R_prime_weighted = W @ R_prime
        
        # Solve least squares problem
        try:
            J = np.linalg.lstsq(R_weighted, R_prime_weighted, rcond=None)[0]
            residuals = R_prime_weighted - R_weighted @ J
            dmin2_values[i] = np.mean(np.sum(residuals**2, axis=1))
        except:
            continue
    
    return dmin2_values

def detect_t1_events(neighbors_t_rmin, neighbors_t_rmax, 
                    neighbors_t1_rmin, neighbors_t1_rmax,
                    data_t, data_t1, box_size, cutoffs, threshold=1.0):
    """Original T1 detection with rmin/rmax logic"""
    r_min, r_max = cutoffs
    t1_events = []
    t1_particles = set()
    N = len(data_t)
    
    for i in range(N):
        # Filter by displacement
        displacement = np.abs(data_t[i, :2] - data_t1[i, :2])
        displacement = np.minimum(displacement, box_size - displacement)
        if np.linalg.norm(displacement) > threshold:
            continue
        
        # Bond breaking: was in r_min at t, outside r_max at t1
        broken = set()
        for j in neighbors_t_rmin[i]:
            if j not in neighbors_t1_rmax[i]:
                broken.add(j)
        
        # Bond formation: was outside r_max at t, inside r_min at t1
        formed = set()
        for j in neighbors_t1_rmin[i]:
            if j not in neighbors_t_rmax[i]:
                formed.add(j)
        
        if broken or formed:
            t1_events.append((i, broken, formed))
            t1_particles.update([i] + list(broken) + list(formed))
    
    return t1_events, t1_particles

def plot_rdf(r, g_r, r_min=None, r_max=None):
    """Plot RDF with cutoff markers"""
    plt.figure(figsize=(8, 6), dpi=200)
    plt.plot(r, g_r, 'b-', linewidth=2)
    
    if r_min is not None:
        plt.axvline(x=r_min, color='g', linestyle='--', label=f'r_min: {r_min:.2f}')
    if r_max is not None:
        plt.axvline(x=r_max, color='r', linestyle='--', label=f'r_max: {r_max:.2f}')
    
    plt.xlabel('Normalized Distance (r/σ)')
    plt.ylabel('g(r)')
    plt.title('Radial Distribution Function')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()

def plot_dmin2_scatter(data, dmin2_values, Lx, Ly):
    """Scatter plot with particle sizes"""
    plt.figure(figsize=(10, 5), dpi=300)  
    sizes = data[:, 2]*100
    
    scatter = plt.scatter(
        data[:, 0], data[:, 1], 
        c=dmin2_values, 
        cmap="viridis", 
        s=sizes, 
        edgecolor='black',
        linewidth=0.5,
        alpha=0.8,
        vmin=0,       
        vmax=0.08,
    )
    
    plt.colorbar(scatter, label=r"$D_{\min}^2$")
    plt.xlim(0, 36)
    plt.ylim(0, 18)
    plt.gca().set_aspect('equal')
    plt.xlabel("X position")
    plt.ylabel("Y position")
    plt.title("Spatial Distribution of $D_{\min}^2$")
    plt.tight_layout()
    plt.show()

def visualize_t1_events(data_t, events, box_size):
    """Highlight T1 event particles in red"""
    t1_mask = np.zeros(len(data_t), dtype=bool)
    for i, lost, gained in events:
        t1_mask[i] = True
        t1_mask[list(lost)] = True
        t1_mask[list(gained)] = True
    
    plt.figure(figsize=(8, 8), dpi=300)
    plt.scatter(
        data_t[~t1_mask, 0], data_t[~t1_mask, 1],
        c='blue', s=data_t[~t1_mask, 2]*70,
        alpha=0.5, label='Normal'
    )
    plt.scatter(
        data_t[t1_mask, 0], data_t[t1_mask, 1],
        c='red', s=data_t[t1_mask, 2]*50,
        alpha=0.9, label='T1 Event'
    )
    plt.xlim(0, 36)
    plt.ylim(0, 18)
    plt.gca().set_aspect('equal')
    plt.legend()
    plt.title("T1 Transition Events")
    plt.show()

if __name__ == "__main__":
    base_path = "Flow_N900_alpha2.500_phi1.400_delta0.010_gamma2.000_w0.500_sigm_obs10.000_K10.000_fx_ext0.000800_dt0.100"
    file_t = os.path.join(base_path, "data_4200")
    file_t1 = os.path.join(base_path, "data_4300")
    
    # Load data
    data_t = load_data(file_t)
    data_t1 = load_data(file_t1)
    
    # Set box size (Lx = 2*Ly)
    Ly = 17.6608220062
    Lx=2*17.6608220062
    box_size = np.array([Lx,Ly])
    
    # Compute neighbor lists and RDF
    neighbors_t_rmin, neighbors_t_rmax, cutoffs, rdf_data = compute_neighbor_lists(data_t, box_size)
    neighbors_t1_rmin, neighbors_t1_rmax, _, _ = compute_neighbor_lists(data_t1, box_size)
    
    # Plot RDF with cutoffs
    plot_rdf(rdf_data[0], rdf_data[1], cutoffs[0], cutoffs[1])
    print(f"First minimum at: {cutoffs[0]+0.1:.3f}")
    print(f"Using r_min: {cutoffs[0]:.3f}, r_max: {cutoffs[1]:.3f}")
    
    # Detect T1 events
    t1_events, t1_particles = detect_t1_events(
        neighbors_t_rmin, neighbors_t_rmax,
        neighbors_t1_rmin, neighbors_t1_rmax,
        data_t, data_t1, box_size, cutoffs
    )
    print(f"\nFound {len(t1_events)} T1 events")
    
    # Compute dmin2
    dmin2_values = compute_dmin2(
        data_t[:, :2], data_t1[:, :2], data_t[:, 2], box_size, cutoffs[1]
    )
    
    # Generate plots
    plot_dmin2_scatter(data_t, dmin2_values, box_size[0], box_size[1])
    visualize_t1_events(data_t, t1_events, box_size)
