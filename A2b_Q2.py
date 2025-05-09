# Import packages
import numpy as np                  # Numpy for array, matrices
import matplotlib.pyplot as plt     # Matplotlib for Post processing 
from matplotlib import ticker, cm 
from matplotlib.colors import LinearSegmentedColormap
import time                         # Time taken comparision

def solve_2d_convection_diffusion(scheme):
    """
    Solves 2D convection-diffusion problem with different schemes
    
    Discretization scheme ('cds', 'upwind', or 'hybrid')
    """
    # Problem parameters
    dx = 0.05  # grid size in x-direction
    dy = 0.05  # grid size in y-direction
    H = 1.0    # height of channel
    L = 40.0   # length of channel
    nx = int(L/dx) + 1  # number of grid points in x
    ny = int(H/dy) + 1  # number of grid points in y
    
    # Physical properties
    k = 1.0     # thermal conductivity
    rho = 1.0   # density
    cp = 100.0  # specific heat
    
    tau = k/(cp)  
    
    # Boundary conditions
    phi_top = 100.0  # top wall temperature
    phi_inlet = 50.0  # inlet temperature
    
    # Solver parameters
    max_iter = 200000  # maximum iterations
    tol = 1e-6  # tolerance
    
    # Create mesh
    y_range = np.linspace(-H/2, H/2, ny)
    x, y = np.meshgrid(np.linspace(0, L, nx), y_range, indexing='ij')
    
    # Initialize temperature field
    phi = np.ones((nx, ny)) * phi_inlet
    
    # Calculate diffusion coefficients
    De = np.ones((nx, ny)) * tau * dy/dx
    Dw = np.ones((nx, ny)) * tau * dy/dx
    Dn = np.ones((nx, ny)) * tau * dx/dy
    Ds = np.ones((nx, ny)) * tau * dx/dy
    
    # Calculate convection coefficients
    Fe = np.ones((nx, ny))
    Fw = np.ones((nx, ny))
    Fn = np.zeros((nx, ny)) # Created using zeros because it is zero anyways
    Fs = np.zeros((nx, ny)) # Created using zeros because it is zero anyways
    
    # Set Velocity Profile & append fluxes
    for j in range(ny):
        V = 1.5 * (1 - 4 * (y_range[j]**2))
        Fe[:, j] = rho * V * dy
        Fw[:, j] = rho * V * dy
    
    # Initial array for coefficients 
    aE = np.zeros((nx, ny))
    aw = np.zeros((nx, ny))
    an = np.zeros((nx, ny))
    aS = np.zeros((nx, ny))
    aP = np.zeros((nx, ny))
    
    scheme = scheme.lower()  # Convert scheme to lowercase to avoid casing errors
    
    # CENTRAL DIFFERENCE SCHEME COEFFICIENTS
    
    if scheme == 'cds':
        for i in range(nx):
            for j in range(ny):
                
                aE[i, j] = De[i, j] - Fe[i, j]/2
                aw[i, j] = Dw[i, j] + Fw[i, j]/2
                an[i, j] = Dn[i, j] - Fn[i, j]/2
                aS[i, j] = Ds[i, j] + Fs[i, j]/2
                
                # Boundary conditions
                if i == nx-1:  
                    aE[i, j] = 0  # Outlet
                if j == 0:     
                    aS[i, j] = 0  # Bottom 
                if j == ny-1:  
                    an[i, j] = 0  # Top 
    
    
    # UPWIND SCHEME COEFFICIENTS
    
    elif scheme == 'upwind':
        for i in range(nx):
            for j in range(ny):
                
                aE[i, j] = De[i, j] + max(0, -Fe[i, j])
                aw[i, j] = Dw[i, j] + max(0, Fw[i, j])
                an[i, j] = Dn[i, j] + max(0, -Fn[i, j])
                aS[i, j] = Ds[i, j] + max(0, Fs[i, j])
                
                # Boundary conditions
                if i == nx-1:
                    aE[i, j] = 0 # Oultet
                if j == 0:
                    aS[i, j] = 0 # Bottom
                if j == ny-1:
                    an[i, j] = 0 # Top
    
    # HYRBID SCHEME COEFFICIENTS
    
    elif scheme == 'hybrid':
        for i in range(nx):
            for j in range(ny):
                
                aE[i, j] = max(-Fe[i, j], De[i, j] - Fe[i, j]/2, 0)
                aw[i, j] = max(Fw[i, j], Dw[i, j] + Fw[i, j]/2, 0)
                an[i, j] = max(-Fn[i, j], Dn[i, j] - Fn[i, j]/2, 0)
                aS[i, j] = max(Fs[i, j], Ds[i, j] + Fs[i, j]/2, 0)
                
                # Boundary conditions
                if i == nx-1:
                    aE[i, j] = 0
                if j == 0:
                    aS[i, j] = 0
                if j == ny-1:
                    an[i, j] = 0
    
    # Calculate aP coefficients
    aP = aE + aw + aS + an
    
    # Start timing
    start_time = time.time()
    
    # Line-by-line TDMA solver
    for iter in range(max_iter):
        phi_old = phi.copy()  # Copy
        
        # Apply bottom wall boundary condition 
        phi[:, 0] = phi[:, 1]
    
        for j in range(1, ny-1):
            
            # Initialize TDMA arrays
            a = np.zeros(nx)
            b = np.zeros(nx)
            c = np.zeros(nx)
            d = np.zeros(nx)
            
            phi_y = np.zeros(nx)  # Temp in that jth column
            
            # Inlet boundary condition
            a[0] = 1
            b[0] = 0
            c[0] = 0
            d[0] = phi_inlet
            
            # Interior points
            for i in range(1, nx-1):
                a[i] = aP[i, j]
                b[i] = aE[i, j]
                c[i] = aw[i, j]
                d[i] = (aS[i, j] * phi_old[i, j-1]) + (an[i, j] * phi_old[i, j+1])
            
            # Outlet boundary condition 
            a[-1] = aP[-1, j]
            b[-1] = 0
            c[-1] = aw[-1, j]
            d[-1] = aS[-1, j] * phi_old[-1, j-1] + an[-1, j] * phi_old[-1, j+1]
            
            P = np.zeros(nx)
            Q = np.zeros(nx)
            
            # TDMA forward sweep
            for i in range(nx):
                if i == 0:
                    P[i] = b[i] / a[i]
                    Q[i] = d[i] / a[i]
                else:
                    P[i] = b[i] / (a[i] - c[i] * P[i-1])
                    Q[i] = (d[i] + c[i] * Q[i-1]) / (a[i] - c[i] * P[i-1])
            
            # TDMA backward substitution
            for i in range(nx-1, -1, -1):
                if i == nx-1:
                    phi_y[i] = Q[i]
                else:
                    phi_y[i] = Q[i] + P[i] * phi_y[i+1]
            
            # Update temperature field
            phi[:, j] = phi_y
        
        # Apply boundary conditions
        phi[0, :]  = phi_inlet
        phi[:, -1] = phi_top

        
        # Check convergence
        error = np.max(np.abs(phi - phi_old))
        if error < tol:
            iter = iter + 1
            print(f'Converged at iteration {iter} with error {error:.6e}')
            break
    
    end_time = time.time()
    execution_time = end_time - start_time
    print(f'Took {execution_time} seconds to solve')
    
    # Calculate bulk mean temperature
    
    phi_avg = np.zeros(nx)
    for i in range(nx):
        numerator = 0
        denominator = 0
        for j in range(ny):
            y_val = y_range[j]
            u_val = 1.5 * (1 - 4 * y_val**2)
            weight = rho * cp * u_val * dy
            numerator += weight * phi[i, j]
            denominator += weight
        phi_avg[i] = numerator / denominator if denominator != 0 else 0
    
    # Calculate heat transfer coefficient and Nusselt number
    dT_dy = (phi[:, -1] - phi[:, -2]) / dy
    h = (k * dT_dy) / (phi_top - phi_avg)
    Nu = h * (2 * H) / k
    
    # Return results
    x_range = np.linspace(0, L, nx)
    return phi, phi_avg, h, Nu, x, y, x_range, y_range

def plot_combined_results(results_dict):
    """
    Plot combined results for all schemes
    """
    schemes = list(results_dict.keys())
    colors = {'cds': 'blue', 'upwind': 'red', 'hybrid': 'green'}
    
    # Plot temperature contours 
    fig, axes = plt.subplots(1, len(schemes), figsize=(18, 6))
    
    for i, scheme in enumerate(schemes):
        phi, _, _, _, x, y, _, _ = results_dict[scheme]
        
        contour = axes[i].contourf(x, y, phi, 20, cmap='Blues')
        
        # Add contour lines
        contour_lines = axes[i].contour(x, y, phi, 20, colors='black', linewidths=0.5)
        axes[i].clabel(contour_lines, inline=True, fontsize=8, fmt='%1.1f')
        
        axes[i].set_title(f'Temperature Distribution ({scheme.upper()})')
        axes[i].set_xlabel('x (m)')
        axes[i].set_ylabel('y (m)')
    
    # Add a single colorbar for all plots
    cbar = fig.colorbar(contour, ax=axes, shrink=0.8)
    cbar.set_label('Temperature (°C)')
    
    plt.savefig('temperature_contours_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot temperature profiles at 3 different axial locations in x-direction
    x_locations = [10, 20, 30]  # Axial locations in meters
    
    for x_loc in x_locations:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for scheme in schemes:
            phi, _, _, _, x, y, x_range, y_range = results_dict[scheme]
            
            # Find closest index to the desired x location
            x_idx = np.abs(x_range - x_loc).argmin()
            
            # Extract temperature profile at this x location
            temp_profile = phi[x_idx, :]
            
            # Plot temperature vs y
            ax.plot(temp_profile, y_range, label=f'{scheme.upper()}', color=colors[scheme], linewidth=2)
        
        ax.set_title(f'Temperature Profile at x = {x_loc} m')
        ax.set_xlabel('Temperature (°C)')
        ax.set_ylabel('y (m)')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'temperature_profile_x{x_loc}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot temperature profiles at 3 different y locations
    y_locations = [-0.25, 0, 0.25]  # y locations in meters
    
    for y_loc in y_locations:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        for scheme in schemes:
            phi, _, _, _, x, y, x_range, y_range = results_dict[scheme]
            
            # Find closest index to the desired y location
            y_idx = np.abs(y_range - y_loc).argmin()
            
            # Extract temperature profile at this y location
            temp_profile = phi[:, y_idx]
            
            # Plot temperature vs x
            ax.plot(x_range, temp_profile, label=f'{scheme.upper()}', color=colors[scheme], linewidth=2)
        
        ax.set_title(f'Temperature Profile at y = {y_loc} m')
        ax.set_xlabel('x (m)')
        ax.set_ylabel('Temperature (°C)')
        ax.grid(True)
        ax.legend()
        
        plt.tight_layout()
        plt.savefig(f'temperature_profile_y{y_loc}.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    # Plot bulk mean temperature for all schemes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scheme in schemes:
        _, phi_avg, _, _, _, _, x_range, _ = results_dict[scheme]
        ax.plot(x_range, phi_avg, label=f'{scheme.upper()}', color=colors[scheme], linewidth=2)
    
    ax.set_title('Bulk Mean Temperature vs. Axial Distance')
    ax.set_xlabel('Axial Distance (m)')
    ax.set_ylabel('Bulk Mean Temperature (°C)')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('bulk_mean_temperature_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot heat transfer coefficient for all schemes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scheme in schemes:
        _, _, h, _, _, _, x_range, _ = results_dict[scheme]
        ax.plot(x_range, h, label=f'{scheme.upper()}', color=colors[scheme], linewidth=2)
    
    ax.set_title('Heat Transfer Coefficient vs. Axial Distance')
    ax.set_xlabel('Axial Distance (m)')
    ax.set_ylabel('Heat Transfer Coefficient')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('heat_transfer_coefficient_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot Nusselt number for all schemes
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for scheme in schemes:
        _, _, _, Nu, _, _, x_range, _ = results_dict[scheme]
        ax.plot(x_range, Nu, label=f'{scheme.upper()}', color=colors[scheme], linewidth=2)
    
    ax.set_title('Nusselt Number vs. Axial Distance')
    ax.set_xlabel('Axial Distance (m)')
    ax.set_ylabel('Nusselt Number')
    ax.grid(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig('nusselt_number_comparison.png', dpi=300, bbox_inches='tight')
    plt.show()

# Run the solver with different schemes and plot combined results
if __name__ == "__main__":
    schemes = ['cds', 'upwind', 'hybrid']
    results = {}
    
    for scheme in schemes:
        print(f"\nSolving with {scheme.upper()} scheme...")
        results[scheme] = solve_2d_convection_diffusion(scheme)
    
    print("\nPlotting combined results...")
    plot_combined_results(results)
    
    print("\nAnalysis complete!")