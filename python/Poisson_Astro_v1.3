# importing all the libraries I need

import numpy as np
import matplotlib.pyplot as plt


# 1. Constants and numerical parameters

L=3.0 # size of the square domain
G=1 # normalized gravitational constant
tol=1e-5 # tolerance for convergence
max_iter=3000 # maximum number of iterations


# 2. User inputs (all asked up front, before the long calculation)

N=0
while N<3:
    N=int(input("How many nodes do you want? (3 minimum) ")) # number of nodes
M=0
while M<=0:
    M=float(input("Enter the mass of the gas cloud: ")) # mass of the gas cloud
cs=0
while cs<=0:
    cs = float(input("Enter the isothermal sound speed cs of the gas: "))# Thermal sound speed


# 3. Grid, density, source term

h=L/(N-1) # grid step
x=np.linspace(-L/2,L/2,N) # defining the X-axis
y=np.linspace(-L/2,L/2,N) # defining the Y-axis
X,Y=np.meshgrid(x, y) # creating the mesh
r=np.sqrt(X**2+Y**2)

sigma=0.15*L # the amount of space occupied by the gas cloud
rho=(M/(2*np.pi*sigma**2))*np.exp(-r**2/(2*sigma**2)) # gaussian distribution of the mass density
source = 4.0 * np.pi * G * (h**2) * rho # constant, so we compute it outside the loop to save time


# checking the mesh

print("Grid shape:", X.shape)
print("Grid step h:", h)


# 4. Phi solver using SOR (Successive Over-Relaxation)

omega = 2.0 / (1.0 + np.sin(np.pi / (N-1)))
print(f"SOR omega: {omega:.4f}")

# calculation of phi
phi=np.zeros((N,N)) # initializing the potential matrix
for k in range(max_iter):
    norm_two_of_R = 0.0
    R_tot=0.0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            # Standard Gauss-Seidel tentative step
            phi_gs = 0.25 * ((phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]) - source[i, j])
            # SOR over-relaxation step
            phi_new = (1.0 - omega) * phi[i, j] + omega * phi_gs
            phi[i, j] = phi_new
            # residual (used instead of the change per sweep, for a better convergence test)
            R_local=((phi[i+1][j]+phi[i-1][j]+phi[i][j+1]+phi[i][j-1]-4*phi[i][j])-source[i][j])/h**2
            R_tot=R_tot+R_local**2
    norm_two_of_R=np.sqrt(R_tot/(N-2)**2)


    # Report progress every 100 sweeps
    if (k + 1) % 100 == 0:
        print(f"Sweep {k+1:4d} | L2 Residual: {norm_two_of_R:.3e}")

    # Exit automatically once converged
    if norm_two_of_R < tol:
        print(f"\nSuccessfully converged after {k+1} sweeps! (Final residual: {norm_two_of_R:.2e})\n")
        break
else:
    print(f"\nReached maximum iterations ({max_iter}) without full convergence.\n")


# 5. Convergence guard, energies calculations and viral analysis

if k==max_iter: # TODO (item 2): this is never true, because k is at most max_iter-1. Use a "converged" flag instead.
    print("Maximum iterations reached without convergence.") # to prevent useless calculation
else:
    gy, gx = np.gradient(-phi, h, h) # note: np.gradient returns the derivative along axis 0 (y) first, then axis 1 (x), that's why we unpack gy, gx instead of gx, gy

    # --- Gravitational potential energy W ---
    # Formulation 1: from matter density and potential
    W_matter = 0.5 * np.sum(rho * phi) * (h**2)

    # Formulation 2: from the gravitational vector field energy
    g2 = gx**2 + gy**2
    W_field = - (1.0 / (8.0 * np.pi * G)) * np.sum(g2) * (h**2)

    print(f"Gravitational Potential Energy (Matter W): {W_matter:.4f}")
    print(f"Gravitational Field Energy (Field W):     {W_field:.4f}")
    print(f"Gravitational Binding Energy |W_matter|:   {abs(W_matter):.4f}")

    # --- Virial stability analysis ---

    # Thermal kinetic energy: K_th = 3/2 * M * cs^2
    K_th = 1.5 * M * (cs**2)
    W_abs = abs(W_matter)

    # Virial critical sound speed threshold: cs_crit = sqrt(|W| / (3*M))
    cs_crit = np.sqrt(W_abs / (3.0 * M))

    print("ASTROPHYSICAL VIRIAL STABILITY ANALYSIS")
    print(f"Total Cloud Mass M:                  {M:.2f}")
    print(f"Gravitational Binding Energy |W_matter|:    {W_abs:.4f}")
    print(f"Thermal Kinetic Energy 2*K_th:       {2.0 * K_th:.4f}")
    print(f"Virial critical sound speed cs_crit:  {cs_crit:.4f}")
    print(f"Actual Gas Sound Speed cs:           {cs:.4f}")

    # Virial decision (a static criterion, not a dynamical simulation)
    if W_abs > 2.0 * K_th:
        print("GRAVITY DOMINATES (collapse-prone by the virial criterion)")
    elif W_abs < 2.0 * K_th:
        print("NOT SELF-GRAVITY BOUND (PRESSURE-SUPPORTED)")
    else:
        print("VIRIAL EQUILIBRIUM")
    print("=" * 65 + "\n")


    # 6. 2x2 DASHBOARD VISUALIZATION

    fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, figsize=(12, 10))

    # 1. Top-left: density (rho)
    contour_rho = ax_tl.contourf(X, Y, rho, levels=25, cmap='viridis')
    cbar_rho = fig.colorbar(contour_rho, ax=ax_tl)
    cbar_rho.set_label(r'Density $\rho(x, y)$')
    ax_tl.set_title("Mass Density")
    ax_tl.set_xlabel("x")
    ax_tl.set_ylabel("y")
    ax_tl.set_aspect('equal')

    # 2. Top-right: gravitational potential Phi
    contour_phi = ax_tr.contourf(X, Y, phi, levels=25, cmap='inferno')
    cbar_phi = fig.colorbar(contour_phi, ax=ax_tr)
    cbar_phi.set_label(r'Potential $\Phi(x, y)$')
    ax_tr.set_title("Gravitational Potential")
    ax_tr.set_xlabel("x")
    ax_tr.set_ylabel("y")
    ax_tr.set_aspect('equal')

    # 3. Bottom-left: gravitational pull vectors g = -grad Phi
    stride = max(1, N // 15)  # subsample arrows so they don't overlap into a giant dot
    ax_bl.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
                gx[::stride, ::stride], gy[::stride, ::stride], 
                color='navy') # the arrows point in the direction of the gravitational pull
    ax_bl.set_title(r"Gravitational Pull Vectors $\vec{g} = -\nabla\Phi$")
    ax_bl.set_xlabel("x")
    ax_bl.set_ylabel("y")
    ax_bl.set_aspect('equal')

    # 4. Bottom-right: the end result (collapsed protostar or dispersed cloud)
    if W_abs > 2.0 * K_th:
        # Physical collapse: the core shrinks to a compact protostar (e.g. 5x denser and 5x narrower)
        sigma_star = sigma * 0.2
        rho_collapsed = (M / (2.0 * np.pi * sigma_star**2)) * np.exp(-r**2 / (2.0 * sigma_star**2))
        
        cnt_star = ax_br.contourf(X, Y, rho_collapsed, levels=30, cmap='plasma')
        cbar_star = fig.colorbar(cnt_star, ax=ax_br)
        cbar_star.set_label(r'Protostellar Density $\rho_{\text{final}}$')
        ax_br.set_title("Result: Collapsed Protostellar Core")
    else:
        # Physical dispersal: cloud expands outward into space
        sigma_dispersed = sigma * 2.5
        rho_dispersed = (M / (2.0 * np.pi * sigma_dispersed**2)) * np.exp(-r**2 / (2.0 * sigma_dispersed**2))
        
        cnt_disp = ax_br.contourf(X, Y, rho_dispersed, levels=25, cmap='Blues')
        cbar_disp = fig.colorbar(cnt_disp, ax=ax_br)
        cbar_disp.set_label(r'Dispersed Density $\rho_{\text{final}}$')
        ax_br.set_title("Result: Cloud Dispersed into ISM")

    ax_br.set_xlabel("x")
    ax_br.set_ylabel("y")
    ax_br.set_aspect('equal')

    plt.tight_layout()
    plt.show()
