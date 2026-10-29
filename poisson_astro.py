#importing all the libraries i need

import numpy as np
import matplotlib.pyplot as plt

#setting up all the ctes and vectros, matrices i need

L=3.0
N=int(input("How many nodes do you want?"))
h=L/(N-1)
x=np.linspace(-L/2,L/2,N)
y=np.linspace(-L/2,L/2,N)
X,Y=np.meshgrid(x, y)
G=1 #normalized gravitational cte
M=float(input("Enter the mass of the gas cloud (e.g., 1.0):"))
sigma=0.15*L
r=np.sqrt(X**2+Y**2)
rho=(M/(2*np.pi*sigma**2))*np.exp(-r**2/(2*sigma**2)) #gaussian distribution of the mass density
tol=1e-5 #tolerance for convergence
max_iter=3000 #maximum number of iterations
source = 4.0 * np.pi * G * (h**2) * rho #cte so we put it outside the loop to save time

#checking the mesh

print("Grid shape:", X.shape)
print("Grid step h:", h)

#calulation of phi
phi=np.zeros((N,N)) #initializing the potential matrix
for k in range(max_iter):
    max_diff = 0.0
    for i in range(1, N - 1):
        for j in range(1, N - 1):
            phi_new = 0.25 * ((phi[i+1, j] + phi[i-1, j] + phi[i, j+1] + phi[i, j-1]) - source[i, j])
            diff = abs(phi_new - phi[i, j])
            if diff > max_diff:
                max_diff = diff
            phi[i, j] = phi_new

    # Report progress every 100 sweeps
    if (k + 1) % 100 == 0:
        print(f"Sweep {k+1:4d} | Max Change: {max_diff:.3e}")

    # Exit automatically once converged
    if max_diff < tol:
        print(f"\nSuccessfully converged after {k+1} sweeps! (Final residual: {max_diff:.2e})\n")
        break
else:
    print(f"\nReached maximum iterations ({max_iter}) without full convergence.\n")

gy, gx = np.gradient(-phi, h, h) #note: this fct calc the derivative with respect to y then x, that's why we wrote gy, gx instead of gx, gy

# --- Gravitational Potential Energy (Binding Energy) ---
# Formulation 1: From matter density and potential
W_matter = 0.5 * np.sum(rho * phi) * (h**2)

# Formulation 2: From the gravitational vector field energy
g2 = gx**2 + gy**2
W_field = - (1.0 / (8.0 * np.pi * G)) * np.sum(g2) * (h**2)

print(f"Gravitational Potential Energy (Matter W): {W_matter:.4f}")
print(f"Gravitational Field Energy (Field W):     {W_field:.4f}")
print(f"Total Gravitational Binding Energy |W|:   {abs(W_matter):.4f}")

# --- Step 3: Virial Theorem & Jeans Collapse Analysis ---
# 1. Thermal sound speed (proxy for gas temperature: cs^2 ~ T)
cs = float(input("Enter the isothermal sound speed cs of the gas:"))

# 2. Thermal Kinetic Energy: K_th = 3/2 * M * cs^2
K_th = 1.5 * M * (cs**2)
W_abs = abs(W_matter)

# 3. Critical Jeans sound speed threshold: cs_crit = sqrt(|W| / (3*M))
cs_crit = np.sqrt(W_abs / (3.0 * M))

print("ASTROPHYSICAL VIRIAL STABILITY ANALYSIS")
print(f"Total Cloud Mass M:                  {M:.2f}")
print(f"Gravitational Binding Energy |W|:    {W_abs:.4f}")
print(f"Thermal Kinetic Energy 2*K_th:       {2.0 * K_th:.4f}")
print(f"Critical Jeans Sound Speed cs_crit:  {cs_crit:.4f}")
print(f"Actual Gas Sound Speed cs:           {cs:.4f}")

# 4. The Dynamical Virial Decision
if W_abs > 2.0 * K_th:
    print("GRAVITATIONAL COLLAPSE")
elif W_abs < 2.0 * K_th:
    print("THERMAL DISPERSAL")
else:
    print("VIRIAL EQUILIBRIUM")
print("=" * 65 + "\n")

# --- 2x2 Dashboard Visualization ---
fig, ((ax_tl, ax_tr), (ax_bl, ax_br)) = plt.subplots(2, 2, figsize=(12, 10))

# 1. Top-Left:Density (rho)
contour_rho = ax_tl.contourf(X, Y, rho, levels=25, cmap='viridis')
cbar_rho = fig.colorbar(contour_rho, ax=ax_tl)
cbar_rho.set_label(r'Density $\rho(x, y)$')
ax_tl.set_title("Mass Density")
ax_tl.set_xlabel("x")
ax_tl.set_ylabel("y")
ax_tl.set_aspect('equal')

# 2. Top-Right: Gravitational Potential Phi
contour_phi = ax_tr.contourf(X, Y, phi, levels=25, cmap='inferno')
cbar_phi = fig.colorbar(contour_phi, ax=ax_tr)
cbar_phi.set_label(r'Potential $\Phi(x, y)$')
ax_tr.set_title("Gravitational Potential")
ax_tr.set_xlabel("x")
ax_tr.set_ylabel("y")
ax_tr.set_aspect('equal')

# 3. Bottom-Left: grad Phi (Gravitational Pull Vectors)
stride = max(1, N // 15)  # Subsample arrows so they don't overlap into a giant dot
ax_bl.quiver(X[::stride, ::stride], Y[::stride, ::stride], 
              gx[::stride, ::stride], gy[::stride, ::stride], 
              color='navy') # this makes arrows that points towards the direction of the gravitational pull
ax_bl.set_title(r"Gravitational Pull Vectors $\vec{g} = -\nabla\Phi$")
ax_bl.set_xlabel("x")
ax_bl.set_ylabel("y")
ax_bl.set_aspect('equal')

# 4. Bottom-Right: The End Result (Collapsed Protostar or Dispersed Cloud)
if W_abs > 2.0 * K_th:
    # Physical Collapse: The core shrinks to a compact protostar (e.g. 5x denser and 5x narrower)
    sigma_star = sigma * 0.2
    rho_collapsed = (M / (2.0 * np.pi * sigma_star**2)) * np.exp(-r**2 / (2.0 * sigma_star**2))
    
    cnt_star = ax_br.contourf(X, Y, rho_collapsed, levels=30, cmap='plasma')
    cbar_star = fig.colorbar(cnt_star, ax=ax_br)
    cbar_star.set_label(r'Protostellar Density $\rho_{\text{final}}$')
    ax_br.set_title("Result: Collapsed Protostellar Core")
else:
    # Physical Dispersal: Cloud expands outward into space
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