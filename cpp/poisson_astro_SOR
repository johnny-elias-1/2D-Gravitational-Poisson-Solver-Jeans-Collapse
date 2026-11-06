//importing all necessary libraries
#include <iostream>  //for std::cout and std::cin
#include <vector>    //for std::vector arrays
#include <cmath>     //for std::sqrt, std::exp, std::abs, std::sin

const double pi = 3.14159265358979323846;

//main fct
int main(){

    //setting up all ctes and parameters
    double L = 3.0; // Length of the domain
    int N; std::cin>> N; // Number of nodes in the grid
    double h = L/(N-1); //grid step
    
    std::vector<double> x(N);
    double dx = L / (N - 1);
    for (int j = 0; j < N; j++) {
        x[j] = -L / 2.0 + j * dx;
    }
    
    std::vector<double> y(N);
    double dy = L / (N - 1);
    for (int i = 0; i < N; i++) {
        y[i] = -L / 2.0 + i * dy;
    }
    
    int G = 1; // normalized gravitational constant
    double M; std::cin >> M; // mass of the gas cloud
    double sigma = 0.15 * L; // the amount of space occupied by the gas cloud
    double tol = 1e-5; // tolerance for convergence
    int max_iter = 30000; // maximum number of iterations

    // Create flat 1D vectors for density (rho) and source term
    std::vector<double> rho(N * N, 0.0);
    std::vector<double> source(N * N, 0.0);

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            double r = std::sqrt(x[j] * x[j] + y[i] * y[i]);
            double density = (M / (2.0 * pi * sigma * sigma)) * std::exp(-(r * r) / (2.0 * sigma * sigma));
            
            rho[i * N + j] = density;
            source[i * N + j] = 4.0 * pi * G * density;
        }
    }

    //checking mesh and step size
    std::cout<< "Grid shape: " << N*N << std::endl;
    std::cout<< "Grid step: " << h << std::endl;

    //SOR factor (using float division)
    double omega = 2.0 / (1.0 + std::sin(pi / N));
    std::cout<< "SOR factor: " << omega << std::endl;

    //calculation of phi
    std::vector<double> phi(N * N, 0.0);
    bool converged = false;

    for (int k = 1; k <= max_iter; k++) {
        double max_diff = 0.0; // variable to track the maximum difference in phi values
        
        for (int i = 1; i < N - 1; i++) {
             for (int j = 1; j < N - 1; j++) {
                // Gauss-Seidel part using flat 1D vector indexing & source term lookup
                double phi_gs = (1.0 / 4.0) * (
                    phi[(i + 1) * N + j] + 
                    phi[(i - 1) * N + j] + 
                    phi[i * N + (j + 1)] + 
                    phi[i * N + (j - 1)] - 
                    source[i * N + j]
                );
                
                // SOR part
                double phi_new = (1.0 - omega) * phi[i * N + j] + omega * phi_gs;
                
                // calculating the difference between the new and old phi values
                double diff = std::abs(phi_new - phi[i * N + j]);
                if (diff > max_diff) {
                    max_diff = diff;
                }
                
                phi[i * N + j] = phi_new; // update phi with the new value
            }
        }

        // Report progress every 100 sweeps (Properly inside the loop)
        if (k % 100 == 0) {
            std::cout << "Sweep " << k << " | Max Change: " << max_diff << std::endl;
        }

        // Exit automatically once converged (Properly inside the loop)
        if (max_diff < tol) {
            std::cout << "\nSuccessfully converged after " << k << " sweeps! (Final residual: " << max_diff << ")\n\n";
            converged = true;
            break;
        }
    } // closes the max_iter for loop

    if (!converged) {
        std::cout << "\nReached maximum iterations (" << max_iter << ") without full convergence.\n\n";
    }

    // --- Gravitational Potential Energy (Binding Energy) ---
    double sum_rho_phi = 0.0;
    double sum_g2 = 0.0;
    
    for (int i = 1; i < N - 1; i++) {
        for (int j = 1; j < N - 1; j++) {
            int idx = i * N + j;
            
            // Finite difference gradients for gx and gy (g = -nabla Phi)
            double d_phi_dx = (phi[i * N + (j + 1)] - phi[i * N + (j - 1)]) / (2.0 * h);
            double d_phi_dy = (phi[(i + 1) * N + j] - phi[(i - 1) * N + j]) / (2.0 * h);
            double g2 = (d_phi_dx * d_phi_dx) + (d_phi_dy * d_phi_dy);
            
            sum_g2 += g2;
            sum_rho_phi += rho[idx] * phi[idx]; // Enabled with the saved rho vector
        }
    }

    // Formulation from matter density and potential
    double W_matter = 0.5 * sum_rho_phi * (h * h);
    double W_field = - (1.0 / (8.0 * pi * G)) * sum_g2 * (h * h);
    double W_abs = std::abs(W_matter);

    std::cout << "Gravitational Potential Energy (Matter W): " << W_matter << std::endl;
    std::cout << "Gravitational Field Energy (Field W):      " << W_field << std::endl;
    std::cout << "Total Gravitational Binding Energy |W|:  " << W_abs << std::endl;

    // --- Step 3: Virial Theorem & Jeans Collapse Analysis ---
    double cs;
    std::cout << "Enter the isothermal sound speed cs of the gas: ";
    std::cin >> cs;

    double K_th = 1.5 * M * (cs * cs);
    double cs_crit = std::sqrt(W_abs / (3.0 * M));

    std::cout << "\nASTROPHYSICAL VIRIAL STABILITY ANALYSIS\n";
    std::cout << "Total Cloud Mass M:                  " << M << "\n";
    std::cout << "Gravitational Binding Energy |W|:    " << W_abs << "\n";
    std::cout << "Thermal Kinetic Energy 2*K_th:       " << (2.0 * K_th) << "\n";
    std::cout << "Critical Jeans Sound Speed cs_crit:  " << cs_crit << "\n";
    std::cout << "Actual Gas Sound Speed cs:           " << cs << "\n";

    // Dynamical Virial Decision
    if (W_abs > 2.0 * K_th) {
        std::cout << "RESULT: GRAVITATIONAL COLLAPSE\n";
    } else if (W_abs < 2.0 * K_th) {
        std::cout << "RESULT: THERMAL DISPERSAL\n";
    } else {
        std::cout << "RESULT: VIRIAL EQUILIBRIUM\n";
    }

    return 0;
}
