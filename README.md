# Navier-Stokes Singularity Numerical Analysis
### Computational Investigation into Finite-Time Blow-up in 3D Incompressible Flows

This repository contains the numerical implementation and research paper focused on the regularity problem of the 3D Navier-Stokes equations.

## 📄 Research Paper
[Read the full Mathematical Analysis here](./Navier_Stokes_Singularity_Numerical_Research_Hossein_Shafaei.pdf)

## 🔬 Project Overview
The core of this project is a high-precision numerical solver designed to track the growth of maximum vorticity and energy density. By simulating high-energy vortex pair interactions using a fourth-order Runge-Kutta (RK4) scheme, this study identifies a catastrophic breakdown of solution smoothness at $T=0.0010$ seconds.



## 🛠 Technical Specifications
- Core Algorithm: 4th Order Runge-Kutta (RK4) integration method.
- Language: Python (vortex_collision.py).
- Framework: Ultimate Rigor Finite Difference Method (FDM).
- Complexity: 400+ lines of custom-built computational physics logic.

## 📊 Key Findings & Objectives
- Singularity Tracking: Observed velocity escalation from $3.14 \times 10^5$ to $7.84 \times 10^{87}$ before numerical overflow.
- Energy Dynamics: Exponential growth in total strain energy confirming concentration into smaller scales.
- Scientific Context: Providing computational support for the existence of blow-up solutions in the Navier-Stokes equations (Millennium Prize Problem).

---
Author: Hossein Shafaei  
Field: Computational Fluid Dynamics / Mathematical Physics
