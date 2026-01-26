# Physics-Informed Neural Network (PINN) for 2D Convection–Diffusion PDE

## Overview
This project implements a **Physics-Informed Neural Network (PINN)** using **TensorFlow** to solve a **2D steady-state convection–diffusion partial differential equation** on a unit square domain.  
The model learns the solution by minimizing the governing PDE residual and enforcing boundary conditions, without requiring labeled solution data.

The implementation demonstrates the complete PINN workflow: domain sampling, automatic differentiation–based residual computation, constrained training, and inference on unseen test points.

---

## Features
- Physics-Informed Neural Network (PINN) formulation
- Solves a 2D convection–diffusion PDE on a unit square
- Automatic differentiation using TensorFlow `GradientTape`
- Interior and boundary point sampling
- Dirichlet boundary condition enforcement
- Custom loss combining PDE residual and boundary loss
- CSV-based inference and submission generation

--

## Problem Definition

### Governing Equation
The network approximates the solution \( u(x, y) \) of the PDE:
$$
\[
-\varepsilon (\nabla^2 u) + 2 \frac{\partial u}{\partial x} + 3 \frac{\partial u}{\partial y} = f(x, y)
\]$$

where:
- $$\( \varepsilon \)$$ is a small diffusion coefficient
- \( f(x, y) \) is a known forcing function
- Domain: \( (x, y) \in [0, 1] \times [0, 1] \)

---

### Boundary Condition
\[
u(x, y) = 0 \quad \text{on the boundary}
\]

---

## Forcing Function
The forcing function \( f(x, y) \) is defined as:

$$
\begin{aligned}
f(x, y) =\;& 2\varepsilon\Big(-x + e^{2(x-1)/\varepsilon}\Big) + xy^2 + 6xy - x \, e^{3(y-1)/\varepsilon} \\
&- y^2 \, e^{2(x-1)/\varepsilon} + 2y^2 - 6y \, e^{2(x-1)/\varepsilon} \\
&- 2 \, e^{3(y-1)/\varepsilon} + e^{(2x + 3y - 5)/\varepsilon}
\end{aligned}
$$

---

## System Design

### High-Level Architecture

| Stage | Description |
|------|------------|
| Domain Sampling | Samples interior and boundary points |
| PINN Model | Fully-connected neural network |
| PDE Loss | Enforces governing equation |
| Boundary Loss | Enforces Dirichlet conditions |
| Training Loop | Gradient-based optimization |
| Inference Module | Predicts solution on test data |
| Submission Generator | Outputs CSV predictions |

---

## Component Breakdown

### 1. Domain Sampling
- **Interior points:** Uniform random sampling inside the domain
- **Boundary points:** Uniform sampling along all four edges

N_interior = 4000
N_boundary = 400

## Neural Network Architecture
- Fully-connected feedforward network  
- Activation function: `tanh`  
- Input: `(x, y)`  
- Output: scalar solution `u(x, y)`  

## Physics-Informed Loss Functions

### PDE Residual Loss
- Uses second-order derivatives via automatic differentiation  
- Penalizes violations of the governing PDE  

### Boundary Condition Loss
- Enforces zero Dirichlet boundary condition  
- Mean squared error between predicted and exact boundary values  

### Total Loss
\[
\mathcal{L} = \mathcal{L}_{PDE} + \beta \mathcal{L}_{BC}
\]

## Training Engine
- Optimizer: Adam  
- Learning rate: `0.01`  
- Epochs: `1000`  
- Periodic loss logging  

## Submission Pipeline
- Load test coordinates  
- Predict solution using trained PINN  
- Save predictions to CSV  

---

## Configuration Parameters

| Parameter | Value |
|---------|-------|
| Domain | `[0,1] × [0,1]` |
| Diffusion Coefficient | `eps = 1e-4` |
| Interior Points | `4000` |
| Boundary Points | `400` |
| Epochs | `1000` |
| Learning Rate | `0.01` |
| Boundary Weight | `beta = 1.0` |

---

## Design Principles
- **Physics-guided learning:** No labeled solution data required  
- **Modularity:** Clear separation of PDE, boundary, and training logic  
- **Numerical stability:** Explicit control over diffusion stiffness  
- **Reproducibility:** Deterministic architecture and sampling strategy  

---

## Scalability and Extensions
- Time-dependent PDEs  
- Coupled or multi-physics systems  
- Adaptive sampling near boundary layers  
- Higher-order PDEs  
- GPU acceleration  
- Integration with OT-PINNs or KAN-based PINNs  

---

## Constraints and Limitations
- Steady-state PDE only  
- Fixed rectangular domain  
- Dirichlet boundary conditions only  
- No adaptive mesh refinement  

---

## Dependencies
- Python 3.8+  
- tensorflow  
- numpy  
- pandas  
- matplotlib  

---

## Key Takeaways
- PINNs can solve stiff PDEs without labeled data  
- Automatic differentiation enables exact PDE residual computation  
- Boundary condition enforcement is critical for convergence  
- Neural solvers scale naturally to higher-dimensional problems  

---

## License
This project is intended for **educational and research use**.  
Please cite relevant Physics-Informed Neural Network literature if used in academic work.
