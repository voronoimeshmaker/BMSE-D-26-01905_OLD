# MMS1TSOMPsiV

`MMS1TSOMPsiV` is a TSOMPsiV adaptation following the MMS program pattern used by
`MMS1BGC`, but with the manufactured solution and two-field system from the
legacy TSOMPsiV implementation in `ModelosOld`.

Run target, once CMake has detected the directory:

```bash
NP=1 make run_MMS1TSOMPsiV
```

`NP=1` is recommended while this TSOMPsiV driver is in development because it uses
the new sequential operator/solver path.

## Input File

The case data are read from `Dados/simulation.dat`. In addition to the MMS
mesh/time controls, this file now contains every TSOMPsiV model parameter used by
the driver:

```text
tf
cdt
nx_list
output_dir
verbose
debug
alpha
rho
theta
lambda_c
lambda_r
v_boundary
sc
```

The aliases `alpha_tsompsiv`, `rho_tsompsiv`, `theta_tsompsiv`, `lambdac`, and `lambdar`
are also accepted by the parser, but the documented file uses the canonical
names above.

The `v_boundary` option belongs to the TSOMPsiV model and selects how the boundary
condition for `V` is interpreted on both sides of the domain:

```text
zero_value        -> V = 0
constant_gradient -> dV/dx = cte
scaled_psi        -> V = (1 - sc) * Psi
```

The `Psi` boundary condition remains generic. The TSOMPsiV implementation has
separated coefficient/RHS paths for `zero_value`, `constant_gradient`, and
`scaled_psi`. The older alias `zero_gradient` is still accepted and maps to
`constant_gradient`. For `scaled_psi`, the parameter `sc` follows the Maple
convention `V = (1 - sc)*Psi`.

## Manufactured Solution

```text
Psi(x,t) = exp(-t) * sin(pi*x)
V(x,t)   = exp(-t) * sin(pi*x)
```

The TSOMPsiV equations are:

```text
Psi_t - d11*Psi_xx - d12*V_xx = S_Psi
V_t - lambdaC*Psi + (lambdaC + lambdaR)*V - d21*Psi_xx - d22*V_xx = S_V
```

For `g(x)=sin(pi*x)`, `g''(x)=-pi^2*sin(pi*x)`:

```text
S_Psi = exp(-t) * [-g - (d11+d12)*g'']
S_V   = exp(-t) * [(-1 + lambdaR)*g - (d21+d22)*g'']
```

Boundary conditions are homogeneous Dirichlet conditions for `Psi`.

## New bgclib TSOMPsiV Path

This driver does not use `ModelosOld`, `paper_mms`, `MMSCommon`, `SimConfig`,
`SimState`, or old PETSc assembly functions at runtime.

It uses the new TSOMPsiV API:

```text
bgc::models::tsompsiv::Constants
bgc::models::tsompsiv::computeCoefficients
bgc::models::tsompsiv::computeBoundaryRHS
bgc::models::tsompsiv::buildBoundaryRHS
bgc::models::tsompsiv::buildOperator
bgc::models::bgc::solveMMSLinearSystem
bgc::computeLocalTruncationError
```

The flattened unknown vector is ordered as:

```text
[ U_0 ... U_{N-1}  V_0 ... V_{N-1} ]
```

## Output Contract

Outputs are written below the directory selected by `output_dir` in
`Dados/simulation.dat`. Relative paths are resolved from the `MMS1TSOMPsiV`
case directory. For example, `output_dir = Caso2` writes:

```text
Caso2/mms1tsompsiv_convergence.csv
Caso2/mms1tsompsiv_fields_N< nx >.dat
Caso2/mms1tsompsiv_lte_N< nx >.dat
```

The setup summary is written in the same directory:

```text
Caso2/mms1tsompsiv_setup.csv
```
