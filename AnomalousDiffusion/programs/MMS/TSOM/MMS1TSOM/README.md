# MMS1TSOM

`MMS1TSOM` is a TSOM adaptation following the MMS program pattern used by
`MMS1BGC`, but with the manufactured solution and two-field system from the
legacy TSOM implementation in `ModelosOld`.

Run target, once CMake has detected the directory:

```bash
NP=1 make run_MMS1TSOM
```

`NP=1` is recommended while this TSOM driver is in development because it uses
the new sequential operator/solver path.

## Input File

The case data are read from `Dados/simulation.dat`. In addition to the MMS
mesh/time controls, this file now contains every TSOM model parameter used by
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

The aliases `alpha_tsom`, `rho_tsom`, `theta_tsom`, `lambdac`, and `lambdar`
are also accepted by the parser, but the documented file uses the canonical
names above.

The `v_boundary` option belongs to the TSOM model and selects how the boundary
condition for `V` is interpreted on both sides of the domain:

```text
zero_value        -> V = 0
constant_gradient -> dV/dx = cte
scaled_psi        -> V = (1 - sc) * Psi
```

The `Psi` boundary condition remains generic. The corresponding boundary
condition for `U` is a consequence of `Psi = U + V`. The current TSOM
implementation has separated coefficient/RHS paths for `zero_value` and
`constant_gradient`. The older alias `zero_gradient` is still accepted and maps
to `constant_gradient`. For `scaled_psi`, the parameter `sc` is the fraction of
`Psi` assigned to `U` at the boundary, so `U = sc*Psi` and
`V = (1 - sc)*Psi`.

## Manufactured Solution

```text
U(x,t)   = exp(-t) * sin(pi*x)
V(x,t)   = exp(-t) * sin(pi*x)
Psi(x,t) = U(x,t) + V(x,t)
```

The TSOM equations are:

```text
U_t + lambdaC*U - lambdaR*V - d11*U_xx - d12*V_xx = S_U
V_t - lambdaC*U + lambdaR*V - d21*U_xx - d22*V_xx = S_V
```

For `g(x)=sin(pi*x)`, `g''(x)=-pi^2*sin(pi*x)`:

```text
S_U = exp(-t) * [(-1 + lambdaC - lambdaR)*g - (d11+d12)*g'']
S_V = exp(-t) * [(-1 - lambdaC + lambdaR)*g - (d21+d22)*g'']
```

Boundary conditions are homogeneous Dirichlet conditions for `Psi`.

## New bgclib TSOM Path

This driver does not use `ModelosOld`, `paper_mms`, `MMSCommon`, `SimConfig`,
`SimState`, or old PETSc assembly functions at runtime.

It uses the new TSOM API:

```text
bgc::models::tsom::Constants
bgc::models::tsom::computeCoefficients
bgc::models::tsom::computeBoundaryRHS
bgc::models::tsom::buildBoundaryRHS
bgc::models::tsom::buildOperator
bgc::models::bgc::solveMMSLinearSystem
bgc::computeLocalTruncationError
```

The flattened unknown vector is ordered as:

```text
[ U_0 ... U_{N-1}  V_0 ... V_{N-1} ]
```

## Output Contract

Outputs are written below the directory selected by `output_dir` in
`Dados/simulation.dat`. Relative paths are resolved from the `MMS1TSOM`
case directory. For example, `output_dir = Caso2` writes:

```text
Caso2/mms1tsom_convergence.csv
Caso2/mms1tsom_fields_N< nx >.dat
Caso2/mms1tsom_lte_N< nx >.dat
```

The setup summary is written in the same directory:

```text
Caso2/mms1tsom_setup.csv
```
