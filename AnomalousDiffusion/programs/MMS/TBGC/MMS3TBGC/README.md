# MMS3TBGC

`MMS3TBGC` is the mixed hyperbolic-trigonometric manufactured-solution test for
the new-library TBGC model. It uses only the new `bgclib` API.

Run target:

```bash
make run_MMS3TBGC
```

## Manufactured Solution

```text
phi(x,t) = exp(-t) * [cosh(2 * (x - 1/3)) + 0.2 * sin(pi * x)]
mu(x,t)  = phi(x,t) - Bv * phi_xx(x,t)
```

The source term follows the TBGC/BGC manufactured equation:

```text
source(x,t) = exp(-t) * (-profile - profile_xx + Bv * profile_xxxx)
```

where:

```text
profile      = cosh(2 * (x - 1/3)) + 0.2 * sin(pi*x)
profile_xx   = 4*cosh(2 * (x - 1/3)) - 0.2*pi^2*sin(pi*x)
profile_xxxx = 16*cosh(2 * (x - 1/3)) + 0.2*pi^4*sin(pi*x)
```

Boundary conditions are non-homogeneous and sampled from the manufactured
solution:

```text
west:
  dphi/dx = exp(-t) * [-2*sinh(2/3) + 0.2*pi]
  phi    = exp(-t) * cosh(2/3)

east:
  dphi/dx = exp(-t) * [2*sinh(4/3) - 0.2*pi]
  phi    = exp(-t) * cosh(4/3)
```

## New bgclib Path

This driver does not use `ModelosOld`, `paper_mms`, `MMSCommon`, `SimConfig`,
`SimState`, or the old PETSc assembly functions.

It builds the system through:

```text
bgc::models::tbgc::computeCoefficients
bgc::models::tbgc::buildBlockOperator
bgc::models::tbgc::flattenBlockOperator
bgc::computeLocalTruncationError
bgc::models::bgc::solveMMSLinearSystem
```

The flattened unknown vector is ordered as:

```text
[ phi_0 ... phi_{N-1}  mu_0 ... mu_{N-1} ]
```

At each implicit step:

```text
b_phi = h * source(t_np1) + (h/dt) * phi_n + boundary_b1
b_mu  = boundary_b2
```

## Output

Outputs are written below `Saida/Bv_<value>/`:

```text
mms3tbgc_convergence.csv
mms3tbgc_fields_N< nx >.dat
mms3tbgc_lte_N< nx >.dat
```

The setup summary is:

```text
Saida/mms3tbgc_setup.csv
```
