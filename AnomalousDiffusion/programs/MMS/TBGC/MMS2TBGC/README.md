# MMS2TBGC

`MMS2TBGC` is the non-homogeneous-boundary manufactured-solution test for the
new-library TBGC model. It replaces the old TBGCS-style driver and intentionally
uses only the new `bgclib` API.

The executable target is created from this directory name, so the run target is:

```bash
make run_MMS2TBGC
```

## Manufactured Solution

The prescribed field is

```text
phi(x,t) = exp(-t) * cosh(2 * (x - 1/3))
```

The auxiliary TBGC field is

```text
mu(x,t) = phi(x,t) - Bv * phi_xx(x,t)
        = (1 - 4*Bv) * phi(x,t)
```

The source term is

```text
source(x,t) = (16*Bv - 5) * phi(x,t)
```

Boundary conditions are non-homogeneous and sampled from the manufactured
solution:

```text
west:
  dphi/dx = -2 * exp(-t) * sinh(2/3)
  phi    =  exp(-t) * cosh(2/3)

east:
  dphi/dx =  2 * exp(-t) * sinh(4/3)
  phi    =  exp(-t) * cosh(4/3)
```

## New bgclib Path

This program does not use `ModelosOld`, `paper_mms`, `MMSCommon`, `SimConfig`,
`SimState`, or the old PETSc assembly functions.

The driver builds the TBGC system through:

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

At each implicit step, the right-hand side is assembled as:

```text
b_phi = h * source(t_np1) + (h/dt) * phi_n + boundary_b1
b_mu  = boundary_b2
```

## Input

The input file is `Dados/simulation.dat`.

Supported keys:

```text
tf
cdt
nx_list
bv_list
output_dir
verbose
debug
```

The time step is computed as in the BGC MMS programs:

```text
h       = 1 / nx
dt_raw  = cdt * h^2
nTimes  = round(tf / dt_raw), at least 1
dt      = tf / nTimes
```

## Output

Outputs are written below `Saida/Bv_<value>/`.

Per `Bv`, the convergence table is:

```text
mms2tbgc_convergence.csv
```

Per mesh, the field comparison files are:

```text
mms2tbgc_fields_N< nx >.dat
```

Columns:

```text
P x phi_num phi_exact phi_err mu_num mu_exact mu_err
```

Per mesh, the local truncation-error files are:

```text
mms2tbgc_lte_N< nx >.dat
```

Columns:

```text
P x_center tau_phi tau_mu
```

The setup summary is:

```text
Saida/mms2tbgc_setup.csv
```
