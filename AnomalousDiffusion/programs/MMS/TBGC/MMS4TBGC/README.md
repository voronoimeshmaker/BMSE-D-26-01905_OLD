# MMS4TBGC

`MMS4TBGC` is a physical Gaussian-pulse spreading campaign for the new-library
TBGC model. It is not a manufactured-solution convergence test: there is no
source term and no analytical final-time reference.

Run target:

```bash
make run_MMS4TBGC
```

## Initial Condition

```text
phi(x,0) = [1/(sigma0*sqrt(2*pi))] * exp[-(x-mu)^2/(2*sigma0^2)]
mu       = 0.5
sigma0   = 0.04
```

The TBGC auxiliary field is initialized analytically from the same pulse:

```text
phi_xx(x,0) = phi(x,0) * [((x-mu)^2 / sigma0^4) - (1 / sigma0^2)]
mu(x,0)     = phi(x,0) - Bv * phi_xx(x,0)
```

Boundary conditions are homogeneous:

```text
dphi/dx = 0
phi    = 0
```

## New bgclib Path

This driver does not use `ModelosOld`, `paper_mms`, `MMSCommon`, `SimConfig`,
`SimState`, or the old PETSc assembly functions.

It builds the system through:

```text
bgc::models::tbgc::computeCoefficients
bgc::models::tbgc::buildBlockOperator
bgc::models::tbgc::flattenBlockOperator
bgc::models::bgc::solveMMSLinearSystem
```

The flattened unknown vector is ordered as:

```text
[ phi_0 ... phi_{N-1}  mu_0 ... mu_{N-1} ]
```

With zero source, each implicit step uses:

```text
b_phi = (h/dt) * phi_n + boundary_b1
b_mu  = boundary_b2
```

## Diagnostics

The time series reports:

```text
step tau sigma2 sigma2_raw F_h mass phi_min phi_max phi_bdyW phi_bdyE
```

where `sigma2` is the mass-normalized centered second moment.

## Output

Outputs are written below `Saida/Bv_<value>/N<nx>/`:

```text
mms4tbgc_timeseries.dat
mms4tbgc_profile_0.dat
mms4tbgc_profile_1.dat
mms4tbgc_profile_2.dat
mms4tbgc_profile_3.dat
mms4tbgc_profile_4.dat
```
