# MMS3TSOMPsiV

MMS3TSOMPsiV checks discrete mass conservation for the TSOMPsiV model.

The test runs without manufactured source terms and uses

```text
Psi(x,0) = sin^2(pi*x)
V(x,0)   = (1 - f*) * Psi(x,0)
U(x,0)   = f* * Psi(x,0)
f*       = rho / (alpha + rho)
```

For each mesh in `Dados/simulation.dat`, the program writes:

- `mms3tsompsiv_mass.csv`: initial/final mass, final drift, maximum step drift, and
  maximum centered mass-balance residual.
- `mms3tsompsiv_fields_N*.dat`: final `U`, `V`, and `Psi` fields.
- `mms3tsompsiv_setup.csv`: parameters used in the run.

The driver uses only the new `bgclib` TSOMPsiV model, operator assembly, PETSc solve
path, and IO helpers.
