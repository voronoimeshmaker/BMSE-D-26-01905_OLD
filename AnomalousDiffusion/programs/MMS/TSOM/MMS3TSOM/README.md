# MMS3TSOM

MMS3TSOM checks discrete mass conservation for the TSOM model.

The test runs without manufactured source terms and uses

```text
Psi(x,0) = sin^2(pi*x)
V(x,0)   = (1 - f*) * Psi(x,0)
U(x,0)   = f* * Psi(x,0)
f*       = rho / (alpha + rho)
```

For each mesh in `Dados/simulation.dat`, the program writes:

- `mms3tsom_mass.csv`: initial/final mass, final drift, maximum step drift, and
  maximum centered mass-balance residual.
- `mms3tsom_fields_N*.dat`: final `U`, `V`, and `Psi` fields.
- `mms3tsom_setup.csv`: parameters used in the run.

The driver uses only the new `bgclib` TSOM model, operator assembly, PETSc solve
path, and IO helpers.
