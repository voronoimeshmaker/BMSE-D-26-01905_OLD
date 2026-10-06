# MMS4TSOMPsiV

MMS4TSOMPsiV records the TSOMPsiV free-energy history for the three splitting
parameters requested in the verification text.

The initial condition is

```text
Psi(x,0) = 1 + sin(pi*x)
V(x,0)   = sin^2(pi*x/2)
U(x,0)   = Psi(x,0) - V(x,0)
```

The discrete free energy written by the program is

```text
F = h/2 * sum_i ( U_i^2 + eta * V_i^2 )
eta = rho / alpha
```

For each `theta` in `theta_list`, the program writes:

- `mms4tsompsiv_energy_history.csv`: `theta`, time step, time, energy, energy
  increment, and mass.
- `mms4tsompsiv_energy_summary.csv`: initial/final energy, total change, minimum and
  maximum increment, number of positive increments, and mass drift.

The default boundary mode is `scaled_psi` with `sc = rho/(alpha+rho)`, matching
`V = (1 - sc) * Psi`.
