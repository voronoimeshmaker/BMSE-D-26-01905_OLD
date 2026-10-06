# Requirements

This document records the functional and design requirements that `bgclib`
should preserve as it becomes more generic.

## Functional Requirements

### Models

- The library must support the BGC, TBGC, and TSOM models.
- Each model must expose constants, coefficient builders, RHS builders when
  needed, and operator builders through public headers.
- Models must not depend on old-library classes or old PETSc assembly helpers.
- Model-specific boundary-condition modes must be part of the model API, not
  MMS-program options hidden inside a driver.

### Boundary Conditions

- Generic scalar boundary conditions are represented as:

  ```text
  alpha * phi + beta * dphi/dx = gamma(t)
  ```

- West and east boundaries must be handled independently.
- If a boundary mode changes matrix rows and RHS terms, both changes must be
  implemented in the model coefficient module.
- Boundary formulas must be organized so first-volume and last-volume equations
  can be checked against symbolic derivations.

### Numerical Assembly

- Model code emits `DiscreteOperator` and `DiscreteRHS` data.
- PETSc matrix/vector allocation happens in generic assembly modules.
- MPI runs must not duplicate full matrix insertion on every rank.
- Repeated triplets must be added with PETSc additive insertion.
- Constant matrices should be assembled once per mesh and reused across time
  steps whenever the model/time discretization permits it.

### Solvers

- Linear systems use PETSc `KSP`.
- Serial runs may use direct LU through `KSPPREONLY`/`PCLU`.
- MPI runs must use a solver/preconditioner combination that works with
  distributed matrices unless the user overrides PETSc options.
- Solver reason and iteration count must be recorded in MMS convergence files.

### MMS and Analysis

- MMS drivers must use shared finite-volume norm functions from `Analysis`.
- Norms must use PETSc `VecNorm` internally.
- Local truncation error must be computed by applying the assembled discrete
  operator to exact values and subtracting the discrete source/RHS.
- Convergence CSV files must include mesh size, time step, time-step count,
  field norms, LTE norms, solver reason, iteration count, and status.
- Field output files must include numerical value, exact value, and pointwise
  error.

### IO

- `simulation.dat` uses a small `key = value` format.
- Keys are case-insensitive after normalization to lower case.
- Lists are comma-separated.
- Output directories may be relative to the case directory or absolute.
- Output files opened through `bgc::openOutputFile` flush automatically so long
  runs can be monitored while they execute.

## Non-functional Requirements

### Maintainability

- Prefer explicit plain data over inheritance.
- Keep formulas close to their model.
- Keep PETSc resource ownership in generic runtime/numerics code.
- Avoid duplicated MMS infrastructure across drivers.
- Keep function names precise enough to show whether they compute coefficients,
  assemble PETSc objects, solve systems, or write diagnostics.

### Parallel Behavior

- PETSc communicators should usually be `PETSC_COMM_WORLD` for solve paths.
- Code that writes files should write only once unless the file is intentionally
  rank-specific.
- Collective PETSc operations must be called collectively by all participating
  ranks.
- Programs should report the number of MPI processes used in their run summary.

### Verification

- Boundary rows require special attention in tests and manual review.
- For MMS cases, inspect both global norms and the first/last volume pointwise
  errors.
- Convergence should be assessed separately for `L1`, `L2`, `Linf`, and LTE.
- Regression checks should compare `U`, `V`, `Psi`, `phi`, or `mu` according to
  the model's documented unknown ordering.

### Documentation

- Every public header must state its responsibility.
- Model headers must document constants, unknown ordering, and boundary modes.
- Coefficient source files should document how formulas are split between
  interior rows, boundary rows, and RHS terms.
- MMS drivers should document only case-specific mathematical choices; reusable
  library behavior belongs in `bgclib` documentation.

