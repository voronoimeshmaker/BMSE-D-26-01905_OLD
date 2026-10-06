# bgclib

`bgclib` is the new C++/PETSc library for one-dimensional anomalous diffusion
models used by the MMS and physical campaign programs in this repository.

The library is intentionally organized around plain data structures and free
functions. Model code produces model-independent descriptions of matrices,
right-hand sides, boundary terms, exact fields, and diagnostics. Generic
numerical modules then assemble PETSc objects, solve linear systems, compute
norms, and write results. This keeps BGC, TBGC, TSOM, and future models from
copying PETSc assembly details into each program.

## Design Goals

- Keep model equations in model modules, not in MMS drivers.
- Keep PETSc allocation and insertion in generic assembly modules.
- Use value types and model tags instead of inheritance-based model classes.
- Make boundary-condition variants explicit parts of a model when they change
  coefficients or source terms.
- Let MMS programs define only the manufactured fields, source formulas, case
  data, and output naming.
- Support MPI execution by assembling distributed PETSc matrices/vectors from
  global model triplets without duplicating rows on every rank.

## Public Modules

### Core

`include/bgclib/Core` contains small value types used across the library:

- `Grid1D`, `TimeConfig`, `BoundaryCondition`, and `BoundarySet` describe the
  mathematical domain and boundary constraints.
- `MatrixLayout`, `SimConfig`, `RunContext`, and solver/diagnostics configs
  describe a run without owning PETSc resources.
- `SimState` owns runtime PETSc handles for programs that use the runtime layer.

These headers should remain lightweight and free of model-specific equations.

### Models

`include/bgclib/Models` and `src/Models` contain model-specific mathematics:

- `BGC`: scalar fourth-order anomalous diffusion model.
- `TBGC`: two-field thermodynamic BGC formulation with `phi` and `mu`.
- `TSOM`: two-field model with `U`, `V`, and `Psi = U + V`.

Each model defines:

- a tag type and `ModelTraits` specialization;
- physical constants;
- coefficient builders;
- boundary source builders when boundary conditions introduce RHS terms;
- operator builders that emit `DiscreteOperator` or model block operators.

Model code must not call `MatSetValue`, `VecSetValue`, or allocate PETSc
objects directly. It should emit plain coefficients and sparse triplets.

### Numerics

`include/bgclib/Numerics` contains model-independent numerical descriptions:

- `StencilRow` stores small local stencil rows.
- `DiscreteOperator` stores sparse matrix triplets in global coordinates.
- `DiscreteRHS` stores sparse vector contributions.
- `Assembly/*` converts those structures into PETSc `Mat` and `Vec` objects.

The assembly layer is responsible for MPI ownership ranges. In parallel, each
rank inserts only the rows it owns.

### Analysis

`include/bgclib/Analysis` contains diagnostics and MMS support:

- finite-volume norms using PETSc `VecNorm`;
- field comparisons;
- cell-centered statistics and centered moments;
- local truncation error;
- exact-solution and manufactured-source wrappers;
- MMS campaign metadata and run summaries.

Individual MMS programs should use these tools instead of embedding their own
norm or truncation-error implementations.

### IO

`include/bgclib/IO` contains small shared IO utilities:

- key/value configuration parsing for `simulation.dat`;
- output path resolution;
- directory creation;
- output file opening with automatic flushing.

The IO layer does not understand model equations. It only parses common formats
and creates file paths/streams.

### Runtime

`include/bgclib/Runtime` is the boundary between compile-time model tags and
runtime program selection. It contains operation tables, a registry, solver
configuration helpers, and transient-run helpers.

This layer is still intentionally thin. MMS programs currently use a more
direct path while the reusable runtime API matures.

## Important Conventions

### Finite-volume grid

Most MMS programs use cell-centered values:

```text
x_i = x0 + (i + 1/2) h,    h = length / nx
```

`Grid1D::dx()` describes nodal spacing for modules that need nodal metadata,
while MMS helpers use `controlVolumeSpacing()` for finite-volume spacing.

### Unknown ordering

Scalar models store one value per control volume:

```text
[ phi_0 ... phi_{N-1} ]
```

Two-field flattened systems use block-by-field ordering:

```text
[ field1_0 ... field1_{N-1}  field2_0 ... field2_{N-1} ]
```

For TBGC this means `[phi ... mu]`. For TSOM this means `[U ... V]`.

### Boundary conditions

Generic boundary conditions use:

```text
alpha * phi + beta * dphi/dx = gamma(t)
```

When a model has extra boundary-condition modes that alter coefficients, those
modes belong to the model constants. TSOM is the current example: the boundary
mode for `V` changes both matrix rows and RHS terms.

### PETSc ownership

Model builders may generate complete global triplet lists because they are
plain data. PETSc assembly routines decide which rank inserts which rows. This
avoids duplicated matrix coefficients in MPI runs.

## Documentation Map

- `docs/ARCHITECTURE.md`: module boundaries and data flow.
- `docs/ADDING_MODELS.md`: steps for adding a new model.
- `docs/REQUIREMENTS.md`: functional and non-functional requirements.

