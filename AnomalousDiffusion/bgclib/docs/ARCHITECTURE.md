# Architecture

This document describes how `bgclib` is organized and where each kind of code
belongs. The central rule is simple:

```text
models describe equations -> numerics assembles PETSc -> analysis checks results
```

MMS programs and physical campaigns should be thin drivers. They should not own
duplicated PETSc assembly, norm computation, output-path logic, or generic
time-stepping logic.

## Layer Overview

```text
programs/
  MMS and physical drivers
      |
      v
bgclib/Models
  model constants, coefficient formulas, boundary formulas, operator builders
      |
      v
bgclib/Numerics
  DiscreteOperator, DiscreteRHS, PETSc Mat/Vec assembly
      |
      v
PETSc/MPI
  distributed matrices, vectors, KSP, PC

bgclib/Analysis and bgclib/IO are side services used by both programs and models.
```

## Core Data

The `Core` module contains the value types that define a run:

- `Grid1D`: one-dimensional domain metadata.
- `TimeConfig`: fixed time step and time interval metadata.
- `BoundaryCondition`: one scalar boundary equation.
- `BoundarySet`: west/east boundary equations.
- `SimConfig`: model id, grid, time, and boundaries.
- `RunContext`: high-level run configuration bundle.
- `SimState`: PETSc runtime state.

Core types should stay small. They should not parse files, build model
coefficients, allocate PETSc matrices, or write output.

## Model Layer

A model module owns mathematical formulas. For a model named `XYZ`, the expected
layout is:

```text
include/bgclib/Models/XYZ/Model.hpp
include/bgclib/Models/XYZ/Coeff.hpp
include/bgclib/Models/XYZ/Operator.hpp
src/Models/XYZ/Coeff.cpp
src/Models/XYZ/Operator.cpp
```

### `Model.hpp`

Defines:

- the model tag;
- model constants;
- optional model-specific enums and parsing helpers;
- the `ModelTraits` specialization.

This file should be cheap to include and should not contain long coefficient
formulas.

### `Coeff.hpp` and `Coeff.cpp`

Define and compute model coefficients. This is where discretized equations,
boundary closures, and model-specific RHS contributions belong.

For boundary-sensitive models, coefficient functions should be split clearly:

```text
interior coefficients
boundary coefficients for mode A
boundary coefficients for mode B
boundary RHS for mode A
boundary RHS for mode B
```

This makes it possible to compare each branch with Maple derivations and avoids
burying boundary-specific source terms inside MMS programs.

### `Operator.hpp` and `Operator.cpp`

Convert model coefficient data into generic operators. Operator builders emit
`DiscreteOperator` or block operators made of `DiscreteOperator`. They should
not allocate PETSc `Mat` objects.

## Numerics Layer

The numerics layer owns model-independent sparse algebra descriptions:

- `StencilRow`: a compact row representation used by model coefficient code.
- `DiscreteOperator`: global sparse matrix entries.
- `DiscreteRHS`: global sparse vector entries.

The assembly submodule converts those plain structures into PETSc objects:

- `createSequentialMatrix()`: creates a PETSc AIJ matrix from global triplets.
- `createSequentialVector()`: creates a PETSc vector from dense global values.
- `copySequentialVector()`: gathers a PETSc vector into a global `std::vector`.
- RHS helpers: create and update PETSc vectors from `DiscreteRHS`.

The historical names still contain `Sequential` in some APIs, but the current
implementation uses `PETSC_COMM_WORLD` and respects PETSc ownership ranges.
Renaming those functions is a future cleanup, not a model responsibility.

## Analysis Layer

The analysis layer should contain reusable diagnostics:

- finite-volume norms;
- local truncation error;
- field comparison tables;
- statistics, moments, and energy-like quantities;
- MMS campaign metadata.

MMS programs should not implement their own versions of these operations unless
the diagnostic is genuinely case-specific.

## IO Layer

The IO layer owns shared filesystem and configuration behavior:

- read `key = value` files;
- parse scalars, booleans, and comma-separated lists;
- resolve output paths relative to a case directory;
- create output directories;
- open output streams with automatic flushing.

The IO layer should not depend on BGC, TBGC, TSOM, or any future model.

## Runtime Layer

The runtime layer is for programs that choose models at runtime. It maps a
runtime model id to operation tables and solve helpers while preserving the
compile-time model implementation style.

The runtime layer should call public model and numerics APIs. It should not
duplicate coefficient formulas.

## Data Flow in an MMS Run

For a typical MMS case:

1. The program reads `Dados/simulation.dat`.
2. The program builds model constants and manufactured field/source functions.
3. The model builds coefficients from grid, time, constants, and boundaries.
4. The model converts coefficients to a `DiscreteOperator`.
5. The generic solver assembles PETSc matrix/vector objects once per mesh.
6. Each time step updates RHS values and calls `KSPSolve`.
7. Analysis computes exact fields, errors, norms, and local truncation error.
8. IO writes fields, LTE, setup, and convergence files.

## Parallel Execution

Model code may build complete global sparse triplet lists on each rank. PETSc
assembly functions then use ownership ranges:

```text
MatGetOwnershipRange -> insert only owned matrix rows
VecGetOwnershipRange -> insert only owned vector rows
```

This avoids the old failure mode where every MPI rank inserted and solved a
complete duplicate of the sequential problem.

## Boundary-condition Ownership

Generic boundary conditions are represented by `BoundaryCondition`, but any
model-specific interpretation belongs to that model. For example, TSOM supports:

```text
V = 0
dV/dx = cte
V = (1 - sc) * Psi
```

Those branches are implemented inside `Models/TSOM/Coeff.cpp` because each
branch can change both matrix coefficients and RHS terms.

